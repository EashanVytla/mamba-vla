from __future__ import annotations

import copy
import itertools
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Iterator, List, Optional, Sequence, TYPE_CHECKING
from omegaconf import DictConfig

from torch.utils.data import DataLoader, DistributedSampler

from vla_scratch.transforms.base import TransformedDataset
from vla_scratch.transforms.common import ToDataSample
from vla_scratch.transforms.normalization import DeNormalize, Normalize, load_norm_stats
from vla_scratch.utils.config import locate_class

if TYPE_CHECKING:
    from vla_scratch.datasets.config import DataConfig
    from vla_scratch.policies.config import PolicyConfig
    from vla_scratch.transforms.base import TransformFn


def instantiate_transform(spec: Any) -> Any:
    """Instantiate a transform object from a spec.

    Accepts either an existing object with a `.compute()` method, or a dict with
    a `_target_` key and constructor kwargs. Returns an object exposing
    `.compute(sample) -> Any`.
    """
    # Already a transform-like object
    if hasattr(spec, "compute") and callable(getattr(spec, "compute")):
        return spec

    # Config mapping
    if isinstance(spec, dict) or isinstance(spec, DictConfig):
        target = spec.get("_target_")
        if target is None:
            raise ValueError("Transform configuration must define '_target_'.")
        kwargs = {k: v for k, v in spec.items() if k != "_target_"}
        cls = locate_class(target)
        obj = cls(**kwargs)
        if hasattr(obj, "compute") and callable(getattr(obj, "compute")):
            return obj
        raise TypeError(f"Instance of '{target}' does not expose a 'compute' method.")

    raise TypeError(f"Unsupported transform specification: {spec!r}")


def make_transforms(specs: Sequence[Any]) -> List["TransformFn"]:
    """Instantiate transform specs into concrete transform objects."""
    return [instantiate_transform(spec) for spec in specs]


def build_input_transforms(
    data_cfg: "DataConfig",
    policy_cfg: "PolicyConfig",
    *,
    add_noise: bool = False,
) -> Sequence["TransformFn"]:
    dataset_tfs = make_transforms(data_cfg.input_transforms)

    norm_tf: List["TransformFn"] = []
    if data_cfg.norm_stats_path is not None:
        stats = load_norm_stats(data_cfg, policy_cfg)
        norm_tf = [
            Normalize(
                norm_stats=stats,
                strict=False,
                noise_cfg=data_cfg.noise_cfg,
                enable_aug=add_noise,
            )
        ]

    policy_tfs = make_transforms(policy_cfg.transforms)
    return dataset_tfs + norm_tf + [ToDataSample()] + policy_tfs


def build_output_transforms(
    data_cfg: "DataConfig",
    policy_cfg: "PolicyConfig",
) -> Sequence["TransformFn"]:
    denorm_tf: List["TransformFn"] = []
    if data_cfg.norm_stats_path is not None:
        stats = load_norm_stats(data_cfg, policy_cfg)
        denorm_tf = [DeNormalize(norm_stats=stats, strict=False)]

    inv_tfs = make_transforms(data_cfg.output_inv_transforms)

    from vla_scratch.transforms.common import ToNumpy

    return denorm_tf + inv_tfs + [ToNumpy()]


def create_dataset(
    data_cfg: "DataConfig",
    policy_cfg: "PolicyConfig",
    *,
    skip_norm_stats: bool = False,
    skip_policy_transforms: bool = False,
    add_noise: bool = False,
) -> TransformedDataset:
    """Create a dataset pipeline applying configured transforms."""
    input_tfs = data_cfg.input_transforms
    output_tfs = data_cfg.output_transforms
    dataset_tfs = make_transforms(list(input_tfs) + list(output_tfs))

    norm_tf: List["TransformFn"] = []
    if (not skip_norm_stats) and data_cfg.norm_stats_path is not None:
        stats = load_norm_stats(data_cfg, policy_cfg)
        norm_tf = [
            Normalize(
                norm_stats=stats,
                noise_cfg=data_cfg.noise_cfg,
                enable_aug=add_noise,
            )
        ]

    policy_tfs = (
        make_transforms(policy_cfg.transforms) if not skip_policy_transforms else []
    )

    pipeline = dataset_tfs + norm_tf + [ToDataSample()] + policy_tfs

    base_dataset = data_cfg.instantiate()
    return TransformedDataset(base_dataset, pipeline)


class PrefetchingEpochIterator(Iterator[Iterator]):
    """Prefetch the next epoch's first batch while the current epoch runs."""

    def __init__(
        self,
        # iterator_fn: Callable[[int], Iterator],
        dataloader: DataLoader,
        num_epochs: int,
        *,
        max_workers: int = 1,
    ):
        # self.iterator_fn = iterator_fn
        self.dataloader = dataloader
        self.num_epochs = num_epochs
        self.epoch_idx = 0
        self.current_iter: Optional[Iterator] = None

        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.prefetched_iter: Optional[Iterator] = None

        self._submit_prefetch(0)

    def __iter__(self) -> "PrefetchingEpochIterator":
        return self

    def __next__(self) -> Iterator:
        if self.epoch_idx >= self.num_epochs:
            raise StopIteration

        self._cleanup_prev_iter()

        prefetched_first_batch = self.prefetch_batch_future.result()
        self.current_iter = self.prefetched_iter

        next_epoch = self.epoch_idx + 1
        self._submit_prefetch(next_epoch)
        self.epoch_idx += 1

        return itertools.chain((prefetched_first_batch,), self.current_iter)

    def _submit_prefetch(self, epoch_idx: int):
        if epoch_idx >= self.num_epochs:
            return
        dataloader = copy.copy(self.dataloader)
        if isinstance(dataloader.sampler, DistributedSampler):
            dataloader.sampler.set_epoch(epoch_idx)
        self.prefetched_iter = iter(dataloader)
        # return self.executor.submit(
        #     lambda: next(self.prefetched_iter),
        # )
        self.prefetch_batch_future = self.executor.submit(
            lambda iter: next(iter),
            self.prefetched_iter,
        )

    def _cleanup_prev_iter(self, final=False) -> None:
        if self.dataloader.persistent_workers and not final:
            # do not shutdown workers if persistent
            return
        if self.current_iter is not None and hasattr(self.current_iter, "_shutdown_workers"):
            self.current_iter._shutdown_workers()  # type: ignore[attr-defined]
        self.current_iter = None

    def finalize(self) -> None:
        self._cleanup_prev_iter()


class EagerEpochIterator(Iterator[Iterator]):
    """Create a fresh iterator each epoch without prefetching."""

    def __init__(
        self,
        dataloader: DataLoader,
        num_epochs: int,
    ):
        self.dataloader = dataloader
        self.num_epochs = num_epochs
        self.epoch_idx = 0
        self.current_iter: Optional[Iterator] = None

    def __iter__(self) -> "EagerEpochIterator":
        return self

    def __next__(self) -> Iterator:
        if self.epoch_idx >= self.num_epochs:
            raise StopIteration

        self._cleanup_prev_iter()
        dataloader = self.dataloader
        if isinstance(dataloader.sampler, DistributedSampler):
            dataloader.sampler.set_epoch(self.epoch_idx)
        iterator = iter(dataloader)
        self.current_iter = iterator
        self.epoch_idx += 1
        return iterator

    def _cleanup_prev_iter(self, final=False) -> None:
        if self.dataloader.persistent_workers and not final:
            # do not shutdown workers if persistent
            return
        if self.current_iter is not None and hasattr(self.current_iter, "_shutdown_workers"):
            self.current_iter._shutdown_workers()  # type: ignore[attr-defined]
        self.current_iter = None

    def finalize(self) -> None:
        self._cleanup_prev_iter()
