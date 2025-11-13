import torch
from typing import Dict


from vla_scratch.transforms.base import TransformFn
from vla_scratch.transforms.data_keys import (
    PROCESSED_ACTION_KEY,
    PROCESSED_IMAGE_KEY,
    PROCESSED_IMAGE_MASK_KEY,
    PROCESSED_STATE_KEY,
)
from vla_scratch.datasets.libero.common import (
    ACTION_KEY,
    FUTURE_STATE_KEY,
    STATE_KEY,
    IMAGE_KEY,
    WRIST_IMAGE_KEY,
)
from vla_scratch.utils.math import (
    matrix_from_quat,
    quat_apply_inverse,
    quat_conjugate,
    quat_from_angle_axis,
    quat_mul,
    unscale_transform,
    quat_from_matrix,
    axis_angle_from_quat,
    quat_apply,
    rotation_matrix_to_6d,
    rotation_6d_to_matrix,
)


class LiberoState(TransformFn):
    """Convert history of Libero states into relative 6D pose deltas."""

    def compute(self, sample: Dict) -> Dict:
        state_seq: torch.Tensor = sample[STATE_KEY]
        history = state_seq.shape[0] - 1

        pos_w = state_seq[:, 0:3]
        rotvec_seq = state_seq[:, 3:6]
        angle = torch.linalg.norm(rotvec_seq, dim=-1)
        quat_w = quat_from_angle_axis(angle, rotvec_seq)

        current_pos_w = pos_w[-1]
        current_quat_w = quat_w[-1]

        history_pos_w = pos_w[:history]
        current_quat_w_hist = current_quat_w.unsqueeze(0).expand(history, -1)
        history_dpos = quat_apply_inverse(
            current_quat_w_hist, history_pos_w - current_pos_w
        )

        history_quat_w = quat_w[:history]
        history_dquat = quat_mul(quat_conjugate(current_quat_w_hist), history_quat_w)
        history_drotmat = matrix_from_quat(history_dquat)
        history_dori6d = rotation_matrix_to_6d(history_drotmat)

        history_grippers = state_seq[:history, 6:8]
        state_vec = torch.cat([history_dpos, history_dori6d, history_grippers], dim=-1)
        sample[PROCESSED_STATE_KEY] = state_vec
        return sample


class LiberoGlobalState(TransformFn):
    """Convert history of Libero states into global 6D pose."""

    def compute(self, sample: Dict) -> Dict:
        state_seq: torch.Tensor = sample[STATE_KEY][1:]  # skip most old state

        pos_w = state_seq[:, 0:3]

        rotvec_seq = state_seq[:, 3:6]
        angle = torch.linalg.norm(rotvec_seq, dim=-1)
        quat_w = quat_from_angle_axis(angle, rotvec_seq)
        ori6d_w = rotation_matrix_to_6d(matrix_from_quat(quat_w))

        grippers = state_seq[:, 6:8]

        state_vec = torch.cat([pos_w, ori6d_w, grippers], dim=-1)
        sample[PROCESSED_STATE_KEY] = state_vec
        return sample


class LiberoAction(TransformFn):
    """
    Convert Libero actions into relative 6D pose deltas.
    """

    actions_low = torch.tensor([-0.05, -0.05, -0.05, -0.5, -0.5, -0.5, -1.0])
    actions_high = torch.tensor([0.05, 0.05, 0.05, 0.5, 0.5, 0.5, 1.0])

    def __init__(self, eef_frame: bool = True) -> None:
        self.eef_frame = eef_frame

    def compute(self, sample: Dict) -> Dict:
        future_states: torch.Tensor = sample[FUTURE_STATE_KEY]
        actions_raw: torch.Tensor = sample[ACTION_KEY]
        actions = unscale_transform(
            actions_raw,
            self.actions_low.to(actions_raw.device, actions_raw.dtype),
            self.actions_high.to(actions_raw.device, actions_raw.dtype),
        )
        horizon = actions.shape[0]

        future_pos_w = future_states[:, 0:3]
        future_rotvec_w = future_states[:, 3:6]
        angle = torch.linalg.norm(future_rotvec_w, dim=-1)
        future_quat_w = quat_from_angle_axis(angle, future_rotvec_w)

        cmd_dpos = actions[:, 0:3]
        cmd_drotvec = actions[:, 3:6]
        cmd_dangle = torch.linalg.norm(cmd_drotvec, dim=-1)
        cmd_dquat = quat_from_angle_axis(cmd_dangle, cmd_drotvec)

        cmd_grippers = actions[:, 6:7]

        target_pos_w = future_pos_w + cmd_dpos
        target_quat_w = quat_mul(cmd_dquat, future_quat_w)

        if self.eef_frame:
            current_pos_w = future_pos_w[-horizon]
            current_quat_w = future_quat_w[-horizon]
            current_quat_w_expand = current_quat_w.unsqueeze(0).expand_as(target_quat_w)

            target_rel_pos = quat_apply_inverse(
                current_quat_w_expand, target_pos_w - current_pos_w
            )
            target_dquat = quat_mul(
                quat_conjugate(current_quat_w_expand), target_quat_w
            )
            target_drotmat = matrix_from_quat(target_dquat)
            target_rel_ori6d = rotation_matrix_to_6d(target_drotmat)

            actions_out = torch.cat(
                [target_rel_pos, target_rel_ori6d, cmd_grippers], dim=-1
            )
        else:
            target_dori6d = rotation_matrix_to_6d(matrix_from_quat(target_quat_w))
            actions_out = torch.cat([target_pos_w, target_dori6d, cmd_grippers], dim=-1)

        sample[PROCESSED_ACTION_KEY] = actions_out
        return sample



class LiberoActionDummy(TransformFn):
    """
    Convert Libero actions into relative 6D pose deltas.
    """

    def compute(self, sample: Dict) -> Dict:
        sample[PROCESSED_ACTION_KEY] = sample[ACTION_KEY]
        return sample


class LiberoImages(TransformFn):
    """Stack Libero camera streams into a standard tensor layout."""

    image_keys = (IMAGE_KEY, WRIST_IMAGE_KEY)
    mask = torch.ones((len(image_keys), 1), dtype=torch.bool)

    def compute(self, sample: Dict) -> Dict:
        images = [sample[key] for key in self.image_keys]
        stacked = torch.stack(images, dim=0).type(torch.uint8)
        # shape: (num_cameras, C, H, W)
        sample[PROCESSED_IMAGE_KEY] = stacked
        sample[PROCESSED_IMAGE_MASK_KEY] = self.mask
        return sample


class LiberoActionToGlobal(TransformFn):
    """Invert network action targets back to dataset action format.

    Expects `sample["actions"]` to contain per-step targets of shape [K, 10]:
      [Δpos_to_target(3), Δori6d_to_target(6), gripper(1)] expressed relative to
    the current pose (last state in window), and un-normalized via UnNormalizeAction.

    Produces dataset-like actions [K, 7]: [dpos(3), axis_angle(3), gripper(1)],
    scaled to dataset action range using the same bounds as training.
    """

    def __init__(self, eef_frame: bool = True) -> None:
        super().__init__()
        self.eef_frame = eef_frame

    def compute(self, sample: Dict) -> Dict:
        actions: torch.Tensor = sample[PROCESSED_ACTION_KEY]

        if self.eef_frame:
            # the last state in the window is the current pose
            current_pos_w = sample[STATE_KEY][-1, 0:3]
            current_rotvec_w = sample[STATE_KEY][-1, 3:6]
            angle = torch.linalg.norm(current_rotvec_w)
            current_quat_w = quat_from_angle_axis(angle, current_rotvec_w)

            target_dpos = actions[:, 0:3]
            target_dori6d = actions[:, 3:9]

            # Convert 6D orientation delta back to quaternion delta (relative to current frame)
            drotmat = rotation_6d_to_matrix(target_dori6d)
            dquat = quat_from_matrix(drotmat)

            current_quat_w_expand = current_quat_w.unsqueeze(0).expand_as(dquat)
            target_pos_w = current_pos_w.unsqueeze(0) + quat_apply(
                current_quat_w_expand, target_dpos
            )
            target_quat_w = quat_mul(current_quat_w_expand, dquat)
            target_axis_angle_w = axis_angle_from_quat(target_quat_w)
        else:
            # actions are already in global frame
            target_pos_w = actions[:, 0:3]
            target_ori6d_w = actions[:, 3:9]
            target_mat_w = rotation_6d_to_matrix(target_ori6d_w)
            target_axis_angle_w = axis_angle_from_quat(quat_from_matrix(target_mat_w))

        grip = actions[:, 9:10]
        actions = torch.cat([target_pos_w, target_axis_angle_w, grip], dim=-1)  # [K, 7]
        sample[ACTION_KEY] = actions
        return sample
