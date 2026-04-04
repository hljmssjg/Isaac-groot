"""
Modality config for Jiangeng/G1_0330 dataset.

G1 upper-body only: left_arm(7) + right_arm(7) + left_hand(7) + right_hand(7) = 28 DOF
3 cameras: cam_head, cam_left_wrist, cam_extra

Use with:
  --embodiment_tag NEW_EMBODIMENT
  --modality_config_path examples/G1_0330/g1_0330_modality_config.py
"""

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)

_ACTION_HORIZON = 16

g1_0330_config = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["cam_head", "cam_left_wrist", "cam_extra"],
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "left_arm",
            "right_arm",
            "left_hand",
            "right_hand",
        ],
    ),
    "action": ModalityConfig(
        delta_indices=list(range(_ACTION_HORIZON)),
        modality_keys=[
            "left_arm",
            "right_arm",
            "left_hand",
            "right_hand",
        ],
        action_configs=[
            # left_arm: relative delta
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            # right_arm: relative delta
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            # left_hand: absolute (dexterous hand positions)
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            # right_hand: absolute (dexterous hand positions)
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.task_description"],
    ),
}

register_modality_config(g1_0330_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
