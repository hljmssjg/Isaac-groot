"""
Modality config aligned with datasets/g1-pick-apple/meta/modality.json.

Use with:
  --embodiment-tag NEW_EMBODIMENT
  --modality-config-path examples/g1-pick-apple/g1_pick_apple_modality_config.py
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

# Horizon 16 matches Gr00tN1d6Config.action_horizon default (see gr00t/configs/model/gr00t_n1d6.py).
_ACTION_HORIZON = 16

g1_pick_apple_config = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["ego_view"],
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "left_leg",
            "right_leg",
            "waist",
            "left_arm",
            "left_hand",
            "right_arm",
            "right_hand",
        ],
    ),
    "action": ModalityConfig(
        delta_indices=list(range(_ACTION_HORIZON)),
        modality_keys=[
            "left_leg",
            "right_leg",
            "waist",
            "left_arm",
            "left_hand",
            "right_arm",
            "right_hand",
        ],
        action_configs=[
            # Legs: relative deltas (dataset has full-body joint targets; matches use_relative_action)
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            # Waist: absolute (same as unitree_g1 waist)
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            # Arms: relative (same as unitree_g1 arms)
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            # Hands: absolute (same as unitree_g1 hands)
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
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

register_modality_config(g1_pick_apple_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
