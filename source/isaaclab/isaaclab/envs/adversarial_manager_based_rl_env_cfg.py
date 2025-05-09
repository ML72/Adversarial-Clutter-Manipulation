# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

from isaaclab.utils import configclass

from .manager_based_rl_env_cfg import ManagerBasedRLEnvCfg


@configclass
class AdversarialManagerBasedRLEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for a reinforcement learning environment with the manager-based workflow."""

    num_clutter_objects: int = MISSING

    positioning_strategy: str = "<CHANGE_THIS_STRATEGY>" # Set this for training, does not matter for evaluation
