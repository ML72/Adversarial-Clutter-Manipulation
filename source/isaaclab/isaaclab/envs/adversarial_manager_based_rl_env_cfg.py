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

    # Set these for training, does not matter for evaluation
    positioning_strategy: str = "domain_rand"
    train_mode: str = "train"
    train_actions_path: str | None = None
    train_positions_path: str | None = None # Not currently used
