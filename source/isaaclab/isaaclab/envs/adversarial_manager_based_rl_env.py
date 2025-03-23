# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# needed to import for allowing type-hinting: np.ndarray | None
from __future__ import annotations

import gymnasium as gym
import math
import numpy as np
import torch
from collections.abc import Sequence
from typing import Any, ClassVar

from isaaclab.managers import CommandManager, CurriculumManager, RewardManager, TerminationManager

from .adversarial_manager_based_rl_env_cfg import AdversarialManagerBasedRLEnvCfg
from .manager_based_rl_env import ManagerBasedRLEnv


# Amplitude of the cube position (to scale a value between -1 and 1)
cube_position_ampl_x = 0.25
cube_position_ampl_y = 0.4
cube_position_ampl_z = 0.1

class AdversarialManagerBasedRLEnv(ManagerBasedRLEnv):
    """The superclass for the manager-based workflow reinforcement learning-based environments."""

    def __init__(self, cfg: AdversarialManagerBasedRLEnvCfg, **kwargs):
        super().__init__(cfg, **kwargs)
        self.num_clutter_objects = cfg.num_clutter_objects
        self.position_dim = 3

        # dim 1 of adversary_action is num_clutter_objects + 1, to account for the object that is placed
        self.adversary_action = torch.zeros((self.num_envs, (self.num_clutter_objects + 1) * self.position_dim)).to(
            self.device
        )

    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        """Execute one time-step of the environment's dynamics and reset terminated environments.

        Unlike the :class:`ManagerBasedEnv.step` class, the function performs the following operations:

        1. Process the actions.
        2. Perform physics stepping.
        3. Perform rendering if gui is enabled.
        4. Update the environment counters and compute the rewards and terminations.
        5. Reset the environments that terminated.
        6. Compute the observations.
        7. Return the observations, rewards, resets and extras.

        Args:
            action: The actions to apply on the environment. Shape is (num_envs, action_dim).

        Returns:
            A tuple containing the observations, rewards, resets (terminated and truncated) and extras.
        """
        # process actions
        self.action_manager.process_action(action.to(self.device))

        self.recorder_manager.record_pre_step()

        # check if we need to do rendering within the physics loop
        # note: checked here once to avoid multiple checks within the loop
        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()

        # perform physics stepping
        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            # set actions into buffers
            self.action_manager.apply_action()
            # set actions into simulator
            self.scene.write_data_to_sim()
            # simulate
            self.sim.step(render=False)
            # render between steps only if the GUI or an RTX sensor needs it
            # note: we assume the render interval to be the shortest accepted rendering interval.
            #    If a camera needs rendering at a faster frequency, this will lead to unexpected behavior.
            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                self.sim.render()
            # update buffers at sim dt
            self.scene.update(dt=self.physics_dt)

        # post-step:
        # -- update env counters (used for curriculum generation)
        self.episode_length_buf += 1  # step in current episode (per env)
        self.common_step_counter += 1  # total step (common for all envs)
        # -- check terminations
        self.reset_buf = self.termination_manager.compute()
        self.reset_terminated = self.termination_manager.terminated
        self.reset_time_outs = self.termination_manager.time_outs
        # -- reward computation
        self.reward_buf = self.reward_manager.compute(dt=self.step_dt)

        if len(self.recorder_manager.active_terms) > 0:
            # update observations for recording if needed
            self.obs_buf = self.observation_manager.compute()
            self.recorder_manager.record_post_step()

        # -- reset envs that terminated/timed-out and log the episode information
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) == self.num_envs:
            # trigger recorder terms for pre-reset calls
            self.recorder_manager.record_pre_reset(reset_env_ids)

            self._reset_idx(reset_env_ids)
            # update articulation kinematics
            self.scene.write_data_to_sim()
            self.sim.forward()

            # if sensors are added to the scene, make sure we render to reflect changes in reset
            if self.sim.has_rtx_sensors() and self.cfg.rerender_on_reset:
                self.sim.render()

            # trigger recorder terms for post-reset calls
            self.recorder_manager.record_post_reset(reset_env_ids)

        # -- update command
        self.command_manager.compute(dt=self.step_dt)
        # -- step interval events
        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)
        # -- compute observations
        # note: done after reset to get the correct observations for reset envs
        self.obs_buf = self.observation_manager.compute()

        # return observations, rewards, resets and extras
        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras

    def _reset_idx(self, env_ids: Sequence[int]):
        """Reset environments based on specified indices.

        Args:
            env_ids: List of environment ids which must be reset
        """
        super()._reset_idx(env_ids)
        self.adversarial_reset(env_ids)

    def adversarial_reset(self, reset_env_ids: Sequence[int]) -> tuple[VecEnvObs, dict]:
        """Reset the environment.

        Returns:
            np.ndarray: The initial observation.
        """
        adversary_pos = self.adversary_action[reset_env_ids]
        adversary_pos = torch.clamp(adversary_pos, -1, 1)

        for asset_name, rigid_object in self.scene._rigid_objects.items():
            # catch both "object" and "clutter_object<i>"
            if "object" in asset_name: 
                object_idx = int(asset_name.split("clutter_object")[-1]) if "clutter_object" in asset_name else 0
                clutter_obj_state = rigid_object.data.default_root_state[
                    reset_env_ids
                ].clone()  # get states of only envs we want to reset
                clutter_obj_state[:, 0:3] += self.scene.env_origins[reset_env_ids]
                root_pose = clutter_obj_state[:, :7]
                root_velocity = clutter_obj_state[:, 7:] * 0.0  # zero out the velocity
                root_pose[:, :3] += torch.stack(
                    [
                        adversary_pos[:, object_idx * 3] * cube_position_ampl_x,
                        adversary_pos[:, object_idx * 3 + 1] * cube_position_ampl_y,
                        torch.abs(adversary_pos[:, object_idx * 3 + 2]) * cube_position_ampl_z + 0.1,
                    ],
                    dim=-1,
                ).to(root_pose.device)
                rigid_object.write_root_link_pose_to_sim(root_pose, env_ids=reset_env_ids)
                rigid_object.write_root_com_velocity_to_sim(root_velocity, env_ids=reset_env_ids)
        self.scene.write_data_to_sim()
