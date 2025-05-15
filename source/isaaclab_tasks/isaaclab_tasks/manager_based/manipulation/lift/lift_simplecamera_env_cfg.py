# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
from dataclasses import MISSING
from typing import List, Union

from pxr import Gf

import isaaclab.sim as sim_utils
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from . import mdp
from .lift_simple_env_cfg import LiftSimpleEnvCfg, ObjectTableSceneCfg


##
# Scene definition
##


@configclass
class ObjectTableCameraSceneCfg(ObjectTableSceneCfg):

    # add camera to the scene
    tiled_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Camera",
        offset=TiledCameraCfg.OffsetCfg(pos=(0.5, 0.0, 2.0), rot=(0.9945, 0.0, 0.1045, 0.0), convention="opengl"),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 20.0)
        ),
        width=200,
        height=200,
    )


##
# MDP settings
##


@configclass
class ResNet18ObservationCfg:
    """Observation specifications for the MDP."""

    @configclass
    class ResNet18FeaturesCameraPolicyCfg(ObsGroup):
        """Observations for policy group with features extracted from RGB images with a frozen ResNet18."""

        image = ObsTerm(
            func=mdp.image_features,
            params={"sensor_cfg": SceneEntityCfg("tiled_camera"), "data_type": "rgb", "model_name": "resnet18"},
        )

    policy: ObsGroup = ResNet18FeaturesCameraPolicyCfg()


##
# Environment configuration
##


@configclass
class LiftSimpleCameraEnvCfg(LiftSimpleEnvCfg):
    """Configuration for the lifting environment."""

    scene: ObjectTableCameraSceneCfg = ObjectTableCameraSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ResNet18ObservationCfg = ResNet18ObservationCfg()

    def __post_init__(self):
        super().__post_init__()

        # remove ground as it obstructs the camera
        self.scene.ground = None

        # viewer settings
        self.viewer.eye = (7.0, 0.0, 2.5)
        self.viewer.lookat = (0.0, 0.0, 2.5)
