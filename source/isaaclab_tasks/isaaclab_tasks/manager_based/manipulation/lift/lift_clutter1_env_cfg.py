# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
from dataclasses import MISSING
from typing import List, Union

from pxr import Gf

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, DeformableObjectCfg, RigidObjectCfg
from isaaclab.envs import AdversarialManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from . import mdp

origin = [0.6, 0, 0.0]


##
# Scene definition
##

box_length = 0.4
box_width = 0.5
box_height = 0.4
lip_height = 0.03
thickness = 0.0010

num_clutter_objects = 6


@configclass
class ObjectTableSceneCfg(InteractiveSceneCfg):
    """Configuration for the lift scene with a robot and a object.
    This is the abstract base implementation, the exact scene is defined in the derived classes
    which need to set the target object, robot and end-effector frames
    """

    # robots: will be populated by agent env cfg
    robot: ArticulationCfg = MISSING
    # end-effector sensor: will be populated by agent env cfg
    ee_frame: FrameTransformerCfg = MISSING
    # target object: will be populated by agent env cfg
    object: RigidObjectCfg | DeformableObjectCfg = MISSING

    # Table
    # table = AssetBaseCfg(
    #     prim_path="{ENV_REGEX_NS}/Table",
    #     init_state=AssetBaseCfg.InitialStateCfg(pos=[0.5, 0, 0], rot=[0.707, 0, 0, 0.707]),
    #     spawn=UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"),
    # )

    # Box
    bottom = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Wall1",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(torch.tensor(origin) + torch.tensor([0, 0, 0])).tolist(),
            rot=[0, 0.7071068, 0, 0.7071068],  # No rotation
        ),
        spawn=sim_utils.CuboidCfg(
            size=(thickness, box_width, box_length),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=10000.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.5, 0.0)),
        ),
    )
    top = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Wall5",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(torch.tensor(origin) + torch.tensor([0, 0, box_height])).tolist(),
            rot=[0, 0.7071068, 0, 0.7071068],  # No rotation
        ),
        spawn=sim_utils.CuboidCfg(
            size=(thickness, box_width, box_length),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=10000.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.5, 0.0)),
        ),
    )
    wall2 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Wall2",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(torch.tensor(origin) + torch.tensor([0, -box_width / 2, box_height / 2])).tolist(),
            rot=[0, 0, 0, 1],  # No rotation
        ),
        spawn=sim_utils.CuboidCfg(
            size=(box_length, thickness, box_height),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=10000.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.5, 0.0)),
        ),
    )
    wall4 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Wall4",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(torch.tensor(origin) + torch.tensor([0, box_width / 2, box_height / 2])).tolist(),
            rot=[0, 0, 0, 1],  # No rotation
        ),
        spawn=sim_utils.CuboidCfg(
            size=(box_length, thickness, box_height),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=10000.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.5, 0.0)),
        ),
    )
    wall3 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Wall3",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(torch.tensor(origin) + torch.tensor([box_length / 2, 0, box_height / 2])).tolist(),
            rot=[0, 0, 0, 1],  # No rotation
        ),
        spawn=sim_utils.CuboidCfg(
            size=(thickness, box_width, box_height),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=10000.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.5, 0.0)),
        ),
    )
    lip = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Wall6",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(torch.tensor(origin) + torch.tensor([-box_length / 2, 0, lip_height / 2])).tolist(),
            rot=[0, 0, 0, 1],  # No rotation
        ),
        spawn=sim_utils.CuboidCfg(
            size=(thickness, box_width, lip_height),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=10000.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.5, 0.0)),
        ),
    )

    # plane
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, -0.3]),
        spawn=GroundPlaneCfg(),
    )

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

    def __post_init__(self):
        """Initialize with a variable number of clutter objects."""
        for i in range(num_clutter_objects):
            setattr(self, f"clutter_object{i+1}", MISSING)


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    object_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name=MISSING,  # will be set by agent env cfg
        resampling_time_range=(5.0, 5.0),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.25, 0.25), pos_y=(0.0, 0.0), pos_z=(0.35, 0.35), roll=(0.0, 0.0), pitch=(0.0, 0.0), yaw=(0.0, 0.0)
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # will be set by agent env cfg
    arm_action: mdp.JointPositionActionCfg | mdp.DifferentialInverseKinematicsActionCfg = MISSING
    gripper_action: mdp.BinaryJointPositionActionCfg = MISSING


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        object_position = ObsTerm(func=mdp.object_position_in_robot_root_frame)
        target_object_position = ObsTerm(func=mdp.generated_commands, params={"command_name": "object_pose"})
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

            """Initialize with a variable number of clutter objects."""
            for i in range(num_clutter_objects):
                clutter_pos = ObsTerm(
                    func=mdp.clutter_position_in_robot_root_frame,
                    params={"object_cfg": SceneEntityCfg(f"clutter_object{i+1}")},
                )
                setattr(self, f"clutter_position{i+1}", clutter_pos)

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_object_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.1, 0.1), "y": (-0.1, 0.1), "z": (0.0, 0.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object", body_names="Object"),
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    reaching_object = RewTerm(func=mdp.object_ee_distance, params={"std": 0.1}, weight=1.0)

    lifting_object = RewTerm(func=mdp.object_is_lifted, params={"minimal_height": 0.04}, weight=15.0)

    object_goal_tracking = RewTerm(
        func=mdp.object_goal_distance,
        params={"std": 0.3, "minimal_height": 0.04, "command_name": "object_pose"},
        weight=16.0,
    )
    object_goal_tracking_fine_grained = RewTerm(
        func=mdp.object_goal_distance,
        params={"std": 0.05, "minimal_height": 0.04, "command_name": "object_pose"},
        weight=5.0,
    )

    # action penalty
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)

    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-4,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    def __post_init__(self):
        for i in range(num_clutter_objects):
            clutter_dropping = RewTerm(
                func=mdp.rew_root_height_below_minimum,
                weight=0,
                params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg(f"clutter_object{i+1}")},
            )
            setattr(self, f"clutter_drop{i+1}", clutter_dropping)


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    object_dropping = DoneTerm(
        func=mdp.root_height_below_minimum, params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("object")}
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    action_rate = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "action_rate", "weight": -1e-1, "num_steps": 25000}
    )

    joint_vel = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "joint_vel", "weight": -1e-1, "num_steps": 25000}
    )

    reaching_object = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "reaching_object", "weight": 0, "num_steps": 25000}
    )

    def __post_init__(self):
        for i in range(num_clutter_objects):
            clutter_dropping = CurrTerm(
                func=mdp.modify_reward_weight,
                params={"term_name": f"clutter_drop{i+1}", "weight": -10, "num_steps": 15000},
            )
            setattr(self, f"clutter_drop{i+1}", clutter_dropping)


##
# Environment configuration
##


@configclass
class LiftClutter1EnvCfg(AdversarialManagerBasedRLEnvCfg):
    """Configuration for the lifting environment."""

    # Scene settings
    scene: ObjectTableSceneCfg = ObjectTableSceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 5.0
        self.num_clutter_objects = num_clutter_objects
        # simulation settings
        self.sim.dt = 0.01  # 100Hz
        self.sim.render_interval = self.decimation

        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024 * 16
        self.sim.physx.gpu_max_rigid_patch_count = 1024 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625
