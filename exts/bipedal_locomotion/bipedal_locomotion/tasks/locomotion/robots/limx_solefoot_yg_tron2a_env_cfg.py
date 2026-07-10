import math

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from bipedal_locomotion.assets.config.solefoot_yg_tron2a_cfg import SOLEFOOT_YG_TRON2A_CFG
from bipedal_locomotion.tasks.locomotion import mdp
from bipedal_locomotion.tasks.locomotion.cfg.SF_TRON2A.terrains_cfg import (
    BLIND_FLAT_TERRAINS_CFG,
    BLIND_FLAT_TERRAINS_PLAY_CFG,
)
from bipedal_locomotion.tasks.locomotion.cfg.SF_TRON2A.training_terrain.tron_camp_training_terrain import (
    TRON2_SPAWN_Z,
    TRON_CAMP_TRAINING_TERRAIN_CFG,
)
from bipedal_locomotion.tasks.locomotion.robots.limx_solefoot_tron2a_env_cfg import (
    SF_TRON2A_BaseEnvCfg,
    SF_TRON2A_BaseEnvCfg_PLAY,
    SF_TRON2A_BlindFlatEnvCfg,
    SF_TRON2A_BlindFlatEnvCfg_PLAY,
)


SFYG_LEG_TORSO_LINKS = [
    "base_Link",
    "proximal_pitch_[LR]_Link",
    "proximal_roll_[LR]_Link",
    "proximal_yaw_[LR]_Link",
    "knee_[LR]_Link",
    "ankle_pitch_[LR]_Link",
]

SFYG_LEG_JOINTS = [
    "proximal_pitch_[LR]_Joint",
    "proximal_roll_[LR]_Joint",
    "proximal_yaw_[LR]_Joint",
    "knee_[LR]_Joint",
    "ankle_pitch_[LR]_Joint",
]


def _exclude_arm_from_events_and_rewards(self):
    if self.events.add_base_mass is not None:
        self.events.add_base_mass.params["asset_cfg"].body_names = "base_Link"

    if self.events.reset_robot_joints is not None:
        self.events.reset_robot_joints.params["asset_cfg"] = SceneEntityCfg("robot", joint_names=SFYG_LEG_JOINTS)

    if hasattr(self.rewards, "dof_pos_limits") and self.rewards.dof_pos_limits is not None:
        self.rewards.dof_pos_limits.params["asset_cfg"].joint_names = SFYG_LEG_JOINTS
    if hasattr(self.rewards, "dof_vel_limits") and self.rewards.dof_vel_limits is not None:
        self.rewards.dof_vel_limits.params["asset_cfg"].joint_names = SFYG_LEG_JOINTS
    if hasattr(self.rewards, "stand_still") and self.rewards.stand_still is not None:
        self.rewards.stand_still.params["asset_cfg"].joint_names = SFYG_LEG_JOINTS


def _use_generated_flat_terrain(self, terrain_generator):
    self.scene.terrain.terrain_type = "generator"
    self.scene.terrain.terrain_generator = terrain_generator
    self.scene.terrain.max_init_terrain_level = None


def _use_tron_camp_training_terrain(self):
    self.scene.terrain = TRON_CAMP_TRAINING_TERRAIN_CFG
    self.scene.robot.init_state.pos = (0.0, 0.0, TRON2_SPAWN_Z)


def _use_forward_walking_commands(self):
    self.commands.base_velocity.heading_command = True
    self.commands.base_velocity.rel_standing_envs = 0.0
    self.commands.base_velocity.rel_heading_envs = 1.0
    self.commands.base_velocity.ranges.lin_vel_x = (0.45, 0.80)
    self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
    self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
    self.commands.base_velocity.ranges.heading = (0.0, 0.0)
    self.commands.gait_command.ranges.frequencies = (0.75, 1.05)
    self.commands.gait_command.ranges.durations = (0.58, 0.68)
    self.commands.gait_command.ranges.swing_height = (0.06, 0.11)


def _use_forward_walking_rewards(self):
    self.rewards.track_lin_vel_x_exp.weight = 4.0
    self.rewards.track_lin_vel_y_exp.weight = 1.5
    self.rewards.track_ang_vel_z_exp.weight = 1.5
    self.rewards.keep_balance.weight = 0.3
    self.rewards.base_height_exp.weight = 0.4
    self.rewards.flat_orientation_l2.weight = -1.5
    self.rewards.gait_reward.weight = 0.45
    self.rewards.feet_distance.weight = 0.7
    self.rewards.feet_air_time.weight = 0.0
    self.rewards.feet_slide.weight = -0.08
    self.rewards.action_rate_l2.weight = -0.02
    self.rewards.action_smoothness.weight = -0.0008
    self.rewards.same_feet_x_position = RewTerm(
        func=mdp.same_feet_x_position,
        weight=-0.20,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="ankle_pitch_.*")},
    )
    self.rewards.centerline_y = RewTerm(
        func=mdp.track_centerline_y,
        weight=1.0,
        params={"std": 0.6},
    )


def _use_track_aligned_resets(self):
    if self.events.reset_robot_base is None:
        return
    self.events.reset_robot_base.params["pose_range"] = {
        "x": (-0.1, 0.1),
        "y": (-0.1, 0.1),
        "yaw": (-0.05, 0.05),
    }
    self.events.reset_robot_base.params["velocity_range"] = {
        "x": (-0.1, 0.1),
        "y": (-0.1, 0.1),
        "z": (-0.1, 0.1),
        "roll": (-0.1, 0.1),
        "pitch": (-0.1, 0.1),
        "yaw": (-0.1, 0.1),
    }


def _use_gait_stabilization_resets(self):
    _use_track_aligned_resets(self)
    if self.events.reset_robot_joints is None:
        return
    self.events.reset_robot_joints.params["position_range"] = (-0.1, 0.1)
    self.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)


def _use_gait_stabilization_commands(self):
    self.commands.base_velocity.heading_command = True
    self.commands.base_velocity.rel_standing_envs = 0.0
    self.commands.base_velocity.rel_heading_envs = 1.0
    self.commands.base_velocity.ranges.lin_vel_x = (0.45, 0.80)
    self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
    self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
    self.commands.base_velocity.ranges.heading = (0.0, 0.0)
    self.commands.gait_command.ranges.frequencies = (0.65, 0.90)
    self.commands.gait_command.ranges.offsets = (0.5, 0.5)
    self.commands.gait_command.ranges.durations = (0.50, 0.58)
    self.commands.gait_command.ranges.swing_height = (0.11, 0.17)


def _use_gait_stabilization_rewards(self):
    self.rewards.track_lin_vel_x_exp.weight = 3.5
    self.rewards.track_lin_vel_x_exp.params["std"] = 0.25
    self.rewards.track_world_lin_vel_x_exp = RewTerm(
        func=mdp.track_world_lin_vel_x_exp,
        weight=4.0,
        params={"command_name": "base_velocity", "std": 0.25},
    )
    self.rewards.track_lin_vel_y_exp.weight = 3.5
    self.rewards.track_lin_vel_y_exp.params["std"] = 0.18
    self.rewards.track_ang_vel_z_exp.weight = 4.0
    self.rewards.track_ang_vel_z_exp.params["std"] = 0.20
    self.rewards.keep_balance.weight = 0.1
    self.rewards.base_height_exp.weight = 0.3
    self.rewards.flat_orientation_l2.weight = -1.7
    self.rewards.lin_vel_z_l2.weight = -0.6
    self.rewards.body_lin_acc.weight = -0.0025
    self.rewards.gait_reward.weight = 0.45
    self.rewards.feet_distance.weight = 1.0
    self.rewards.feet_air_time.weight = 0.0
    self.rewards.feet_slide.weight = -0.12
    self.rewards.feet_ang_slide_penalty.weight = -0.15
    self.rewards.action_rate_l2.weight = -0.012
    self.rewards.action_smoothness.weight = -0.00045
    self.rewards.knee_joint_orientation.weight = -0.6
    self.rewards.torques_smoothness_penalty.weight = -3.0e-7
    self.rewards.same_feet_x_position = RewTerm(
        func=mdp.same_feet_x_position,
        weight=-0.012,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="ankle_pitch_.*")},
    )
    self.rewards.centerline_y = RewTerm(
        func=mdp.track_centerline_y,
        weight=3.0,
        params={"std": 0.35},
    )
    self.rewards.heading_alignment = RewTerm(
        func=mdp.heading_alignment_exp,
        weight=2.0,
        params={"target_heading": 0.0, "std": 0.18},
    )
    self.rewards.heading_error_abs = RewTerm(
        func=mdp.heading_error_deadband,
        weight=-6.0,
        params={"target_heading": 0.0, "deadband": math.radians(8.0)},
    )
    self.rewards.world_lin_vel_y_abs = RewTerm(
        func=mdp.world_lin_vel_y_abs,
        weight=-3.0,
    )
    self.rewards.world_pos_y_abs = RewTerm(
        func=mdp.world_pos_y_abs,
        weight=-1.0,
    )
    self.rewards.forward_speed_shortfall = RewTerm(
        func=mdp.lin_vel_x_shortfall_penalty,
        weight=-4.0,
        params={"command_name": "base_velocity", "deadband": 0.05},
    )
    self.rewards.single_stance = RewTerm(
        func=mdp.single_stance_reward,
        weight=0.45,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="ankle_pitch_.*")},
    )
    self.rewards.swing_foot_clearance = RewTerm(
        func=mdp.swing_foot_clearance_reward,
        weight=0.75,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="ankle_pitch_.*"),
            "asset_cfg": SceneEntityCfg("robot", body_names="ankle_pitch_.*"),
            "target_height": 0.12,
            "std": 0.05,
        },
    )
    self.rewards.double_flight = RewTerm(
        func=mdp.double_flight_penalty,
        weight=-1.2,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="ankle_pitch_.*")},
    )
    self.rewards.double_stance_feet_x_alignment = RewTerm(
        func=mdp.double_stance_feet_x_alignment_reward,
        weight=0.25,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="ankle_pitch_.*"),
            "asset_cfg": SceneEntityCfg("robot", body_names="ankle_pitch_.*"),
            "std": 0.10,
        },
    )
    self.rewards.single_stance_feet_x_separation = RewTerm(
        func=mdp.single_stance_feet_x_separation_reward,
        weight=0.12,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="ankle_pitch_.*"),
            "asset_cfg": SceneEntityCfg("robot", body_names="ankle_pitch_.*"),
            "target_distance": 0.18,
            "std": 0.08,
        },
    )
    self.rewards.balanced_feet_x_symmetry = RewTerm(
        func=mdp.balanced_feet_x_symmetry_reward,
        weight=0.45,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="ankle_pitch_.*"),
            "std": 0.16,
            "max_tilt": 0.35,
        },
    )


def _use_gait_heading_termination(self):
    self.terminations.bad_heading = DoneTerm(
        func=mdp.bad_heading,
        params={"target_heading": 0.0, "limit_angle": math.radians(25.0)},
    )


@configclass
class SFYG_TRON2A_BaseEnvCfg(SF_TRON2A_BaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = SOLEFOOT_YG_TRON2A_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        _exclude_arm_from_events_and_rewards(self)


@configclass
class SFYG_TRON2A_BaseEnvCfg_PLAY(SF_TRON2A_BaseEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = SOLEFOOT_YG_TRON2A_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        _exclude_arm_from_events_and_rewards(self)


@configclass
class SFYG_TRON2A_BlindFlatEnvCfg(SF_TRON2A_BlindFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _use_generated_flat_terrain(self, BLIND_FLAT_TERRAINS_CFG)
        self.scene.robot = SOLEFOOT_YG_TRON2A_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        _exclude_arm_from_events_and_rewards(self)


@configclass
class SFYG_TRON2A_BlindFlatEnvCfg_PLAY(SF_TRON2A_BlindFlatEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        _use_generated_flat_terrain(self, BLIND_FLAT_TERRAINS_PLAY_CFG)
        self.scene.robot = SOLEFOOT_YG_TRON2A_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        _exclude_arm_from_events_and_rewards(self)


@configclass
class SFYG_TRON2A_BlindTronCampEnvCfg(SFYG_TRON2A_BlindFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _use_tron_camp_training_terrain(self)


@configclass
class SFYG_TRON2A_BlindTronCampEnvCfg_PLAY(SFYG_TRON2A_BlindFlatEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        _use_tron_camp_training_terrain(self)


@configclass
class SFYG_TRON2A_BlindTronCampForwardEnvCfg(SFYG_TRON2A_BlindTronCampEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _use_forward_walking_commands(self)
        _use_forward_walking_rewards(self)
        _use_track_aligned_resets(self)


@configclass
class SFYG_TRON2A_BlindTronCampForwardEnvCfg_PLAY(SFYG_TRON2A_BlindTronCampEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        _use_forward_walking_commands(self)
        _use_forward_walking_rewards(self)
        _use_track_aligned_resets(self)
        self.scene.num_envs = 1


@configclass
class SFYG_TRON2A_BlindGaitEnvCfg(SFYG_TRON2A_BlindFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _use_gait_stabilization_commands(self)
        _use_gait_stabilization_rewards(self)
        _use_gait_heading_termination(self)
        _use_gait_stabilization_resets(self)


@configclass
class SFYG_TRON2A_BlindGaitEnvCfg_PLAY(SFYG_TRON2A_BlindFlatEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        _use_gait_stabilization_commands(self)
        _use_gait_stabilization_rewards(self)
        _use_gait_heading_termination(self)
        _use_gait_stabilization_resets(self)
        self.scene.num_envs = 1
