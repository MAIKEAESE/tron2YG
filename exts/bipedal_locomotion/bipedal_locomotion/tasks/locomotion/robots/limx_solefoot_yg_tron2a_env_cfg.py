import math

from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import RayCasterCfg, patterns
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


def _use_base_height_scanner(self):
    self.scene.height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_Link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(
            resolution=0.10,
            size=(1.6, 1.0),
        ),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    self.scene.height_scanner.update_period = self.decimation * self.sim.dt


def _use_policy_height_scan(self):
    self.observations.policy.height_scan = ObsTerm(
        func=mdp.height_scan,
        params={"sensor_cfg": SceneEntityCfg("height_scanner"), "offset": 0.8},
        clip=(-1.0, 1.0),
        scale=5.0,
    )


def _use_perceptive_stair_clearance(self):
    self.commands.gait_command.ranges.swing_height = (0.15, 0.24)
    self.rewards.single_stance.weight = 0.65
    self.rewards.swing_foot_clearance = RewTerm(
        func=mdp.terrain_adaptive_swing_foot_clearance_reward,
        weight=1.50,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="ankle_pitch_.*"),
            "asset_cfg": SceneEntityCfg("robot", body_names="ankle_pitch_.*"),
            "ray_sensor_cfg": SceneEntityCfg("height_scanner"),
            "base_height": 0.16,
            "obstacle_margin": 0.10,
            "max_target_height": 0.34,
            "std": 0.06,
            "front_min_x": 0.15,
            "near_abs_x": 0.25,
            "obstacle_threshold": 0.025,
        },
    )
    self.rewards.double_flight.weight = -3.2
    self.rewards.same_feet_x_position.weight = -0.004
    self.rewards.single_stance_feet_x_separation.weight = 0.0
    self.rewards.single_stance_feet_x_separation.params["target_distance"] = 0.23
    self.rewards.single_stance_feet_x_separation.params["std"] = 0.10
    self.rewards.stance_foot_stability = RewTerm(
        func=mdp.stance_foot_stability_reward,
        weight=0.50,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="ankle_pitch_.*"),
            "asset_cfg": SceneEntityCfg("robot", body_names="ankle_pitch_.*"),
            "foot_vel_std": 0.12,
            "max_tilt": 0.45,
        },
    )
    self.rewards.swing_foot_lateral_deviation = RewTerm(
        func=mdp.swing_foot_lateral_deviation,
        weight=-1.20,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="ankle_pitch_.*"),
            "asset_cfg": SceneEntityCfg("robot", body_names="ankle_pitch_.*"),
            "max_abs_y": 0.18,
        },
    )
    self.rewards.swing_knee_flexion = RewTerm(
        func=mdp.swing_knee_flexion_reward,
        weight=0.35,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="ankle_pitch_.*"),
            "asset_cfg": SceneEntityCfg("robot", joint_names="knee_[LR]_Joint"),
            "target_flexion": 0.35,
            "std": 0.20,
        },
    )


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


def _use_tron_camp_curriculum_resets(self):
    if self.events.reset_robot_base is None:
        return
    self.events.reset_robot_base.func = mdp.reset_root_state_tron_camp_curriculum
    self.events.reset_robot_base.params = {
        "z_range": (TRON2_SPAWN_Z, TRON2_SPAWN_Z + 0.04),
        "y_range": (-0.18, 0.18),
        "yaw_range": (-0.08, 0.08),
        "velocity_range": {
            "x": (0.0, 0.15),
            "y": (-0.04, 0.04),
            "z": (-0.02, 0.02),
            "roll": (-0.05, 0.05),
            "pitch": (-0.05, 0.05),
            "yaw": (-0.05, 0.05),
        },
        "curriculum_segments": (
            # Local x offsets are relative to each generated track origin.
            # They target feature entries, not feature tops, so the spawn z
            # remains valid across randomized terrain cells.
            {"name": "start_flat", "x_range": (-0.2, 0.8), "weight": 0.30},
            {"name": "up_ramp_entry", "x_range": (2.0, 4.8), "weight": 0.14},
            {"name": "downhill_exit", "x_range": (10.0, 17.0), "weight": 0.16},
            {"name": "bumps_entry", "x_range": (16.0, 27.0), "weight": 0.16},
            {"name": "stairs_entry", "x_range": (24.0, 39.0), "weight": 0.18},
            {"name": "platform_entry", "x_range": (42.0, 58.0), "weight": 0.06},
        ),
    }


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
    self.commands.gait_command.ranges.durations = (0.52, 0.60)
    self.commands.gait_command.ranges.swing_height = (0.10, 0.15)


def _use_gait_stabilization_rewards(self):
    self.rewards.track_lin_vel_x_exp.weight = 4.0
    self.rewards.track_lin_vel_x_exp.params["std"] = 0.25
    self.rewards.track_world_lin_vel_x_exp = RewTerm(
        func=mdp.track_world_lin_vel_x_exp,
        weight=4.5,
        params={"command_name": "base_velocity", "std": 0.25},
    )
    self.rewards.track_lin_vel_y_exp.weight = 3.5
    self.rewards.track_lin_vel_y_exp.params["std"] = 0.18
    self.rewards.track_ang_vel_z_exp.weight = 4.0
    self.rewards.track_ang_vel_z_exp.params["std"] = 0.20
    self.rewards.keep_balance.weight = 0.1
    self.rewards.base_height_exp.weight = 0.3
    self.rewards.flat_orientation_l2.weight = -2.3
    self.rewards.lin_vel_z_l2.weight = -0.6
    self.rewards.body_lin_acc.weight = -0.003
    self.rewards.gait_reward.weight = 0.45
    self.rewards.feet_distance.weight = 0.5
    self.rewards.feet_air_time.weight = 0.0
    self.rewards.feet_slide.weight = -0.24
    self.rewards.feet_ang_slide_penalty.weight = -0.28
    self.rewards.action_rate_l2.weight = -0.005
    self.rewards.action_smoothness.weight = -0.00020
    self.rewards.knee_joint_orientation.weight = -0.8
    self.rewards.torques_smoothness_penalty.weight = -3.0e-7
    self.rewards.same_feet_x_position = RewTerm(
        func=mdp.same_feet_x_position,
        weight=-0.012,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="ankle_pitch_.*")},
    )
    self.rewards.centerline_y = RewTerm(
        func=mdp.track_centerline_y,
        weight=8.5,
        params={"std": 0.22},
    )
    self.rewards.heading_alignment = RewTerm(
        func=mdp.heading_alignment_exp,
        weight=0.3,
        params={"target_heading": 0.0, "std": 0.25},
    )
    self.rewards.heading_error_abs = RewTerm(
        func=mdp.heading_error_deadband,
        weight=-1.0,
        params={"target_heading": 0.0, "deadband": math.radians(10.0)},
    )
    self.rewards.world_lin_vel_y_abs = RewTerm(
        func=mdp.world_lin_vel_y_abs,
        weight=-3.5,
    )
    self.rewards.world_pos_y_abs = RewTerm(
        func=mdp.world_pos_y_deadband,
        weight=-9.0,
        params={"deadband": 0.06},
    )
    self.rewards.predicted_path_lateral_error = RewTerm(
        func=mdp.predicted_path_lateral_error,
        weight=-3.0,
        params={
            "target_heading": 0.0,
            "hip_weight": 0.03,
            "step_weight": 0.65,
            "same_direction_weight": 0.05,
            "deadband": 0.015,
            "feet_cfg": SceneEntityCfg("robot", body_names="ankle_pitch_.*"),
        },
    )
    self.rewards.lateral_correction_velocity = RewTerm(
        func=mdp.lateral_correction_velocity_exp,
        weight=5.0,
        params={
            "gain": 0.8,
            "far_gain": 1.2,
            "far_threshold": 0.25,
            "max_correction_speed": 0.38,
            "std": 0.18,
            "deadband": 0.05,
        },
    )
    self.rewards.forward_speed_shortfall = RewTerm(
        func=mdp.lin_vel_x_shortfall_penalty,
        weight=-8.0,
        params={"command_name": "base_velocity", "deadband": 0.05},
    )
    self.rewards.forward_overspeed = RewTerm(
        func=mdp.forward_overspeed_penalty,
        weight=-3.0,
        params={"command_name": "base_velocity", "margin": 0.08},
    )
    self.rewards.single_stance = RewTerm(
        func=mdp.single_stance_reward,
        weight=0.45,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="ankle_pitch_.*")},
    )
    self.rewards.swing_foot_clearance = RewTerm(
        func=mdp.swing_foot_clearance_reward,
        weight=0.70,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="ankle_pitch_.*"),
            "asset_cfg": SceneEntityCfg("robot", body_names="ankle_pitch_.*"),
            "target_height": 0.11,
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
        weight=0.08,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="ankle_pitch_.*"),
            "asset_cfg": SceneEntityCfg("robot", body_names="ankle_pitch_.*"),
            "target_distance": 0.15,
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


def _use_gait_heading_termination(self, limit_angle=math.radians(35.0)):
    self.terminations.bad_heading = DoneTerm(
        func=mdp.bad_heading,
        params={"target_heading": 0.0, "limit_angle": limit_angle},
    )


def _use_gait_lateral_termination(self, limit_distance=0.50):
    self.terminations.bad_lateral_drift = DoneTerm(
        func=mdp.bad_lateral_drift,
        params={"limit_distance": limit_distance},
    )


def _use_competition_exposure_training(self):
    # Keep fall/contact terminations, but let imperfect policies survive long enough
    # to collect experience on the downhill exit and stair sections.
    _use_gait_heading_termination(self, limit_angle=math.radians(50.0))
    _use_gait_lateral_termination(self, limit_distance=1.00)
    self.episode_length_s = 30.0


def _use_random_direction_terrain_training(self):
    self.commands.base_velocity.heading_command = True
    self.commands.base_velocity.rel_standing_envs = 0.0
    self.commands.base_velocity.rel_heading_envs = 1.0
    self.commands.base_velocity.ranges.lin_vel_x = (0.35, 0.75)
    self.commands.base_velocity.ranges.lin_vel_y = (-0.10, 0.10)
    self.commands.base_velocity.ranges.ang_vel_z = (-0.35, 0.35)
    self.commands.base_velocity.ranges.heading = (-0.75, 0.75)

    self.rewards.track_lin_vel_x_exp.weight = 4.0
    self.rewards.track_lin_vel_y_exp.weight = 2.0
    self.rewards.track_ang_vel_z_exp.weight = 2.0
    self.rewards.forward_speed_shortfall.weight = -3.0
    self.rewards.forward_overspeed.weight = -2.0

    # World-frame route rewards conflict with random heading pretraining.
    self.rewards.track_world_lin_vel_x_exp.weight = 0.0
    self.rewards.centerline_y.weight = 0.0
    self.rewards.heading_alignment.weight = 0.0
    self.rewards.heading_error_abs.weight = 0.0
    self.rewards.world_lin_vel_y_abs.weight = 0.0
    self.rewards.world_pos_y_abs.weight = 0.0
    self.rewards.predicted_path_lateral_error.weight = 0.0
    self.rewards.lateral_correction_velocity.weight = 0.0

    self.terminations.bad_heading = None
    self.terminations.bad_lateral_drift = None
    self.episode_length_s = 30.0


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
        _use_gait_lateral_termination(self)
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


@configclass
class SFYG_TRON2A_BlindGaitTronCampEnvCfg(SFYG_TRON2A_BlindTronCampEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _use_gait_stabilization_commands(self)
        _use_gait_stabilization_rewards(self)
        _use_gait_heading_termination(self)
        _use_gait_lateral_termination(self)
        _use_gait_stabilization_resets(self)


@configclass
class SFYG_TRON2A_BlindGaitTronCampEnvCfg_PLAY(SFYG_TRON2A_BlindTronCampEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        _use_gait_stabilization_commands(self)
        _use_gait_stabilization_rewards(self)
        _use_gait_heading_termination(self, limit_angle=math.radians(35.0))
        _use_gait_stabilization_resets(self)
        self.scene.num_envs = 1


@configclass
class SFYG_TRON2A_PerceptiveGaitTronCampEnvCfg(SFYG_TRON2A_BlindGaitTronCampEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _use_base_height_scanner(self)
        _use_policy_height_scan(self)
        _use_perceptive_stair_clearance(self)
        _use_competition_exposure_training(self)
        _use_tron_camp_curriculum_resets(self)


@configclass
class SFYG_TRON2A_PerceptiveGaitTronCampEnvCfg_PLAY(SFYG_TRON2A_BlindGaitTronCampEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        _use_base_height_scanner(self)
        _use_policy_height_scan(self)
        _use_perceptive_stair_clearance(self)
        self.scene.num_envs = 1


@configclass
class SFYG_TRON2A_PerceptiveRandomDirTronCampEnvCfg(SFYG_TRON2A_PerceptiveGaitTronCampEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _use_random_direction_terrain_training(self)


@configclass
class SFYG_TRON2A_PerceptiveRandomDirTronCampEnvCfg_PLAY(SFYG_TRON2A_PerceptiveGaitTronCampEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        _use_random_direction_terrain_training(self)
        self.scene.num_envs = 1
