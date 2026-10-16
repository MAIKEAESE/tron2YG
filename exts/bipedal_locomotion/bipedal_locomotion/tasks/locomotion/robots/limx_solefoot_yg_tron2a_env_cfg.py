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
    self.commands.base_velocity.rel_standing_envs = 0.22
    self.commands.base_velocity.ranges.lin_vel_x = (0.12, 0.50)
    self.commands.gait_command.ranges.frequencies = (0.45, 0.75)
    self.commands.gait_command.ranges.durations = (0.58, 0.72)
    self.commands.gait_command.ranges.swing_height = (0.13, 0.23)
    self.rewards.single_stance.weight = 0.90
    self.rewards.swing_foot_clearance = RewTerm(
        func=mdp.terrain_adaptive_swing_foot_clearance_reward,
        weight=3.0,
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
            "min_forward_speed": 0.12,
            "max_tilt": 0.50,
            "allow_obstacle_stationary": True,
            "min_base_clearance": 0.62,
            "max_stance_foot_speed": 0.28,
            "high_obstacle_threshold": 0.18,
            "max_high_obstacle_yaw": math.radians(25.0),
            # Observed obstacle height -> absolute swing-clearance target.
            "height_bands": (
                (0.025, 0.07),
                (0.080, 0.13),
                (0.140, 0.21),
            ),
        },
    )
    self.rewards.swing_foot_liftoff = RewTerm(
        func=mdp.swing_foot_liftoff_reward,
        weight=0.40,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="ankle_pitch_.*"),
            "asset_cfg": SceneEntityCfg("robot", body_names="ankle_pitch_.*"),
            "ray_sensor_cfg": SceneEntityCfg("height_scanner"),
            "min_liftoff": 0.025,
            "target_liftoff": 0.055,
            "max_liftoff": 0.08,
            "std": 0.025,
            "max_lateral_abs_y": 0.18,
            "max_tilt": 0.50,
            "high_obstacle_threshold": 0.18,
            "max_high_obstacle_yaw": math.radians(25.0),
        },
    )
    self.rewards.walking_single_leg_air_time = RewTerm(
        func=mdp.walking_single_leg_air_time_reward,
        weight=5.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="ankle_pitch_.*"),
            "asset_cfg": SceneEntityCfg("robot", body_names="ankle_pitch_.*"),
            "command_name": "base_velocity",
            "min_command_speed": 0.08,
            "min_forward_speed": 0.00,
            "target_air_time": 0.30,
            "air_time_std": 0.12,
            "min_air_time": 0.14,
            "max_air_time": 0.40,
            "max_tilt": 0.80,
            "min_base_height": 0.58,
            "max_lateral_abs_y": 0.4,
            "max_stance_foot_speed": 0.25,
            "min_stance_force": 80.0,
            "max_heading_error": math.radians(40.0),
        },
    )
    self.rewards.single_leg_air_time_overrun = RewTerm(
        func=mdp.single_leg_air_time_overrun_penalty,
        weight=-3.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="ankle_pitch_.*"),
            "max_air_time": 0.40,
        },
    )
    self.rewards.obstacle_forward_swing = RewTerm(
        func=mdp.obstacle_forward_swing_reward,
        weight=4.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="ankle_pitch_.*"),
            "asset_cfg": SceneEntityCfg("robot", body_names="ankle_pitch_.*"),
            "ray_sensor_cfg": SceneEntityCfg("height_scanner"),
            "non_foot_sensor_cfg": SceneEntityCfg("contact_forces", body_names="(?!ankle_pitch_).*"),
            "obstacle_threshold": 0.06,
            "max_obstacle_height": 0.34,
            "target_forward_delta": 0.16,
            "forward_std": 0.08,
            "min_forward_delta": 0.04,
            "liftoff_margin": 0.02,
            "min_liftoff": 0.05,
            "max_liftoff": 0.12,
            "liftoff_std": 0.06,
            "max_stance_foot_speed": 0.08,
            "max_tilt": 0.50,
            "max_yaw": math.radians(25.0),
        },
    )
    self.rewards.high_obstacle_approach = RewTerm(
        func=mdp.high_obstacle_approach_reward,
        weight=0.65,
        params={
            "ray_sensor_cfg": SceneEntityCfg("height_scanner"),
            "non_foot_sensor_cfg": SceneEntityCfg("contact_forces", body_names="(?!ankle_pitch_).*"),
            "min_obstacle_height": 0.05,
            "target_speed": 0.18,
            "speed_std": 0.14,
            "max_speed": 0.42,
            "max_yaw": math.radians(25.0),
            "yaw_std": math.radians(14.0),
            "max_tilt": 0.50,
            "min_base_height": 0.64,
            "non_foot_threshold": 5.0,
        },
    )
    self.rewards.obstacle_approach_progress = RewTerm(
        func=mdp.obstacle_approach_progress_reward,
        weight=1.00,
        params={
            "ray_sensor_cfg": SceneEntityCfg("height_scanner"),
            "non_foot_sensor_cfg": SceneEntityCfg("contact_forces", body_names="(?!ankle_pitch_).*"),
            "obstacle_threshold": 0.06,
            "target_heading": 0.0,
            "target_speed": 0.28,
            "speed_std": 0.16,
            "max_speed": 0.52,
            "yaw_std": math.radians(12.0),
            "max_yaw": math.radians(35.0),
            "lateral_speed_std": 0.10,
            "lateral_deadband": 0.08,
            "lateral_pos_std": 0.35,
            "max_tilt": 0.55,
            "non_foot_threshold": 5.0,
        },
    )
    self.rewards.obstacle_heading_error = RewTerm(
        func=mdp.obstacle_heading_error_deadband,
        weight=-1.0,
        params={
            "ray_sensor_cfg": SceneEntityCfg("height_scanner"),
            "obstacle_threshold": 0.05,
            "target_heading": 0.0,
            "deadband": math.radians(10.0),
        },
    )
    self.rewards.obstacle_heading_alignment = RewTerm(
        func=mdp.obstacle_heading_alignment_exp,
        weight=2.2,
        params={
            "ray_sensor_cfg": SceneEntityCfg("height_scanner"),
            "obstacle_threshold": 0.05,
            "target_heading": 0.0,
            "std": math.radians(14.0),
        },
    )
    self.rewards.undesired_contacts.weight = -3.5
    self.rewards.undesired_contacts.params["threshold"] = 5.0
    self.rewards.keep_balance.weight = 0.85
    self.rewards.base_height_exp.weight = 0.95
    self.rewards.base_height_exp.params["target_height"] = 0.74
    self.rewards.feet_distance.weight = 0.4
    # feet_distance only constrains lateral Y stance width; forward X stride is shaped separately below.
    self.rewards.feet_distance.params["max_dist"] = 0.35
    self.rewards.feet_distance.params["desired_dist"] = 0.25
    self.rewards.double_flight.weight = -1.5
    self.rewards.flat_orientation_l2.weight = 0.0
    self.rewards.forward_speed_shortfall.weight = 0.0
    self.rewards.world_lin_vel_y_abs.weight = -0.1
    self.rewards.predicted_path_lateral_error.weight = -0.2
    self.rewards.action_rate_l2.weight = -0.05
    self.rewards.dof_vel_l2.weight = -2.0e-4
    self.rewards.dof_acc_l2.weight = -2.0e-6
    self.rewards.same_feet_x_position.weight = 0.0
    self.rewards.single_stance_feet_x_separation.weight = 0.45
    self.rewards.single_stance_feet_x_separation.params["target_distance"] = 0.30
    self.rewards.single_stance_feet_x_separation.params["std"] = 0.14
    self.rewards.balanced_feet_x_symmetry.weight = 0.80
    self.rewards.balanced_feet_x_symmetry.params["std"] = 0.12
    self.rewards.foot_joint_orientation.weight = -0.11
    self.rewards.stance_foot_stability = RewTerm(
        func=mdp.stance_foot_stability_reward,
        weight=2.00,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="ankle_pitch_.*"),
            "asset_cfg": SceneEntityCfg("robot", body_names="ankle_pitch_.*"),
            "foot_vel_std": 0.08,
            "max_tilt": 0.45,
        },
    )
    self.rewards.swing_foot_lateral_deviation = RewTerm(
        func=mdp.swing_foot_lateral_deviation,
        weight=0.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="ankle_pitch_.*"),
            "asset_cfg": SceneEntityCfg("robot", body_names="ankle_pitch_.*"),
            "max_abs_y": 0.15,
        },
    )
    self.rewards.swing_knee_flexion = RewTerm(
        func=mdp.swing_knee_flexion_reward,
        weight=0.08,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="ankle_pitch_.*"),
            "asset_cfg": SceneEntityCfg("robot", joint_names="knee_[LR]_Joint"),
            "target_flexion": 0.30,
            "std": 0.20,
            "min_forward_speed": 0.12,
            "max_tilt": 0.55,
        },
    )
    self.rewards.forward_swing_knee_lift = RewTerm(
        func=mdp.forward_swing_knee_lift_reward,
        weight=0.45,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="ankle_pitch_.*"),
            "feet_cfg": SceneEntityCfg("robot", body_names="ankle_pitch_.*"),
            "knee_cfg": SceneEntityCfg("robot", joint_names="knee_[LR]_Joint"),
            "min_forward_delta": 0.04,
            "target_lift": 0.15,
            "lift_std": 0.05,
            "target_knee_flexion": 0.5,
            "knee_std": 0.20,
            "min_lift": 0.04,
            "max_tilt": 0.50,
            "min_base_height": 0.64,
            "max_heading_error": math.radians(25.0),
            "max_stance_foot_speed": 0.12,
            "min_stance_force": 320.0,
        },
    )
    self.rewards.stance_knee_extension = RewTerm(
        func=mdp.stance_knee_extension_reward,
        weight=0.75,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="ankle_pitch_.*"),
            "asset_cfg": SceneEntityCfg("robot", joint_names="knee_[LR]_Joint"),
            "std": 0.16,
        },
    )


def _use_stand_lift_fast_convergence(self):
    """Fresh stand-checkpoint line: strict width, early lift, and staged obstacle exposure."""
    self.commands.base_velocity.rel_standing_envs = 0.08
    self.commands.base_velocity.ranges.lin_vel_x = (0.14, 0.42)
    self.commands.gait_command.ranges.frequencies = (0.45, 0.68)
    self.commands.gait_command.ranges.durations = (0.60, 0.74)
    self.commands.gait_command.ranges.swing_height = (0.14, 0.24)

    self.rewards.keep_balance.weight = 0.75
    self.rewards.base_height_exp.weight = 0.90
    self.rewards.flat_orientation_l2.weight = -0.15
    self.rewards.track_world_lin_vel_x_exp.weight = 4.0
    self.rewards.track_lin_vel_x_exp.weight = 2.8
    self.rewards.forward_speed_shortfall.weight = -1.0
    self.rewards.predicted_path_lateral_error.weight = -0.2
    self.rewards.world_lin_vel_y_abs.weight = -0.1
    self.rewards.world_pos_y_abs.weight = -0.35
    self.rewards.world_pos_y_abs.params["deadband"] = 0.24

    self.rewards.feet_distance.weight = 0.85
    self.rewards.feet_distance.params["min_dist"] = 0.16
    self.rewards.feet_distance.params["max_dist"] = 0.35
    self.rewards.feet_distance.params["desired_dist"] = 0.28
    self.rewards.feet_distance.params["std"] = 0.08
    self.rewards.single_stance_feet_x_separation.weight = 0.45
    self.rewards.single_stance_feet_x_separation.params["target_distance"] = 0.30
    self.rewards.single_stance_feet_x_separation.params["std"] = 0.14
    self.rewards.feet_slide.weight = -0.18
    self.rewards.foot_contact_force.weight = -0.0004
    self.rewards.double_flight.weight = -1.2

    self.rewards.single_stance.weight = 1.05
    self.rewards.walking_single_leg_air_time.weight = 5.0
    self.rewards.walking_single_leg_air_time.params["target_air_time"] = 0.30
    self.rewards.walking_single_leg_air_time.params["min_air_time"] = 0.14
    self.rewards.walking_single_leg_air_time.params["max_air_time"] = 0.40
    self.rewards.walking_single_leg_air_time.params["min_base_height"] = 0.64
    self.rewards.walking_single_leg_air_time.params["min_stance_force"] = 130.0
    self.rewards.single_leg_air_time_overrun = RewTerm(
        func=mdp.single_leg_air_time_overrun_penalty,
        weight=-3.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="ankle_pitch_.*"),
            "max_air_time": 0.40,
        },
    )
    self.rewards.stance_foot_stability.weight = 2.4
    self.rewards.stance_knee_extension.weight = 0.95
    self.rewards.action_rate_l2.weight = -0.05
    self.rewards.dof_vel_l2.weight = -2.0e-4
    self.rewards.dof_acc_l2.weight = -2.0e-6
    self.rewards.balanced_feet_x_symmetry.weight = 0.85
    self.rewards.balanced_feet_x_symmetry.params["std"] = 0.12
    self.rewards.foot_joint_orientation.weight = -0.11

    self.rewards.swing_foot_clearance.weight = 2.2
    self.rewards.swing_foot_liftoff.weight = 0.55
    if not hasattr(self.rewards, "linear_forward_swing_lift"):
        self.rewards.linear_forward_swing_lift = RewTerm(
            func=mdp.linear_forward_swing_lift_reward,
            weight=0.015,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names="ankle_pitch_.*"),
                "asset_cfg": SceneEntityCfg("robot", body_names="ankle_pitch_.*"),
                "min_forward_delta": 0.03,
                "min_lift": 0.015,
                "max_lift": 0.18,
                "max_tilt": 0.80,
                "min_base_height": 0.60,
                "max_heading_error": math.radians(40.0),
                "max_stance_foot_speed": 0.25,
                "min_stance_force": 80.0,
            },
        )
    else:
        self.rewards.linear_forward_swing_lift.weight = 0.015
        self.rewards.linear_forward_swing_lift.params["min_lift"] = 0.015
        self.rewards.linear_forward_swing_lift.params["max_lift"] = 0.18
        self.rewards.linear_forward_swing_lift.params["min_base_height"] = 0.60
    self.rewards.forward_swing_knee_lift.weight = 0.35
    self.rewards.forward_swing_knee_lift.params["target_lift"] = 0.20
    self.rewards.forward_swing_knee_lift.params["min_lift"] = 0.035
    self.rewards.forward_swing_knee_lift.params["min_base_height"] = 0.60
    self.rewards.forward_swing_knee_lift.params["min_stance_force"] = 180.0
    self.rewards.swing_knee_flexion.weight = 0.12
    self.rewards.swing_knee_flexion.params["target_flexion"] = 0.36

    self.rewards.obstacle_forward_swing.weight = 2.4
    self.rewards.obstacle_approach_progress.weight = 0.75
    self.rewards.obstacle_heading_error.weight = -0.8
    self.rewards.obstacle_heading_alignment.weight = 1.4
    self.rewards.high_obstacle_approach.weight = 0.45


def _use_scratch_gait_rewards(self):
    """From-scratch gait line: no stand-prior checkpoint, standing as a command mode.

    Design constraints (see doc/GAIT_SCRATCH_RUNPLAN.md):
    - Reward stack = 27 active terms: 21 task-level design terms set below plus 6
      inherited safety/physics terms pinned at their base values. The foot-posture
      shapers and lateral band-aids accumulated while escaping the stand-still local
      optimum are zeroed: they pulled foot orientation in conflicting directions
      (in-toe gait, lateral drift) in every resumed line.
    - Stair skills flow through the single terrain-adaptive swing-clearance term
      (yaw-referenced, height-banded) instead of the five obstacle rescue terms.
    """
    # Commands: straight-track walking; standing enters as the zero-velocity command.
    self.commands.base_velocity.heading_command = False
    self.commands.base_velocity.rel_standing_envs = 0.20
    self.commands.base_velocity.rel_heading_envs = 0.0
    self.commands.base_velocity.ranges.lin_vel_x = (0.12, 0.50)
    self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
    self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
    self.commands.base_velocity.ranges.heading = (0.0, 0.0)
    self.commands.gait_command.ranges.frequencies = (0.50, 0.80)
    self.commands.gait_command.ranges.durations = (0.58, 0.70)
    self.commands.gait_command.ranges.swing_height = (0.15, 0.24)

    # Velocity tracking.
    self.rewards.track_lin_vel_x_exp.weight = 4.0
    self.rewards.track_lin_vel_x_exp.params["std"] = 0.25
    self.rewards.track_lin_vel_y_exp.weight = 1.5
    self.rewards.track_lin_vel_y_exp.params["std"] = 0.18
    self.rewards.track_ang_vel_z_exp.weight = 2.0
    self.rewards.track_ang_vel_z_exp.params["std"] = 0.20
    self.rewards.forward_speed_shortfall.weight = -1.5
    self.rewards.forward_speed_shortfall.params["deadband"] = 0.05

    # Posture.
    self.rewards.keep_balance.weight = 0.6
    self.rewards.base_height_exp.weight = 0.9
    self.rewards.base_height_exp.params["target_height"] = 0.74
    self.rewards.flat_orientation_l2.weight = -0.5

    # Gait core; terrain-adaptive clearance keeps its yaw-referenced height bands.
    self.rewards.single_stance.weight = 0.4
    self.rewards.walking_single_leg_air_time.weight = 3.0
    self.rewards.single_leg_air_time_overrun.weight = -2.0
    self.rewards.swing_foot_clearance.weight = 3.0
    self.rewards.swing_foot_liftoff.weight = 0.35
    self.rewards.double_flight.weight = -1.5

    # Slip and contact safety.
    self.rewards.feet_slide.weight = -0.25
    self.rewards.feet_ang_slide_penalty.weight = -0.25
    self.rewards.undesired_contacts.weight = -1.5
    self.rewards.undesired_contacts.params["threshold"] = 5.0

    # Route discipline: two terms replace six overlapping lateral shapers.
    self.rewards.world_pos_y_abs.weight = -0.5
    self.rewards.world_pos_y_abs.params["deadband"] = 0.25
    self.rewards.heading_error_abs.weight = -0.8
    self.rewards.heading_error_abs.params["deadband"] = math.radians(10.0)

    # Action regularizers.
    self.rewards.action_rate_l2.weight = -0.02
    self.rewards.dof_vel_l2.weight = -2.0e-4
    self.rewards.dof_acc_l2.weight = -1.0e-6

    # Inherited safety/physics limit terms: kept, weights pinned at their base
    # values so the scratch line is self-documenting (see modification report).
    self.rewards.ang_vel_xy_l2.weight = -0.15
    self.rewards.dof_pos_limits.weight = -0.2
    self.rewards.dof_vel_limits.weight = -0.025
    self.rewards.applied_torque_limits.weight = -1.0e-2
    self.rewards.dof_power_l1.weight = -2.5e-7
    self.rewards.foot_contact_force.weight = -0.002

    # Zeroed: base posture priors — joint pull toward the default (standing) pose
    # and knee yaw orientation are the same shaping family as the A-group terms.
    self.rewards.joint_deviation_l1.weight = 0.0
    self.rewards.orientation_exp_knee.weight = 0.0
    # Zeroed: already zero in the gait parent chain; pinned explicitly.
    self.rewards.feet_air_time.weight = 0.0
    self.rewards.dof_torques_l2.weight = 0.0

    # Zeroed: duplicate world-frame tracking and overshoot guards.
    self.rewards.track_world_lin_vel_x_exp.weight = 0.0
    self.rewards.forward_overspeed.weight = 0.0
    self.rewards.gait_reward.weight = 0.0
    # Zeroed: foot-posture shapers that fought each other (in-toe / drift artifacts).
    self.rewards.knee_joint_orientation.weight = 0.0
    self.rewards.same_feet_x_position.weight = 0.0
    self.rewards.feet_distance.weight = 0.3
    self.rewards.feet_distance.params["desired_dist"] = 0.22
    self.rewards.foot_joint_orientation.weight = 0.0
    self.rewards.single_stance_feet_x_separation.weight = 0.0
    self.rewards.balanced_feet_x_symmetry.weight = 0.0
    self.rewards.double_stance_feet_x_alignment.weight = 0.0
    self.rewards.stance_foot_stability.weight = 0.0
    self.rewards.stance_knee_extension.weight = 0.0
    self.rewards.swing_knee_flexion.weight = 0.0
    self.rewards.forward_swing_knee_lift.weight = 0.0
    self.rewards.swing_foot_lateral_deviation.weight = -0.2
    # Zeroed: stand-escape lateral band-aids.
    self.rewards.centerline_y.weight = 0.0
    self.rewards.heading_alignment.weight = 0.0
    self.rewards.world_lin_vel_y_abs.weight = 0.0
    self.rewards.predicted_path_lateral_error.weight = 0.0
    self.rewards.lateral_correction_velocity.weight = 0.0
    # Zeroed: obstacle rescue terms; the adaptive clearance term owns stair skills.
    self.rewards.obstacle_forward_swing.weight = 0.0
    self.rewards.obstacle_approach_progress.weight = 0.0
    self.rewards.obstacle_heading_error.weight = 0.0
    self.rewards.obstacle_heading_alignment.weight = 0.0
    self.rewards.high_obstacle_approach.weight = 0.0
    # Zeroed: redundant smoothness penalties (action_rate + dof_acc cover them).
    self.rewards.action_smoothness.weight = 0.0
    self.rewards.torques_smoothness_penalty.weight = 0.0
    self.rewards.body_lin_acc.weight = 0.0
    self.rewards.lin_vel_z_l2.weight = 0.0


def _use_scratch_curriculum_resets(self):
    """Bumps-free TronCamp curriculum for the scratch line.

    The official evaluation course has no bumps section, so bumps_entry gets zero
    spawn weight at every phase. Segment lookup in the reset function is
    name-based and sampling normalizes internally, so dropping the key equals
    zero weight and the remaining segments keep their relative ratios.
    Caveat: the generated track still contains the bumps feature between the
    downhill and stairs sections — zero spawn weight does not prevent traversal.
    """
    _use_tron_camp_curriculum_resets(self)
    params = self.events.reset_robot_base.params
    params["curriculum_segments"] = tuple(
        {**segment, "weight": 0.0} if segment["name"] == "bumps_entry" else segment
        for segment in params["curriculum_segments"]
    )
    params["phase_schedule"] = tuple(
        {
            "step": phase["step"],
            "weights": {name: weight for name, weight in phase["weights"].items() if name != "bumps_entry"},
        }
        for phase in params["phase_schedule"]
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
    self.rewards.feet_slide.weight = -0.18
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
        "adaptive_curriculum": True,
        "promote_distance": 4.0,
        "demote_distance": 1.2,
        "max_adaptive_level": 4,
        "randomize_mastered_levels": True,
        "phase_schedule": (
            {
                "step": 0,
                "weights": {
                    "start_flat": 0.80,
                    "up_ramp_entry": 0.10,
                    "downhill_exit": 0.10,
                },
            },
            {
                "step": 300 * 24,
                "weights": {
                    "start_flat": 0.32,
                    "up_ramp_entry": 0.14,
                    "downhill_exit": 0.12,
                    "bumps_entry": 0.30,
                    "stairs_entry": 0.12,
                },
            },
            {
                "step": 600 * 24,
                "weights": {
                    "start_flat": 0.24,
                    "up_ramp_entry": 0.10,
                    "downhill_exit": 0.10,
                    "bumps_entry": 0.30,
                    "stairs_entry": 0.25,
                    "platform_entry": 0.01,
                },
            },
            {
                # 熔断点 1：减缓难度爬升，锁定在温和台阶比例
                "step": 1000 * 24,
                "weights": {
                    "start_flat": 0.24,
                    "up_ramp_entry": 0.10,
                    "downhill_exit": 0.10,
                    "bumps_entry": 0.25,
                    "stairs_entry": 0.30, # 限制台阶比例不超过 0.30
                    "platform_entry": 0.01,
                },
            },
            {
                # 熔断点 2：1400 轮不再升级难度，继续巩固
                "step": 1400 * 24,
                "weights": {
                    "start_flat": 0.24,
                    "up_ramp_entry": 0.10,
                    "downhill_exit": 0.10,
                    "bumps_entry": 0.25,
                    "stairs_entry": 0.30, 
                    "platform_entry": 0.01,
                },
            },
            {
                # 熔断点 3：1800 轮同样不升级，彻底固化步态
                "step": 1800 * 24,
                "weights": {
                    "start_flat": 0.24,
                    "up_ramp_entry": 0.10,
                    "downhill_exit": 0.10,
                    "bumps_entry": 0.25,
                    "stairs_entry": 0.30, 
                    "platform_entry": 0.01,
                },

            },
        ),
    }


def _use_stand_lift_curriculum_resets(self):
    _use_tron_camp_curriculum_resets(self)
    if self.events.reset_robot_base is None:
        return
    self.events.reset_robot_base.params["promote_distance"] = 3.0
    self.events.reset_robot_base.params["demote_distance"] = 0.8
    self.events.reset_robot_base.params["max_adaptive_level"] = 3
    self.events.reset_robot_base.params["phase_schedule"] = (
        {
            "step": 0,
            "weights": {
                "start_flat": 0.70,
                "up_ramp_entry": 0.18,
                "downhill_exit": 0.12,
            },
        },
        {
            "step": 200 * 24,
            "weights": {
                "start_flat": 0.42,
                "up_ramp_entry": 0.18,
                "downhill_exit": 0.15,
                "bumps_entry": 0.15,
                "stairs_entry": 0.10,
            },
        },
        {
            "step": 600 * 24,
            "weights": {
                "start_flat": 0.22,
                "up_ramp_entry": 0.12,
                "downhill_exit": 0.12,
                "bumps_entry": 0.24,
                "stairs_entry": 0.28,
                "platform_entry": 0.02,
            },
        },
        {
            "step": 1000 * 24,
            "weights": {
                "start_flat": 0.18,
                "up_ramp_entry": 0.10,
                "downhill_exit": 0.10,
                "bumps_entry": 0.22,
                "stairs_entry": 0.36,
                "platform_entry": 0.04,
            },
        },
    )


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
    self.rewards.keep_balance.weight = 0.35
    self.rewards.base_height_exp.weight = 0.70
    self.rewards.flat_orientation_l2.weight = -1.15
    self.rewards.lin_vel_z_l2.weight = 0.0
    self.rewards.body_lin_acc.weight = -0.004
    self.rewards.gait_reward.weight = 0.45
    self.rewards.feet_distance.weight = 0.5
    self.rewards.feet_air_time.weight = 0.0
    self.rewards.feet_slide.weight = -0.24
    self.rewards.feet_ang_slide_penalty.weight = -0.28
    self.rewards.action_rate_l2.weight = -0.001
    self.rewards.dof_torques_l2.weight = 0.0
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
        weight=2.5,
        params={"std": 0.28},
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
        weight=-2.0,
    )
    self.rewards.world_pos_y_abs = RewTerm(
        func=mdp.world_pos_y_deadband,
        weight=-0.5,
        params={"deadband": 0.20},
    )
    self.rewards.predicted_path_lateral_error = RewTerm(
        func=mdp.predicted_path_lateral_error,
        weight=-1.5,
        params={
            "target_heading": 0.0,
            "hip_weight": 0.03,
            "step_weight": 0.65,
            "same_direction_weight": 0.05,
            "deadband": 0.03,
            "feet_cfg": SceneEntityCfg("robot", body_names="ankle_pitch_.*"),
        },
    )
    self.rewards.lateral_correction_velocity = RewTerm(
        func=mdp.lateral_correction_velocity_exp,
        weight=2.5,
        params={
            "gain": 0.8,
            "far_gain": 1.0,
            "far_threshold": 0.30,
            "max_correction_speed": 0.28,
            "std": 0.24,
            "deadband": 0.08,
        },
    )
    self.rewards.forward_speed_shortfall = RewTerm(
        func=mdp.lin_vel_x_shortfall_penalty,
        weight=-3.2,
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
        weight=3.0,
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


def _use_obstacle_heading_termination(
    self,
    limit_angle=math.radians(35.0),
    obstacle_threshold=0.05,
):
    self.terminations.bad_obstacle_heading = DoneTerm(
        func=mdp.bad_obstacle_heading,
        params={
            "ray_sensor_cfg": SceneEntityCfg("height_scanner"),
            "obstacle_threshold": obstacle_threshold,
            "limit_angle": limit_angle,
            "target_heading": 0.0,
        },
    )


def _use_competition_exposure_training(self):
    # Keep fall/contact terminations, but let imperfect policies survive long enough
    # to collect experience on the downhill exit and stair sections.
    _use_gait_heading_termination(self, limit_angle=math.radians(50.0))
    _use_gait_lateral_termination(self, limit_distance=1.00)
    _use_obstacle_heading_termination(
        self,
        limit_angle=math.radians(55.0),
        obstacle_threshold=0.12,
    )
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


def _use_perceptive_rough_base_training(self):
    """Rough-terrain baseline branch: prioritize balance and clean contacts over route-tight gait shaping."""
    self.commands.base_velocity.heading_command = True
    self.commands.base_velocity.rel_standing_envs = 0.08
    self.commands.base_velocity.rel_heading_envs = 1.0
    self.commands.base_velocity.ranges.lin_vel_x = (0.20, 0.48)
    self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
    self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
    self.commands.base_velocity.ranges.heading = (0.0, 0.0)
    self.commands.gait_command.ranges.frequencies = (0.55, 0.80)
    self.commands.gait_command.ranges.durations = (0.56, 0.68)
    self.commands.gait_command.ranges.swing_height = (0.11, 0.18)

    self.rewards.track_lin_vel_x_exp.weight = 2.6
    self.rewards.track_world_lin_vel_x_exp.weight = 2.2
    self.rewards.track_lin_vel_y_exp.weight = 1.0
    self.rewards.track_ang_vel_z_exp.weight = 1.4
    self.rewards.keep_balance.weight = 1.15
    self.rewards.base_height_exp.weight = 1.05
    self.rewards.flat_orientation_l2.weight = -0.85
    self.rewards.lin_vel_z_l2.weight = -1.10
    self.rewards.body_lin_acc.weight = -0.006
    self.rewards.gait_reward.weight = 0.25
    self.rewards.feet_slide.weight = -0.32
    self.rewards.feet_ang_slide_penalty.weight = -0.30
    self.rewards.action_rate_l2.weight = -0.008
    self.rewards.action_smoothness.weight = -0.00025
    self.rewards.knee_joint_orientation.weight = -0.45

    self.rewards.centerline_y.weight = 0.8
    self.rewards.heading_alignment.weight = 0.25
    self.rewards.heading_error_abs.weight = -0.5
    self.rewards.world_lin_vel_y_abs.weight = -0.7
    self.rewards.world_pos_y_abs.weight = -0.9
    self.rewards.world_pos_y_abs.params["deadband"] = 0.28
    self.rewards.predicted_path_lateral_error.weight = -0.35
    self.rewards.lateral_correction_velocity.weight = 0.8

    self.rewards.forward_speed_shortfall.weight = -1.2
    self.rewards.forward_overspeed.weight = -1.6
    self.rewards.single_stance.weight = 0.35
    self.rewards.swing_foot_clearance.weight = 0.45
    self.rewards.swing_foot_liftoff.weight = 0.22
    self.rewards.obstacle_forward_swing.weight = 0.0
    self.rewards.high_obstacle_approach.weight = 0.45
    self.rewards.obstacle_heading_error.weight = -1.0
    self.rewards.undesired_contacts.weight = -3.0
    self.rewards.double_flight.weight = -2.2
    self.rewards.single_stance_feet_x_separation.weight = 0.03
    self.rewards.stance_foot_stability.weight = 1.35
    self.rewards.swing_foot_lateral_deviation.weight = -0.55
    self.rewards.swing_knee_flexion.weight = 0.04
    self.rewards.stance_knee_extension.weight = 0.90
    self.rewards.balanced_feet_x_symmetry.weight = 0.35

    _use_gait_heading_termination(self, limit_angle=math.radians(55.0))
    _use_gait_lateral_termination(self, limit_distance=1.20)
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
        _use_obstacle_heading_termination(
            self,
            limit_angle=math.radians(50.0),
            obstacle_threshold=0.12,
        )
        self.scene.num_envs = 1


@configclass
class SFYG_TRON2A_PerceptiveStandLiftTronCampEnvCfg(SFYG_TRON2A_BlindGaitTronCampEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _use_base_height_scanner(self)
        _use_policy_height_scan(self)
        _use_perceptive_stair_clearance(self)
        _use_stand_lift_fast_convergence(self)
        _use_competition_exposure_training(self)
        _use_stand_lift_curriculum_resets(self)


@configclass
class SFYG_TRON2A_PerceptiveStandLiftTronCampEnvCfg_PLAY(SFYG_TRON2A_BlindGaitTronCampEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        _use_base_height_scanner(self)
        _use_policy_height_scan(self)
        _use_perceptive_stair_clearance(self)
        _use_stand_lift_fast_convergence(self)
        _use_obstacle_heading_termination(
            self,
            limit_angle=math.radians(50.0),
            obstacle_threshold=0.12,
        )
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


@configclass
class SFYG_TRON2A_PerceptiveRoughBaseTronCampEnvCfg(SFYG_TRON2A_PerceptiveGaitTronCampEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _use_perceptive_rough_base_training(self)


@configclass
class SFYG_TRON2A_PerceptiveRoughBaseTronCampEnvCfg_PLAY(SFYG_TRON2A_PerceptiveGaitTronCampEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        _use_perceptive_rough_base_training(self)
        self.scene.num_envs = 1


@configclass
class SFYG_TRON2A_PerceptiveGaitScratchTronCampEnvCfg(SFYG_TRON2A_PerceptiveGaitTronCampEnvCfg):
    """From-scratch perceptive gait: standing is a command mode, lean reward stack.

    Train WITHOUT --resume: the whole point is to drop the stand-prior checkpoint.
    """

    def __post_init__(self):
        super().__post_init__()
        _use_scratch_gait_rewards(self)
        _use_scratch_curriculum_resets(self)


@configclass
class SFYG_TRON2A_PerceptiveGaitScratchTronCampEnvCfg_PLAY(SFYG_TRON2A_BlindGaitTronCampEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        _use_base_height_scanner(self)
        _use_policy_height_scan(self)
        _use_perceptive_stair_clearance(self)
        _use_scratch_gait_rewards(self)
        # Same curriculum reset function as training so the Play smoke test
        # exercises the bumps-free segment weights for real.
        _use_scratch_curriculum_resets(self)
        self.scene.num_envs = 1
