from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from bipedal_locomotion.assets.config.solefoot_yg_tron2a_cfg import SOLEFOOT_YG_TRON2A_CFG
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
    self.commands.base_velocity.heading_command = False
    self.commands.base_velocity.rel_standing_envs = 0.0
    self.commands.base_velocity.rel_heading_envs = 0.0
    self.commands.base_velocity.ranges.lin_vel_x = (0.75, 1.10)
    self.commands.base_velocity.ranges.lin_vel_y = (-0.05, 0.05)
    self.commands.base_velocity.ranges.ang_vel_z = (-0.10, 0.10)
    self.commands.base_velocity.ranges.heading = (0.0, 0.0)
    self.commands.gait_command.ranges.frequencies = (1.0, 1.35)


def _use_forward_walking_rewards(self):
    self.rewards.track_lin_vel_x_exp.weight = 4.0
    self.rewards.track_lin_vel_y_exp.weight = 0.3
    self.rewards.track_ang_vel_z_exp.weight = 0.3
    self.rewards.keep_balance.weight = 0.3
    self.rewards.base_height_exp.weight = 0.4
    self.rewards.gait_reward.weight = 0.8
    self.rewards.feet_air_time.weight = 0.10


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


@configclass
class SFYG_TRON2A_BlindTronCampForwardEnvCfg_PLAY(SFYG_TRON2A_BlindTronCampEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        _use_forward_walking_commands(self)
        _use_forward_walking_rewards(self)
        self.scene.num_envs = 1
