from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from bipedal_locomotion.assets.config.wheelfoot_yg_tron2a_cfg import WHEELFOOT_YG_TRON2A_CFG
from bipedal_locomotion.tasks.locomotion.robots.limx_wheelfoot_tron2a_env_cfg import (
    WF_TRON2A_BaseEnvCfg,
    WF_TRON2A_BaseEnvCfg_PLAY,
    WF_TRON2A_BlindFlatEnvCfg,
    WF_TRON2A_BlindFlatEnvCfg_PLAY,
)


WFYG_LEG_TORSO_LINKS = [
    "base_Link",
    "proximal_pitch_[LR]_Link",
    "proximal_roll_[LR]_Link",
    "proximal_yaw_[LR]_Link",
    "knee_[LR]_Link",
    "wheel_[LR]_Link",
]

WFYG_LEG_JOINTS = [
    "proximal_pitch_[LR]_Joint",
    "proximal_roll_[LR]_Joint",
    "proximal_yaw_[LR]_Joint",
    "knee_[LR]_Joint",
    "wheel_[LR]_Joint",
]


def _exclude_arm_from_events_and_rewards(self):
    if self.events.add_base_mass is not None:
        self.events.add_base_mass.params["asset_cfg"].body_names = "base_Link"

    if self.events.reset_robot_joints is not None:
        self.events.reset_robot_joints.params["asset_cfg"] = SceneEntityCfg("robot", joint_names=WFYG_LEG_JOINTS)

    if hasattr(self.rewards, "non_wheel_pos_limits") and self.rewards.non_wheel_pos_limits is not None:
        self.rewards.non_wheel_pos_limits.params["asset_cfg"].joint_names = WFYG_LEG_JOINTS


@configclass
class WFYG_TRON2A_BaseEnvCfg(WF_TRON2A_BaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = WHEELFOOT_YG_TRON2A_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        _exclude_arm_from_events_and_rewards(self)


@configclass
class WFYG_TRON2A_BaseEnvCfg_PLAY(WF_TRON2A_BaseEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = WHEELFOOT_YG_TRON2A_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        _exclude_arm_from_events_and_rewards(self)


@configclass
class WFYG_TRON2A_BlindFlatEnvCfg(WF_TRON2A_BlindFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = WHEELFOOT_YG_TRON2A_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        _exclude_arm_from_events_and_rewards(self)


@configclass
class WFYG_TRON2A_BlindFlatEnvCfg_PLAY(WF_TRON2A_BlindFlatEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = WHEELFOOT_YG_TRON2A_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        _exclude_arm_from_events_and_rewards(self)
