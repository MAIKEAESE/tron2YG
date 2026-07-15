import gymnasium as gym

from bipedal_locomotion.tasks.locomotion.agents.limx_rsl_rl_ppo_cfg import (
    SF_TRON2AFlatPPORunnerCfg, WF_TRON2AFlatPPORunnerCfg,
    SFYG_TRON2AFlatPPORunnerCfg, SFYG_TRON2ATronCampPPORunnerCfg,
    SFYG_TRON2ATronCampForwardPPORunnerCfg, SFYG_TRON2AGaitPPORunnerCfg,
    SFYG_TRON2APerceptiveGaitPPORunnerCfg, SFYG_TRON2APerceptiveRandomDirPPORunnerCfg,
    SFYG_TRON2APerceptiveRoughBasePPORunnerCfg,
    WFYG_TRON2AFlatPPORunnerCfg,
)

from . import (
    limx_solefoot_tron2a_env_cfg,
    limx_wheelfoot_tron2a_env_cfg,
    limx_solefoot_yg_tron2a_env_cfg,
    limx_wheelfoot_yg_tron2a_env_cfg,
)

##
# Create PPO runners for RSL-RL
##

limx_sf_tron2a_blind_flat_runner_cfg = SF_TRON2AFlatPPORunnerCfg()

limx_wf_tron2a_blind_flat_runner_cfg = WF_TRON2AFlatPPORunnerCfg()

limx_sfyg_tron2a_blind_flat_runner_cfg = SFYG_TRON2AFlatPPORunnerCfg()

limx_sfyg_tron2a_blind_troncamp_runner_cfg = SFYG_TRON2ATronCampPPORunnerCfg()

limx_sfyg_tron2a_blind_troncamp_forward_runner_cfg = SFYG_TRON2ATronCampForwardPPORunnerCfg()

limx_sfyg_tron2a_blind_gait_runner_cfg = SFYG_TRON2AGaitPPORunnerCfg()

limx_sfyg_tron2a_perceptive_gait_runner_cfg = SFYG_TRON2APerceptiveGaitPPORunnerCfg()

limx_sfyg_tron2a_perceptive_random_dir_runner_cfg = SFYG_TRON2APerceptiveRandomDirPPORunnerCfg()

limx_sfyg_tron2a_perceptive_rough_base_runner_cfg = SFYG_TRON2APerceptiveRoughBasePPORunnerCfg()

limx_wfyg_tron2a_blind_flat_runner_cfg = WFYG_TRON2AFlatPPORunnerCfg()


##
# Register Gym environments
##


######################################
# SF_TRON2A Blind Flat Environment
######################################
gym.register(
    id="Isaac-Limx-SF-TRON2A-Blind-Flat-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": limx_solefoot_tron2a_env_cfg.SF_TRON2A_BlindFlatEnvCfg,
        "rsl_rl_cfg_entry_point": limx_sf_tron2a_blind_flat_runner_cfg,
    },
)

gym.register(
    id="Isaac-Limx-SF-TRON2A-Blind-Flat-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": limx_solefoot_tron2a_env_cfg.SF_TRON2A_BlindFlatEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": limx_sf_tron2a_blind_flat_runner_cfg,
    },
)


######################################
# WF_TRON2A Blind Flat Environment
######################################
gym.register(
    id="Isaac-Limx-WF-TRON2A-Blind-Flat-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": limx_wheelfoot_tron2a_env_cfg.WF_TRON2A_BlindFlatEnvCfg,
        "rsl_rl_cfg_entry_point": limx_wf_tron2a_blind_flat_runner_cfg,
    },
)

gym.register(
    id="Isaac-Limx-WF-TRON2A-Blind-Flat-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": limx_wheelfoot_tron2a_env_cfg.WF_TRON2A_BlindFlatEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": limx_wf_tron2a_blind_flat_runner_cfg,
    },
)


######################################
# SFYG_TRON2A Blind Flat Environment
######################################
gym.register(
    id="Isaac-Limx-SFYG-TRON2A-Blind-Flat-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": limx_solefoot_yg_tron2a_env_cfg.SFYG_TRON2A_BlindFlatEnvCfg,
        "rsl_rl_cfg_entry_point": limx_sfyg_tron2a_blind_flat_runner_cfg,
    },
)

gym.register(
    id="Isaac-Limx-SFYG-TRON2A-Blind-Flat-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": limx_solefoot_yg_tron2a_env_cfg.SFYG_TRON2A_BlindFlatEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": limx_sfyg_tron2a_blind_flat_runner_cfg,
    },
)


gym.register(
    id="Isaac-Limx-SFYG-TRON2A-Blind-TronCamp-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": limx_solefoot_yg_tron2a_env_cfg.SFYG_TRON2A_BlindTronCampEnvCfg,
        "rsl_rl_cfg_entry_point": limx_sfyg_tron2a_blind_troncamp_runner_cfg,
    },
)

gym.register(
    id="Isaac-Limx-SFYG-TRON2A-Blind-TronCamp-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": limx_solefoot_yg_tron2a_env_cfg.SFYG_TRON2A_BlindTronCampEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": limx_sfyg_tron2a_blind_troncamp_runner_cfg,
    },
)


gym.register(
    id="Isaac-Limx-SFYG-TRON2A-Blind-TronCamp-Forward-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": limx_solefoot_yg_tron2a_env_cfg.SFYG_TRON2A_BlindTronCampForwardEnvCfg,
        "rsl_rl_cfg_entry_point": limx_sfyg_tron2a_blind_troncamp_forward_runner_cfg,
    },
)

gym.register(
    id="Isaac-Limx-SFYG-TRON2A-Blind-TronCamp-Forward-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": limx_solefoot_yg_tron2a_env_cfg.SFYG_TRON2A_BlindTronCampForwardEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": limx_sfyg_tron2a_blind_troncamp_forward_runner_cfg,
    },
)


gym.register(
    id="Isaac-Limx-SFYG-TRON2A-Blind-Gait-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": limx_solefoot_yg_tron2a_env_cfg.SFYG_TRON2A_BlindGaitEnvCfg,
        "rsl_rl_cfg_entry_point": limx_sfyg_tron2a_blind_gait_runner_cfg,
    },
)

gym.register(
    id="Isaac-Limx-SFYG-TRON2A-Blind-Gait-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": limx_solefoot_yg_tron2a_env_cfg.SFYG_TRON2A_BlindGaitEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": limx_sfyg_tron2a_blind_gait_runner_cfg,
    },
)


gym.register(
    id="Isaac-Limx-SFYG-TRON2A-Blind-Gait-TronCamp-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": limx_solefoot_yg_tron2a_env_cfg.SFYG_TRON2A_BlindGaitTronCampEnvCfg,
        "rsl_rl_cfg_entry_point": limx_sfyg_tron2a_blind_gait_runner_cfg,
    },
)

gym.register(
    id="Isaac-Limx-SFYG-TRON2A-Blind-Gait-TronCamp-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": limx_solefoot_yg_tron2a_env_cfg.SFYG_TRON2A_BlindGaitTronCampEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": limx_sfyg_tron2a_blind_gait_runner_cfg,
    },
)


gym.register(
    id="Isaac-Limx-SFYG-TRON2A-Perceptive-Gait-TronCamp-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": limx_solefoot_yg_tron2a_env_cfg.SFYG_TRON2A_PerceptiveGaitTronCampEnvCfg,
        "rsl_rl_cfg_entry_point": limx_sfyg_tron2a_perceptive_gait_runner_cfg,
    },
)

gym.register(
    id="Isaac-Limx-SFYG-TRON2A-Perceptive-Gait-TronCamp-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": limx_solefoot_yg_tron2a_env_cfg.SFYG_TRON2A_PerceptiveGaitTronCampEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": limx_sfyg_tron2a_perceptive_gait_runner_cfg,
    },
)


gym.register(
    id="Isaac-Limx-SFYG-TRON2A-Perceptive-RandomDir-TronCamp-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": limx_solefoot_yg_tron2a_env_cfg.SFYG_TRON2A_PerceptiveRandomDirTronCampEnvCfg,
        "rsl_rl_cfg_entry_point": limx_sfyg_tron2a_perceptive_random_dir_runner_cfg,
    },
)

gym.register(
    id="Isaac-Limx-SFYG-TRON2A-Perceptive-RandomDir-TronCamp-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": limx_solefoot_yg_tron2a_env_cfg.SFYG_TRON2A_PerceptiveRandomDirTronCampEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": limx_sfyg_tron2a_perceptive_random_dir_runner_cfg,
    },
)


gym.register(
    id="Isaac-Limx-SFYG-TRON2A-Perceptive-RoughBase-TronCamp-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": limx_solefoot_yg_tron2a_env_cfg.SFYG_TRON2A_PerceptiveRoughBaseTronCampEnvCfg,
        "rsl_rl_cfg_entry_point": limx_sfyg_tron2a_perceptive_rough_base_runner_cfg,
    },
)

gym.register(
    id="Isaac-Limx-SFYG-TRON2A-Perceptive-RoughBase-TronCamp-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": limx_solefoot_yg_tron2a_env_cfg.SFYG_TRON2A_PerceptiveRoughBaseTronCampEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": limx_sfyg_tron2a_perceptive_rough_base_runner_cfg,
    },
)


######################################
# WFYG_TRON2A Blind Flat Environment
######################################
gym.register(
    id="Isaac-Limx-WFYG-TRON2A-Blind-Flat-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": limx_wheelfoot_yg_tron2a_env_cfg.WFYG_TRON2A_BlindFlatEnvCfg,
        "rsl_rl_cfg_entry_point": limx_wfyg_tron2a_blind_flat_runner_cfg,
    },
)

gym.register(
    id="Isaac-Limx-WFYG-TRON2A-Blind-Flat-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": limx_wheelfoot_yg_tron2a_env_cfg.WFYG_TRON2A_BlindFlatEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": limx_wfyg_tron2a_blind_flat_runner_cfg,
    },
)
