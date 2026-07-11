from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg
from bipedal_locomotion.utils.wrappers.rsl_rl.rl_mlp_cfg import EncoderCfg, RslRlPpoAlgorithmMlpCfg

import os
robot_type = os.getenv("ROBOT_TYPE")

#-----------------------------------------------------------------
@configclass
class SF_TRON2AFlatPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 15000
    save_interval = 500
    experiment_name = "sf_tron_2a_flat"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmMlpCfg(
        class_name="PPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        obs_history_len=10,
    )
    encoder = EncoderCfg(
        output_detach = True,
        num_output_dim = 3,
        hidden_dims = [256, 128],
        activation = "elu",
        orthogonal_init = False,
    )

#-----------------------------------------------------------------
@configclass
class WF_TRON2AFlatPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 15000
    save_interval = 500
    experiment_name = "wf_tron_2a_flat"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmMlpCfg(
        class_name="PPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        obs_history_len=10,
    )
    encoder = EncoderCfg(
        output_detach = True,
        num_output_dim = 3,
        hidden_dims = [256, 128],
        activation = "elu",
        orthogonal_init = False,
    )

#-----------------------------------------------------------------
@configclass
class SFYG_TRON2AFlatPPORunnerCfg(SF_TRON2AFlatPPORunnerCfg):
    experiment_name = "sfyg_tron_2a_flat"


#-----------------------------------------------------------------
@configclass
class SFYG_TRON2ATronCampPPORunnerCfg(SFYG_TRON2AFlatPPORunnerCfg):
    experiment_name = "sfyg_tron_2a_troncamp"


#-----------------------------------------------------------------
@configclass
class SFYG_TRON2ATronCampForwardPPORunnerCfg(SFYG_TRON2ATronCampPPORunnerCfg):
    max_iterations = 3000
    save_interval = 250
    experiment_name = "sfyg_tron_2a_troncamp_forward"
    algorithm = RslRlPpoAlgorithmMlpCfg(
        class_name="PPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        obs_history_len=10,
    )


#-----------------------------------------------------------------
@configclass
class SFYG_TRON2AGaitPPORunnerCfg(SFYG_TRON2AFlatPPORunnerCfg):
    max_iterations = 5000
    save_interval = 100
    experiment_name = "sfyg_tron_2a_gait"
    algorithm = RslRlPpoAlgorithmMlpCfg(
        class_name="PPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.003,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        obs_history_len=10,
    )


#-----------------------------------------------------------------
@configclass
class SFYG_TRON2APerceptiveGaitPPORunnerCfg(SFYG_TRON2AGaitPPORunnerCfg):
    max_iterations = 6000
    save_interval = 100
    experiment_name = "sfyg_tron_2a_perceptive_gait"


#-----------------------------------------------------------------
@configclass
class WFYG_TRON2AFlatPPORunnerCfg(WF_TRON2AFlatPPORunnerCfg):
    experiment_name = "wfyg_tron_2a_flat"
