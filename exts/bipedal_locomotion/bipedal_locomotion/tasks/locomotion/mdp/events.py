from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Literal

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg

def prepare_quantity_for_tron(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    foot_radius = 0.127,
):
    asset: Articulation = env.scene[asset_cfg.name]
    env._foot_radius = foot_radius

def apply_external_force_torque_stochastic(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    force_range: dict[str, tuple[float, float]],
    torque_range: dict[str, tuple[float, float]],
    probability: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Randomize the external forces and torques applied to the bodies.

    This function creates a set of random forces and torques sampled from the given ranges. The number of forces
    and torques is equal to the number of bodies times the number of environments. The forces and torques are
    applied to the bodies by calling ``asset.set_external_force_and_torque``. The forces and torques are only
    applied when ``asset.write_data_to_sim()`` is called in the environment.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    # clear the existing forces and torques
    try:
        asset._external_force_b *= 0
        asset._external_torque_b *= 0
    except:
        asset.permanent_wrench_composer.reset()

    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)

    random_values = torch.rand(env_ids.shape, device=env_ids.device)
    mask = random_values < probability
    masked_env_ids = env_ids[mask]

    if len(masked_env_ids) == 0:
        return

    # resolve number of bodies
    num_bodies = len(asset_cfg.body_ids) if isinstance(asset_cfg.body_ids, list) else asset.num_bodies

    # sample random forces and torques
    size = (len(masked_env_ids), num_bodies, 3)
    force_range_list = [force_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]]
    force_range = torch.tensor(force_range_list, device=asset.device)
    forces = math_utils.sample_uniform(force_range[:, 0], force_range[:, 1], size, asset.device)
    torque_range_list = [torque_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]]
    torque_range = torch.tensor(torque_range_list, device=asset.device)
    torques = math_utils.sample_uniform(torque_range[:, 0], torque_range[:, 1], size, asset.device)
    # set the forces and torques into the buffers
    # note: these are only applied when you call: `asset.write_data_to_sim()`
    asset.set_external_force_and_torque(forces, torques, env_ids=masked_env_ids, body_ids=asset_cfg.body_ids)


def reset_root_state_tron_camp_curriculum(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    curriculum_segments: tuple[dict, ...],
    z_range: tuple[float, float],
    phase_schedule: tuple[dict, ...] | None = None,
    y_range: tuple[float, float] = (-0.15, 0.15),
    yaw_range: tuple[float, float] = (-0.08, 0.08),
    velocity_range: dict[str, tuple[float, float]] | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset the robot near mixed TronCamp terrain features.

    The generated TronCamp track has a stable local feature order, but each
    terrain cell samples exact lengths and heights. Spawn before feature entries
    instead of on top of obstacles so the default base height stays valid across
    randomized cells.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)

    num_resets = len(env_ids)
    device = asset.device
    root_states = asset.data.default_root_state[env_ids].clone()
    env_origins = getattr(env.scene, "env_origins", None)
    if env_origins is None:
        env_origin_pos = torch.zeros((num_resets, 3), device=device)
    else:
        env_origin_pos = env_origins[env_ids].to(device)

    weights = _resolve_tron_camp_curriculum_weights(env, curriculum_segments, phase_schedule, device)
    weights = weights / torch.sum(weights)
    segment_ids = torch.multinomial(weights, num_resets, replacement=True)

    x_offsets = torch.zeros(num_resets, device=device)
    for segment_index, segment in enumerate(curriculum_segments):
        mask = segment_ids == segment_index
        if not torch.any(mask):
            continue
        x_min, x_max = segment["x_range"]
        x_offsets[mask] = torch.empty(int(torch.sum(mask).item()), device=device).uniform_(float(x_min), float(x_max))

    y_offsets = torch.empty(num_resets, device=device).uniform_(float(y_range[0]), float(y_range[1]))
    yaw = torch.empty(num_resets, device=device).uniform_(float(yaw_range[0]), float(yaw_range[1]))
    z = torch.empty(num_resets, device=device).uniform_(float(z_range[0]), float(z_range[1]))

    root_states[:, 0] = env_origin_pos[:, 0] + x_offsets
    root_states[:, 1] = env_origin_pos[:, 1] + y_offsets
    root_states[:, 2] = env_origin_pos[:, 2] + z
    root_states[:, 3:7] = math_utils.quat_from_euler_xyz(
        torch.zeros_like(yaw),
        torch.zeros_like(yaw),
        yaw,
    )

    velocity_range = velocity_range or {}
    for index, key in enumerate(("x", "y", "z", "roll", "pitch", "yaw")):
        low, high = velocity_range.get(key, (0.0, 0.0))
        root_states[:, 7 + index] = torch.empty(num_resets, device=device).uniform_(float(low), float(high))

    asset.write_root_pose_to_sim(root_states[:, :7], env_ids=env_ids)
    asset.write_root_velocity_to_sim(root_states[:, 7:], env_ids=env_ids)


def _resolve_tron_camp_curriculum_weights(
    env: ManagerBasedEnv,
    curriculum_segments: tuple[dict, ...],
    phase_schedule: tuple[dict, ...] | None,
    device: torch.device,
) -> torch.Tensor:
    """Resolve segment sampling weights, optionally interpolating over training progress."""
    segment_names = [str(segment["name"]) for segment in curriculum_segments]
    if not phase_schedule:
        return torch.tensor([float(segment["weight"]) for segment in curriculum_segments], device=device)

    current_step = float(getattr(env, "common_step_counter", 0))
    phases = sorted(phase_schedule, key=lambda phase: float(phase["step"]))

    if current_step <= float(phases[0]["step"]):
        weights_by_name = phases[0]["weights"]
    elif current_step >= float(phases[-1]["step"]):
        weights_by_name = phases[-1]["weights"]
    else:
        lower_phase = phases[0]
        upper_phase = phases[-1]
        for index in range(len(phases) - 1):
            left = phases[index]
            right = phases[index + 1]
            if float(left["step"]) <= current_step <= float(right["step"]):
                lower_phase = left
                upper_phase = right
                break

        lower_step = float(lower_phase["step"])
        upper_step = float(upper_phase["step"])
        alpha = 0.0 if upper_step <= lower_step else (current_step - lower_step) / (upper_step - lower_step)
        weights_by_name = {
            name: (1.0 - alpha) * float(lower_phase["weights"].get(name, 0.0))
            + alpha * float(upper_phase["weights"].get(name, 0.0))
            for name in segment_names
        }

    weights = torch.tensor([float(weights_by_name.get(name, 0.0)) for name in segment_names], device=device)
    if torch.sum(weights) <= 0.0:
        weights = torch.ones(len(curriculum_segments), device=device)
    return weights


def randomize_rigid_body_mass_inertia(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    mass_inertia_distribution_params: tuple[float, float],
    operation: Literal["add", "scale", "abs"],
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    """Randomize the inertia of the bodies by adding, scaling, or setting random values.

    This function allows randomizing the mass of the bodies of the asset. The function samples random values from the
    given distribution parameters and adds, scales, or sets the values into the physics simulation based on the operation.

    .. tip::
        This function uses CPU tensors to assign the body masses. It is recommended to use this function
        only during the initialization of the environment.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    # resolve body indices
    if asset_cfg.body_ids == slice(None):
        body_ids = torch.arange(asset.num_bodies, dtype=torch.int, device="cpu")
    else:
        body_ids = torch.tensor(asset_cfg.body_ids, dtype=torch.int, device="cpu")

    # get the current inertias of the bodies (num_assets, num_bodies)
    inertias = asset.root_physx_view.get_inertias().clone()
    masses = asset.root_physx_view.get_masses().clone()

    masses = _randomize_prop_by_op(
        masses, mass_inertia_distribution_params, env_ids, body_ids, operation=operation, distribution=distribution
    )
    scale = masses / asset.root_physx_view.get_masses()
    inertias *= scale.unsqueeze(-1)

    asset.root_physx_view.set_masses(masses, env_ids)
    asset.root_physx_view.set_inertias(inertias, env_ids)


def randomize_rigid_body_coms(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    com_distribution_params: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
    operation: Literal["add", "scale", "abs"],
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    """
    Randomize the center of mass (COM) of the bodies by adding, scaling, or setting random values for each dimension.
    """
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    if asset_cfg.body_ids == slice(None):
        body_ids = torch.arange(asset.num_bodies, dtype=torch.int, device="cpu")
    else:
        body_ids = torch.tensor(asset_cfg.body_ids, dtype=torch.int, device="cpu")

    coms = asset.root_physx_view.get_coms().clone()

    # Apply randomization to each dimension separately
    for dim in range(3):  # 0=x, 1=y, 2=z
        coms[..., dim] = _randomize_prop_by_op(
            coms[..., dim],
            com_distribution_params[dim],
            env_ids,
            body_ids,
            operation=operation,
            distribution=distribution,
        )

    asset.root_physx_view.set_coms(coms, env_ids)


"""
Internal helper functions.
"""


def _randomize_prop_by_op(
    data: torch.Tensor,
    distribution_parameters: tuple[float | torch.Tensor, float | torch.Tensor],
    dim_0_ids: torch.Tensor | None,
    dim_1_ids: torch.Tensor | slice,
    operation: Literal["add", "scale", "abs"],
    distribution: Literal["uniform", "log_uniform", "gaussian"],
) -> torch.Tensor:
    """Perform data randomization based on the given operation and distribution.

    Args:
        data: The data tensor to be randomized. Shape is (dim_0, dim_1).
        distribution_parameters: The parameters for the distribution to sample values from.
        dim_0_ids: The indices of the first dimension to randomize.
        dim_1_ids: The indices of the second dimension to randomize.
        operation: The operation to perform on the data. Options: 'add', 'scale', 'abs'.
        distribution: The distribution to sample the random values from. Options: 'uniform', 'log_uniform'.

    Returns:
        The data tensor after randomization. Shape is (dim_0, dim_1).

    Raises:
        NotImplementedError: If the operation or distribution is not supported.
    """
    # resolve shape
    # -- dim 0
    if dim_0_ids is None:
        n_dim_0 = data.shape[0]
        dim_0_ids = slice(None)
    else:
        n_dim_0 = len(dim_0_ids)
        if not isinstance(dim_1_ids, slice):
            dim_0_ids = dim_0_ids[:, None]
    # -- dim 1
    if isinstance(dim_1_ids, slice):
        n_dim_1 = data.shape[1]
    else:
        n_dim_1 = len(dim_1_ids)

    # resolve the distribution
    if distribution == "uniform":
        dist_fn = math_utils.sample_uniform
    elif distribution == "log_uniform":
        dist_fn = math_utils.sample_log_uniform
    elif distribution == "gaussian":
        dist_fn = math_utils.sample_gaussian
    else:
        raise NotImplementedError(
            f"Unknown distribution: '{distribution}' for joint properties randomization."
            " Please use 'uniform', 'log_uniform', 'gaussian'."
        )
    # perform the operation
    if operation == "add":
        data[dim_0_ids, dim_1_ids] += dist_fn(*distribution_parameters, (n_dim_0, n_dim_1), device=data.device)
    elif operation == "scale":
        data[dim_0_ids, dim_1_ids] *= dist_fn(*distribution_parameters, (n_dim_0, n_dim_1), device=data.device)
    elif operation == "abs":
        data[dim_0_ids, dim_1_ids] = dist_fn(*distribution_parameters, (n_dim_0, n_dim_1), device=data.device)
    else:
        raise NotImplementedError(
            f"Unknown operation: '{operation}' for property randomization. Please use 'add', 'scale', or 'abs'."
        )
    return data

def action_out_of_limits(
    env: ManagerBasedRLEnv, threshold: float = 5.0
) -> torch.Tensor:
    """Terminate when the action values exceed the specified threshold.
    
    This function checks if any action value exceeds the threshold limit,
    which can help prevent training instability caused by extremely large actions.
    
    Args:
        env: The environment instance.
        threshold: Maximum allowed absolute value for actions. Default is 5.0.
        
    Returns:
        Boolean tensor indicating which environments should be terminated.
    """
    # Get the current actions from the environment
    actions = env.action_manager.action
    
    # Check if any action exceeds the threshold (in absolute value)
    action_magnitude = torch.abs(actions)
    max_action_per_env = torch.max(action_magnitude, dim=1)[0]
    
    # Return True for environments where max action exceeds threshold
    return max_action_per_env > threshold
