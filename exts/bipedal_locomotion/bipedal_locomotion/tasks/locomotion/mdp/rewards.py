"""This sub-module contains the reward functions that can be used for LimX Point Foot's locomotion task.

The functions can be passed to the :class:`isaaclab.managers.RewardTermCfg` object to
specify the reward function and its parameters.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.sensors import ContactSensor, RayCaster
from isaaclab.utils.math import quat_apply_inverse, yaw_quat
import isaaclab.utils.math as math_utils
from bipedal_locomotion.utils.math import CubicSpline


if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import RewardTermCfg


def normalize_angle(x):
    return torch.atan2(torch.sin(x), torch.cos(x))


def stay_alive(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Reward for staying alive."""
    return torch.ones(env.num_envs, device=env.device)


def track_lin_vel_xy_yaw_frame_exp(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) in the gravity aligned robot frame using exponential kernel."""
    asset: RigidObject = env.scene[asset_cfg.name]
    vel_yaw = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - vel_yaw[:, :2]), dim=1
    )
    return torch.exp(-lin_vel_error / std**2)


def track_lin_vel_x_yaw_frame_exp(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward tracking of the x-axis linear velocity command in the gravity-aligned (yaw) robot frame using an exponential kernel.

    Split-out of :func:`track_lin_vel_xy_yaw_frame_exp` so that the longitudinal (x) and
    lateral (y) tracking rewards can carry independent weights / kernel widths.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    vel_yaw = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
    lin_vel_error = torch.square(
        env.command_manager.get_command(command_name)[:, 0] - vel_yaw[:, 0]
    )
    return torch.exp(-lin_vel_error / std**2)


def lin_vel_x_shortfall_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    deadband: float = 0.05,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize failing to move forward at the commanded x velocity."""
    asset: RigidObject = env.scene[asset_cfg.name]
    vel_yaw = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
    target_vel_x = env.command_manager.get_command(command_name)[:, 0]
    return torch.clamp(target_vel_x - vel_yaw[:, 0] - deadband, min=0.0)


def track_lin_vel_y_yaw_frame_exp(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward tracking of the y-axis linear velocity command in the gravity-aligned (yaw) robot frame using an exponential kernel.

    Split-out of :func:`track_lin_vel_xy_yaw_frame_exp` so the lateral channel can be
    tuned (weight / std) independently from the longitudinal channel — useful when
    pure-lateral motion is under-trained or when symmetry constraints make the
    network insensitive to vy commands.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    vel_yaw = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
    lin_vel_error = torch.square(
        env.command_manager.get_command(command_name)[:, 1] - vel_yaw[:, 1]
    )
    return torch.exp(-lin_vel_error / std**2)


def heading_alignment_exp(
    env: ManagerBasedRLEnv,
    std: float,
    target_heading: float = 0.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward aligning the robot yaw with a target heading."""
    asset: RigidObject = env.scene[asset_cfg.name]
    _, _, yaw = math_utils.euler_xyz_from_quat(asset.data.root_quat_w)
    yaw_error = normalize_angle(yaw - target_heading)
    return torch.exp(-torch.square(yaw_error) / std**2)


def heading_error_abs(
    env: ManagerBasedRLEnv,
    target_heading: float = 0.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize absolute yaw error from the target heading."""
    asset: RigidObject = env.scene[asset_cfg.name]
    _, _, yaw = math_utils.euler_xyz_from_quat(asset.data.root_quat_w)
    return torch.abs(normalize_angle(yaw - target_heading))


def world_lin_vel_y_abs(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize lateral drift velocity in the world frame."""
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.abs(asset.data.root_lin_vel_w[:, 1])


def world_pos_y_abs(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize lateral displacement from the terrain cell centerline in the world frame."""
    asset: RigidObject = env.scene[asset_cfg.name]
    env_origins = getattr(env.scene, "env_origins", None)
    if env_origins is None:
        target_y = torch.zeros(env.num_envs, device=env.device)
    else:
        target_y = env_origins[:, 1]
    return torch.abs(asset.data.root_pos_w[:, 1] - target_y)


def joint_powers_l1(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize joint powers on the articulation using L1-kernel"""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.abs(torch.mul(asset.data.applied_torque, asset.data.joint_vel)), dim=1)


def joint_deviation_from_default_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize selected joints deviating from their default (initial) positions using L2 kernel."""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_error = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    return torch.sum(torch.square(joint_error), dim=1)


def leg_symmetry(
    env: ManagerBasedRLEnv,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward regulate abad joint position."""
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    feet_pos_w = asset.data.body_link_pos_w[:, asset_cfg.body_ids]
    base_quat = asset.data.root_link_quat_w.unsqueeze(1).expand(-1, 2, -1)
    base_pos = asset.data.root_link_state_w[:, :3].unsqueeze(1).expand(-1, 2, -1)
    feet_pos_b = math_utils.quat_apply_inverse(
        base_quat,
        feet_pos_w - base_pos,
    )
    leg_symmetry_err = torch.abs(feet_pos_b[:, 0, 1]) - torch.abs(feet_pos_b[:, 1, 1])

    return torch.exp(-leg_symmetry_err ** 2 / std**2)


def same_feet_x_position(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Reward regulate abad joint position."""
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    feet_pos_w = asset.data.body_link_pos_w[:, asset_cfg.body_ids]
    base_quat = asset.data.root_link_quat_w.unsqueeze(1).expand(-1, 2, -1)
    base_pos = asset.data.root_link_state_w[:, :3].unsqueeze(1).expand(-1, 2, -1)
    feet_pos_b = math_utils.quat_apply_inverse(
        base_quat,
        feet_pos_w - base_pos,
    )
    feet_x_distance = torch.abs(feet_pos_b[:, 0, 0] - feet_pos_b[:, 1, 0])
    return feet_x_distance


def contact_forces(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize contact forces as the amount of violations of the net contact force."""
    asset: Articulation = env.scene[asset_cfg.name]
    robot_links_mass = asset.root_physx_view.get_masses()
    robot_mass = torch.sum(robot_links_mass, dim=-1, keepdim=True)
    robot_mass = robot_mass.to(env.device)

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    violation = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] - robot_mass * 9.8
    return torch.sum(violation.clip(min=0.0), dim=1)


def distance_aligned(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    min_dist: float,
    max_dist: float,
    desired_dist: float,
    std: float,
    command_name: str = "base_velocity",
    vy_max: float = 0.8,
    decay_power: float = 1.0,
) -> torch.Tensor:
    """Penalize feet distance from the desired distance."""
    asset: RigidObject = env.scene[asset_cfg.name]

    left_idx = asset_cfg.body_ids[0]
    right_idx = asset_cfg.body_ids[1]
    base_quat = asset.data.root_quat_w
    heading_aligned = math_utils.yaw_quat(base_quat)

    left_pos = math_utils.quat_apply_inverse(heading_aligned, asset.data.body_pos_w[:, left_idx])
    right_pos = math_utils.quat_apply_inverse(heading_aligned, asset.data.body_pos_w[:, right_idx])

    distance_y = torch.abs(left_pos[:, 1] - right_pos[:, 1])
    d_min = torch.where(distance_y < min_dist, min_dist - distance_y, torch.tensor(0.0, device=distance_y.device))
    d_max = torch.where(distance_y > max_dist, distance_y - max_dist, torch.tensor(0.0, device=distance_y.device))

    reward_1 = torch.exp(-(d_min + d_max) / std**2)
    reward_2 = torch.exp(-torch.square((distance_y - desired_dist) / std**2))

    vy = env.command_manager.get_command(command_name)[:, 1]
    x = torch.clamp(torch.abs(vy) / vy_max, 0.0, 1.0)
    vy_weight = (1.0 - x) ** decay_power

    return (reward_1 + vy_weight * reward_2) / 2


def stand_still(
    env,
    lin_threshold: float = 0.05,
    ang_threshold: float = 0.05,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize linear and angular motion when command velocities are near zero."""
    asset = env.scene[asset_cfg.name]
    base_lin_vel = asset.data.root_lin_vel_w[:, :2]
    base_ang_vel = asset.data.root_ang_vel_w[:, -1]

    commands = env.command_manager.get_command("base_velocity")

    lin_commands = commands[:, :2]
    ang_commands = commands[:, 2]

    reward_lin = torch.sum(
        torch.abs(base_lin_vel) * (torch.norm(lin_commands, dim=1, keepdim=True) < lin_threshold), dim=-1
    )

    reward_ang = torch.abs(base_ang_vel) * (torch.abs(ang_commands) < ang_threshold)

    total_reward = reward_lin + reward_ang
    return total_reward


def base_height_exp(
    env: ManagerBasedRLEnv,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    target_height: float = 0.72,
) -> torch.Tensor:
    """Penalize base height from the target height using L2 squared kernel."""
    asset: RigidObject = env.scene[asset_cfg.name]
    base_height_error = torch.square(asset.data.root_pos_w[:, 2] - target_height)
    return torch.exp(-base_height_error / std**2)


def base_projection_at_feet_midpoint(
    env: ManagerBasedRLEnv, 
    std: float, 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    feet_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="wheel_.*")
) -> torch.Tensor:
    """Reward base projection at the feet midpoint."""
    asset: Articulation = env.scene[asset_cfg.name]
    feet_pos_w = asset.data.body_pos_w[:, feet_cfg.body_ids, :2]
    midpoint_xy = torch.mean(feet_pos_w, dim=1)
    base_xy = asset.data.root_pos_w[:, :2]
    error_sq = torch.sum(torch.square(base_xy - midpoint_xy), dim=1)
    return torch.exp(-error_sq / std**2)


def track_centerline_y(
    env: ManagerBasedRLEnv,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward staying near the terrain cell centerline in world Y."""
    asset: Articulation = env.scene[asset_cfg.name]
    env_origins = getattr(env.scene, "env_origins", None)
    if env_origins is None:
        target_y = torch.zeros(env.num_envs, device=env.device)
    else:
        target_y = env_origins[:, 1]
    lateral_error = asset.data.root_pos_w[:, 1] - target_y
    return torch.exp(-torch.square(lateral_error) / std**2)


def single_stance_reward(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 1.0,
) -> torch.Tensor:
    """Reward having exactly one foot in contact with the ground."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = (
        contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > threshold
    )
    return (torch.sum(contacts.int(), dim=1) == 1).float()


def double_flight_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 1.0,
) -> torch.Tensor:
    """Penalize both feet leaving the ground at the same time."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = (
        contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > threshold
    )
    return (torch.sum(contacts.int(), dim=1) == 0).float()


def swing_foot_clearance_reward(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    target_height: float = 0.06,
    std: float = 0.03,
    threshold: float = 1.0,
) -> torch.Tensor:
    """Reward the non-contact foot clearing the stance foot by a modest height."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = (
        contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > threshold
    )
    single_stance = torch.sum(contacts.int(), dim=1) == 1

    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    foot_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
    clearance = torch.max(foot_z, dim=1)[0] - torch.min(foot_z, dim=1)[0]
    reward = torch.exp(-torch.square(clearance - target_height) / std**2)
    return torch.where(single_stance, reward, torch.zeros_like(reward))


def double_stance_feet_x_alignment_reward(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    std: float = 0.06,
    threshold: float = 1.0,
) -> torch.Tensor:
    """Reward aligned foot x positions only when both feet are in contact."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = (
        contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > threshold
    )
    double_stance = torch.sum(contacts.int(), dim=1) == 2

    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    feet_pos_w = asset.data.body_link_pos_w[:, asset_cfg.body_ids]
    base_quat = asset.data.root_link_quat_w.unsqueeze(1).expand(-1, 2, -1)
    base_pos = asset.data.root_link_state_w[:, :3].unsqueeze(1).expand(-1, 2, -1)
    feet_pos_b = math_utils.quat_apply_inverse(base_quat, feet_pos_w - base_pos)
    feet_x_error = feet_pos_b[:, 0, 0] - feet_pos_b[:, 1, 0]
    reward = torch.exp(-torch.square(feet_x_error) / std**2)
    return torch.where(double_stance, reward, torch.zeros_like(reward))


def single_stance_feet_x_separation_reward(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    target_distance: float = 0.16,
    std: float = 0.06,
    threshold: float = 1.0,
) -> torch.Tensor:
    """Reward a forward/back foot split during single stance to encourage longer steps."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = (
        contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > threshold
    )
    single_stance = torch.sum(contacts.int(), dim=1) == 1

    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    feet_pos_w = asset.data.body_link_pos_w[:, asset_cfg.body_ids]
    base_quat = asset.data.root_link_quat_w.unsqueeze(1).expand(-1, 2, -1)
    base_pos = asset.data.root_link_state_w[:, :3].unsqueeze(1).expand(-1, 2, -1)
    feet_pos_b = math_utils.quat_apply_inverse(base_quat, feet_pos_w - base_pos)
    feet_x_distance = torch.abs(feet_pos_b[:, 0, 0] - feet_pos_b[:, 1, 0])
    reward = torch.exp(-torch.square(feet_x_distance - target_distance) / std**2)
    return torch.where(single_stance, reward, torch.zeros_like(reward))


def balanced_feet_x_symmetry_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    std: float = 0.16,
    max_tilt: float = 0.35,
) -> torch.Tensor:
    """Reward loose fore-aft foot symmetry only while the base is reasonably balanced."""
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    feet_pos_w = asset.data.body_link_pos_w[:, asset_cfg.body_ids]
    base_quat = asset.data.root_link_quat_w.unsqueeze(1).expand(-1, 2, -1)
    base_pos = asset.data.root_link_state_w[:, :3].unsqueeze(1).expand(-1, 2, -1)
    feet_pos_b = math_utils.quat_apply_inverse(base_quat, feet_pos_w - base_pos)

    # If the two feet are fore-aft balanced around the body, x_left + x_right stays near zero.
    symmetry_error = feet_pos_b[:, 0, 0] + feet_pos_b[:, 1, 0]
    reward = torch.exp(-torch.square(symmetry_error) / std**2)

    projected_gravity = asset.data.projected_gravity_b
    balanced = torch.norm(projected_gravity[:, :2], dim=1) < max_tilt
    return torch.where(balanced, reward, torch.zeros_like(reward))


class FeetSlidePenaltyWrapper:
    """A wrapper class for calculating feet slide penalty."""
    def __init__(self):
        self.count = 0
        self.friction = None
        self.__name__ = "feet_slide_penalty"

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        sensor_cfg: SceneEntityCfg,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
        """Penalize feet slide."""

        asset: RigidObject | Articulation = env.scene[asset_cfg.name]

        if self.count <= 1:
            self.friction = asset.root_physx_view.get_material_properties()[..., 0].to(device=asset.device)
            self.num_shapes_per_body = []
            for link_path in asset.root_physx_view.link_paths[0]:
                link_physx_view = asset._physics_sim_view.create_rigid_body_view(link_path)  # type: ignore
                self.num_shapes_per_body.append(link_physx_view.max_shapes)

            # sample material properties from the given ranges
            body_count = 0
            self.body_ids = []
            for body_ids, valid in enumerate(self.num_shapes_per_body):
                if valid:
                    if isinstance(asset_cfg.body_ids, slice):
                        asset_cfg.body_ids = list(range(len(asset_cfg.body_ids)))
                    if body_ids in asset_cfg.body_ids:
                        self.body_ids.append(body_count)
                    body_count += 1

        self.count += 1

        contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
        contacts = (
            contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
        )
        asset = env.scene[asset_cfg.name]
        feet_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
        reward = torch.sum(torch.square(feet_vel.norm(dim=-1)) * contacts * self.friction[:, self.body_ids], dim=1)

        return reward


feet_slide_penalty = FeetSlidePenaltyWrapper()


def base_com_height(
    env: ManagerBasedRLEnv,
    target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Penalize asset height from its target using L2 squared kernel.

    Note:
        For flat terrain, target height is in the world frame. For rough terrain,
        sensor readings can adjust the target height to account for the terrain.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    if sensor_cfg is not None:
        sensor: RayCaster = env.scene[sensor_cfg.name]
        # Adjust the target height using the sensor data
        adjusted_target_height = target_height + torch.mean(sensor.data.ray_hits_w[..., 2], dim=1)
    else:
        # Use the provided target height directly for flat terrain
        adjusted_target_height = target_height
    # Compute the L2 squared penalty
    return torch.abs(asset.data.root_pos_w[:, 2] - adjusted_target_height)


def stand_still_reg(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    exclude_joints_name: list[str] = [r"J\d"],
) -> torch.Tensor:
    """Regulate the stand still"""
    asset: Articulation = env.scene[asset_cfg.name]

    exclude_joints_idx = asset.find_joints(exclude_joints_name)[0]
    all_joints_idx = range(asset.num_joints)
    vel_idx_exclude_arm = [i for i in all_joints_idx if i not in exclude_joints_idx]

    joint_vel = asset.data.joint_vel[:, vel_idx_exclude_arm]

    # stand still env , set to zero
    is_standing_env = env.command_manager.get_term("base_velocity").is_standing_env  # type: ignore

    not_standing_env_ids = (~is_standing_env).nonzero(as_tuple=False).flatten()

    reward = torch.sum(torch.abs(joint_vel), dim=1)

    reward[not_standing_env_ids] = 0.0

    return reward


class TorquesSmoothnessPenaltyWrapper:
    """A wrapper class for calculating torques smoothness penalty."""

    def __init__(self):
        self.prev_torques = None
        self.__name__ = "torques_smoothness_penalty"

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
        """Penalize large instantaneous changes in the torques output"""
        asset: Articulation = env.scene[asset_cfg.name]
        torques = asset.data.applied_torque[:, asset_cfg.joint_ids].clone()
        if self.prev_torques is None:
            self.prev_torques = torques
            return torch.zeros(torques.shape[0], device=torques.device)
        reward = torch.sum(torch.square(torques - self.prev_torques), dim=1)
        self.prev_torques = torques
        return reward


torques_smoothness_penalty = TorquesSmoothnessPenaltyWrapper()


def body_orientation_yaw_exp(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    target_yaw: float = 0.0,
) -> torch.Tensor:
    """Penalize body orientation yaw error."""
    asset: RigidObject = env.scene[asset_cfg.name]
    num_body = len(asset_cfg.body_ids)
    base_quat = asset.data.root_quat_w
    inverse_base_quat = math_utils.quat_inv(base_quat).unsqueeze(1).expand(-1, num_body, -1)
    body_quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids, :]
    body_quat_b = math_utils.quat_mul(inverse_base_quat, body_quat_w).flatten(0, 1)
    r, p, y = math_utils.euler_xyz_from_quat(body_quat_b.squeeze(1))

    yaw_error = normalize_angle(y - target_yaw)

    ruler_angle = torch.stack([normalize_angle(r), normalize_angle(p), yaw_error], dim=-1).reshape(-1, num_body, 3)
    quat_mismatch = torch.exp(-torch.abs(ruler_angle[:, :, 2]) * 10)
    return torch.mean(quat_mismatch, dim=1)


def feet_air_time_positive_biped(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds.

    This function rewards the agent for taking steps up to a specified threshold and also keep one foot at
    a time in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    # stand still env , set to zero
    is_standing_env = env.command_manager.get_term("base_velocity").is_standing_env  # type: ignore
    no_gait_env_ids = is_standing_env.nonzero(as_tuple=False).flatten()
    reward[no_gait_env_ids] = 0.0
    return reward


def joint_orientation_l1_symmetric(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    joint_error = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    reward = torch.sum(torch.abs(joint_error), dim=-1)
    return reward


def joint_orientation_l1(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    knee_joint_error = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    reward = torch.sum(torch.abs(knee_joint_error) * (knee_joint_error > 0), dim=-1)
    return reward


def weighted_joint_deviation_l1(
    env: ManagerBasedRLEnv,
    deviation_weight: dict[str, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize joint positions that deviate from the default one."""
    asset: Articulation = env.scene[asset_cfg.name]
    # compute out of limits constraints
    angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]

    weighted_joint_deviation = torch.zeros_like(asset.data.joint_pos)

    for joint_name, w in deviation_weight.items():
        joint_idx = asset.find_joints(joint_name)[0]
        weighted_joint_deviation[:, joint_idx] = torch.abs(angle[:, joint_idx]) * w
    return torch.sum(weighted_joint_deviation, dim=1)


def weighted_joint_power_l1(
    env: ManagerBasedRLEnv,
    power_weight: dict[str, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize joint power applied on the articulation using L1 kernel.

    NOTE: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their joint torques contribute to the term.
    """
    asset: Articulation = env.scene[asset_cfg.name]

    weighted_power = torch.zeros_like(asset.data.applied_torque)

    for joint_name, w in power_weight.items():
        joint_idx = asset.find_joints(joint_name)[0]
        weighted_power[:, joint_idx] = (
            torch.abs(asset.data.applied_torque[:, joint_idx] * asset.data.joint_vel[:, joint_idx]) * w
        )

    return torch.sum(weighted_power, dim=1)


def feet_angle_slide(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize feet sliding.

    This function penalizes the agent for sliding its feet on the ground. The reward is computed as the
    norm of the angler velocity of the feet multiplied by a binary contact sensor. This ensures that the
    agent is penalized only when the feet are in contact with the ground.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset = env.scene[asset_cfg.name]
    feet_ang = asset.data.body_ang_vel_w[:, asset_cfg.body_ids, 2]
    command = env.command_manager.get_command(command_name)
    feet_ang_reward = torch.sum(torch.abs(feet_ang) * contacts, dim=1)
    reward = torch.where(torch.abs(command[:, 2]) > 0.1, feet_ang_reward, feet_ang_reward * 0.1)
    return reward


class GaitReward(ManagerTermBase):
    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        """Initialize the term.

        Args:
            cfg: The configuration of the reward.
            env: The RL environment instance.
        """
        super().__init__(cfg, env)

        self.sensor_cfg = cfg.params["sensor_cfg"]
        self.asset_cfg = cfg.params["asset_cfg"]


        self.contact_sensor: ContactSensor = env.scene.sensors[self.sensor_cfg.name]
        self.asset: Articulation = env.scene[self.asset_cfg.name]

        # Store configuration parameters
        self.force_scale = float(cfg.params["tracking_contacts_shaped_force"])
        self.vel_scale = float(cfg.params["tracking_contacts_shaped_vel"])
        self.height_scale = float(cfg.params["tracking_contacts_shaped_height"])
        self.force_sigma = cfg.params["gait_force_sigma"]
        self.vel_sigma = cfg.params["gait_vel_sigma"]
        self.height_sigma = cfg.params["gait_height_sigma"]
        self.touch_down_vel = float(cfg.params["touch_down_vel"])
        self.kappa_gait_probs = cfg.params["kappa_gait_probs"]
        self.command_name = cfg.params["command_name"]
        self.dt = env.step_dt
        self.use_reference_motion = cfg.params["use_reference_motion"]

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        tracking_contacts_shaped_force,
        tracking_contacts_shaped_vel,
        tracking_contacts_shaped_height,
        gait_force_sigma,
        gait_vel_sigma,
        gait_height_sigma,
        touch_down_vel,
        kappa_gait_probs,
        command_name,
        sensor_cfg,
        asset_cfg,
        use_reference_motion,
    ) -> torch.Tensor:
        """Compute the reward.

        The reward combines force-based and velocity-based terms to encourage desired gait patterns.

        Args:
            env: The RL environment instance.

        Returns:
            The reward value.
        """
        gait_params = env.command_manager.get_command(self.command_name)  # type: ignore
        gait_indices = env.command_manager.get_term(self.command_name).gait_indices  # type: ignore

        # Update contact targets
        desired_contact_states = self.compute_contact_targets(gait_params)

        # Update foot height targets
        self.compute_desired_foot_height(gait_params, gait_indices)

        # Force-based reward
        foot_forces = torch.norm(
            self.contact_sensor.data.net_forces_w[:, self.sensor_cfg.body_ids], dim=-1
        )
        force_reward = self._compute_force_reward(foot_forces, desired_contact_states)

        total_reward = force_reward

        # Velocity-based reward
        if self.vel_scale != 0:
            foot_velocities = self.asset.data.body_lin_vel_w[:, self.asset_cfg.body_ids]
            velocity_reward = self._compute_velocity_reward(
                foot_velocities, self.des_foot_velocity_z, desired_contact_states
            )
            total_reward += velocity_reward

        # Height-based reward
        if self.height_scale != 0:
            foot_heights = self.asset.data.body_pos_w[:, self.asset_cfg.body_ids, 2]
            height_reward = self._compute_height_reward(foot_heights, self.des_foot_height, desired_contact_states)
            total_reward += height_reward

        # stand still env , set to zero
        is_standing_env = env.command_manager.get_term("base_velocity").is_standing_env  # type: ignore

        no_gait_env_ids = is_standing_env.nonzero(as_tuple=False).flatten()

        total_reward[no_gait_env_ids] = 0.0
        return total_reward

    def compute_contact_targets(self, gait_params):
        """Calculate desired contact states for the current timestep."""
        frequencies = gait_params[:, 0]
        offsets = gait_params[:, 1]
        durations = torch.cat(
            [
                gait_params[:, 2].view(self.num_envs, 1),
                gait_params[:, 2].view(self.num_envs, 1),
            ],
            dim=1,
        )

        assert torch.all(frequencies > 0), "Frequencies must be positive"
        assert torch.all(
            (offsets >= 0) & (offsets <= 1)
        ), "Offsets must be between 0 and 1"
        assert torch.all(
            (durations > 0) & (durations < 1)
        ), "Durations must be between 0 and 1"

        # use gait indices from command
        command_term = self._env.command_manager.get_term("gait_command")  # type: ignore
        gait_indices = command_term.gait_indices  # type: ignore

        # Calculate foot indices
        foot_indices = torch.remainder(
            torch.cat(
                [
                    gait_indices.view(self.num_envs, 1),
                    (gait_indices + offsets + 1).view(self.num_envs, 1),
                ],
                dim=1,
            ),
            1.0,
        )

        # Determine stance and swing phases
        stance_idxs = foot_indices < durations
        swing_idxs = foot_indices > durations

        # Adjust foot indices based on phase
        foot_indices[stance_idxs] = torch.remainder(foot_indices[stance_idxs], 1) * (
            0.5 / durations[stance_idxs]
        )
        foot_indices[swing_idxs] = 0.5 + (
            torch.remainder(foot_indices[swing_idxs], 1) - durations[swing_idxs]
        ) * (0.5 / (1 - durations[swing_idxs]))

        # Calculate desired contact states using von mises distribution
        smoothing_cdf_start = torch.distributions.normal.Normal(
            0, self.kappa_gait_probs
        ).cdf
        desired_contact_states = smoothing_cdf_start(foot_indices) * (
            1 - smoothing_cdf_start(foot_indices - 0.5)
        ) + smoothing_cdf_start(foot_indices - 1) * (
            1 - smoothing_cdf_start(foot_indices - 1.5)
        )

        return desired_contact_states

    def compute_desired_foot_height(self, gait_params, gait_indices):
        """Calculate desired foot height for the current timestep."""
        frequencies = gait_params[:, 0]
        mask_0 = (gait_indices < 0.25) & (gait_indices >= 0.0)  # lift up
        mask_1 = (gait_indices < 0.5) & (gait_indices >= 0.25)  # touch down
        mask_2 = (gait_indices < 0.75) & (gait_indices >= 0.5)  # lift up
        mask_3 = (gait_indices <= 1.0) & (gait_indices >= 0.75)  # touch down
        swing_start_time = torch.zeros(self.num_envs, device=self.device)
        swing_start_time[mask_1] = 0.25 / frequencies[mask_1]
        swing_start_time[mask_2] = 0.5 / frequencies[mask_2]
        swing_start_time[mask_3] = 0.75 / frequencies[mask_3]
        swing_end_time = swing_start_time + 0.25 / frequencies
        swing_start_pos = torch.ones(self.num_envs, device=self.device)
        swing_start_pos[mask_0] = 0.0
        swing_start_pos[mask_2] = 0.0
        swing_end_pos = torch.ones(self.num_envs, device=self.device)
        swing_end_pos[mask_1] = 0.0
        swing_end_pos[mask_3] = 0.0
        swing_end_vel = torch.ones(self.num_envs, device=self.device)
        swing_end_vel[mask_0] = 0.0
        swing_end_vel[mask_2] = 0.0
        swing_end_vel[mask_1] = self.touch_down_vel
        swing_end_vel[mask_3] = self.touch_down_vel

        # generate desire foot z trajectory
        swing_height = gait_params[:, 3]

        start = {
            'time': swing_start_time,
            'position': swing_start_pos * swing_height,
            'velocity': torch.zeros(self.num_envs, device=self.device),
        }
        end = {
            'time': swing_end_time,
            'position': swing_end_pos * swing_height,
            'velocity': swing_end_vel,
        }
        cubic_spline = CubicSpline(start, end)
        self.des_foot_height = cubic_spline.position(gait_indices / frequencies)
        self.des_foot_velocity_z = cubic_spline.velocity(gait_indices / frequencies)

    def _compute_force_reward(
        self, forces: torch.Tensor, desired_contacts: torch.Tensor
    ) -> torch.Tensor:
        """Compute force-based reward component."""
        reward = torch.zeros_like(forces[:, 0])
        if self.force_scale < 0:  # Negative scale means penalize unwanted contact
            for i in range(forces.shape[1]):
                reward += (1 - desired_contacts[:, i]) * (
                    1 - torch.exp(-forces[:, i] ** 2 / self.force_sigma)
                )
        else:  # Positive scale means reward desired contact
            for i in range(forces.shape[1]):
                reward += (1 - desired_contacts[:, i]) * torch.exp(
                    -forces[:, i] ** 2 / self.force_sigma
                )

        return (reward / forces.shape[1]) * self.force_scale

    def _compute_velocity_reward(
        self, foot_velocities: torch.Tensor, des_foot_velocities_z: torch.Tensor, desired_contacts: torch.Tensor
    ) -> torch.Tensor:
        """Compute velocity-based reward component."""
        foot_velocity_norm = torch.norm(foot_velocities, dim=-1)
        reward = torch.zeros_like(foot_velocity_norm[:, 0])
        if self.vel_scale < 0:  # Negative scale means penalize movement during contact
            for i in range(foot_velocity_norm.shape[1]):
                reward += desired_contacts[:, i] * (
                    1 - torch.exp(-foot_velocity_norm[:, i] ** 2 / self.vel_sigma)
                )
                if self.use_reference_motion:
                    swing_phase = 1 - desired_contacts[:, i]
                    reward += swing_phase * (
                        1 - torch.exp(-((foot_velocities[:, i, 2] - des_foot_velocities_z) ** 2) / self.vel_sigma)
                    )
        else:  # Positive scale means reward movement during swing
            for i in range(foot_velocity_norm.shape[1]):
                reward += desired_contacts[:, i] * torch.exp(
                    -foot_velocity_norm[:, i] ** 2 / self.vel_sigma
                )
                if self.use_reference_motion:
                    swing_phase = 1 - desired_contacts[:, i]
                    reward += swing_phase * torch.exp(
                        -((foot_velocities[:, i, 2] - des_foot_velocities_z) ** 2) / self.vel_sigma
                    )

        return (reward / foot_velocity_norm.shape[1]) * self.vel_scale

    def _compute_height_reward(
        self, foot_heights: torch.Tensor, des_foot_height: torch.Tensor, desired_contacts: torch.Tensor
    ) -> torch.Tensor:
        """Compute height-based reward component."""
        reward = torch.zeros_like(foot_heights[:, 0])
        if self.height_scale < 0:  # Negative scale means penalize movement during contact
            for i in range(foot_heights.shape[1]):
                if self.use_reference_motion:
                    swing_phase = 1 - desired_contacts[:, i]
                    # if self.cfg.terrain.mesh_type == "plane":
                    reward += swing_phase * (
                        1 - torch.exp(-(foot_heights[:, i] - des_foot_height) ** 2 / self.height_sigma)
                    )
                stand_phase = desired_contacts[:, i]
                reward += stand_phase * (1 - torch.exp(-(foot_heights[:, i]) ** 2 / self.height_sigma))
        else:  # Positive scale means reward movement during swing
            for i in range(foot_heights.shape[1]):
                if self.use_reference_motion:
                    swing_phase = 1 - desired_contacts[:, i]
                    # if self.cfg.terrain.mesh_type == "plane":
                    reward += swing_phase * torch.exp(
                        -(foot_heights[:, i] - des_foot_height) ** 2 / self.height_sigma
                    )
                stand_phase = desired_contacts[:, i]
                reward += stand_phase * torch.exp(-(foot_heights[:, i]) ** 2 / self.height_sigma)

        return (reward / foot_heights.shape[1]) * self.height_scale


class ActionSmoothnessPenalty(ManagerTermBase):
    """A reward term for penalizing large instantaneous changes in the network action output.

    This penalty encourages smoother actions over time.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        """Initialize the term.

        Args:
            cfg: The configuration of the reward term.
            env: The RL environment instance.
        """
        super().__init__(cfg, env)
        self.dt = env.step_dt
        self.prev_prev_action = None
        self.prev_action = None

    def __call__(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        """Compute the action smoothness penalty.

        Args:
            env: The RL environment instance.

        Returns:
            The penalty value based on the action smoothness.
        """
        current_action = env.action_manager.action.clone()
        if self.prev_action is None:
            self.prev_action = current_action
            return torch.zeros(current_action.shape[0], device=current_action.device)
        if self.prev_prev_action is None:
            self.prev_prev_action = self.prev_action
            self.prev_action = current_action
            return torch.zeros(current_action.shape[0], device=current_action.device)
        penalty = torch.sum(torch.square(current_action - 2 * self.prev_action + self.prev_prev_action), dim=1)
        self.prev_prev_action = self.prev_action
        self.prev_action = current_action
        startup_env_mask = env.episode_length_buf < 3
        penalty[startup_env_mask] = 0
        return penalty
