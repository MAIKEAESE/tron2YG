"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import csv
import math
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../rsl_rl")))

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--checkpoint_path", type=str, default=None, help="Relative path to checkpoint file.")
parser.add_argument("--follow_camera", action="store_true", default=False, help="Follow env-0 robot while recording.")
parser.add_argument("--camera_distance", type=float, default=5.0, help="Follow camera distance behind the robot.")
parser.add_argument("--camera_height", type=float, default=2.6, help="Follow camera height above the robot.")
parser.add_argument("--camera_side", type=float, default=0.8, help="Follow camera lateral offset.")
parser.add_argument("--camera_lookahead", type=float, default=2.5, help="Follow camera target distance ahead.")
parser.add_argument("--pose_log", action="store_true", default=False, help="Write env-0 pose diagnostics to CSV.")
parser.add_argument("--pose_log_interval", type=int, default=10, help="Pose log interval in simulation steps.")
parser.add_argument("--pose_log_path", type=str, default=None, help="Optional CSV path for pose diagnostics.")

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""


import gymnasium as gym
import os
import torch

from rsl_rl.runner import OnPolicyRunner

from isaaclab.envs import ManagerBasedRLEnvCfg,DirectMARLEnv, multi_agent_to_single_agent
import isaaclab.utils.math as math_utils
from isaaclab.utils.dict import print_dict
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
# Import extensions to set up environment tasks
import bipedal_locomotion  # noqa: F401
from bipedal_locomotion.utils.wrappers.rsl_rl import RslRlPpoAlgorithmMlpCfg, export_mlp_as_onnx, export_policy_as_jit


def update_follow_camera(env):
    """Keep the render camera behind env-0 and looking ahead along the track."""
    if not args_cli.follow_camera:
        return

    base_env = env.unwrapped
    robot = base_env.scene["robot"]
    root_pos = robot.data.root_pos_w[0].detach().cpu()
    root_x, root_y, root_z = (float(root_pos[0]), float(root_pos[1]), float(root_pos[2]))

    eye = [
        root_x - args_cli.camera_distance,
        root_y + args_cli.camera_side,
        root_z + args_cli.camera_height,
    ]
    target = [
        root_x + args_cli.camera_lookahead,
        root_y,
        root_z + 0.7,
    ]
    base_env.sim.set_camera_view(eye=eye, target=target)


def _angle_diff_rad(angle, reference):
    return math.atan2(math.sin(angle - reference), math.cos(angle - reference))


def _find_body_id(body_names, candidates):
    for candidate in candidates:
        for idx, name in enumerate(body_names):
            if name == candidate:
                return idx, name
    for candidate in candidates:
        for idx, name in enumerate(body_names):
            if candidate in name:
                return idx, name
    return None, None


def _find_joint_id(joint_names, candidates):
    for candidate in candidates:
        for idx, name in enumerate(joint_names):
            if name == candidate:
                return idx, name
    for candidate in candidates:
        for idx, name in enumerate(joint_names):
            if candidate in name:
                return idx, name
    return None, None


def _csv_float(value):
    if value is None:
        return ""
    if torch.is_tensor(value):
        value = value.detach().cpu()
        if value.numel() != 1:
            return ""
        value = float(value)
    try:
        if math.isnan(float(value)):
            return ""
    except TypeError:
        return ""
    return float(value)


class PoseCsvLogger:
    """Write env-0 heading, hip, and foot diagnostics for video review."""

    def __init__(self, env, log_dir):
        self.env = env
        self.step = 0
        self.interval = max(args_cli.pose_log_interval, 1)
        self.path = args_cli.pose_log_path or os.path.join(log_dir, "pose_metrics.csv")
        self.file = open(self.path, "w", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(
            self.file,
            fieldnames=[
                "step",
                "root_x",
                "root_y",
                "root_z",
                "base_yaw_deg",
                "base_forward_x",
                "base_forward_y",
                "vel_x",
                "vel_y",
                "speed_xy",
                "velocity_yaw_deg",
                "velocity_minus_base_deg",
                "target_heading_error_deg",
                "hip_line_yaw_deg",
                "hip_forward_yaw_deg",
                "hip_forward_minus_base_deg",
                "left_foot_x_b",
                "left_foot_y_b",
                "left_foot_z_b",
                "right_foot_x_b",
                "right_foot_y_b",
                "right_foot_z_b",
                "left_foot_vel_x_b",
                "left_foot_vel_y_b",
                "left_foot_vel_z_b",
                "right_foot_vel_x_b",
                "right_foot_vel_y_b",
                "right_foot_vel_z_b",
                "left_foot_contact",
                "right_foot_contact",
                "left_foot_contact_force",
                "right_foot_contact_force",
                "foot_x_diff_b",
                "foot_y_diff_b",
                "foot_z_diff_b",
                "foot_x_sum_b",
                "foot_y_sum_b",
                "swing_foot_liftoff",
                "swing_foot_forward_delta_x",
                "stance_foot_speed_xy",
                "left_knee_angle",
                "right_knee_angle",
                "left_knee_angle_delta",
                "right_knee_angle_delta",
                "obstacle_height_front",
                "terrain_level",
                "terrain_segment",
                "non_foot_contact",
            ],
        )
        self.writer.writeheader()
        self._resolve_body_ids()
        print(f"[INFO] Writing pose diagnostics to: {self.path}")

    def _resolve_body_ids(self):
        robot = self.env.unwrapped.scene["robot"]
        body_names = list(getattr(robot, "body_names", []))
        self.left_hip_id, self.left_hip_name = _find_body_id(
            body_names,
            ["proximal_pitch_L_Link", "proximal_roll_L_Link", "proximal_yaw_L_Link"],
        )
        self.right_hip_id, self.right_hip_name = _find_body_id(
            body_names,
            ["proximal_pitch_R_Link", "proximal_roll_R_Link", "proximal_yaw_R_Link"],
        )
        self.left_foot_id, self.left_foot_name = _find_body_id(body_names, ["ankle_pitch_L_Link"])
        self.right_foot_id, self.right_foot_name = _find_body_id(body_names, ["ankle_pitch_R_Link"])
        joint_names = list(getattr(robot, "joint_names", []))
        self.left_knee_id, self.left_knee_name = _find_joint_id(joint_names, ["knee_L_Joint"])
        self.right_knee_id, self.right_knee_name = _find_joint_id(joint_names, ["knee_R_Joint"])
        missing = [
            name
            for body_id, name in [
                (self.left_hip_id, "left hip"),
                (self.right_hip_id, "right hip"),
                (self.left_foot_id, "left foot"),
                (self.right_foot_id, "right foot"),
            ]
            if body_id is None
        ]
        if missing:
            raise RuntimeError(f"Pose log could not resolve body ids for: {', '.join(missing)}")
        print(
            "[INFO] Pose log bodies: "
            f"hip=({self.left_hip_name}, {self.right_hip_name}), "
            f"foot=({self.left_foot_name}, {self.right_foot_name}), "
            f"knee=({self.left_knee_name}, {self.right_knee_name})"
        )

    def _foot_contact_metrics(self):
        sensors = getattr(self.env.unwrapped.scene, "sensors", {})
        contact_sensor = sensors.get("contact_forces") if hasattr(sensors, "get") else None
        if contact_sensor is None:
            return {
                "left_foot_contact": "",
                "right_foot_contact": "",
                "left_foot_contact_force": "",
                "right_foot_contact_force": "",
                "non_foot_contact": "",
            }

        forces = contact_sensor.data.net_forces_w_history[0]
        force_norms = torch.norm(forces, dim=-1).max(dim=0)[0]
        sensor_body_names = list(getattr(contact_sensor, "body_names", []))
        if not sensor_body_names:
            sensor_body_names = list(getattr(contact_sensor.data, "body_names", []))

        def sensor_body_index(body_name):
            for idx, name in enumerate(sensor_body_names):
                if name == body_name:
                    return idx
            for idx, name in enumerate(sensor_body_names):
                if body_name in name or name in body_name:
                    return idx
            return None

        left_idx = sensor_body_index(self.left_foot_name)
        right_idx = sensor_body_index(self.right_foot_name)
        left_force = force_norms[left_idx] if left_idx is not None else None
        right_force = force_norms[right_idx] if right_idx is not None else None
        foot_indices = {idx for idx in (left_idx, right_idx) if idx is not None}
        non_foot_contact = ""
        if sensor_body_names and force_norms.numel() == len(sensor_body_names):
            non_foot_mask = torch.ones_like(force_norms, dtype=torch.bool)
            for idx in foot_indices:
                non_foot_mask[idx] = False
            non_foot_contact = bool(torch.any(force_norms[non_foot_mask] > 1.0).item())

        return {
            "left_foot_contact": int(bool(left_force is not None and left_force.item() > 1.0)),
            "right_foot_contact": int(bool(right_force is not None and right_force.item() > 1.0)),
            "left_foot_contact_force": _csv_float(left_force),
            "right_foot_contact_force": _csv_float(right_force),
            "non_foot_contact": int(non_foot_contact) if isinstance(non_foot_contact, bool) else "",
        }

    def _obstacle_height_front(self, robot):
        sensors = getattr(self.env.unwrapped.scene, "sensors", {})
        ray_sensor = sensors.get("height_scanner") if hasattr(sensors, "get") else None
        if ray_sensor is None:
            return ""
        ray_hits = ray_sensor.data.ray_hits_w[0:1]
        valid_hits = torch.isfinite(ray_hits).all(dim=-1)
        yaw_frame = math_utils.yaw_quat(robot.data.root_quat_w[0:1]).unsqueeze(1).expand(-1, ray_hits.shape[1], -1)
        rel_hits = math_utils.quat_apply_inverse(yaw_frame, ray_hits - robot.data.root_pos_w[0:1].unsqueeze(1))
        rel_x = rel_hits[..., 0]
        front_mask = valid_hits & (rel_x > 0.15)
        near_mask = valid_hits & (torch.abs(rel_x) < 0.25)
        low = torch.full_like(ray_hits[..., 2], -1.0e6)
        front_z = torch.max(torch.where(front_mask, ray_hits[..., 2], low), dim=1)[0]
        near_count = torch.sum(near_mask.float(), dim=1).clamp(min=1.0)
        near_z = torch.sum(torch.where(near_mask, ray_hits[..., 2], torch.zeros_like(ray_hits[..., 2])), dim=1) / near_count
        obstacle_height = torch.clamp(front_z - near_z, min=0.0)
        return _csv_float(obstacle_height[0])

    def _terrain_metrics(self):
        base_env = self.env.unwrapped
        scene = base_env.scene
        terrain_level = ""
        terrain_segment = ""
        for owner in (base_env, scene):
            for attr_name in ("_tron_camp_curriculum_levels", "terrain_levels", "terrain_level"):
                value = getattr(owner, attr_name, None)
                if value is not None:
                    try:
                        terrain_level = int(value[0].detach().cpu()) if torch.is_tensor(value) else int(value[0])
                    except (TypeError, IndexError, ValueError):
                        terrain_level = ""
                    break
            if terrain_level != "":
                break
        for owner in (base_env, scene):
            for attr_name in ("_tron_camp_last_segment_ids", "terrain_segments", "terrain_segment", "terrain_types"):
                value = getattr(owner, attr_name, None)
                if value is not None:
                    try:
                        terrain_segment = int(value[0].detach().cpu()) if torch.is_tensor(value) else int(value[0])
                    except (TypeError, IndexError, ValueError):
                        terrain_segment = ""
                    break
            if terrain_segment != "":
                break
        return terrain_level, terrain_segment

    def record(self):
        self.step += 1
        if self.step % self.interval != 0:
            return

        robot = self.env.unwrapped.scene["robot"]
        root_pos = robot.data.root_pos_w[0].detach()
        root_vel = robot.data.root_lin_vel_w[0].detach()
        root_quat = robot.data.root_quat_w[0:1].detach()
        _, _, yaw = math_utils.euler_xyz_from_quat(root_quat)
        base_yaw = float(yaw[0].detach().cpu())
        forward = math_utils.quat_apply(
            root_quat, torch.tensor([[1.0, 0.0, 0.0]], device=robot.device)
        )[0]

        vel_x = float(root_vel[0].detach().cpu())
        vel_y = float(root_vel[1].detach().cpu())
        speed_xy = math.hypot(vel_x, vel_y)
        velocity_yaw = math.atan2(vel_y, vel_x) if speed_xy > 1e-4 else float("nan")

        body_pos = robot.data.body_link_pos_w[0].detach()
        left_hip = body_pos[self.left_hip_id]
        right_hip = body_pos[self.right_hip_id]
        hip_line = left_hip - right_hip
        hip_line_yaw = math.atan2(float(hip_line[1].cpu()), float(hip_line[0].cpu()))
        hip_forward_yaw = hip_line_yaw - math.pi / 2.0

        foot_ids = torch.tensor([self.left_foot_id, self.right_foot_id], device=robot.device, dtype=torch.long)
        base_quat = robot.data.root_link_quat_w[:, None, :].expand(-1, 2, -1)
        base_pos = robot.data.root_link_state_w[:, None, :3].expand(-1, 2, -1)
        feet_pos_b = math_utils.quat_apply_inverse(base_quat, robot.data.body_link_pos_w[:, foot_ids] - base_pos)[0]
        feet_vel_b = math_utils.quat_apply_inverse(base_quat, robot.data.body_lin_vel_w[:, foot_ids])[0]
        left_foot_b = feet_pos_b[0].detach().cpu()
        right_foot_b = feet_pos_b[1].detach().cpu()
        left_foot_vel_b = feet_vel_b[0].detach().cpu()
        right_foot_vel_b = feet_vel_b[1].detach().cpu()
        contact_metrics = self._foot_contact_metrics()
        left_contact = contact_metrics["left_foot_contact"] == 1
        right_contact = contact_metrics["right_foot_contact"] == 1
        liftoff = abs(float(left_foot_b[2] - right_foot_b[2]))
        if left_contact and not right_contact:
            swing_forward_delta_x = float(right_foot_b[0] - left_foot_b[0])
            stance_speed_xy = math.hypot(float(left_foot_vel_b[0]), float(left_foot_vel_b[1]))
        elif right_contact and not left_contact:
            swing_forward_delta_x = float(left_foot_b[0] - right_foot_b[0])
            stance_speed_xy = math.hypot(float(right_foot_vel_b[0]), float(right_foot_vel_b[1]))
        else:
            swing_forward_delta_x = ""
            stance_speed_xy = ""
        left_knee = robot.data.joint_pos[0, self.left_knee_id] if self.left_knee_id is not None else None
        right_knee = robot.data.joint_pos[0, self.right_knee_id] if self.right_knee_id is not None else None
        left_knee_default = robot.data.default_joint_pos[0, self.left_knee_id] if self.left_knee_id is not None else None
        right_knee_default = robot.data.default_joint_pos[0, self.right_knee_id] if self.right_knee_id is not None else None
        terrain_level, terrain_segment = self._terrain_metrics()

        row = {
                "step": self.step,
                "root_x": float(root_pos[0].cpu()),
                "root_y": float(root_pos[1].cpu()),
                "root_z": float(root_pos[2].cpu()),
                "base_yaw_deg": math.degrees(base_yaw),
                "base_forward_x": float(forward[0].cpu()),
                "base_forward_y": float(forward[1].cpu()),
                "vel_x": vel_x,
                "vel_y": vel_y,
                "speed_xy": speed_xy,
                "velocity_yaw_deg": math.degrees(velocity_yaw) if not math.isnan(velocity_yaw) else "",
                "velocity_minus_base_deg": math.degrees(_angle_diff_rad(velocity_yaw, base_yaw))
                if not math.isnan(velocity_yaw)
                else "",
                "target_heading_error_deg": math.degrees(_angle_diff_rad(base_yaw, 0.0)),
                "hip_line_yaw_deg": math.degrees(hip_line_yaw),
                "hip_forward_yaw_deg": math.degrees(hip_forward_yaw),
                "hip_forward_minus_base_deg": math.degrees(_angle_diff_rad(hip_forward_yaw, base_yaw)),
                "left_foot_x_b": float(left_foot_b[0]),
                "left_foot_y_b": float(left_foot_b[1]),
                "left_foot_z_b": float(left_foot_b[2]),
                "right_foot_x_b": float(right_foot_b[0]),
                "right_foot_y_b": float(right_foot_b[1]),
                "right_foot_z_b": float(right_foot_b[2]),
                "left_foot_vel_x_b": float(left_foot_vel_b[0]),
                "left_foot_vel_y_b": float(left_foot_vel_b[1]),
                "left_foot_vel_z_b": float(left_foot_vel_b[2]),
                "right_foot_vel_x_b": float(right_foot_vel_b[0]),
                "right_foot_vel_y_b": float(right_foot_vel_b[1]),
                "right_foot_vel_z_b": float(right_foot_vel_b[2]),
                "foot_x_diff_b": float(left_foot_b[0] - right_foot_b[0]),
                "foot_y_diff_b": float(left_foot_b[1] - right_foot_b[1]),
                "foot_z_diff_b": float(left_foot_b[2] - right_foot_b[2]),
                "foot_x_sum_b": float(left_foot_b[0] + right_foot_b[0]),
                "foot_y_sum_b": float(left_foot_b[1] + right_foot_b[1]),
                "swing_foot_liftoff": liftoff,
                "swing_foot_forward_delta_x": swing_forward_delta_x,
                "stance_foot_speed_xy": stance_speed_xy,
                "left_knee_angle": _csv_float(left_knee),
                "right_knee_angle": _csv_float(right_knee),
                "left_knee_angle_delta": _csv_float(left_knee - left_knee_default)
                if left_knee is not None and left_knee_default is not None
                else "",
                "right_knee_angle_delta": _csv_float(right_knee - right_knee_default)
                if right_knee is not None and right_knee_default is not None
                else "",
                "obstacle_height_front": self._obstacle_height_front(robot),
                "terrain_level": terrain_level,
                "terrain_segment": terrain_segment,
            }
        row.update(contact_metrics)
        self.writer.writerow(row)
        self.file.flush()

    def close(self):
        self.file.close()


def main():
    """Play with RSL-RL agent."""
    # parse configuration
    env_cfg: ManagerBasedRLEnvCfg = parse_env_cfg(
        task_name=args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs
    )
    agent_cfg: RslRlPpoAlgorithmMlpCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    env_cfg.seed = agent_cfg.seed

    # specify directory for logging experiments
    if args_cli.checkpoint_path is None:
        log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
        log_root_path = os.path.abspath(log_root_path)
        print(f"[INFO] Loading experiment from directory: {log_root_path}")
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    else:
        resume_path = args_cli.checkpoint_path
    log_dir = os.path.dirname(resume_path)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)
    # load previously trained model
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)
    encoder = ppo_runner.get_inference_encoder(device=env.unwrapped.device)

    # export policy to onnx
    if EXPORT_POLICY:
        export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
        export_policy_as_jit(
            ppo_runner.alg.actor_critic, export_model_dir
        )
        print("Exported policy as jit script to: ", export_model_dir)
        export_mlp_as_onnx(
            ppo_runner.alg.actor_critic.actor, 
            export_model_dir, 
            "policy",
            ppo_runner.alg.actor_critic.num_actor_obs,
        )
        export_mlp_as_onnx(
            ppo_runner.alg.encoder,
            export_model_dir,
            "encoder",
            ppo_runner.alg.encoder.num_input_dim,
        )
    # reset environment
    obs_dict = env.get_observations()
    obs = obs_dict["policy"]
    obs_history = obs_dict.get("obsHistory")
    obs_history = obs_history.flatten(start_dim=1)
    commands = obs_dict.get("commands") 
    pose_logger = PoseCsvLogger(env, log_dir) if args_cli.pose_log else None
    update_follow_camera(env)
 
    # simulate environment
    try:
        while simulation_app.is_running():
            # run everything in inference mode
            with torch.inference_mode():
                # agent stepping
                est = encoder(obs_history)
                actions = policy(torch.cat((est, obs, commands), dim=-1).detach())
                # env stepping
                obs_dict, _, _, infos = env.step(actions)
                obs = obs_dict["policy"]
                obs_history = obs_dict.get("obsHistory")
                obs_history = obs_history.flatten(start_dim=1)
                commands = obs_dict.get("commands") 
                update_follow_camera(env)
                if pose_logger is not None:
                    pose_logger.record()
    finally:
        if pose_logger is not None:
            pose_logger.close()
        # close the simulator
        env.close()


if __name__ == "__main__":
    EXPORT_POLICY = True
    # run the main execution
    main()
    # close sim app
    simulation_app.close()
