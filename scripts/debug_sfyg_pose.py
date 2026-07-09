"""Inspect SFYG reset pose, body frames, foot symmetry, and joint defaults."""

from __future__ import annotations

import argparse
import os
import re
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../rsl_rl")))

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Debug SFYG robot pose and frame alignment.")
parser.add_argument(
    "--task",
    type=str,
    default="Isaac-Limx-SFYG-TRON2A-Blind-Gait-Play-v0",
    help="Task used to instantiate the robot.",
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to create.")
parser.add_argument("--foot_body_regex", type=str, default="ankle_pitch_.*", help="Regex used to select foot bodies.")
parser.add_argument("--step_count", type=int, default=5, help="Number of zero-action steps before inspection.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from isaaclab_tasks.utils import parse_env_cfg
import isaaclab.utils.math as math_utils

import bipedal_locomotion  # noqa: F401


def _as_list(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return list(value)


def _tensor_row(tensor, row=0):
    return tensor.detach().cpu()[row]


def _fmt_tensor(tensor, precision=5):
    values = tensor.detach().cpu().flatten().tolist()
    return "[" + ", ".join(f"{value:.{precision}f}" for value in values) + "]"


def _find_names(names, pattern):
    regex = re.compile(pattern)
    return [(idx, name) for idx, name in enumerate(names) if regex.fullmatch(name) or regex.search(name)]


def _print_section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def _print_names(title, names):
    _print_section(title)
    for idx, name in enumerate(names):
        print(f"{idx:03d}: {name}")


def _print_joint_defaults(robot):
    joint_names = _as_list(getattr(robot, "joint_names", []))
    default_joint_pos = _tensor_row(robot.data.default_joint_pos)
    joint_pos = _tensor_row(robot.data.joint_pos)

    _print_section("Joint default/current positions")
    for idx, name in enumerate(joint_names):
        print(f"{idx:03d}: {name:32s} default={default_joint_pos[idx]: .6f} current={joint_pos[idx]: .6f}")


def _print_lr_joint_pairs(robot):
    joint_names = _as_list(getattr(robot, "joint_names", []))
    default_joint_pos = _tensor_row(robot.data.default_joint_pos)

    left = {name.replace("_L_", "_?_"): idx for idx, name in enumerate(joint_names) if "_L_" in name}
    right = {name.replace("_R_", "_?_"): idx for idx, name in enumerate(joint_names) if "_R_" in name}

    _print_section("L/R joint default comparison")
    for key in sorted(left.keys() & right.keys()):
        left_idx = left[key]
        right_idx = right[key]
        print(
            f"{joint_names[left_idx]:32s} {default_joint_pos[left_idx]: .6f} | "
            f"{joint_names[right_idx]:32s} {default_joint_pos[right_idx]: .6f} | "
            f"sum={default_joint_pos[left_idx] + default_joint_pos[right_idx]: .6f} "
            f"diff={default_joint_pos[left_idx] - default_joint_pos[right_idx]: .6f}"
        )


def _print_base_pose(robot):
    root_pos = _tensor_row(robot.data.root_pos_w)
    root_quat = _tensor_row(robot.data.root_quat_w)
    roll, pitch, yaw = math_utils.euler_xyz_from_quat(robot.data.root_quat_w)
    forward = math_utils.quat_apply(robot.data.root_quat_w, torch.tensor([[1.0, 0.0, 0.0]], device=robot.device))
    left = math_utils.quat_apply(robot.data.root_quat_w, torch.tensor([[0.0, 1.0, 0.0]], device=robot.device))

    _print_section("Base pose/frame")
    print(f"root_pos_w              = {_fmt_tensor(root_pos)}")
    print(f"root_quat_w             = {_fmt_tensor(root_quat)}")
    print(f"root_rpy_rad            = {_fmt_tensor(torch.stack((roll[0], pitch[0], yaw[0])))}")
    print(f"root_rpy_deg            = {_fmt_tensor(torch.rad2deg(torch.stack((roll[0], pitch[0], yaw[0]))))}")
    print(f"base +X forward in world = {_fmt_tensor(forward[0])}")
    print(f"base +Y left in world    = {_fmt_tensor(left[0])}")


def _print_foot_geometry(robot, foot_body_regex):
    body_names = _as_list(getattr(robot, "body_names", []))
    foot_matches = _find_names(body_names, foot_body_regex)

    _print_section(f"Foot body matches: {foot_body_regex}")
    if not foot_matches:
        print("No foot body matched. Use --foot_body_regex to inspect another body set.")
        return
    for idx, name in foot_matches:
        print(f"{idx:03d}: {name}")

    if len(foot_matches) < 2:
        print("Need at least two foot bodies for symmetry diagnostics.")
        return

    selected = foot_matches[:2]
    body_ids = torch.tensor([idx for idx, _ in selected], device=robot.device, dtype=torch.long)
    names = [name for _, name in selected]

    foot_pos_w = robot.data.body_link_pos_w[0, body_ids].detach()
    base_quat = robot.data.root_link_quat_w[:, None, :].expand(-1, len(body_ids), -1)
    base_pos = robot.data.root_link_state_w[:, None, :3].expand(-1, len(body_ids), -1)
    foot_pos_b = math_utils.quat_apply_inverse(base_quat, robot.data.body_link_pos_w[:, body_ids] - base_pos)[0]

    _print_section("Foot geometry")
    for i, name in enumerate(names):
        print(f"{name:32s} world={_fmt_tensor(foot_pos_w[i])} base={_fmt_tensor(foot_pos_b[i])}")

    diff_b = foot_pos_b[0] - foot_pos_b[1]
    sum_x = foot_pos_b[0, 0] + foot_pos_b[1, 0]
    sum_y = foot_pos_b[0, 1] + foot_pos_b[1, 1]
    print("\nDiagnostics:")
    print(f"foot0 - foot1 in base frame = {_fmt_tensor(diff_b)}")
    print(f"abs x difference            = {abs(float(diff_b[0])):.6f}  (want near 0 at neutral stance)")
    print(f"abs y difference            = {abs(float(diff_b[1])):.6f}  (stance width)")
    print(f"x sum around body           = {float(sum_x): .6f}  (want near 0 if fore-aft symmetric)")
    print(f"y sum around body           = {float(sum_y): .6f}  (want near 0 if left/right symmetric)")


def main():
    env_cfg = parse_env_cfg(task_name=args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env = gym.make(args_cli.task, cfg=env_cfg)
    env.reset()

    base_env = env.unwrapped
    robot = base_env.scene["robot"]

    actions = torch.zeros((base_env.num_envs, base_env.action_manager.total_action_dim), device=base_env.device)
    for _ in range(max(args_cli.step_count, 0)):
        env.step(actions)

    body_names = _as_list(getattr(robot, "body_names", []))
    joint_names = _as_list(getattr(robot, "joint_names", []))

    _print_section("Task")
    print(f"task                  = {args_cli.task}")
    print(f"num_envs              = {base_env.num_envs}")
    print(f"device                = {base_env.device}")
    print(f"foot_body_regex       = {args_cli.foot_body_regex}")
    print(f"zero_action_steps     = {args_cli.step_count}")

    _print_names("Robot body names", body_names)
    _print_names("Robot joint names", joint_names)
    _print_base_pose(robot)
    _print_foot_geometry(robot, args_cli.foot_body_regex)
    _print_joint_defaults(robot)
    _print_lr_joint_pairs(robot)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
