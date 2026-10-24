"""Record a finite TRON2 clip. Copy beside the official scripts/rsl_rl/play.py.

Uses the existing IsaacLab/PyTorch environment. Does not train or export models.
Test headless rendering separately; an external wall-clock timeout is still required.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


def run_steps(app, env, policy, encoder, video_length):
    """Bound simulator calls even when the application stays running indefinitely."""
    obs_dict = env.get_observations()
    completed = 0
    for _ in range(video_length):
        if not app.is_running():
            break
        obs = obs_dict["policy"]
        history = obs_dict.get("obsHistory")
        commands = obs_dict.get("commands")
        if history is None or commands is None:
            raise RuntimeError("obsHistory or commands is missing; recording stopped.")
        with torch.inference_mode():
            latent = encoder(history.flatten(start_dim=1))
            actions = policy(torch.cat((latent, obs, commands), dim=-1))
            obs_dict, _, _, _ = env.step(actions)
        completed += 1
    return completed


def main(args, simulation_app):
    import gymnasium as gym
    from isaaclab_tasks.utils import parse_env_cfg
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    import bipedal_locomotion  # noqa: F401; registers the official task.
    from rsl_rl.runner import OnPolicyRunner

    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
    agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
    env_cfg.seed = args.seed
    agent_cfg.seed = args.seed
    agent_cfg.device = args.device

    # Fixed recording commands; the policy and training configuration on disk are untouched.
    velocity = env_cfg.commands.base_velocity
    velocity.heading_command = False
    velocity.rel_heading_envs = 0.0
    velocity.rel_standing_envs = 0.0
    velocity.ranges.lin_vel_x = (args.vx, args.vx)
    velocity.ranges.lin_vel_y = (0.0, 0.0)
    velocity.ranges.ang_vel_z = (0.0, 0.0)
    velocity.resampling_time_range = (1000000.0, 1000000.0)
    # Make the camera's side/oblique view meaningful; retain other reset randomization.
    reset = env_cfg.events.reset_robot_base
    if reset is not None and "pose_range" in reset.params:
        reset.params["pose_range"]["yaw"] = (0.0, 0.0)

    env_cfg.viewer.origin_type = "asset_root"
    env_cfg.viewer.asset_name = "robot"
    env_cfg.viewer.env_index = 0
    env_cfg.viewer.resolution = (960, 720)
    env_cfg.viewer.eye = (0.0, 3.0, 0.5) if args.view == "side" else (2.2, 2.5, 0.7)
    env_cfg.viewer.lookat = (0.0, 0.0, -0.25)

    output = Path(args.output_dir).expanduser().resolve()
    # Fail before launching an environment if this recording would reuse an output folder.
    output.mkdir(parents=True, exist_ok=False)
    env = None
    completed = 0
    try:
        env = gym.make(args.task, cfg=env_cfg, render_mode="rgb_array")
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=str(output),
            step_trigger=lambda step: step == 0,
            video_length=args.video_length,
            name_prefix="tron2_play",
            disable_logger=True,
        )
        env = RslRlVecEnvWrapper(env)
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=args.device)
        checkpoint = torch.load(args.checkpoint_path, map_location="cpu", weights_only=True)
        # Recording requires the complete matching policy, including its velocity estimator.
        runner.alg.actor_critic.load_state_dict(checkpoint["model_state_dict"], strict=True)
        runner.alg.encoder.load_state_dict(checkpoint["encoder_state_dict"], strict=True)
        policy = runner.get_inference_policy(device=env.unwrapped.device)
        encoder = runner.get_inference_encoder(device=env.unwrapped.device)
        commands = env.unwrapped.command_manager.get_command("base_velocity")
        print("[PLAY_RECORD] actual command:", commands.detach().cpu().tolist(), flush=True)
        if not torch.isfinite(commands).all() or not torch.allclose(
            commands, torch.tensor([[args.vx, 0.0, 0.0]], device=commands.device), atol=1e-5
        ):
            raise RuntimeError("Actual velocity command differs from the requested recording command.")
        completed = run_steps(simulation_app, env, policy, encoder, args.video_length)
    finally:
        if env is not None:
            env.close()  # Flush frames and finish the MP4 before reporting success.

    videos = sorted(output.glob("*.mp4"))
    if completed != args.video_length or not videos or any(p.stat().st_size == 0 for p in videos):
        raise RuntimeError(f"Recording incomplete: {completed} steps, videos={videos}")
    metadata = {
        "task": args.task,
        "checkpoint": str(Path(args.checkpoint_path).resolve()),
        "seed": args.seed,
        "vx": args.vx,
        "view": args.view,
        "steps": completed,
        "control_dt": env_cfg.decimation * env_cfg.sim.dt,
        "nominal_sim_seconds": completed * env_cfg.decimation * env_cfg.sim.dt,
        "videos": [str(p) for p in videos],
        "note": "A saved MP4 does not establish non-black frames or locomotion quality; inspect it.",
    }
    (output / "recording.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("[PLAY_RECORD_DONE]", json.dumps(metadata), flush=True)


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[2]
    if not (repo_root / "rsl_rl" / "rsl_rl").is_dir():
        raise RuntimeError("Copy this file into the official repository's scripts/rsl_rl/ directory first.")
    sys.path.insert(0, str(repo_root / "rsl_rl"))
    sys.path.insert(0, str(repo_root / "exts" / "bipedal_locomotion"))

    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="Isaac-Limx-SFYG-TRON2A-Blind-Flat-Play-v0")
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--output_dir", required=True, help="A new recording folder, not an existing directory.")
    parser.add_argument("--video_length", type=int, default=100, help="Maximum control steps, not wall-clock seconds.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--vx", type=float, default=0.5)
    parser.add_argument("--view", choices=("side", "oblique"), default="oblique")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if not 1 <= args.video_length <= 1500:
        parser.error("video_length must be 1..1500; use a short clip first.")
    if not math.isfinite(args.vx) or not 0.0 <= args.vx <= 1.0:
        parser.error("vx must be finite and between 0 and 1 m/s.")
    if not Path(args.checkpoint_path).is_file():
        parser.error("checkpoint_path does not point to an existing file.")
    if Path(args.output_dir).expanduser().exists():
        parser.error("output_dir already exists; choose a new name to preserve previous recordings.")
    args.enable_cameras = True
    app = AppLauncher(args)
    try:
        import torch
        main(args, app.app)
    finally:
        app.app.close()
