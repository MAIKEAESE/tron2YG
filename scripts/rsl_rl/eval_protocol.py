"""Fixed evaluation protocol: the same ruler for every checkpoint.

Why this exists: training-reward curves only say "converged", not "effective"
(a conservative local optimum also plateaus). This script runs a deterministic
policy (mean action, no exploration noise) on a FIXED seed and reports the
same metrics for any checkpoint, so runs are comparable and changes are
attributable:

  - max local-x progress per episode (track distance)
  - segment pass rates (ramp / downhill / bumps / stairs / platform)
  - survival steps and termination cause (time_out vs failure)

Usage (cloud):

    python scripts/rsl_rl/eval_protocol.py \
        --task Isaac-Limx-SFYG-TRON2A-Perceptive-GaitScratch-TronCamp-Play-v0 \
        --checkpoint_path logs/rsl_rl/sfyg_tron_2a_perceptive_gait_scratch/scratch_a_xxx/model_1000.pt \
        --num_envs 50 --repeat 2 --eval_seed 42

Run it every ~200 training iterations on the same seed; plot `progress_mean`
and `stairs_pass_pct` over checkpoints — that second curve is the one that
tells you whether the policy is improving on what you actually care about.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../rsl_rl")))

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Fixed-seed evaluation of a checkpoint.")
parser.add_argument("--task", type=str, required=True, help="Name of the task (use the -Play- variant).")
parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the model .pt to evaluate.")
parser.add_argument("--num_envs", type=int, default=50, help="Parallel eval envs (episodes per repeat).")
parser.add_argument("--repeat", type=int, default=2, help="How many passes over all envs.")
parser.add_argument("--eval_seed", type=int, default=42, help="Fixed seed: THE ruler. Keep it constant across checkpoints.")
parser.add_argument("--max_episode_steps", type=int, default=2000, help="Safety cap per episode (~40 s at 50 Hz).")
parser.add_argument("--output_dir", type=str, default="eval", help="Directory for eval results.")
parser.add_argument(
    "--history_csv", type=str, default="eval/eval_history.csv",
    help="Append-mode history file: one row per run, plot curves over checkpoints from it.",
)
# segment boundaries along local-x of the generated TronCamp track (see mixed_terrain.py)
parser.add_argument("--x_ramp", type=float, default=2.0, help="Local-x where the up-ramp begins.")
parser.add_argument("--x_downhill", type=float, default=10.0, help="Local-x where the downhill exit begins.")
parser.add_argument("--x_bumps", type=float, default=16.0, help="Local-x where the bumps section begins.")
parser.add_argument("--x_stairs", type=float, default=24.0, help="Local-x where the stairs section begins.")
parser.add_argument("--x_past_stairs", type=float, default=39.0, help="Local-x past the last stair.")
parser.add_argument("--x_platform", type=float, default=42.0, help="Local-x where the final platform begins.")
# append RSL-RL cli args: parse_rsl_rl_cfg expects flags like --resume on the namespace
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

from rsl_rl.runner import OnPolicyRunner

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

# Import extension to set up environment tasks
import bipedal_locomotion  # noqa: F401

from bipedal_locomotion.utils.wrappers.rsl_rl import RslRlPpoAlgorithmMlpCfg


def main():
    env_cfg: ManagerBasedRLEnvCfg = parse_env_cfg(
        task_name=args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs
    )
    agent_cfg: RslRlPpoAlgorithmMlpCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # THE ruler: same seed across every checkpoint ever evaluated with this script.
    env_cfg.seed = args_cli.eval_seed
    torch.manual_seed(args_cli.eval_seed)

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)
    device = env.unwrapped.device

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(args_cli.checkpoint_path)
    policy = runner.get_inference_policy(device=device)
    encoder = runner.get_inference_encoder(device=device)

    base_env = env.unwrapped
    robot = base_env.scene["robot"]
    env_origins = base_env.scene.env_origins.to(device)

    progress_max = torch.zeros(args_cli.num_envs, device=device)
    episode_step = torch.zeros(args_cli.num_envs, dtype=torch.long, device=device)

    records = []  # one dict per finished episode

    obs_dict, _ = env.reset()
    obs = obs_dict["policy"]
    obs_history = obs_dict.get("obsHistory")
    if obs_history is not None:
        obs_history = obs_history.flatten(start_dim=1)
    commands = obs_dict.get("commands")

    total_target = args_cli.num_envs * args_cli.repeat
    seg_keys = [
        ("ramp_pass", args_cli.x_ramp),
        ("downhill_pass", args_cli.x_downhill),
        ("bumps_pass", args_cli.x_bumps),
        ("stairs_reach", args_cli.x_stairs),
        ("stairs_pass", args_cli.x_past_stairs),
        ("platform_reach", args_cli.x_platform),
    ]

    with torch.inference_mode():
        while len(records) < total_target and simulation_app.is_running():
            est = encoder(obs_history)
            actions = policy(torch.cat((est, obs, commands), dim=-1).detach())
            obs_dict, _, dones, _ = env.step(actions)
            obs = obs_dict["policy"]
            obs_history = obs_dict.get("obsHistory")
            if obs_history is not None:
                obs_history = obs_history.flatten(start_dim=1)
            commands = obs_dict.get("commands")

            episode_step += 1
            local_x = robot.data.root_pos_w[:, 0] - env_origins[:, 0]
            progress_max = torch.maximum(progress_max, local_x)

            done_mask = torch.as_tensor(dones, device=device).bool()
            if episode_step.max().item() >= args_cli.max_episode_steps:
                done_mask |= torch.ones_like(done_mask)  # safety cap counts as stuck

            if done_mask.any():
                time_out = base_env.termination_manager.time_out[done_mask]
                for env_i in torch.nonzero(done_mask).flatten().tolist():
                    prog = progress_max[env_i].item()
                    rec = {
                        "env": env_i,
                        "steps": int(episode_step[env_i].item()),
                        "progress_max": round(prog, 3),
                        "terminated": not bool(base_env.termination_manager.time_out[env_i].item()),
                    }
                    for key, bound in seg_keys:
                        rec[key] = int(prog >= bound)
                    records.append(rec)
                progress_max[done_mask] = 0.0
                episode_step[done_mask] = 0
                # silence unused warning for time_out bulk read
                _ = time_out

    # ---- summary ----
    if not records:
        print("[ERROR] no episodes finished; check env/task")
        env.close()
        return

    def pct(key):
        return 100.0 * sum(r[key] for r in records) / len(records)

    progs = sorted(r["progress_max"] for r in records)
    n = len(progs)
    summary = {
        "checkpoint": args_cli.checkpoint_path,
        "task": args_cli.task,
        "seed": args_cli.eval_seed,
        "episodes": n,
        "progress_mean": round(sum(progs) / n, 2),
        "progress_p25": round(progs[n // 4], 2),
        "progress_p50": round(progs[n // 2], 2),
        "progress_best": round(progs[-1], 2),
        "ramp_pass_pct": round(pct("ramp_pass"), 1),
        "downhill_pass_pct": round(pct("downhill_pass"), 1),
        "bumps_pass_pct": round(pct("bumps_pass"), 1),
        "stairs_reach_pct": round(pct("stairs_reach"), 1),
        "stairs_pass_pct": round(pct("stairs_pass"), 1),
        "platform_reach_pct": round(pct("platform_reach"), 1),
        "failure_rate_pct": round(100.0 * sum(r["terminated"] for r in records) / n, 1),
        "steps_mean": round(sum(r["steps"] for r in records) / n, 1),
    }

    os.makedirs(args.output_dir, exist_ok=True)
    tag = os.path.splitext(os.path.basename(args_cli.checkpoint_path))[0]

    with open(os.path.join(args.output_dir, f"eval_{tag}_episodes.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    header_needed = not os.path.exists(args_cli.history_csv)
    os.makedirs(os.path.dirname(args_cli.history_csv) or ".", exist_ok=True)
    with open(args_cli.history_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        if header_needed:
            writer.writeheader()
        writer.writerow(summary)

    # Mirror the audit script: stdout may not reach the platform log on normal
    # exit (block buffering); stderr has proven reliable.
    lines = ["==== fixed-seed eval ===="]
    lines += [f"  {key:>20}: {value}" for key, value in summary.items()]
    lines.append(
        f"[INFO] history appended to {os.path.abspath(args_cli.history_csv)} — plot progress_mean / stairs_pass_pct across checkpoints"
    )
    report = "\n".join(lines)
    print(report, flush=True)
    print(report, file=sys.stderr, flush=True)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
