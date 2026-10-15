"""Audit per-term reward scale statistics on the untrained (random-action) policy.

Motivation (see doc/GAIT_SCRATCH_MODIFICATION_REPORT.md): unbounded linear/L2
penalties are fine in theory, but numerical training stability depends on no
single term dominating the return via outliers. This script measures, per active
reward term: raw and weighted P50/P95/P99/max, activation ratio, per-episode
contribution share, and failure-moment (pre-fall) peaks.

Usage (cloud, after Play smoke test):

    python scripts/rsl_rl/reward_audit.py \
        --task Isaac-Limx-SFYG-TRON2A-Perceptive-GaitScratch-TronCamp-v0 \
        --num_envs 256 --steps 2500 \
        --output reward_audit_scratch.md

Notes:
- Random actions approximate the from-scratch early phase, which is exactly the
  regime where outlier penalties are worst. Auditing a trained policy requires
  loading a checkpoint (future extension; play.py has the loading pattern).
- Per-term values are recomputed after env.step() from the same post-physics
  state the RewardManager used, so they match the trained reward exactly.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Audit per-term reward scale statistics.")
parser.add_argument("--task", type=str, required=True, help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=256, help="Number of environments to simulate.")
parser.add_argument("--steps", type=int, default=2500, help="Env steps to roll out (~50 s at 50 Hz).")
parser.add_argument("--output", type=str, default="reward_audit.md", help="Markdown report path.")
parser.add_argument("--pre_fall_steps", type=int, default=25, help="Window before a failure reset used for pre-fall peaks.")
parser.add_argument("--dominant_threshold", type=float, default=1.0, help="|weighted P99| above this flags 'dominant'.")
parser.add_argument("--spiky_ratio", type=float, default=100.0, help="max / |P99| above this flags 'spiky'.")
parser.add_argument("--pre_fall_burst_ratio", type=float, default=5.0, help="pre-fall P95 over |P99| above this flags 'burst'.")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import RewardTermCfg
from isaaclab_tasks.utils import parse_env_cfg

# Import extension to set up environment tasks
import bipedal_locomotion  # noqa: F401


def quantiles(values: torch.Tensor) -> dict:
    flat = values.flatten().float()
    if flat.numel() == 0:
        return {"p50": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0}
    return {
        "p50": torch.quantile(flat, 0.50).item(),
        "p95": torch.quantile(flat, 0.95).item(),
        "p99": torch.quantile(flat, 0.99).item(),
        "max": flat.abs().max().item() * (1.0 if flat.max() >= -flat.min() else -1.0),
    }


def main():
    env_cfg: ManagerBasedRLEnvCfg = parse_env_cfg(
        task_name=args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs
    )
    env = gym.make(args_cli.task, cfg=env_cfg)
    device = env.unwrapped.device
    dt = env.unwrapped.step_dt

    reward_manager = env.unwrapped.reward_manager
    # IsaacLab 2.3.x reshaped RewardManager.active_terms; iterating the config
    # object's attributes is stable across versions and includes every RewTerm
    # mounted on the RewardsCfg (dynamically added ones included).
    terms = []
    for term_name, term_cfg in vars(reward_manager.cfg).items():
        if not isinstance(term_cfg, RewardTermCfg):
            continue
        if term_cfg.weight == 0.0:
            continue
        if term_cfg.func is None or hasattr(term_cfg.func, "__func__"):
            print(f"[WARN] skipping class-based/missing term '{term_name}' (not supported by audit)")
            continue
        terms.append((term_name, term_cfg))
    terms.sort(key=lambda item: item[0])
    num_terms = len(terms)
    print(f"[INFO] auditing {num_terms} active reward terms on random policy, {args_cli.num_envs} envs x {args_cli.steps} steps")

    num_envs = args_cli.num_envs
    action_dim = env.unwrapped.action_manager.total_action_dim

    raw_all = torch.zeros(num_terms, args_cli.steps * num_envs, device=device)
    weighted_all = torch.zeros(num_terms, args_cli.steps * num_envs, device=device)
    ep_sum = torch.zeros(num_envs, num_terms, device=device)
    contrib_sum = torch.zeros(num_terms, device=device)
    contrib_abs_total = 0.0
    episodes_done = 0

    ring = torch.zeros(args_cli.pre_fall_steps, num_envs, num_terms, device=device)
    ring_ptr = 0
    pre_fall_max: list[torch.Tensor] = []  # per failing env: [num_terms] max over window

    obs, _ = env.reset()
    for step in range(args_cli.steps):
        actions = torch.rand(num_envs, action_dim, device=device) * 2.0 - 1.0
        obs, _, terminated, truncated, _ = env.step(actions)

        step_raw = torch.zeros(num_envs, num_terms, device=device)
        for term_idx, (_, term_cfg) in enumerate(terms):
            value = term_cfg.func(env.unwrapped, **term_cfg.params) * term_cfg.weight * dt
            step_raw[:, term_idx] = value
        raw_all[:, step * num_envs : (step + 1) * num_envs] = step_raw
        weighted_all[:, step * num_envs : (step + 1) * num_envs] = step_raw
        ep_sum += step_raw

        ring[ring_ptr] = step_raw
        ring_ptr = (ring_ptr + 1) % args_cli.pre_fall_steps

        fail_mask = torch.as_tensor(terminated, device=device).bool()
        done_mask = fail_mask | torch.as_tensor(truncated, device=device).bool()
        if fail_mask.any():
            pre_fall_max.append(ring[:, fail_mask, :].amax(dim=0))  # [n_fail, num_terms]
        if done_mask.any():
            contrib_sum += ep_sum[done_mask].sum(dim=0)
            contrib_abs_total += ep_sum[done_mask].abs().sum().item()
            episodes_done += int(done_mask.sum().item())
            ep_sum[done_mask] = 0.0

    # ---- report ----
    lines = []
    lines.append(f"# Reward scale audit — {args_cli.task}")
    lines.append("")
    lines.append(
        f"random policy, {num_envs} envs x {args_cli.steps} steps (dt={dt:.4f}s), "
        f"{episodes_done} episodes finished. Values below are per-step, weight and dt included."
    )
    lines.append("")
    lines.append(
        "| term | weight | P50 | P95 | P99 | max | act% | ep-contrib% | pre-fall P95 | flags |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|")

    for term_idx, (term_name, term_cfg) in enumerate(terms):
        weighted = weighted_all[term_idx]
        stats = quantiles(weighted)
        activation = (weighted.abs() > 1.0e-6).float().mean().item() * 100.0
        contrib = 0.0 if contrib_abs_total == 0.0 else contrib_sum[term_idx].item() / contrib_abs_total * 100.0
        if pre_fall_max:
            bursts = torch.stack(pre_fall_max)[:, term_idx]  # per failing episode max
            pre_p95 = torch.quantile(bursts.float(), 0.95).item() if bursts.numel() > 0 else 0.0
        else:
            pre_p95 = 0.0

        flags = []
        if max(abs(stats["p99"]), 0.0) > args_cli.dominant_threshold:
            flags.append("DOMINANT")
        if stats["max"] != 0.0 and abs(stats["max"]) > args_cli.spiky_ratio * max(abs(stats["p99"]), 1.0e-9):
            flags.append("SPIKY")
        if abs(pre_p95) > args_cli.pre_fall_burst_ratio * max(abs(stats["p99"]), 1.0e-9):
            flags.append("PRE-FALL-BURST")
        if activation < 1.0:
            flags.append("DEAD")

        lines.append(
            f"| `{term_name}` | {term_cfg.weight:+g} | {stats['p50']:.4f} | {stats['p95']:.4f} | "
            f"{stats['p99']:.4f} | {stats['max']:.4f} | {activation:.1f} | {contrib:+.1f} | "
            f"{pre_p95:.4f} | {', '.join(flags) if flags else ''} |"
        )

    lines.append("")
    lines.append(
        "Trigger conditions for treatment (adjust weight / add deadband / normalize; "
        "do NOT globally bound functions): DOMINANT = |P99| above main-task reward scale; "
        "SPIKY = max >> P99 (few frames carry the episode); PRE-FALL-BURST = order-of-magnitude "
        "explosion right before failure resets; DEAD = gate almost never opens."
    )
    report = "\n".join(lines)
    print(report)
    with open(args_cli.output, "w", encoding="utf-8") as f:
        f.write(report + "\n")
    print(f"[INFO] report written to {os.path.abspath(args_cli.output)}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
