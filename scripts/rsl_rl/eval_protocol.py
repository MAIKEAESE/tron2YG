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
parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to the model .pt to evaluate (container-local).")
parser.add_argument("--checkpoint_url", type=str, default=None, help="Signed OSS URL; downloaded to a temp file when checkpoint_path is absent (the resume-mount fields only apply to trainType=2 tasks).")
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
import math

import torch

import isaaclab.utils.math as math_utils

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

    if args_cli.checkpoint_url and not args_cli.checkpoint_path:
        import tempfile
        import urllib.request

        dest = os.path.join(tempfile.gettempdir(), "eval_ckpt.pt")
        print(f"[INFO] downloading checkpoint from signed URL -> {dest}", file=sys.stderr, flush=True)
        urllib.request.urlretrieve(args_cli.checkpoint_url, dest)
        args_cli.checkpoint_path = dest

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

    # Posture telemetry: gait-quality metrics (duty-factor asymmetry, stance
    # swap rate, touchdown inward bias, step width, roll RMS, base-height std).
    # Measurement only — these NEVER feed back into training rewards.
    from isaaclab.managers import SceneEntityCfg

    feet_sensor_cfg = SceneEntityCfg("contact_forces", body_names="ankle_pitch_.*")
    feet_sensor_cfg.resolve(base_env.scene)
    contact_sensor = base_env.scene.sensors[feet_sensor_cfg.name]
    feet_contact_ids = list(feet_sensor_cfg.body_ids)

    feet_body_cfg = SceneEntityCfg("robot", body_names="ankle_pitch_.*")
    feet_body_cfg.resolve(base_env.scene)
    foot_body_ids = list(feet_body_cfg.body_ids)

    # Align both id lists by body name: index 0 = left foot, 1 = right foot in
    # both the contact sensor and the body pose buffers.
    body_names = robot.data.body_names
    feet_contact_ids = [feet_contact_ids[i] for i in sorted(range(len(feet_contact_ids)), key=lambda i: body_names[feet_contact_ids[i]])]
    foot_body_ids = [foot_body_ids[i] for i in sorted(range(len(foot_body_ids)), key=lambda i: body_names[foot_body_ids[i]])]

    dt = env.unwrapped.step_dt
    # Inward convention (body frame): left foot -y, right foot +y => positive = crossing inward.
    inward_sign = torch.tensor([-1.0, 1.0], device=device).unsqueeze(0)

    prev_contact = torch.zeros(args_cli.num_envs, len(feet_contact_ids), dtype=torch.bool, device=device)
    prev_stance_foot = torch.full((args_cli.num_envs,), -1, dtype=torch.long, device=device)
    stance_count = torch.zeros(args_cli.num_envs, len(feet_contact_ids), device=device)
    swap_count = torch.zeros(args_cli.num_envs, device=device)
    td_in_sum = torch.zeros(args_cli.num_envs, device=device)
    td_count = torch.zeros(args_cli.num_envs, device=device)
    ds_gap_sum = torch.zeros(args_cli.num_envs, device=device)
    ds_count = torch.zeros(args_cli.num_envs, device=device)
    roll_sq_sum = torch.zeros(args_cli.num_envs, device=device)
    h_sum = torch.zeros(args_cli.num_envs, device=device)
    h_sq_sum = torch.zeros(args_cli.num_envs, device=device)

    # Debounced (history-min) contact: true continuous stance, immune to
    # single-frame force dips. Used to measure true swap rate and the longest
    # continuous pillar stance per episode — decides whether a stance-duration
    # penalty could ever fire.
    stable_t = torch.zeros(args_cli.num_envs, len(feet_contact_ids), device=device)
    pillar_max = torch.zeros(args_cli.num_envs, device=device)
    prev_deb_stance = torch.full((args_cli.num_envs,), -1, dtype=torch.long, device=device)
    deb_swap_count = torch.zeros(args_cli.num_envs, device=device)

    # Scenario identity for cross-checkpoint pairing: the wrapper resets an env
    # inside step() when it is done, so the pre-step state of an env that was
    # done last iteration IS its spawn state for the new episode.
    ep_count = {}
    spawn_sig = {}
    prev_done = torch.zeros(args_cli.num_envs, dtype=torch.bool, device=device)

    def _capture_spawns(env_ids):
        root_pos = robot.data.root_pos_w
        _, _, yaw = math_utils.euler_xyz_from_quat(robot.data.root_quat_w)
        for env_i in env_ids:
            spawn_sig[env_i] = (
                round(float(root_pos[env_i, 0] - env_origins[env_i, 0].item()), 2),
                round(float(root_pos[env_i, 1] - env_origins[env_i, 1].item()), 2),
                round(float(yaw[env_i].item()), 3),
            )

    with torch.inference_mode():
        while len(records) < total_target and simulation_app.is_running():
            if not ep_count:  # first episode of every env starts at reset()
                _capture_spawns(range(args_cli.num_envs))
                for env_i in range(args_cli.num_envs):
                    ep_count[env_i] = 1
            elif prev_done.any():  # envs done last step were reset inside step()
                restarted = torch.nonzero(prev_done).flatten().tolist()
                _capture_spawns(restarted)
                for env_i in restarted:
                    ep_count[env_i] = ep_count.get(env_i, 0) + 1

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

            # ---- posture telemetry (per step, per env) ----
            forces = contact_sensor.data.net_forces_w[:, feet_contact_ids, :].norm(dim=-1)
            contact = forces > 1.0  # (N, 2) same threshold as single-stance terms
            stance_count += contact.float()

            foot_pos_w = robot.data.body_link_pos_w[:, foot_body_ids, :]
            base_q = robot.data.root_link_quat_w.unsqueeze(1).expand(-1, foot_pos_w.shape[1], -1)
            base_p = robot.data.root_link_state_w[:, :3].unsqueeze(1).expand(-1, foot_pos_w.shape[1], -1)
            foot_y_b = math_utils.quat_apply_inverse(base_q, foot_pos_w - base_p)[..., 1]  # (N, 2)

            touch = (~prev_contact) & contact
            if touch.any():
                td_in_sum += (foot_y_b * inward_sign * touch.float()).sum(dim=1)
                td_count += touch.float().sum(dim=1)

            ds_mask = contact.all(dim=1)
            if ds_mask.any():
                ds_gap_sum[ds_mask] += (foot_y_b[ds_mask, 0] - foot_y_b[ds_mask, 1]).abs()
                ds_count[ds_mask] += 1.0

            single_now = contact.sum(dim=1) == 1
            stance_foot = torch.where(single_now, contact.long().argmax(dim=1), prev_stance_foot)
            swap_count += ((stance_foot != prev_stance_foot) & (stance_foot >= 0) & (prev_stance_foot >= 0)).float()
            prev_stance_foot = stance_foot
            prev_contact = contact

            forces_hist = contact_sensor.data.net_forces_w_history[:, :, feet_contact_ids, :].norm(dim=-1)
            deb_contact = forces_hist.min(dim=1)[0] > 1.0  # maintained through the whole window
            stable_t = torch.where(deb_contact, stable_t + dt, torch.zeros_like(stable_t))
            pillar_max = torch.maximum(pillar_max, stable_t.max(dim=1)[0])
            deb_single = deb_contact.sum(dim=1) == 1
            deb_stance = torch.where(deb_single, deb_contact.long().argmax(dim=1), prev_deb_stance)
            deb_swap_count += ((deb_stance != prev_deb_stance) & (deb_stance >= 0) & (prev_deb_stance >= 0)).float()
            prev_deb_stance = deb_stance

            roll_cur = math_utils.euler_xyz_from_quat(robot.data.root_quat_w)[0]
            roll_sq_sum += roll_cur.square()
            h_cur = robot.data.root_pos_w[:, 2]
            h_sum += h_cur
            h_sq_sum += h_cur.square()

            done_mask = torch.as_tensor(dones, device=device).bool()
            if episode_step.max().item() >= args_cli.max_episode_steps:
                done_mask |= torch.ones_like(done_mask)  # safety cap counts as stuck

            if done_mask.any():
                # cap the batch append so n never overshoots the target
                remaining = total_target - len(records)
                done_list = torch.nonzero(done_mask).flatten().tolist()[:remaining]
                for env_i in done_list:
                    prog = progress_max[env_i].item()
                    sig = spawn_sig.get(env_i, (None, None, None))
                    steps_f = max(int(episode_step[env_i].item()), 1)
                    duty = (stance_count[env_i] / steps_f).tolist()
                    td_n = max(td_count[env_i].item(), 1.0)
                    ds_n = max(ds_count[env_i].item(), 1.0)
                    rec = {
                        "env": env_i,
                        "ep_idx": ep_count.get(env_i, 1) - 1,
                        "spawn": sig,
                        "steps": int(episode_step[env_i].item()),
                        "progress_max": round(prog, 3),
                        # IsaacLab >= 2.3 renamed the termination buffer to time_outs
                        "terminated": not bool(base_env.termination_manager.time_outs[env_i].item()),
                        # posture metrics
                        "duty_L": round(duty[0], 3),
                        "duty_R": round(duty[1], 3),
                        "duty_asym": round(abs(duty[0] - duty[1]), 3),
                        "swap_rate": round(swap_count[env_i].item() / (steps_f * dt), 2),
                        "td_inward": round(td_in_sum[env_i].item() / td_n, 3),
                        "step_width": round(ds_gap_sum[env_i].item() / ds_n, 3),
                        "roll_rms": round(math.sqrt(roll_sq_sum[env_i].item() / steps_f), 3),
                        "base_h_std": round(math.sqrt(max(h_sq_sum[env_i].item() / steps_f - (h_sum[env_i].item() / steps_f) ** 2, 0.0)), 4),
                        # debounced posture metrics
                        "deb_swap_rate": round(deb_swap_count[env_i].item() / (steps_f * dt), 2),
                        "pillar_max_s": round(pillar_max[env_i].item(), 2),
                    }
                    for key, bound in seg_keys:
                        rec[key] = int(prog >= bound)
                    records.append(rec)
                progress_max[done_mask] = 0.0
                episode_step[done_mask] = 0
                stance_count[done_mask] = 0.0
                swap_count[done_mask] = 0.0
                td_in_sum[done_mask] = 0.0
                td_count[done_mask] = 0.0
                ds_gap_sum[done_mask] = 0.0
                ds_count[done_mask] = 0.0
                roll_sq_sum[done_mask] = 0.0
                h_sum[done_mask] = 0.0
                h_sq_sum[done_mask] = 0.0
                prev_contact[done_mask] = False
                prev_stance_foot[done_mask] = -1
                stable_t[done_mask] = 0.0
                pillar_max[done_mask] = 0.0
                prev_deb_stance[done_mask] = -1
                deb_swap_count[done_mask] = 0.0
            prev_done = done_mask.clone()

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
        # posture aggregates
        "duty_asym_mean": round(sum(r["duty_asym"] for r in records) / n, 3),
        "swap_rate_mean": round(sum(r["swap_rate"] for r in records) / n, 2),
        "td_inward_mean": round(sum(r["td_inward"] for r in records) / n, 3),
        "step_width_mean": round(sum(r["step_width"] for r in records) / n, 3),
        "roll_rms_mean": round(sum(r["roll_rms"] for r in records) / n, 3),
        "base_h_std_mean": round(sum(r["base_h_std"] for r in records) / n, 4),
        "deb_swap_rate_mean": round(sum(r["deb_swap_rate"] for r in records) / n, 2),
        "pillar_max_s_mean": round(sum(r["pillar_max_s"] for r in records) / n, 2),
        "pillar_over_1s_pct": round(100.0 * sum(1 for r in records if r["pillar_max_s"] >= 1.0) / n, 1),
    }

    os.makedirs(args_cli.output_dir, exist_ok=True)
    tag = os.path.splitext(os.path.basename(args_cli.checkpoint_path))[0]

    with open(os.path.join(args_cli.output_dir, f"eval_{tag}_episodes.csv"), "w", newline="", encoding="utf-8") as f:
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
    # Per-episode series for cross-checkpoint pairing. Each record carries its
    # scenario signature (env id, episode index within env, spawn pose); pair
    # records across checkpoints by matching signatures, NOT by output order.
    print("episoderec: " + ";".join(
        f"{r['env']},{r['ep_idx']},{r['spawn'][0]},{r['spawn'][1]},{r['spawn'][2]},{r['progress_max']:.2f},{r['steps']},{int(r['terminated'])},"
        f"{r['duty_L']},{r['duty_R']},{r['duty_asym']},{r['swap_rate']},{r['td_inward']},{r['step_width']},{r['roll_rms']},{r['base_h_std']},"
        f"{r['deb_swap_rate']},{r['pillar_max_s']}"
        for r in records
    ), file=sys.stderr, flush=True)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
