#!/usr/bin/env python
"""官方基线评估（官方代码在 IsaacSim 5.1 / IsaacLab 2.3.1 下的复现）
纪律：预步采样防终态污染；逐帧记录实际命令(heading 每步重算)；MAE 分起步/稳定段；
固定配额每 env 取首个完成回合；跌倒计入失败。"""
import argparse, os, sys, json, math, time
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../rsl_rl")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../exts/bipedal_locomotion")))

import cli_args as rsl_cli_args

parser = argparse.ArgumentParser()
rsl_cli_args.add_rsl_rl_args(parser)
parser.add_argument("--task", type=str, default="Isaac-Limx-SFYG-TRON2A-Blind-Flat-Play-v0")
parser.add_argument("--checkpoint_path", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=20)
parser.add_argument("--episode_s", type=float, default=20.0)
parser.add_argument("--startup_s", type=float, default=2.0)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--quick", action="store_true")
parser.add_argument("--out", type=str, default="official_eval")
parser.add_argument("--scenario", type=str, default=None, help="single scenario: stand/low/mid/turn")
args_cli = parser.parse_args()

from isaaclab.app import AppLauncher
launcher = AppLauncher(headless=True)
simulation_app = launcher.app

import gymnasium as gym
from isaaclab_tasks.utils import parse_env_cfg
from bipedal_locomotion.utils.wrappers.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runner import OnPolicyRunner

SCENARIOS = {
    "stand": dict(vx=0.0, heading=0.0),
    "low":   dict(vx=0.5, heading=0.0),
    "mid":   dict(vx=0.8, heading=0.0),
    "turn":  dict(vx=0.0, heading=0.0, yaw=0.5),
}
THRESH = {"low": 0.10, "mid": 0.15}

if args_cli.scenario:
    SCENARIOS = {k: v for k, v in SCENARIOS.items() if k == args_cli.scenario}
    assert SCENARIOS, f"unknown scenario {args_cli.scenario}"

if args_cli.quick:
    args_cli.num_envs = 2
    args_cli.episode_s = 5.0
    SCENARIOS = {"mid": SCENARIOS["mid"]}


def run_scenario(name, sc, env_cfg, resume_path, agent_train_cfg):
    term = env_cfg.commands.base_velocity
    term.ranges.lin_vel_x = (sc["vx"], sc["vx"])
    term.ranges.lin_vel_y = (0.0, 0.0)
    term.ranges.ang_vel_z = (0.0, 0.0)
    term.ranges.heading = (sc["heading"], sc["heading"])
    if "yaw" in sc:  # 转向直令：关 heading 控制，直接给偏航速率
        term.heading_command = False
        term.ranges.ang_vel_z = (sc["yaw"], sc["yaw"])
    else:
        term.ranges.ang_vel_z = (0.0, 0.0)
    if hasattr(term, "resampling_time_range"):
        term.resampling_time_range = (100.0, 100.0)
    env_cfg.seed = args_cli.seed

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)
    runner = OnPolicyRunner(env, agent_train_cfg, log_dir=None, device=env.unwrapped.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    encoder = runner.get_inference_encoder(device=env.unwrapped.device)

    unw = env.unwrapped
    robot = unw.scene["robot"]
    cmd_term = unw.command_manager.get_term("base_velocity")
    contact = unw.scene.sensors["contact_forces"]
    feet_ids = contact.find_bodies("ankle_pitch_.*")[0]
    feet_body = robot.find_bodies("ankle_pitch_.*")[0]
    knee_ids = robot.find_joints("knee_.*_Joint")[0]
    dt = unw.step_dt
    max_steps = int(round(args_cli.episode_s / dt))
    N = unw.num_envs

    def rec():
        return dict(steps=0, cmd_vx=[], cmd_yaw=[], vel_x=[], start_x=None,
                    air=[False] * N, swing_start=[dict(), dict()], swings=[0, 0],
                    swing_disp=[[], []], knee_min=[9e9, 9e9], knee_max=[-9e9, -9e9],
                    slip=0.0, slip_n=0, roll_sq=0.0, pitch_sq=0.0, base_z=[],
                    fell=False, timeout=False, last_x=0.0, knee0=(0.0, 0.0))
    eps = [rec() for _ in range(N)]
    done_once = [False] * N

    obs_dict = env.get_observations()
    obs = obs_dict["policy"]
    obs_history = obs_dict.get("obsHistory").flatten(start_dim=1)
    commands = obs_dict.get("commands")
    root_now = None
    for t in range(max_steps + 40):
        # ---- 预步采样：当前状态属于正在进行的回合 ----
        cmd_now = cmd_term.vel_command_b.detach().clone()
        vel_now = robot.data.root_lin_vel_b.detach().clone()
        root_now = robot.data.root_pos_w.detach().clone()
        grav = robot.data.projected_gravity_b.detach().clone()
        fz = contact.data.net_forces_w_history[:, -1, :, :].norm(dim=-1)[:, feet_ids]  # (N,2)
        fv = robot.data.body_lin_vel_w.detach().clone()[:, feet_body, :]              # (N,2,3)
        knee = robot.data.joint_pos.detach().clone()[:, knee_ids]

        with torch.inference_mode():
            est = encoder(obs_history)
            actions = policy(torch.cat((est, obs, commands), dim=-1).detach())
        obs_dict, _, dones, _ = env.step(actions)
        obs = obs_dict["policy"]
        obs_history = obs_dict.get("obsHistory").flatten(start_dim=1)
        commands = obs_dict.get("commands")
        time_outs = unw.termination_manager.time_outs

        for i in range(N):
            if done_once[i]:
                continue
            e = eps[i]
            e["steps"] += 1
            e["last_x"] = root_now[i, 0].item()
            e["cmd_vx"].append(cmd_now[i, 0].item())
            e["cmd_yaw"].append(cmd_now[i, 2].item())
            e["vel_x"].append(vel_now[i, 0].item())
            if e["start_x"] is None:
                e["start_x"] = root_now[i, 0].item()
                e["knee0"] = (knee[i, 0].item(), knee[i, 1].item())
            for f in range(2):
                airborne = fz[i, f].item() < 5.0
                if airborne and not e["air"][i]:
                    pass
                if airborne and (i not in e["swing_start"][f]):
                    e["swing_start"][f][i] = (t, root_now[i, 0].item())
                elif not airborne and (i in e["swing_start"][f]):
                    st = e["swing_start"][f].pop(i)
                    if t - st[0] >= 3:
                        e["swings"][f] += 1
                        e["swing_disp"][f].append(root_now[i, 0].item() - st[1])
                if not airborne:
                    e["slip"] += float(fv[i, f, :2].norm().item())
                    e["slip_n"] += 1
                e["knee_min"][f] = min(e["knee_min"][f], knee[i, f].item())
                e["knee_max"][f] = max(e["knee_max"][f], knee[i, f].item())
            e["roll_sq"] += float(grav[i, 1].item()) ** 2
            e["pitch_sq"] += float(grav[i, 0].item()) ** 2
            e["base_z"].append(root_now[i, 2].item())
            if dones[i]:
                done_once[i] = True
                e["fell"] = not bool(time_outs[i].item())
                e["timeout"] = bool(time_outs[i].item())
        if all(done_once):
            break

    rows = []
    s0 = int(args_cli.startup_s / dt)
    for i in range(N):
        e = eps[i]
        n = max(e["steps"], 1)
        cv, vv = e["cmd_vx"], e["vel_x"]
        def mae(lo, hi):
            seg = [(c, v) for c, v in zip(cv[lo:hi], vv[lo:hi])]
            return float(np.mean([abs(c - v) for c, v in seg])) if seg else None
        rows.append(dict(
            env=i, scenario=name, steps=e["steps"], fell=e["fell"], timeout=e["timeout"],
            disp_x=e["last_x"] - e["start_x"],
            vx_mae_startup=mae(0, min(s0, len(cv))), vx_mae_steady=mae(min(s0, len(cv)), len(cv)),
            cmd_vx_mean=float(np.mean(e["cmd_vx"])) if e["cmd_vx"] else 0.0,
            cmd_vx_max=float(np.max(e["cmd_vx"])) if e["cmd_vx"] else 0.0,
            init_knee_L=e["knee0"][0], init_knee_R=e["knee0"][1],
            yaw_cmd_mean_abs=float(np.mean(np.abs(e["cmd_yaw"]))) if e["cmd_yaw"] else 0.0,
            yaw_cmd_max_abs=float(np.max(np.abs(e["cmd_yaw"]))) if e["cmd_yaw"] else 0.0,
            swings_L=e["swings"][0], swings_R=e["swings"][1],
            swing_disp_L=float(np.mean(e["swing_disp"][0])) if e["swing_disp"][0] else 0.0,
            swing_disp_R=float(np.mean(e["swing_disp"][1])) if e["swing_disp"][1] else 0.0,
            knee_rng_L=e["knee_max"][0] - e["knee_min"][0],
            knee_rng_R=e["knee_max"][1] - e["knee_min"][1],
            slip_mean=e["slip"] / max(e["slip_n"], 1),
            roll_rms=math.sqrt(e["roll_sq"] / n), pitch_rms=math.sqrt(e["pitch_sq"] / n),
            base_z_mean=float(np.mean(e["base_z"])) if e["base_z"] else 0.0))
    env.close()
    return rows


def main():
    env_cfg = parse_env_cfg(args_cli.task, device="cuda:0", num_envs=args_cli.num_envs)
    agent_cfg = rsl_cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    agent_train_cfg = agent_cfg.to_dict()
    all_rows = []
    import copy
    for name, sc in SCENARIOS.items():
        t0 = time.time()
        env_cfg_run = copy.deepcopy(env_cfg)
        rows = run_scenario(name, sc, env_cfg_run, args_cli.checkpoint_path, agent_train_cfg)
        all_rows.extend(rows)
        alive = sum(1 for r in rows if not r["fell"])
        sw = sum(r["swings_L"] + r["swings_R"] for r in rows)
        print(f"[{name}] {alive}/{len(rows)} 存活 摆动L/R={sum(r['swings_L'] for r in rows)}/{sum(r['swings_R'] for r in rows)} 用时{time.time()-t0:.0f}s", flush=True)
    os.makedirs(args_cli.out, exist_ok=True)
    import csv
    mode = "a" if args_cli.scenario else "w"
    import os as _os
    ep_path = _os.path.join(args_cli.out, "episodes.csv")
    write_header = not _os.path.exists(ep_path) or mode == "w" or _os.path.getsize(ep_path) == 0
    with open(ep_path, mode, newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        if write_header:
            w.writeheader()
        w.writerows(all_rows)
    summary = {}
    for name in SCENARIOS:
        rs = [r for r in all_rows if r["scenario"] == name]
        if not rs:
            continue
        summary[name] = dict(
            survive_pct=100 * sum(1 for r in rs if not r["fell"]) / len(rs),
            vx_mae_steady_mean=float(np.mean([r["vx_mae_steady"] for r in rs if r["vx_mae_steady"] is not None])),
            vx_mae_startup_mean=float(np.mean([r["vx_mae_startup"] for r in rs if r["vx_mae_startup"] is not None])),
            swings_L=sum(r["swings_L"] for r in rs), swings_R=sum(r["swings_R"] for r in rs),
            knee_rng_L=float(np.mean([r["knee_rng_L"] for r in rs])),
            knee_rng_R=float(np.mean([r["knee_rng_R"] for r in rs])),
            disp_x_mean=float(np.mean([r["disp_x"] for r in rs])))
        if name in THRESH:
            summary[name]["pass_prelim"] = bool(
                summary[name]["survive_pct"] >= 90 and
                summary[name]["vx_mae_steady_mean"] <= THRESH[name] and
                summary[name]["swings_L"] > 0 and summary[name]["swings_R"] > 0)
    suffix = f"_{args_cli.scenario}" if args_cli.scenario else ""
    with open(os.path.join(args_cli.out, f"summary{suffix}.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=1, ensure_ascii=False), flush=True)
    print("EVAL_DONE", flush=True)


if __name__ == "__main__":
    main()
simulation_app.close()
