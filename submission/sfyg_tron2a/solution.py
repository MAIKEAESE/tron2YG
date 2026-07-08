from __future__ import annotations

import math
import os
from collections import deque
from typing import Any

import torch
import torch.nn as nn


POLICY_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "policy.pt")

CONTROL_DT = 0.02
HISTORY_LEN = 10
POLICY_OBS_DIM = 42
ENCODER_INPUT_DIM = HISTORY_LEN * POLICY_OBS_DIM
ENCODER_OUTPUT_DIM = 3
ACTOR_INPUT_DIM = ENCODER_OUTPUT_DIM + POLICY_OBS_DIM + 3
LEG_ACTION_DIM = 10
FULL_ACTION_DIM = 18

GAIT_FREQUENCY = 0.9
GAIT_COMMAND = (GAIT_FREQUENCY, 0.5, 0.5, 0.15)


class ActorNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(ACTOR_INPUT_DIM, 512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, LEG_ACTION_DIM),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.actor(x)


class EncoderNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(ENCODER_INPUT_DIM, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, ENCODER_OUTPUT_DIM),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


def _load_prefixed_state(module: nn.Module, state_dict: dict[str, torch.Tensor], prefix: str) -> None:
    target = module.state_dict()
    filtered = {}
    for key, value in state_dict.items():
        if not key.startswith(prefix):
            continue
        if key in target and target[key].shape == value.shape:
            filtered[key] = value
    missing = set(target) - set(filtered)
    if missing:
        raise RuntimeError(f"missing keys for {prefix.rstrip('.')}: {sorted(missing)}")
    module.load_state_dict(filtered, strict=True)


class AlgSolution:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor = ActorNet().to(self.device).eval()
        self.encoder = EncoderNet().to(self.device).eval()
        self.obs_history: deque[torch.Tensor] = deque(maxlen=HISTORY_LEN)
        self.gait_index = 0.0

        checkpoint = torch.load(POLICY_PATH, map_location=self.device)
        if "model_state_dict" not in checkpoint or "encoder_state_dict" not in checkpoint:
            raise RuntimeError("policy.pt must be an RSL-RL checkpoint with model_state_dict and encoder_state_dict")

        _load_prefixed_state(self.actor, checkpoint["model_state_dict"], "actor.")
        _load_prefixed_state(self.encoder, checkpoint["encoder_state_dict"], "encoder.")

    def reset(self, **kwargs):
        self.obs_history.clear()
        self.gait_index = 0.0

    def get_action_spec(self) -> dict[str, dict[str, Any]]:
        return {
            "leg": {
                "mode": "position",
                "scale": 0.25,
                "stiffness": [159.67, 159.67, 53.22, 159.67, 53.22, 159.67, 159.67, 53.22, 159.67, 53.22],
                "damping": [10.16, 10.16, 3.39, 10.16, 3.39, 10.16, 10.16, 3.39, 10.16, 3.39],
            },
            "arm": {
                "mode": "position",
                "scale": 0.5,
                "stiffness": [80.0] * 8,
                "damping": [4.0] * 8,
            },
        }

    def _policy_obs_from_proprio(self, proprio: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        base_ang_vel = proprio[:, 3:6] * 0.25
        projected_gravity = proprio[:, 9:12]

        joint_pos = proprio[:, 12:22]
        joint_vel = proprio[:, 30:40] * 0.05
        last_action = proprio[:, 48:58]

        sin_phase = math.sin(2.0 * math.pi * self.gait_index)
        cos_phase = math.cos(2.0 * math.pi * self.gait_index)
        gait_phase = torch.tensor([[sin_phase, cos_phase]], dtype=torch.float32, device=self.device)
        gait_command = torch.tensor([GAIT_COMMAND], dtype=torch.float32, device=self.device)

        policy_obs = torch.cat(
            [base_ang_vel, projected_gravity, joint_pos, joint_vel, last_action, gait_phase, gait_command],
            dim=-1,
        )
        commands = proprio[:, 6:9]
        return policy_obs, commands

    def predicts(self, obs, current_score) -> dict:
        proprio = torch.as_tensor(obs["proprio"], dtype=torch.float32, device=self.device)
        if proprio.ndim == 1:
            proprio = proprio.unsqueeze(0)

        policy_obs, commands = self._policy_obs_from_proprio(proprio)

        if not self.obs_history:
            for _ in range(HISTORY_LEN):
                self.obs_history.append(policy_obs.squeeze(0).detach())
        else:
            self.obs_history.append(policy_obs.squeeze(0).detach())

        history = torch.stack(list(self.obs_history), dim=0).reshape(1, -1)

        with torch.inference_mode():
            est = self.encoder(history)
            leg_action = self.actor(torch.cat([est, policy_obs, commands], dim=-1))

        full_action = torch.zeros((1, FULL_ACTION_DIM), dtype=torch.float32, device=self.device)
        full_action[:, :LEG_ACTION_DIM] = leg_action

        self.gait_index = (self.gait_index + CONTROL_DT * GAIT_FREQUENCY) % 1.0

        return {"action": full_action.squeeze(0).detach().cpu().tolist(), "giveup": False}
