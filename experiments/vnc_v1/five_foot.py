"""The official five-legged gait as a motor program inside the posture task.

`tasks/jumper/five_foot` (model_86600) walks and turns on five feet with the left-front arm held
stowed, at 0.81 of a 0.3 m/s command in its own task. Run inside the posture task it reads the
posture observation and writes a posture action:

- its 407 inputs are built from the posture observation's (415): the same IMU and gravity, and,
  frame by frame (five frames 20 ms apart in both tasks: the posture task's four control steps at
  200 Hz are its one at 50 Hz), the 16 other joints' positions, velocities and torques, with the
  left-front arm's five joints read as held at its stow pose (zero), as they always were in its
  training -- a dead leg here hangs instead, which it never felt; its own action history; the
  twist command clipped to its training range (+-0.5 m/s, +-2 rad/s; `vmax` moves the linear
  bound); a neutral body pose;
- it runs at 50 Hz, so each robot's action is recomputed every fourth control step and held;
- its 16 outputs are the posture action's last 16, in the same order, scale (0.25) and default
  pose; the left-front leg's four are left at zero.

With `mirror=True` it walks a robot whose dead leg is the right-front one: the observation is
mirrored left to right before it reads it and its action mirrored back, with the posture task's own
mirror (`tasks/jumper/common/mdp/symmetry.py`: joints permuted and signed, the angular velocity's x
and z and gravity's y negated, the command's vy and wz negated), so the leg it walks without is the
right one.
"""

from __future__ import annotations

import torch

CHECKPOINT = "tasks/jumper/five_foot/out/example/model_86600.pt"
FRAMES, JOINTS = 5, 20  # the posture observation's history frames and joints per frame
# Posture observation offsets (layout.json): joint_pos 6, joint_vel 106, actions 206, command 306,
# actuator_force 309.
JP, JV, ACT, CMD, FORCE = 6, 106, 206, 306, 309
LF = 4  # the left-front leg's joints come first in the posture task's joint order


class FiveFootProgram:
    """Per-robot state (its action history, the held action, its 50 Hz clock) is allocated on
    the first call, for that call's batch, on that call's device."""

    def __init__(self, num_envs: int = 0, device="cpu", checkpoint: str = CHECKPOINT, mirror: bool = False,
                 vmax: float = 0.5) -> None:
        from dataclasses import asdict

        from tensordict import TensorDict

        import tasks
        from rsl_rl.models.mlp_model import MLPModel

        kw = asdict(tasks.load_agent_cfg("jumper.five_foot"))["actor"]
        kw.pop("class_name", None)
        for opt in ("cnn_cfg", "distribution_cfg"):
            if kw.get(opt) is None:
                kw.pop(opt, None)
        if kw.get("rnn_type") is None:
            for opt in ("rnn_type", "rnn_hidden_dim", "rnn_num_layers"):
                kw.pop(opt, None)
        probe = TensorDict({"actor": torch.zeros(1, 407)}, batch_size=[1])
        self.model = MLPModel(probe, {"actor": ["actor"]}, "actor", 16, **kw).to(device)
        self.model.load_state_dict(torch.load(checkpoint, weights_only=False, map_location=device)["actor_state_dict"])
        self.model.eval()
        self.mirror = mirror
        self.vmax = vmax
        if mirror:
            from tasks.jumper.common.mdp.symmetry import JOINT_PERM, JOINT_SIGN

            self.perm = torch.tensor(JOINT_PERM, device=device)
            self.sign = torch.tensor(JOINT_SIGN, device=device)
        self.ring = self.hold = self.tick = None
        if num_envs:
            self._alloc(num_envs, torch.device(device))

    def _alloc(self, n: int, device) -> None:
        self.ring = torch.zeros(FRAMES, n, 16, device=device)  # its own actions, oldest first
        self.hold = torch.zeros(n, 16, device=device)
        self.tick = torch.zeros(n, dtype=torch.long, device=device)

    def to(self, device) -> "FiveFootProgram":
        self.model.to(device)
        if self.mirror:
            self.perm, self.sign = self.perm.to(device), self.sign.to(device)
        if self.ring is not None:
            self.ring, self.hold, self.tick = self.ring.to(device), self.hold.to(device), self.tick.to(device)
        return self

    def reset(self, mask: torch.Tensor) -> None:
        """Start the program afresh for the robots in `mask` (bool, num_envs)."""
        if self.ring is None or self.ring.shape[1] != mask.shape[0]:
            return
        self.ring = torch.where(mask[None, :, None], torch.zeros_like(self.ring), self.ring)
        self.hold = torch.where(mask[:, None], torch.zeros_like(self.hold), self.hold)
        self.tick = torch.where(mask, torch.zeros_like(self.tick), self.tick)

    def inputs(self, x: torch.Tensor) -> torch.Tensor:
        n = x.shape[0]
        zeros5 = x.new_zeros(n, FRAMES, 5)

        def joints(offset: int) -> torch.Tensor:
            block = x[:, offset:offset + FRAMES * JOINTS].view(n, FRAMES, JOINTS)
            return torch.cat([zeros5, block[:, :, LF:]], dim=-1).reshape(n, -1)  # 5 x 21

        cmd = x[:, CMD:CMD + 3].clone()
        cmd[:, :2] = cmd[:, :2].clamp(-self.vmax, self.vmax)
        cmd[:, 2] = cmd[:, 2].clamp(-2.0, 2.0)
        return torch.cat([x[:, :6], joints(JP), joints(JV), self.ring.permute(1, 0, 2).reshape(n, -1), cmd,
                          joints(FORCE), x.new_zeros(n, 3)], dim=-1)

    def mirrored(self, x: torch.Tensor) -> torch.Tensor:
        """The posture observation's terms this program reads, mirrored left to right."""
        n = x.shape[0]
        y = x.clone()
        y[:, 0], y[:, 2] = -x[:, 0], -x[:, 2]  # angular velocity: a pseudovector
        y[:, 4] = -x[:, 4]  # gravity: a vector
        for off in (JP, JV, ACT, FORCE):
            block = x[:, off:off + FRAMES * JOINTS].view(n, FRAMES, JOINTS)
            y[:, off:off + FRAMES * JOINTS] = (block[:, :, self.perm] * self.sign).reshape(n, -1)
        y[:, CMD + 1], y[:, CMD + 2] = -x[:, CMD + 1], -x[:, CMD + 2]
        return y

    @torch.no_grad()
    def __call__(self, x: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
        """The posture action (num_envs x 20) for the robots in `active`; advances their clocks."""
        if self.ring is None or self.ring.shape[1] != x.shape[0] or self.ring.device != x.device:
            self._alloc(x.shape[0], x.device)
        if self.mirror:
            x = self.mirrored(x)
        due = active & (self.tick % 4 == 0)
        if bool(due.any()):
            a = self.model.mlp(self.model.obs_normalizer(self.inputs(x)))
            self.hold = torch.where(due[:, None], a, self.hold)
            pushed = torch.cat([self.ring[1:], a[None]], dim=0)
            self.ring = torch.where(due[None, :, None], pushed, self.ring)
        self.tick = torch.where(active, self.tick + 1, self.tick)
        out = x.new_zeros(x.shape[0], JOINTS)
        out[:, LF:] = self.hold
        if self.mirror:
            out = out[:, self.perm] * self.sign
        return out
