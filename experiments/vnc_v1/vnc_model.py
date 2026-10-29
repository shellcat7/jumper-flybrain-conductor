"""The VNC brain as an rsl_rl actor: the official posture MLP, frozen, plus a residual
from the fly's thoracic nerve cord wired leg to leg.

Each Jumper leg's own observations drive that leg's sensory neurons (the fly leg
nerve on the same side); the shared ones (IMU, commands, gait phase) drive the
descending neurons and the haltere afferents; each leg's correction is read from
its own motor-neuron pool. The cord is a rate model on the signed MaleCNS subgraph
from `build_packet.py`, one step per control step (200 Hz):

    s <- (1 - a) * s + a * tanh(g * (W s) + b + u)

with W fixed (signed, incoming-normalised) and a, g, b learned per neuron. The
readout starts at zero, so the policy starts as the official one; the residual is
`residual_scale * tanh(.)` per action.

Recurrent state is the whole cord (nodes x width per robot). rsl_rl would store it
at every rollout step; `FirstStepHiddenStorage` keeps only the rollout's first step,
which is all its trajectory batches use (a trajectory starts either there or right
after a reset, where the state is zero).
"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.models.mlp_model import MLPModel
from rsl_rl.modules import HiddenState
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import unpad_trajectories

LEGS = ("LF", "RF", "LM", "RM", "LR", "RR")


class FixedMatmul(torch.autograd.Function):
    """W @ x with W constant; the gradient goes back through W^T."""

    @staticmethod
    def forward(ctx, x, w, wt):
        ctx.wt = wt
        return w @ x

    @staticmethod
    def backward(ctx, grad):
        return ctx.wt @ grad.contiguous(), None, None


class LearnedMatmul(torch.autograd.Function):
    """W @ x with W's nonzero values learnable on the fly's sparsity pattern.

    The gradient to x goes back through W^T (the same values, permuted into the
    transpose's CSR order); the gradient to the values is (grad @ x^T) sampled at the
    pattern, which `torch.sparse.sampled_addmm` computes without forming the dense
    product.
    """

    @staticmethod
    def forward(ctx, x, values, crow, col, crow_t, col_t, perm):
        n = crow.numel() - 1
        ctx.save_for_backward(x, values, crow, col, crow_t, col_t, perm)
        return torch.sparse_csr_tensor(crow, col, values, (n, n)) @ x

    @staticmethod
    def backward(ctx, grad):
        x, values, crow, col, crow_t, col_t, perm = ctx.saved_tensors
        n = crow.numel() - 1
        grad = grad.contiguous()
        gx = torch.sparse_csr_tensor(crow_t, col_t, values[perm], (n, n)) @ grad
        pattern = torch.sparse_csr_tensor(crow, col, torch.zeros_like(values), (n, n))
        gv = torch.sparse.sampled_addmm(pattern, grad, x.t().contiguous()).values()
        return gx, gv, None, None, None, None, None


def load_cord(packet: str, device, shuffle_seed: int | None = None) -> tuple[torch.Tensor, torch.Tensor, dict]:
    z = np.load(packet)
    n = len(z["indptr"]) - 1
    counts = z["counts"].astype(np.float32)
    rows = np.repeat(np.arange(n), np.diff(z["indptr"]))
    cols = z["indices"].astype(np.int64) & 0xFFFF  # int16 storage, nodes < 32768
    if shuffle_seed is not None:
        # The shuffled-wiring control: every synapse keeps its target and contact count,
        # its source is drawn by permuting all sources. Same neurons, ports, signs (a
        # source keeps its transmitter), in- and out-degrees; random wiring. A pair drawn
        # twice is summed by the coalesce below.
        cols = cols[np.random.default_rng(shuffle_seed).permutation(len(cols))]
    insum = np.bincount(rows, weights=counts, minlength=n)
    values = (z["sign"][cols] * counts / np.maximum(insum[rows], 1.0)).astype(np.float32)
    idx = torch.tensor(np.stack([rows, cols]))
    w = torch.sparse_coo_tensor(idx, torch.tensor(values), (n, n)).coalesce().to_sparse_csr().to(device)
    wt = w.to_sparse_coo().t().coalesce().to_sparse_csr()
    ports = {k: torch.tensor(z[k].astype(np.int64)) for k in z.files if k not in ("indptr", "indices", "counts", "sign", "body_ids")}
    return w, wt, ports


class VNCModel(MLPModel):
    """Official posture MLP (frozen) + fly-cord residual; see the module docstring."""

    is_recurrent: bool = True

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        hidden_dims: tuple[int, ...] | list[int] = (512, 256, 128, 64),
        activation: str = "elu",
        obs_normalization: bool = True,
        distribution_cfg: dict | None = None,
        packet: str = "",
        layout: str = "",
        base_checkpoint: str = "",
        width: int = 1,
        residual_scale: float = 0.5,
        shuffle_seed: int | None = None,
        residual_l2: float = 0.0,
        brain_hidden: int = 0,
        learn_synapses: bool = False,
        command_readout: bool = False,
        mlp_anchor: float | None = None,
        command_l2: float = 0.0,
        experts: tuple[str, ...] | list[str] = (),
        adapters: tuple[str, ...] | list[str] = (),
        adapter_vmax: float = 0.5,
        obs_mirror: str | None = None,
        gate_bias: float = 3.0,
        gf_reflex: dict | None = None,
        threat_cord: bool = True,
        gf_jump: dict | None = None,
        **_unused,
    ) -> None:
        expert_dist_cfg = copy.deepcopy(distribution_cfg)  # MLPModel pops from the one it gets
        # A "threat" group (the escape task's looming signal) is for the cord only: the
        # official MLP keeps its 415 inputs and never sees it.
        groups = {k: list(v) for k, v in obs_groups.items()}
        self_threat = "threat" in groups.get(obs_set, [])
        if self_threat:
            groups[obs_set] = [g for g in groups[obs_set] if g != "threat"]
        super().__init__(obs, groups, obs_set, output_dim, hidden_dims, activation,
                         obs_normalization, distribution_cfg)
        self.threat_group = "threat" if self_threat else None
        # The official actor: normaliser statistics, action std and MLP, frozen but the std.
        state = torch.load(base_checkpoint, weights_only=False, map_location="cpu")["actor_state_dict"]
        self.load_state_dict(state, strict=False)
        # The hybrid (mlp_anchor set): the official MLP learns too, each weight pulled back
        # toward its official value by mlp_anchor * (w - w0)^2 through a gradient hook. On
        # 10/23 a free MLP fine-tune learnt the dead legs but gave up nominal walking speed;
        # the anchor is what is meant to keep the official gait where nothing calls for a
        # change. None keeps the MLP frozen.
        self.mlp_anchor = mlp_anchor
        for i, p in enumerate(self.mlp.parameters()):
            if mlp_anchor is None:
                p.requires_grad_(False)
            else:
                self.register_buffer(f"mlp_w0_{i}", p.detach().clone(), persistent=False)
                p.register_hook(lambda g, p=p, i=i: g + 2.0 * self.mlp_anchor * (p.detach() - getattr(self, f"mlp_w0_{i}")))
        lay = json.loads(Path(layout).read_text())
        assert [leg["name"] for leg in lay["legs"]] == list(LEGS) and lay["obs_dim"] == self.obs_dim
        w, wt, ports = load_cord(packet, "cpu", shuffle_seed)
        self.n = w.shape[0]
        self.width = width
        self.residual_scale = residual_scale
        self.residual_l2 = residual_l2
        self._w, self._wt = w, wt  # moved with the module in _apply
        # Learnable synapses (the tuning the fly wiring is a starting point for): every
        # synapse keeps the fly's sign -- excitatory stays excitatory -- and learns its
        # magnitude as exp(log_mag), starting from the fly's normalised weight. A synapse
        # whose source has no known sign stays at zero.
        self.learn_synapses = learn_synapses
        if learn_synapses:
            v0 = w.values()
            self.register_buffer("syn_sign", torch.sign(v0), persistent=False)
            self.register_buffer("syn_crow", w.crow_indices(), persistent=False)
            self.register_buffer("syn_col", w.col_indices(), persistent=False)
            order = torch.sparse_coo_tensor(
                torch.stack([w.to_sparse_coo().indices()[0], w.to_sparse_coo().indices()[1]]),
                torch.arange(v0.numel(), dtype=torch.float64), w.shape).t().coalesce().to_sparse_csr()
            self.register_buffer("syn_crow_t", order.crow_indices(), persistent=False)
            self.register_buffer("syn_col_t", order.col_indices(), persistent=False)
            self.register_buffer("syn_perm", order.values().long(), persistent=False)
            self.syn_log_mag = nn.Parameter(torch.log(v0.abs().clamp_min(1e-6)))
        self.leak_logit = nn.Parameter(torch.zeros(self.n, 1))  # a = sigmoid(.) = 0.5 at start
        self.gain = nn.Parameter(torch.ones(self.n, 1))
        self.bias = nn.Parameter(torch.zeros(self.n, 1))
        self.leg_obs, self.leg_act, self.sens, self.motor = [], [], [], []
        self.leg_in = nn.ModuleList()
        self.leg_out = nn.ModuleList()
        for leg, spec in zip(LEGS, lay["legs"], strict=True):
            sens, motor = ports[f"sensory_{leg}"], ports[f"motor_{leg}"]
            self.register_buffer(f"obs_{leg}", torch.tensor(spec["observation_indices"]), persistent=False)
            self.register_buffer(f"act_{leg}", torch.tensor(spec["action_indices"]), persistent=False)
            self.register_buffer(f"sens_{leg}", sens, persistent=False)
            self.register_buffer(f"motor_{leg}", motor, persistent=False)
            inp = nn.Linear(len(spec["observation_indices"]), len(sens) * width)
            nn.init.normal_(inp.weight, std=1.0 / math.sqrt(len(spec["observation_indices"])))
            nn.init.zeros_(inp.bias)
            self.leg_in.append(inp)
            out = nn.Linear(len(motor) * width, len(spec["action_indices"]))
            nn.init.zeros_(out.weight)
            nn.init.zeros_(out.bias)
            self.leg_out.append(out)
        glob = lay["global_indices"]
        self.register_buffer("obs_global", torch.tensor(glob), persistent=False)
        self.register_buffer("dn", ports["descending"], persistent=False)
        halt = torch.cat([ports["haltere_L"], ports["haltere_R"]])
        self.register_buffer("haltere", halt, persistent=False)
        self.dn_in = nn.Linear(len(glob), len(ports["descending"]) * width)
        self.haltere_in = nn.Linear(6, len(halt) * width)  # base_ang_vel and projected_gravity
        for layer in (self.dn_in, self.haltere_in):
            nn.init.normal_(layer.weight, std=1.0 / math.sqrt(layer.in_features))
            nn.init.zeros_(layer.bias)
        # Optional brain stub: the ascending neurons' state, read through a small learned
        # layer, drives the descending neurons on the next step -- a loop through the
        # fly's own ascending and descending populations that lets one leg's trouble
        # reach the others' commands. Its output starts at zero.
        self.brain_hidden = brain_hidden
        if brain_hidden:
            self.register_buffer("an", ports["ascending"], persistent=False)
            self.brain_in = nn.Linear(len(ports["ascending"]) * width, brain_hidden)
            nn.init.normal_(self.brain_in.weight, std=1.0 / math.sqrt(self.brain_in.in_features))
            nn.init.zeros_(self.brain_in.bias)
            self.brain_out = nn.Linear(brain_hidden, len(ports["descending"]) * width)
            nn.init.zeros_(self.brain_out.weight)
            nn.init.zeros_(self.brain_out.bias)
        # The looming signal reaches every descending neuron through a learned map, and the
        # giant fibres (DNp01) besides through a direct path on the expansion rate, the
        # way a fly's looming detectors drive them. `threat_cord=False` keeps the signal out
        # of the cord (for the hard-wired reflex below on a cord trained without it).
        self.threat_cord = bool(self.threat_group) and threat_cord
        if self.threat_cord:
            k = obs[self.threat_group].shape[-1]
            self.threat_in = nn.Linear(k, len(ports["descending"]) * width)
            nn.init.normal_(self.threat_in.weight, std=1.0 / math.sqrt(k))
            nn.init.zeros_(self.threat_in.bias)
            self.register_buffer("giant_fibre", ports["type_DNp01"], persistent=False)
            self.gf_gain = nn.Parameter(torch.ones(1))
        # Descending commands: the descending neurons' state, read out to offsets on the
        # official MLP's own commands (vx, vy, wz; roll, pitch, twist, height), so the cord
        # can order a sidestep, a duck or a lean and the official gait carries it out.
        # Starts at zero.
        self.command_readout = command_readout
        # The pull toward zero on the commands, as `residual_l2` is on the residual. flyesc-s1
        # (10/24) had none: at iteration 200 it sent commands with a spread of 0.10-0.14 (m/s,
        # rad/s) on flat ground with nothing wrong, walked at 0.78 of the command, and used
        # the penalised joint residual hardly at all (mean 0.003). Applied to the offsets in
        # units of their scale, so each command costs alike.
        self.command_l2 = command_l2
        if command_readout:
            names = {t["name"]: t for t in lay["terms"]}
            idx = list(range(names["command"]["offset"], names["command"]["offset"] + 3))
            idx += list(range(names["posture_command"]["offset"], names["posture_command"]["offset"] + 4))
            self.register_buffer("cmd_idx", torch.tensor(idx), persistent=False)
            self.register_buffer("cmd_scale", torch.tensor([0.5, 0.5, 2.0, 0.4, 0.4, 0.4, 0.04]), persistent=False)
            self.cmd_out = nn.Linear(len(ports["descending"]) * width, 7)
            nn.init.zeros_(self.cmd_out.weight)
            nn.init.zeros_(self.cmd_out.bias)
        # Motor programs (the director, 10/25): besides the official MLP, frozen MLP gaits
        # trained for troubles. A readout of the ascending and descending neurons' state
        # chooses among them (a softmax), and the action is the mixture: the cord selects the
        # motor program the situation calls for, as a fly's descending neurons switch its
        # behaviour, instead of writing the joint corrections itself. Program 0 is the
        # official MLP; the gate starts on it. An expert whose file is missing here is built
        # empty and gets its weights from the checkpoint loaded afterwards. An expert named
        # "<file>#mirror" is that gait mirrored left to right: it reads the mirrored observation
        # (`mirror_matrix.py`'s signed permutation, the file `obs_mirror`) and its action is
        # mirrored back, so a gait trained for a dead right-rear leg walks a dead left-rear one.
        self.experts = nn.ModuleList()
        self.experts_paths = list(experts)
        self.expert_mirror = [path.endswith("#mirror") for path in experts]
        for path, mirror in zip(experts, self.expert_mirror):
            file = path[: -len("#mirror")] if mirror else path
            m = MLPModel(obs, groups, obs_set, output_dim, hidden_dims, activation, obs_normalization,
                         copy.deepcopy(expert_dist_cfg))
            if Path(file).exists():
                m.load_state_dict(torch.load(file, weights_only=False, map_location="cpu")["actor_state_dict"])
            for q in m.parameters():
                q.requires_grad_(False)
            self.experts.append(m)
        if any(self.expert_mirror):
            import numpy as np

            from tasks.jumper.common.mdp.symmetry import JOINT_PERM, JOINT_SIGN

            assert obs_mirror, "a mirrored expert needs obs_mirror (mirror_matrix.py's npz)"
            spec = np.load(obs_mirror)
            self.register_buffer("mirror_obs_perm", torch.as_tensor(spec["perm"]), persistent=False)
            self.register_buffer("mirror_obs_sign", torch.as_tensor(spec["sign"]), persistent=False)
            self.register_buffer("mirror_act_perm", torch.tensor(JOINT_PERM), persistent=False)
            self.register_buffer("mirror_act_sign", torch.tensor(JOINT_SIGN, dtype=torch.float32), persistent=False)
        # Programs from other official tasks, run through an adapter that maps the posture
        # observation and action onto theirs (`five_foot.py`): "five_foot" walks on five feet
        # without the left-front leg, "five_foot_mirror" without the right-front one. They follow
        # the experts in the program order and are kept out of the state dict. `adapter_vmax` is
        # their linear command bound: their training ceiling is 0.5 m/s, but with 0.6 the official
        # gait handed a dead front leg ran a 0.6 m/s command at 0.91 instead of 0.79, no falls
        # (PC, 64 robots per case, `eval_official.py --five-foot --five-foot-vmax 0.6`).
        self.adapter_names = list(adapters)
        if self.adapter_names:
            from five_foot import FiveFootProgram

            self.__dict__["_adapters"] = [FiveFootProgram(mirror=(name == "five_foot_mirror"), vmax=adapter_vmax)
                                          for name in self.adapter_names]
        self.has_gate = bool(experts) or bool(self.adapter_names)
        if self.has_gate:
            self.register_buffer("gate_nodes", torch.cat([ports["ascending"], ports["descending"]]), persistent=False)
            self.gate_out = nn.Linear(len(self.gate_nodes) * width, 1 + len(experts) + len(self.adapter_names))
            nn.init.zeros_(self.gate_out.weight)
            with torch.no_grad():
                self.gate_out.bias.zero_()
                self.gate_out.bias[0] = gate_bias
        # The giant-fibre escape, hard-wired (10/25, `tune_reflex.py`): when the looming rate
        # 1/tau that drives DNp01 first passes `thr` (1/s), the fibre fires once and latches
        # an escape command -- a step of `speed` m/s in a direction fixed at that moment from
        # the ball's bearing (`perp` across its path, `away` from it, or `blend`), and a duck
        # of `duck` m -- added to the commands every motor program reads, until the ball stops
        # closing in. Nothing in it is learnt.
        # The posture task's gait clock runs in the environment at the tempo of the env's
        # command, so a step the cord orders would otherwise keep the old cadence (at 0.3 s
        # throws the reflex then missed 0.96 of its dodges against 0.77 when the env's command
        # moved). With `cadence` given (the task's tempo law), the cord keeps its own phase
        # offset -- a fly's stepping rhythm is generated in its nerve cord -- advanced by the
        # extra tempo the escape command calls for, and the motor programs read the shifted
        # phase. The offset only grows while the reflex is on.
        self.gf_reflex = dict(gf_reflex) if gf_reflex else None
        if self.gf_reflex:
            assert self.threat_group, "the giant-fibre reflex reads the escape task's looming signal"
            names = {t["name"]: t for t in lay["terms"]}
            idx = list(range(names["command"]["offset"], names["command"]["offset"] + 3))
            idx += list(range(names["posture_command"]["offset"], names["posture_command"]["offset"] + 4))
            self.register_buffer("reflex_idx", torch.tensor(idx), persistent=False)
            phase = names["gait_phase"]["offset"]
            self.register_buffer("phase_idx", torch.tensor([phase, phase + 1]), persistent=False)
            self.control_dt = float(lay["control_dt"])
        self._gf_fired: torch.Tensor | None = None
        self._gf_dodge: torch.Tensor | None = None
        self._gf_ball: torch.Tensor | None = None  # with several balls: how many were closing in at the last fire
        self._gf_lean: torch.Tensor | None = None  # the latched lean (pitch, roll)
        self._gf_phase: torch.Tensor | None = None
        self._jump_t: torch.Tensor | None = None
        self._jump_aring: torch.Tensor | None = None
        self._jump_fring: torch.Tensor | None = None
        self._guard_t: torch.Tensor | None = None  # control steps into the claw guard (-1: off)
        self._guard_side: torch.Tensor | None = None  # which claw: True the right (a bolt from the right)
        # The giant fibre's other escape, the jump (10/25, `try_jump.py`): for a ball from the
        # front or behind (where a sidestep needs a wide move) the fibre waits until contact is
        # `tau` s away and hands the robot to the official reference-free jump for `steps`
        # control steps -- the jump the fly's giant fibre drives through the TTMn. That program
        # runs at kp 20 (the posture task's is 10), so the model only flags `jump_active` and
        # the caller sets the servo gains; it reads the posture observation's shared terms with
        # its own action history from zero and a torque history primed with the go its training
        # episodes began with (`go`, 100 values), which real torques then push out. Kept out of
        # the state dict: checkpoints stay as they are.
        self.gf_jump = dict(gf_jump) if gf_jump else None
        self.jump_active: torch.Tensor | None = None
        if self.gf_jump:
            assert self.gf_reflex, "the jump is one of the giant-fibre escape's two moves"
            jump_cfg = self.gf_jump
            m = MLPModel(TensorDict({"actor": torch.zeros(1, 406)}, batch_size=[1]), {"actor": ["actor"]}, "actor",
                         output_dim, tuple(jump_cfg.get("hidden_dims", (512, 256, 128, 64))), "elu", True,
                         {"class_name": "GaussianDistribution", "init_std": 0.3, "std_type": "scalar"})
            m.load_state_dict(torch.load(jump_cfg["checkpoint"], weights_only=False, map_location="cpu")["actor_state_dict"])
            for q in m.parameters():
                q.requires_grad_(False)
            self.__dict__["_jump_mlp"] = m.eval()
            self.__dict__["_jump_go"] = torch.tensor(np.load(jump_cfg["go"]), dtype=torch.float32).view(5, 20)[-1]
            names = {t["name"]: t for t in lay["terms"]}
            self.jump_force_newest = names["actuator_force"]["offset"] + 80  # the newest of five frames
        self.hidden_state: torch.Tensor | None = None

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse)
        self._w, self._wt = fn(self._w), fn(self._wt)
        if "_jump_mlp" in self.__dict__:
            self.__dict__["_jump_mlp"]._apply(fn)
            self.__dict__["_jump_go"] = fn(self.__dict__["_jump_go"])
        for prog in self.__dict__.get("_adapters", []):
            prog.to(self.leak_logit.device)
        return self

    # -- the cord --------------------------------------------------------------------

    def _stimulus(self, xn: torch.Tensor, s_prev: torch.Tensor, threat: torch.Tensor | None) -> torch.Tensor:
        """Inputs for one step, nodes x (robots*width); s_prev is the cord state it starts from."""
        b = xn.shape[0]
        u = xn.new_zeros(self.n, b, self.width)
        for k, leg in enumerate(LEGS):
            x = xn.index_select(1, getattr(self, f"obs_{leg}"))
            u[getattr(self, f"sens_{leg}")] = self.leg_in[k](x).view(b, -1, self.width).transpose(0, 1)
        dn = self.dn_in(xn.index_select(1, self.obs_global))
        if self.brain_hidden:
            an = s_prev.view(self.n, b, self.width)[self.an].permute(1, 0, 2).reshape(b, -1)
            dn = dn + self.brain_out(torch.tanh(self.brain_in(an)))
        if threat is not None:
            dn = dn + self.threat_in(threat)
        u[self.dn] = dn.view(b, -1, self.width).transpose(0, 1)
        if threat is not None:
            u[self.giant_fibre] = u[self.giant_fibre] + self.gf_gain * threat[:, 3].view(1, b, 1)
        u[self.haltere] = self.haltere_in(xn[:, :6]).view(b, -1, self.width).transpose(0, 1)
        return u.reshape(self.n, b * self.width)

    def _step(self, xn: torch.Tensor, h: torch.Tensor, threat: torch.Tensor | None = None):
        """One control step. xn: robots x 415 (normalised); h: robots x (nodes*width).
        Returns the residual, the new state and (with `command_readout`) the command offsets."""
        b = xn.shape[0]
        s = h.view(b, self.n, self.width).permute(1, 0, 2).reshape(self.n, b * self.width)
        a = torch.sigmoid(self.leak_logit)
        if self.learn_synapses:
            values = self.syn_sign * torch.exp(self.syn_log_mag)
            msg = LearnedMatmul.apply(s, values, self.syn_crow, self.syn_col, self.syn_crow_t,
                                      self.syn_col_t, self.syn_perm)
        else:
            msg = FixedMatmul.apply(s, self._w, self._wt)
        s = (1 - a) * s + a * torch.tanh(self.gain * msg + self.bias + self._stimulus(xn, s, threat))
        out = xn.new_zeros(b, len(self.leg_act_all()))
        s3 = s.view(self.n, b, self.width)
        for k, leg in enumerate(LEGS):
            pool = s3[getattr(self, f"motor_{leg}")].permute(1, 0, 2).reshape(b, -1)
            out[:, getattr(self, f"act_{leg}")] = self.leg_out[k](pool)
        h = s3.permute(1, 0, 2).reshape(b, self.n * self.width)
        cmd = None
        if self.command_readout:
            cmd = self.cmd_scale * torch.tanh(self.cmd_out(s3[self.dn].permute(1, 0, 2).reshape(b, -1)))
        return self.residual_scale * torch.tanh(out), h, cmd

    def _base(self, x: torch.Tensor, cmd: torch.Tensor | None) -> torch.Tensor:
        """The official MLP's action; with descending commands, on its commands plus the offsets
        (the MLP stays frozen, but the gradient goes through it to the readout)."""
        if cmd is None and self.mlp_anchor is None:
            with torch.no_grad():
                return self.mlp(self.obs_normalizer(x))
        if cmd is None:
            return self.mlp(self.obs_normalizer(x))
        x = x.clone()
        x[..., self.cmd_idx] = x[..., self.cmd_idx] + cmd
        return self.mlp(self.obs_normalizer(x))

    def gate(self, h: torch.Tensor) -> torch.Tensor:
        """The director's choice as logits, robots x programs, from the cord state h."""
        b = h.shape[0]
        return self.gate_out(h.view(b, self.n, self.width)[:, self.gate_nodes].reshape(b, -1))

    def programs(self, x: torch.Tensor) -> torch.Tensor:
        """Every motor program's action (programs x ... x actions): the official MLP, the experts,
        then the adapted programs. Those keep per-robot state and run on one step for every robot
        (robots x obs), every step, chosen or not, so that a switch finds them current; on padded
        trajectories (PPO's update) they read as zero."""
        with torch.no_grad():
            acts = [self.mlp(self.obs_normalizer(x))]
            for e, mirror in zip(self.experts, self.expert_mirror):
                if mirror:
                    xm = x[..., self.mirror_obs_perm] * self.mirror_obs_sign
                    acts.append(e.mlp(e.obs_normalizer(xm))[..., self.mirror_act_perm] * self.mirror_act_sign)
                else:
                    acts.append(e.mlp(e.obs_normalizer(x)))
            for prog in self.__dict__.get("_adapters", []):
                if x.dim() == 2:
                    acts.append(prog(x, torch.ones(x.shape[0], dtype=torch.bool, device=x.device)))
                else:
                    acts.append(torch.zeros_like(acts[0]))
            return torch.stack(acts)

    def reflex(self, threat: torch.Tensor) -> torch.Tensor:
        """The giant-fibre escape's offsets on the commands, robots x 7 (vx, vy, wz; the
        posture command's four, height last); updates the latch."""
        b = threat.shape[0]
        if self._gf_fired is None or self._gf_fired.shape[0] != b:
            self._gf_fired = torch.zeros(b, dtype=torch.bool, device=threat.device)
            self._gf_dodge = threat.new_zeros(b, 2)
        cfg = self.gf_reflex
        live = threat.abs().sum(-1) > 0  # a ball closing in (the signal is zero otherwise)
        d = threat[:, :2] / threat[:, :2].norm(dim=-1, keepdim=True).clamp_min(1e-3)
        side_ok = torch.ones_like(live)
        jump_new = torch.zeros_like(live)
        multi = threat.shape[1] >= 9
        nb = (threat.shape[1] - 9) // 3 if multi else 0
        crowd = torch.zeros_like(live)  # a volley no step clears, low enough to leap over
        clear = None
        if multi and cfg.get("lines") and nb >= 2:
            each = threat[:, 9:9 + 3 * nb].reshape(b, nb, 3)
            clear, cleared = self._clear_lines(each, cfg["speed"], cfg["lines"])
            if self.gf_jump:
                # (10/26) Airborne, the robot is clear of every low bolt at once, from any side: the
                # official jump has all feet 12 cm up from 0.35 to 0.64 s after it starts. So where
                # the best step leaves a ball uncleared, the fibre waits and jumps, if every bolt is
                # low and the last arrives inside that window.
                rate = each[..., 2] * 10.0
                inbound = rate > 0
                elev = (1.0 - (each[..., :2] ** 2).sum(-1)).clamp_min(0.0).sqrt()
                low_all = ((elev < float(self.gf_jump.get("max_elev", 0.1))) | ~inbound).all(dim=1)
                last = torch.where(inbound, 1.0 / rate.clamp_min(1e-3), torch.zeros_like(rate)).max(dim=1).values
                crowd = ((threat[:, 8] >= 2) & (cleared < threat[:, 8]) & low_all
                         & (last <= float(self.gf_jump.get("window", 0.62))))
        if self.gf_jump:
            if self._jump_t is None or self._jump_t.shape[0] != b:
                self._jump_t = torch.full((b,), -1, dtype=torch.long, device=threat.device)
                self._jump_aring = threat.new_zeros(17, b, self.leg_act_count)
                self._jump_fring = threat.new_zeros(17, b, self.leg_act_count)
            # Only a low ball from the front or behind gets the jump: a lobbed one comes down
            # steeply onto a robot jumping in place (0.6 / 0.9 s throws from the front, hit 0.99 /
            # 0.99 with the jump against 0.00 with the sidestep), so it is stepped away from.
            front_back = d[:, 0].abs() >= d[:, 1].abs()
            if self.gf_jump.get("ahead_only"):
                # (10/26) Not for a ball from behind: aimed where a running robot will be, it crosses
                # where the robot is before the jump has lifted the feet (0.4 s throws from behind
                # while running: hit 0.92 with the jump, 0.00 with the step).
                front_back = d[:, 0] >= d[:, 1].abs()
            low = threat[:, 2] < float(self.gf_jump.get("max_elev", 0.1))
            leap = (front_back & low) | crowd
            side_ok = ~leap
            jump_new = (live & ~self._gf_fired & leap & (self._jump_t < 0)
                        & (threat[:, 3] * 10.0 > 1.0 / self.gf_jump["tau"]))
            self._jump_t = torch.where(jump_new, torch.zeros_like(self._jump_t), self._jump_t)
        # The claw guard (`guard`, 10/26, the user's idea): a lone low bolt from ahead is met with the
        # left claw held up on the body's centreline (the five-legged gait walks on the other five,
        # `guard_override`) instead of a step or a jump -- a step would carry the claw off the line.
        guard_new = torch.zeros_like(live)
        g = cfg.get("guard")
        if g:
            if self._guard_t is None or self._guard_t.shape[0] != b:
                self._guard_t = torch.full((b,), -1, dtype=torch.long, device=threat.device)
            alone = threat[:, 8] <= 1 if multi else torch.ones_like(live)
            ahead = (d[:, 0] >= float(g.get("cone", 0.8))) & (threat[:, 2] < float(g.get("max_elev", 0.3))) & alone
            side_ok = side_ok & ~ahead
            jump_new = jump_new & ~ahead
            if self._jump_t is not None:
                self._jump_t = torch.where(ahead & (self._jump_t == 0), torch.full_like(self._jump_t, -1), self._jump_t)
            guard_new = live & ~self._gf_fired & ahead & (self._guard_t < 0) & (threat[:, 3] * 10.0 > cfg["thr"])
            self._guard_t = torch.where(guard_new, torch.zeros_like(self._guard_t), self._guard_t)
            if self._guard_side is None or self._guard_side.shape[0] != b:
                self._guard_side = torch.zeros(b, dtype=torch.bool, device=threat.device)
            # The claw on the bolt's side: the left one alone let 0.30-0.40 of bolts 10 degrees to the
            # right past it (blocked 0.45-0.59), against 1.00 dead ahead and from the left.
            self._guard_side = torch.where(guard_new, d[:, 1] < 0, self._guard_side)
        new = live & ~self._gf_fired & side_ok & (threat[:, 3] * 10.0 > cfg["thr"])
        # Several balls (`escape.looming_multi`: where the others are, and how many are closing
        # in): the step goes to the side away from the other balls in flight, and the fibre fires
        # again only when a ball joins those closing in -- not whenever the most urgent one
        # changes: balls thrown together are equally urgent, and re-aiming at whichever leads
        # by a hair each step left the robot shaking in place (three at once, 120 degrees apart:
        # 0.99 of the balls hit, against 0.006 with their arrivals 0.15 s apart).
        if multi:
            count = threat[:, 8]
            if self._gf_ball is None or self._gf_ball.shape[0] != b:
                self._gf_ball = threat.new_zeros(b)  # balls closing in when the fibre last fired
            new = new | (self._gf_fired & live & side_ok & (count > self._gf_ball)
                         & (threat[:, 3] * 10.0 > cfg["thr"]))
        perp = torch.stack([-d[:, 1], d[:, 0]], dim=-1)
        perp = torch.where((perp[:, :1] < 0) | ((perp[:, :1] == 0) & (perp[:, 1:] < 0)), -perp, perp)
        if multi:
            rest = threat[:, 6:8]
            away = torch.where((perp * rest).sum(-1, keepdim=True) > 0, -perp, perp)
            perp = torch.where(rest.norm(dim=-1, keepdim=True) > 1e-6, away, perp)
            # `lines` (10/26): with two or more closing in, the step clears every ball's line, not
            # only the most urgent one's. Straight bolts fired together from 3 m (0.4 s) hit 0.50
            # of the balls in pairs 90 degrees apart and 0.65 in threes 120 apart against 0.00 for
            # one: a step across one bolt's line ran along the other's.
            if clear is not None:
                perp = torch.where((count >= 2)[:, None], clear, perp)
            self._gf_ball = torch.where(new, count, torch.minimum(self._gf_ball, count))
        if cfg["mode"] == "perp":
            u = perp
        elif cfg["mode"] == "away":
            u = -d
        else:
            u = (perp - d) / (perp - d).norm(dim=-1, keepdim=True).clamp_min(1e-3)
        self._gf_dodge = torch.where(new[:, None], cfg["speed"] * u, self._gf_dodge)
        # The lean (`lean`, rad; a Matrix-style dodge): the trunk tilts away from the ball, latched
        # with the step. posture_command's pitch is + nose down and its roll + left side up
        # (POSTURE_COMMAND.md), so a ball from the front tips the nose up and one from the left
        # raises the left side.
        lean = float(cfg.get("lean", 0.0))
        if lean:
            if self._gf_lean is None or self._gf_lean.shape[0] != b:
                self._gf_lean = threat.new_zeros(b, 2)
            self._gf_lean = torch.where(new[:, None], torch.stack([-lean * d[:, 0], lean * d[:, 1]], dim=-1),
                                        self._gf_lean)
        self._gf_dodge = torch.where(jump_new[:, None], torch.zeros_like(self._gf_dodge), self._gf_dodge)
        self._gf_fired = (self._gf_fired & live) | new | jump_new | guard_new
        out = threat.new_zeros(b, 7)
        on = self._gf_fired[:, None]
        out[:, :2] = torch.where(on, self._gf_dodge, torch.zeros_like(self._gf_dodge))
        out[:, 6] = torch.where(self._gf_fired, -float(cfg.get("duck", 0.0)), 0.0)
        if lean:
            out[:, 4:6] = torch.where(self._gf_fired[:, None], self._gf_lean, torch.zeros_like(self._gf_lean))
        return out

    @staticmethod
    def _clear_lines(each: torch.Tensor, speed: float, opt: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """The step direction (robots x 2) that clears the most closing balls' lines, and how many
        it clears (robots): `each` is
        `escape.looming_multi`'s per-ball channels (robots x balls x 3: direction x, y and 1/tau / 10).
        Of `n` directions round the circle, each ball's miss is how far the step carries the robot
        across its line before it arrives -- `gain` x `speed` m/s from `t0` s after the fibre fires
        (10/26, measured on recorded dodges at 1.2 m/s: 0.03, 0.11, 0.21 and 0.33 m beyond the walk
        0.1-0.4 s after firing) -- and a ball is cleared at `cap` m (single bolts: hit 0.68 / 0.08 /
        0.00 at 0.3 / 0.35 / 0.4 s of warning). The direction clearing the most wins, then the one
        whose smallest miss is largest, ties going away from where the balls come from; where only
        one of two can be cleared, it clears one rather than splitting the step and missing both.
        A straight line's miss does not depend on which way along it one steps."""
        n, t0, cap = int(opt.get("n", 64)), float(opt.get("t0", 0.07)), float(opt.get("cap", 0.28))
        gain = float(opt.get("gain", 0.85))
        d = each[..., :2] / each[..., :2].norm(dim=-1, keepdim=True).clamp_min(1e-3)
        rate = each[..., 2] * 10.0
        live = rate > 0
        reach = gain * speed * (1.0 / rate.clamp_min(1e-3) - t0).clamp_min(0.0)
        ang = torch.arange(n, device=each.device, dtype=each.dtype) * (2.0 * math.pi / n)
        u = torch.stack([torch.cos(ang), torch.sin(ang)], dim=-1)
        across = (d[..., None, 0] * u[:, 1] - d[..., None, 1] * u[:, 0]).abs()  # robots x balls x n
        miss = torch.where(live[..., None], (reach[..., None] * across).clamp_max(cap), cap) / cap
        score = (miss >= 1.0).sum(1) + 0.9 * miss.min(dim=1).values
        toward = (rate[..., None] * (d @ u.T)).sum(1)  # robots x n
        best = (score - 1e-3 * toward).argmax(dim=1)
        return u[best], (miss >= 1.0).sum(1).gather(1, best[:, None])[:, 0].to(each.dtype)

    def guard_override(self, progs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """While the claw guard runs: the five-legged gait's action, its left-front four (the arm it
        walks without) set to the guard pose (`guard["pose"]`, posture-action units); then the
        director's again after `guard["steps"]` control steps."""
        g = self.gf_reflex["guard"]
        on = self._guard_t >= 0
        k0 = 1 + len(self.experts_paths)
        pose = torch.tensor(g["pose"], device=action.device, dtype=action.dtype)
        left = progs[k0 + self.adapter_names.index("five_foot")].clone()
        left[:, :4] = pose
        ff = left
        if "five_foot_mirror" in self.adapter_names and self._guard_side is not None:
            # the right claw: the mirrored gait walks without it, and the pose mirrored (J0, J2, J3
            # change sign, as HOME's do between the two arms)
            right = progs[k0 + self.adapter_names.index("five_foot_mirror")].clone()
            right[:, 4:8] = pose * torch.tensor([-1.0, 1.0, -1.0, -1.0], device=pose.device, dtype=pose.dtype)
            ff = torch.where(self._guard_side[:, None], right, left)
        t = torch.where(on, self._guard_t + 1, self._guard_t)
        self._guard_t = torch.where(t >= int(g.get("steps", 120)), torch.full_like(t, -1), t)
        return torch.where(on[:, None], ff, action)

    @property
    def leg_act_count(self) -> int:
        return len(self.leg_act_all())

    def jump_override(self, x: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Robots mid-jump take the jump program's action (in the posture task's units); sets
        `jump_active` for the caller's servo gains and advances the jump timers."""
        active = self._jump_t >= 0
        self.jump_active = active
        if not bool(active.any()):
            return action
        start = self._jump_t == 0
        aring, fring = self._jump_aring, self._jump_fring
        aring = torch.where(start[None, :, None], torch.zeros_like(aring), aring)
        newest = x[:, self.jump_force_newest:self.jump_force_newest + self.leg_act_count]
        pushed = torch.cat([fring[1:], newest[None]], dim=0)
        fring = torch.where((active & ~start)[None, :, None], pushed, fring)
        fring = torch.where(start[None, :, None], self._jump_go.view(1, 1, -1).expand_as(fring), fring)
        frames = [0, 4, 8, 12, 16]
        hist = aring[frames].permute(1, 0, 2).reshape(x.shape[0], -1)
        force = fring[frames].permute(1, 0, 2).reshape(x.shape[0], -1)
        xj = torch.cat([x[:, :206], hist, force], dim=-1)
        with torch.no_grad():
            m = self.__dict__["_jump_mlp"]
            a = m.mlp(m.obs_normalizer(xj))
        self._jump_aring = torch.where(active[None, :, None], torch.cat([aring[1:], a[None]], dim=0), aring)
        self._jump_fring = fring
        t = torch.where(active, self._jump_t + 1, self._jump_t)
        self._jump_t = torch.where(t >= int(self.gf_jump["steps"]), torch.full_like(t, -1), t)
        return torch.where(active[:, None], a * float(self.gf_jump.get("unit", 2.0)), action)

    def _tempo(self, cmd: torch.Tensor) -> torch.Tensor:
        """The posture task's cadence law, Hz, for commands robots x 3 (vx, vy, wz)."""
        c = self.gf_reflex["cadence"]
        v = cmd[:, :2].norm(dim=-1) + cmd[:, 2].abs() * c["turn_radius"]
        return (v / (2.0 * c["stride"])).clamp(c["freq_min"], c["freq_max"])

    def escape_inputs(self, x: torch.Tensor, threat: torch.Tensor) -> torch.Tensor:
        """The motor programs' inputs under the reflex: its command offsets and, with a
        cadence, the gait phase advanced by the cord's own offset."""
        off = self.reflex(threat)
        y = x.clone()
        y[..., self.reflex_idx] += off
        if "cadence" in self.gf_reflex:
            if self._gf_phase is None or self._gf_phase.shape[0] != x.shape[0]:
                self._gf_phase = x.new_zeros(x.shape[0])
            cmd = x[..., self.reflex_idx[:3]]
            self._gf_phase = self._gf_phase + (self._tempo(cmd + off[:, :3]) - self._tempo(cmd)) * self.control_dt
            a = 2.0 * math.pi * self._gf_phase
            s, c = x[..., self.phase_idx[0]], x[..., self.phase_idx[1]]
            y[..., self.phase_idx[0]] = s * torch.cos(a) + c * torch.sin(a)
            y[..., self.phase_idx[1]] = c * torch.cos(a) - s * torch.sin(a)
        return y

    def leg_act_all(self) -> range:
        return range(sum(len(getattr(self, f"act_{leg}")) for leg in LEGS))

    # -- rsl_rl interface ------------------------------------------------------------

    def forward(self, obs: TensorDict, masks: torch.Tensor | None = None, hidden_state: HiddenState = None,
                stochastic_output: bool = False) -> torch.Tensor:
        x = torch.cat([obs[g] for g in self.obs_groups], dim=-1)
        xn = self.obs_normalizer(x)
        threat = obs[self.threat_group] if self.threat_group else None
        cord_threat = threat if self.threat_cord else None
        if masks is None:  # rollout: one step for every robot
            if self.hidden_state is None or self.hidden_state.shape[1] != xn.shape[0]:
                self.hidden_state = xn.new_zeros(1, xn.shape[0], self.n * self.width)
            res, h, cmd = self._step(xn, self.hidden_state[0], cord_threat)
            self.hidden_state = h.unsqueeze(0)
            x_raw = x
            if self.gf_reflex:  # the motor programs read the escape's commands; the cord, the real ones
                x = self.escape_inputs(x, threat)
            if self.has_gate:
                progs = self.programs(x)
                base = torch.einsum("kba,bk->ba", progs, torch.softmax(self.gate(h), dim=-1))
                if self._guard_t is not None and bool((self._guard_t >= 0).any()):
                    base = self.guard_override(progs, base)
            else:
                base = self._base(x, cmd)
        else:  # update: padded trajectories, time first
            if hidden_state is None:
                raise ValueError("hidden state not passed during the update")
            h = hidden_state[0] if not isinstance(hidden_state, tuple) else hidden_state[0][0]
            steps, cmds, gates = [], [], []
            for t in range(xn.shape[0]):
                r, h, c = self._step(xn[t], h, None if cord_threat is None else cord_threat[t])
                steps.append(r)
                cmds.append(c)
                if self.has_gate:
                    gates.append(torch.softmax(self.gate(h), dim=-1))
            cmd_all = torch.stack(cmds) if self.command_readout else None
            if cmd_all is not None and self.command_l2 > 0 and cmd_all.requires_grad:
                n_cmd = max(int(masks.sum()), 1)
                cmd_all.register_hook(lambda g, c=cmd_all.detach(), n=n_cmd: g + (2.0 * self.command_l2 / n)
                                      * c * masks.unsqueeze(-1) / self.cmd_scale ** 2)
            if self.has_gate:
                base = torch.einsum("ktba,tbk->tba", self.programs(x), torch.stack(gates))
            else:
                base = self._base(x, cmd_all)
            res = unpad_trajectories(torch.stack(steps), masks)
            base = unpad_trajectories(base, masks)
            if self.residual_l2 > 0 and res.requires_grad:
                # The gradient of residual_l2 * mean over samples of sum(res^2), added
                # through a hook so PPO's own loss code stays untouched: the cord then
                # corrects only where the advantage pays for it.
                n = res.numel() // res.shape[-1]
                res.register_hook(lambda g, r=res.detach(): g + (2.0 * self.residual_l2 / n) * r)
        mean = base + res
        if masks is None and self.gf_jump:
            mean = self.jump_override(x_raw, mean)
        if self.distribution is not None:
            if stochastic_output:
                self.distribution.update(mean)
                return self.distribution.sample()
            return self.distribution.deterministic_output(mean)
        return mean

    def update_normalization(self, obs: TensorDict) -> None:
        """The official normaliser stays as trained; the residual sees what the MLP sees."""

    def reset(self, dones: torch.Tensor | None = None, hidden_state: HiddenState = None) -> None:
        if dones is None:
            self.hidden_state = hidden_state
        elif self.hidden_state is not None:
            self.hidden_state[:, dones == 1] = 0.0
        if dones is not None and self._gf_fired is not None:
            self._gf_fired = self._gf_fired & (dones != 1)
        if dones is not None and self._gf_phase is not None:
            self._gf_phase = torch.where(dones == 1, torch.zeros_like(self._gf_phase), self._gf_phase)
        if dones is not None and self._guard_t is not None:
            self._guard_t = torch.where(dones == 1, torch.full_like(self._guard_t, -1), self._guard_t)
        if dones is not None:
            for prog in self.__dict__.get("_adapters", []):
                prog.reset(dones == 1)

    def get_hidden_state(self) -> HiddenState:
        return self.hidden_state

    def detach_hidden_state(self, dones: torch.Tensor | None = None) -> None:
        if self.hidden_state is not None:
            self.hidden_state = self.hidden_state.detach()

    def as_onnx(self, verbose: bool = False) -> nn.Module:
        raise NotImplementedError("the VNC actor is not exported yet")

    def as_jit(self) -> nn.Module:
        raise NotImplementedError("the VNC actor is not exported yet")


class FirstStepHiddenStorage(RolloutStorage):
    """RolloutStorage that keeps the actor's recurrent state only at the rollout's first
    step. The trajectory batches only need the state where a trajectory starts: the
    first step, or a step right after a reset, where the actor's state is zero. For the
    cord (24k nodes per robot) this is the difference between 1 and 48 copies."""

    def _save_hidden_states(self, hidden_states: tuple[HiddenState, HiddenState]) -> None:
        h = hidden_states[0]
        if h is None:  # the very first step of training: the state is zero
            return
        if getattr(self, "_h0", None) is None or self._h0.shape != h.shape:
            # Allocated outside inference mode so the update can build a graph on it;
            # left at zero when the first step had no state yet.
            with torch.inference_mode(False):
                self._h0 = torch.zeros(h.shape, device=self.device)
            self.saved_hidden_state_a = [self._h0]  # marks the actor as recurrent
        if self.step == 0:
            self._h0.copy_(h)

    def recurrent_mini_batch_generator(self, num_mini_batches: int, num_epochs: int = 8):
        from rsl_rl.utils import split_and_pad_trajectories

        padded_obs, masks = split_and_pad_trajectories(self.observations, self.dones)
        mini_batch_size = self.num_envs // num_mini_batches
        dones = self.dones.squeeze(-1)
        last_was_done = torch.zeros_like(dones, dtype=torch.bool)
        last_was_done[1:] = dones[:-1]
        last_was_done[0] = True
        starts = last_was_done.permute(1, 0).nonzero()  # (env, t), env-major like the obs split
        for _ in range(num_epochs):
            first_traj = 0
            for i in range(num_mini_batches):
                start, stop = i * mini_batch_size, (i + 1) * mini_batch_size
                last_traj = first_traj + int(last_was_done[:, start:stop].sum())
                rows = starts[first_traj:last_traj]
                h = torch.zeros(1, len(rows), self._h0.shape[-1], device=self.device)
                at0 = rows[:, 1] == 0
                h[0, at0] = self._h0[0, rows[at0, 0]]
                yield RolloutStorage.Batch(
                    observations=padded_obs[:, first_traj:last_traj],
                    actions=self.actions[:, start:stop],
                    values=self.values[:, start:stop],
                    advantages=self.advantages[:, start:stop],
                    returns=self.returns[:, start:stop],
                    old_actions_log_prob=self.actions_log_prob[:, start:stop],
                    old_distribution_params=tuple(p[:, start:stop] for p in self.distribution_params),
                    hidden_states=(h, None),
                    masks=masks[:, first_traj:last_traj],
                )
                first_traj = last_traj
