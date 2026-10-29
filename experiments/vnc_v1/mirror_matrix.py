"""The posture actor observation's left-right mirror as a signed permutation, for mirrored experts.

The official mirror (`tasks/jumper/common/mdp/symmetry.py`, `_mirror_group`) reads each term's
slice and history from the environment's observation manager. Every term the posture actor
observes mirrors linearly (vectors, pseudovectors, joints, the twist, the posture command, the
gait clock's half shift), so the mirror is a 415 x 415 matrix; fed the identity it returns that
matrix, and this checks that it is a signed permutation and saves it as (perm, sign):

    mirrored[:, i] = sign[i] * x[:, perm[i]]

It also checks the result on the official policy, which was trained with the mirror loss: the
mirrored policy (mirror the observation, act, mirror the action back) should act as it does.

    python experiments/vnc_v1/mirror_matrix.py --out runs/vnc-v1/packet-1/mirror_actor.npz
"""

from __future__ import annotations

import argparse


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--checkpoint", default="tasks/jumper/posture/out/example/model_74800.pt")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    from dataclasses import asdict

    import numpy as np
    import torch

    import tasks
    from mjrl.backend.resolve import resolve
    from mjrl.backend.select import use_backend

    res = resolve(backend="native", device="cpu", num_envs=8)
    use_backend(res)
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from rsl_rl.models.mlp_model import MLPModel

    from tasks.jumper.common.mdp.symmetry import JOINT_PERM, JOINT_SIGN, _mirror_group

    cfg = tasks.load_env_cfg("jumper.posture")
    cfg.scene.num_envs = res.num_envs
    env = ManagerBasedRlEnv(cfg=cfg, device=res.device)
    try:
        wrapped = RslRlVecEnvWrapper(env)
        obs = wrapped.get_observations()
        x = obs["actor"]
        d = x.shape[1]
        m = _mirror_group(env, "actor", torch.eye(d))  # row i: the mirror of basis vector i
        nonzero = (m != 0).sum(dim=0)
        assert bool((nonzero == 1).all()), "the actor mirror is not a signed permutation"
        perm = m.abs().argmax(dim=0)  # mirrored[:, j] = sum_i x[:, i] * m[i, j]
        sign = m[perm, torch.arange(d)]
        assert bool((sign.abs() == 1).all())
        mirrored = x[:, perm] * sign
        assert torch.allclose(mirrored, _mirror_group(env, "actor", x)), "the matrix disagrees with the mirror"
        assert torch.allclose(mirrored[:, perm] * sign, x), "the mirror is not an involution"

        # The official policy, trained with the mirror loss, should be (nearly) mirror-equivariant.
        agent_cfg = tasks.load_agent_cfg("jumper.posture")
        kw = asdict(agent_cfg)["actor"]
        kw.pop("class_name", None)
        for opt in ("cnn_cfg", "distribution_cfg"):
            if kw.get(opt) is None:
                kw.pop(opt, None)
        if kw.get("rnn_type") is None:
            for opt in ("rnn_type", "rnn_hidden_dim", "rnn_num_layers"):
                kw.pop(opt, None)
        groups = asdict(agent_cfg)["obs_groups"]
        mlp = MLPModel(obs, groups, "actor", wrapped.num_actions, **kw)
        mlp.load_state_dict(torch.load(args.checkpoint, weights_only=False, map_location="cpu")["actor_state_dict"])
        jp, js = torch.tensor(JOINT_PERM), torch.tensor(JOINT_SIGN)
        with torch.no_grad():
            for _ in range(20):  # walk a little so the observation is not the reset pose
                obs, _, _, _ = wrapped.step(mlp.mlp(mlp.obs_normalizer(obs["actor"])))
            x = obs["actor"]
            a = mlp.mlp(mlp.obs_normalizer(x))
            b = mlp.mlp(mlp.obs_normalizer(x[:, perm] * sign))[:, jp] * js
        err = float((a - b).abs().mean() / a.abs().mean().clamp_min(1e-6))
        print(f"[mirror] actor {d} dims; official policy mirror error {err:.3f} (relative, mean |a| {float(a.abs().mean()):.3f})")
        np.savez(args.out, perm=perm.numpy().astype(np.int64), sign=sign.numpy().astype(np.float32))
        print(f"MIRROR_SAVED {args.out}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
