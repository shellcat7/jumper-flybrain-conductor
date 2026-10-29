# Method

Everything here runs in simulation: the official Jumper model in MuJoCo, stepped on the GPU with
mujoco-warp, at the official posture task's 200 Hz control rate.

## 1. The cord packet (`experiments/vnc_v1/build_packet.py`)

- Source: MaleCNS v1.0 (Janelia FlyEM), body annotations at confidence ≥ 0.5 and predicted
  neurotransmitters, from <https://male-cns.janelia.org/download/>.
- Nodes: the thoracic ventral nerve cord's intrinsic neurons (20,429), plus the descending neurons (1,314)
  and ascending neurons (1,846) that link it to the brain, and the cord's sensory and motor neurons:
  24,154 in all. Nodes are reordered so that neurons near each other in the cord (segment, side, class)
  sit near each other in memory.
- Edges: synapses with at least 5 contacts, 1.14 M edges (`vnc_ge5.npz`, CSR).
- Sign: acetylcholine +1, GABA and glutamate −1 (36% of edges inhibitory); other transmitters start at 0.
- Ports (`layout.json`): for each Jumper leg, the sensory and motor neurons of the fly leg nerve in the
  same place (LF/RF: prothoracic, LM/RM: mesothoracic, LR/RR: metathoracic, left and right). The haltere
  afferents take the IMU; the descending neurons take the command and gait clock.

## 2. The cord model (`experiments/vnc_v1/vnc_model.py`)

A rate model, one step per control step:

```
s ← (1 − a) · s + a · tanh(g · (W s) + b + u)
```

- `W`: the packet, normalised by each neuron's incoming weight. With `learn_synapses` the magnitudes are
  trained and the signs kept.
- `a`, `g`, `b`: per neuron, trained.
- `u`: the inputs at the ports. Each leg's joint angles, velocities, torques and contact go to that leg's
  sensory pool; the shared observations go to the descending neurons and haltere afferents.
- State: the whole cord per robot (24,154 floats), carried through time. Training backpropagates through
  the cord over the 48-step window.

## 3. The conductor, or "director" (`method: distill-gate`)

The cord does not write joint targets. A linear readout of the ascending and descending neurons' states
gives a softmax over twelve frozen motor programs, and the action is the softmax-weighted mixture of the
programs' actions:

| # | program | what it is for |
|---|---|---|
| 0 | official posture policy (`model_74800`) | nothing wrong; a dead middle leg |
| 1 | `mlp-d` | low foot friction |
| 2 | `mlp-b` | a dead front or rear leg |
| 3 | `sprint16` | commands above 1.35 m/s |
| 4, 5 | `mlpRR`, mirrored | a dead right-rear / left-rear leg |
| 6, 7 | `mlpLM`, mirrored | a dead left-middle / right-middle leg while turning |
| 8, 9 | `mlpLMslow`, mirrored | the same, walking slowly |
| 10, 11 | official five-legged gait, mirrored, through an adapter | a dead front leg (the film's wounded claw) |

Programs 1-9 are the official policy fine-tuned with PPO under each trouble (`experiments/reflex_v1/train_mlp.py`).
A mirrored program reads the left-right mirrored observation and its action is mirrored back
(`mirror_matrix.py`). Programs 10-11 come from the official five-legged task and are mapped onto the posture
task's observation and action by `five_foot.py` (`adapter_vmax 0.6`).

**Training** (`experiments/vnc_v1/distill_vnc.py --mode gate`): DAgger. The student drives (its own action
plus a little noise) through randomly timed troubles (a leg's torque set to zero, foot friction dropped). At
every visited state a rule written from earlier evaluations labels the right program. The readout is fitted
by cross-entropy, and the gradient also trains the cord (a, g, b, synapse magnitudes). The trouble flags
are given only to the labelling rule; the student sees only the official policy's 415 observations. It has
to recognise the trouble from how the legs move. dgate6: 4,096 robots, 300 iterations, lr 1e-3, started from
the dgate5 checkpoint.

**Why distillation.** PPO fine-tuning of the cord (2026-10-24 and 25) either drifted from the official gait or,
held near it by a penalty, learnt nothing. Choosing among programs is a much easier target, and it is what
a fly's descending neurons are thought to do: switch between behaviours rather than drive each muscle.

## 4. The escape reflex (`experiments/vnc_v1/escape.py`, `reflex_actor.py`)

Hand-designed after the fly's giant-fibre pathway (looming detectors → DNp01 → jump muscle):

- **Looming signal**: 1/τ, the projectile's closing speed over its distance, read from the simulator.
- **Fire**: when 1/τ > 0.8 /s for a projectile on a collision course.
- **Sidestep**: at 1.5 m/s, perpendicular to the incoming line, in a direction latched when it fires. With
  several projectiles it samples 64 step directions and takes the one that clears every line by 0.28 m.
- **Duck**: lowers the body 3 cm while stepping.
- **Jump**: the official reference-free jump policy (`ref_free_jump/model_5999`), only for projectiles ahead.
- **Claw guard** (the film's parry): for a frontal projectile it sets the left claw to a fixed pose
  (`[-1.875, -2.444, -2.148, 3.773]` rad). The deflection is contact physics; nothing aims the claw.

The motor programs receive the reflex's step as their command, while the cord keeps the real command.

## 5. Evaluation (`experiments/reflex_v1/eval_official.py`, `experiments/vnc_v1/eval_escape.py`)

- **Troubles**: 8 troubles (dead LF, RF, LM, RR, LM+LR, LM+RR; foot friction 0.03 or 0.06) × 3 commands
  (walk 0.3 m/s; run 0.6 m/s; corner 0.5 m/s at 1.5 rad/s) = 24 cells, plus 3 flat-ground cells. The
  trouble strikes at 2 s of an 8 s episode. 64 robots per cell, the training's sensor noise. Score: mean
  forward speed after the strike over the commanded speed.
- **Top speed**: flat ground, commands 0.3-2.0 m/s.
- **Thrown balls**: a 25 mm, 30 g ball thrown from 1 m on a ballistic arc with 0.3/0.4/0.6/0.9 s of flight,
  from six bearings, at walking and running robots, 32 per case, four throws each. Score: share of throws
  that touch the robot.

## 6. The shuffled-wiring control

`dgshuf6-s3`: the same recipe on a cord with shuffled synapses (`shuffle_seed`). Every synapse keeps its
target and contact count, and its source is drawn by permuting all sources. The neurons, ports, in- and
out-degrees stay the same, and each source keeps its transmitter sign; only the wiring is random. It scores
as well as the real wiring:

| | troubles, mean of 24 | flat ground |
|---|---|---|
| official | 0.485 | 0.951 |
| dgate6-s1 (fly wiring) | 0.717 | 0.951 |
| dgate6-s2 (fly wiring, second seed) | 0.717 | 0.951 |
| dgshuf6-s3 (shuffled wiring) | 0.719 | 0.948 |

So these results do not show that the fly's wiring matters. What they show is that a recurrent network
that picks among motor programs from body signals alone recovers most of the speed lost to a dead leg. A
real test of the wiring would need a task the programs cannot solve without it.

## 7. Known weaknesses

- Two dead legs on one side (LM+LR): 0.04-0.32, no better than the official policy.
- The fastest throws (0.3 s of flight from 1 m): about half still hit.
- With five legs the reflex never ducked under the film's shutter in the re-recordings (it stepped back or
  was hit), so the film's last pass under the shutter is a plain run.
- Sensing is simulator truth. On the real robot, with its forward camera and dToF, the looming signal would
  be partial, noisy and late.
