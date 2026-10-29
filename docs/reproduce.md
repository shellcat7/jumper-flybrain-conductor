# Reproducing the numbers

Everything was run on Google Colab with an NVIDIA RTX PRO 6000 (the "G4" runtime). Any CUDA GPU that
mujoco-warp supports should work. GPU simulation is not bit-exact between runs, so expect differences in the
second decimal place.

## 1. Setup (Colab, GPU runtime)

```bash
JUMPER_SHA=61d065219fca767f3142c8f10aff59eae5a5a004
cd /content
git clone -q https://github.com/KingKongRobotics/jumper.git
git -C jumper checkout -q "$JUMPER_SHA"
git clone -q https://github.com/shellcat7/jumper-flybrain-conductor.git repo
pip install -q torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
pip install -q -e ./jumper mujoco==3.11.0 mujoco-warp==3.11.0 warp-lang==1.18.0 onnxruntime==1.30.0

# the weights (release v1) go to /content/up, where the configs look for them
mkdir -p /content/up /content/results
wget -q https://github.com/shellcat7/jumper-flybrain-conductor/releases/download/v1/jumper-flybrain-conductor-weights-v1.zip
unzip -jo jumper-flybrain-conductor-weights-v1.zip -d /content/up
cd /content/up && sha256sum -c SHA256SUMS
```

The official posture policy (`model_74800.pt`) and jump policy (`model_5999.pt`) come from the official
repository at that commit. Run everything below from `/content/jumper`, because the configs refer to those
policies by paths relative to it.

| config | what it is |
|---|---|
| `configs/eval-brain-vnc-config.json` | dgate6 with the reflex: the scoreboard |
| `configs/film-brain-vnc-config.json` | the same, plus the claw guard: the film |
| `configs/shuffled-control-vnc-config.json` | the shuffled-wiring control (`shuffle_seed: 2`) |

## 2. Troubles (about 4 min per brain on one GPU)

```bash
cd /content/jumper
E=/content/repo/experiments
python -u $E/reflex_v1/eval_official.py --backend warp --envs-per-case 64 --out /content/results/official-f64
python -u $E/reflex_v1/eval_official.py --backend warp --envs-per-case 64 \
    --checkpoint /content/up/dgate6-s1-299.pt --vnc-config /content/repo/configs/eval-brain-vnc-config.json \
    --out /content/results/dgate6-s1-299-f64
python -u $E/reflex_v1/eval_official.py --backend warp --envs-per-case 64 \
    --checkpoint /content/up/dgshuf6-s3-299.pt --vnc-config /content/repo/configs/shuffled-control-vnc-config.json \
    --out /content/results/dgshuf6-s3-299-f64
```

No projectiles are thrown in these runs, so the reflex never fires. The published trouble numbers were
measured with the same checkpoints and the reflex block left out.

## 3. Top speed and thrown balls

```bash
B="--checkpoint /content/up/dgate6-s1-299.pt --vnc-config /content/repo/configs/eval-brain-vnc-config.json"
python -u $E/reflex_v1/eval_official.py --backend warp --envs-per-case 64 --only nominal \
    --commands walk,run,sprint,sprint12,sprint15,sprint18,sprint20 --out /content/results/official-sprint20-f64
python -u $E/reflex_v1/eval_official.py --backend warp --envs-per-case 64 --only nominal \
    --commands walk,run,sprint,sprint12,sprint15,sprint18,sprint20 $B --out /content/results/dgate6-sprint-f64
for f in 0.3 0.4 0.6 0.9; do
  python -u $E/vnc_v1/eval_escape.py --robots-per-case 32 --flight $f --out /content/results/escape-official-f$f
  python -u $E/vnc_v1/eval_escape.py --robots-per-case 32 --flight $f $B --out /content/results/dgate6-escape-f$f
done
```

## 4. The scoreboard

```bash
R=/content/results
python -I $E/vnc_v1/scoreboard.py --name "dgate6-s1 299 + reflex" \
    --troubles $R/official-f64 $R/dgate6-s1-299-f64 \
    --sprint $R/official-sprint20-f64 $R/dgate6-sprint-f64 \
    --escape $R/escape-official-f0.3:$R/dgate6-escape-f0.3 $R/escape-official-f0.4:$R/dgate6-escape-f0.4 \
             $R/escape-official-f0.6:$R/dgate6-escape-f0.6 $R/escape-official-f0.9:$R/dgate6-escape-f0.9
```

The published summaries are in `results/eval/`, and the published report is
`results/scoreboard-dgate6-299-gfmja.md`.

## 5. Training the conductor (dgate6)

dgate6 was fine-tuned from `dgate5-s2-399.pt` (in the weights zip). Earlier rounds that led to dgate5 are not
part of this release.

```bash
python -u -B $E/vnc_v1/distill_vnc.py --mode gate --learn-synapses \
    --packet /content/up/vnc_ge5.npz --layout /content/up/layout.json \
    --experts /content/up/mlp-d-75100-weights.pt /content/up/mlp-b-75200-weights.pt /content/up/sprint16-75399.pt \
              /content/up/mlpRR-75100.pt /content/up/mlpRR-75100.pt#mirror \
              /content/up/mlpLM-75200.pt /content/up/mlpLM-75200.pt#mirror \
              /content/up/mlpLMslow-75599.pt /content/up/mlpLMslow-75599.pt#mirror \
    --adapters five_foot five_foot_mirror --adapter-vmax 0.6 --obs-mirror /content/up/mirror_actor.npz \
    --front-left 10 --front-right 11 --rear-right 4 --rear-left 5 --rear-always \
    --middle-left 8 --middle-right 9 --middle-left-slow 8 --middle-right-slow 9 \
    --middle-left-turn 6 --middle-right-turn 7 --turn-rate 0.5 --grip 1 --sprint 3 --sprint-speed 1.35 \
    --ladder-top 1.6 --slow-official 0.45 --init /content/up/dgate5-s2-399.pt --init-map 0,1,2,3,4,5,6,7,10,11 \
    --num-envs 4096 --iterations 300 --lr 1e-3 --seed 61 --out /content/results/dgate6-s1
```

For the control, add `--shuffle-seed 2`, `--seed 63` and `--init /content/up/dgshuf5-s3-399.pt`. The flags
each run used are also written into its `vnc-config.json`. Compare with the configs above.

## 6. The cord packet from MaleCNS

`vnc_ge5.npz` and `layout.json` are in the weights zip. `experiments/vnc_v1/build_packet.py` shows how they
were made. It reads the MaleCNS v1.0 body annotations (`body-annotations-male-cns-v1.0-minconf-0.5.feather`)
and neurotransmitter predictions (`body-neurotransmitters-male-cns-v1.0.feather`) from
<https://male-cns.janelia.org/download/>. It also reads a whole-CNS contact-count matrix
(`classified-all-contact-counts.npz`, `body-ids.npy`) from an earlier inventory step that is not in this
repository. Rebuilding the packet from scratch means first building that matrix from the MaleCNS
connectivity table.

## 7. The film

See [film.md](film.md). The render scripts in `film/previz/` assume a Google Drive layout
(`/content/drive/MyDrive/jumper-flybrain-colab/...`) holding the recorded takes, the PLATEAU tiles and the
asset caches. The takes (several hundred MB) are not in this release. Ask if you need them.
