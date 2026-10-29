# How the film was made

The film is 37.9 s at 1920×1080 and 25 fps. All of the robot's motion comes from one simulation recording.
Everything else (the set, the drones, the camera, the sound) was built around that recording afterwards.

## 1. The recording (`experiments/vnc_v1/record_scene.py`, scenario `alley_parry`)

24 robots run the same script side by side with the film brain (`configs/film-brain-vnc-config.json`:
dgate6-s1-299 with the reflex and the claw guard). Each one's bodies are logged at 200 Hz, along with the
cord's activity and the projectiles. Take 21 was used.

| time | event | what the controller does |
|---|---|---|
| 1.5 s | a bolt from the left | sidestep |
| 2.8 s | a bolt from the front | claw guard; the bolt glances off and goes back about 160° (3.2 s) |
| 4.3 s | three bolts at once (front, back left, right) | step that clears all three lines, ducking 3 cm |
| 5.8 s | a bolt from the left | sidestep |
| 6.6 s | a wave of four | the official jump policy |
| 7.3 s | a bolt hits the left claw on landing; the script sets that leg's torque to zero | the conductor, from its body signals, switches to the five-legged gait |
| 7.4 s | command: run at 0.6 m/s | |

The last part was recorded again (`film/sim/run_weave.py`, variant c) with the same seed and a weaving
command from 8.0 s. Take 21 stayed within 1.4 cm of the first recording up to 7.4 s. The weave is a
command. The bins and the dumpster were placed in Blender where the path swerved hardest
(`film/blender/props.py: obstacles()`), so they were not in the simulation.

## 2. Rendering (Blender 5.0.1, Cycles, Colab GPUs)

- `film/blender/import_take.py`: rebuilds the robot from the official MJCF and keys each body's recorded
  pose. Nothing in the robot is animated by hand.
- `alley.py`, `photoreal.py`, `detail.py`, `props.py`: the back-alley set, dressed with CC0 scans and textures
  from Poly Haven. Signs are generic Japanese words drawn for this film (`film/blender/signs/`).
- `drone.py`, `fx.py`: each drone is placed where its bolt was launched, flies between its launches, and
  chases the robot after the last volley. The bolts follow the recorded projectiles, with light and sparks
  added.
- `shutter.py`: the roller shutter and the drone that hits it are keyframed.
- `city.py`: the opening aerial over Shinjuku at night, from Project PLATEAU 3D city models (signs in the
  photo textures blurred by `film/post/sign_scrub.py`), with footways, kerbs, lane lines, trees and cars
  added from PLATEAU's road outlines.

Render scripts: `film/previz/render_short.sh` (one shot, split across machines by `--shard i/n`),
`run_all_colab.sh`, `rescue_colab.sh`, `finish_colab.sh`.

## 3. Post (`film/post/`)

- `brain_inset.py`: the upper-right inset. All 24,154 nodes are drawn where their neurons sit in the MaleCNS
  nerve cord. Each node's strength is how far its recorded state (the take's 50 Hz cord log) is above its own
  mean over the take, in its own standard deviations, averaged over a quarter second. Dots are sized and
  coloured by strength (blue, cyan, yellow, white), and faint nodes are left out. Below the inset is the
  motor program the conductor has chosen at that moment.
- `edit_short.py`: the HUD (lower left: six leg lamps, lit while each foot is down, and the playback factor),
  the legend card, the end card and the cut, with speed ramps per shot (from x0.125 round the parry to x2 on
  the run).
- `sound.py`: everything is synthesised with numpy from the recorded events (footfalls from foot contacts,
  servo whine from joint load, bolts, deflections, hits). No sound samples are used.

The cut itself runs on Colab with ffmpeg: `film/previz/cut_short.sh`.

## 4. What was keyframed or placed by hand

- the drones' flight (their launch points and timing are the simulation's projectiles);
- the roller shutter and the drone that crashes into it;
- the camera;
- the bins and the dumpster on the final run;
- light, sparks and smoke on the bolts.
