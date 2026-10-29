# Colab G4: the round-1 short's shots from the alley_parry take (take 21 of
# runs/vnc-v1/scenes/alley_parry_gfmgc: a side bolt stepped from, a lone bolt parried 160 degrees back,
# three at once ducked under, the wave that wounds the left claw on landing, the run on five legs), in
# Cycles with the photo-real set (film/blender/photoreal.py) and the scanned props (props.py). Frames go
# to Drive; several machines share a shot by --shard i/n, and frames already there are skipped.
# Usage: bash render_short.sh <establish|parry|volley|wound|run|shutter> [i/n] [first,last]
#
# (10/28 afternoon, the user: the enemy was never in sight, the layout unreadable, the robot seemed to
# walk and lie down on its own, the ground blotchy and blurred.) Every shot now frames the drones with
# the robot, the lens is stopped down (f/5.6-8) so the alley reads, fewer pebbles, scanned props, and
# after their last volley the drones chase the robot down the alley (and climb away before the shutter).
set -u
S=$1
SHARD=${2:-0/1}
I=/content/drive/MyDrive/jumper-flybrain-colab
OUT=${OUTDIR:-$I/film-2026-10-28c/$S}  # (28b: the takes with the smaller drones)
mkdir -p "$OUT" /content/out
COMMON="--scene ${SCENE:-/content/out/alley_parry_slim.npz} --take 21 --mjcf /content/jumper/assets/jumper/jumper.xml
  --projectile laser --impacts --drones --drone-style hostile --drone-scale 3.9 --chase --pull-up 9.9 --set backalley --engine cycles
  --follow --cam-local --res ${RES:-1920x1080} --haze 0 --photoreal /content/ph_cache
  --props /content/props_cache --pebbles 1500 --shard $SHARD"
case $S in
  # frame f of --every E is control step (f - 1) * E (200 Hz): t = (f - 1) * E / 200 s
  # the alley from behind and above, real time: the robot walking in, the drones dropping in from above,
  # the side bolt it steps from (1.5 s), the front drone's shot (2.8 s) -- cut as the bolt leaves
  establish) ARGS="--every 8 --range 1,78 --cam-offset=-3.2,-0.35,1.55 --aim-offset=1.4,0.3,0 --lens 24 --dof 8" ;;
  # over the robot's shoulder at the front drone 3 m ahead, 1/8 speed: the bolt comes at it, the claw
  # turns it 160 degrees back (3.2 s)
  parry)  ARGS="--every 1 --range 573,681 --cam-offset=-0.95,0.42,0.24 --aim-offset=1.3,-0.1,0.05 --lens 32 --dof 5.6" ;;
  # from above, 1/4 speed: three drones round it fire at once (4.3 s); it ducks and all three pass over
  volley) ARGS="--every 2 --range 411,491 --cam-offset=-1.2,-0.7,3.1 --aim-offset=0.3,0.1,0 --lens 16" ;;
  # the jump over the four-bolt wave (6.6 s), bullet time at its apex (7.1 s, step 1419) swinging round,
  # wide enough for the drones round it, then the bolt that catches the left claw (7.3 s) and the landing
  wound)  ARGS="--every 4 --range 345,390 --cam-offset=-1.25,0.9,0.42 --aim-offset=0,0,0.12 --lens 24 --dof 8 --bullet-time 356,50,-100 --follow-smooth 0.45" ;;
  # ahead of it, looking back up the alley: the run on five legs with the drones on its tail
  run)    ARGS="--every 8 --range 185,250 --cam-offset=1.5,-0.35,0.32 --aim-offset=-1.2,0,0.15 --lens 28 --dof 8" ;;
  # (10/28 evening, the user: the run was a straight line, why it gets away unclear -- weave round bins and
  # dumpsters) the re-recorded take (film/sim/run_weave.py, SCENE=...): from ahead and above, the weave
  # between the obstacles with the drones overhead; the alley laid out from the first take
  runweave) ARGS="--every 8 --range 185,285 --cam-offset=2.3,0.25,1.75 --aim-offset=-0.9,0,0.35 --lens 24 --dof 8 --obstacles 7.8,11.4 --alley-ref /content/out/alley_parry_slim.npz" ;;
  shutterweave) ARGS="--every 4 --range 569,649 --shutter ${SHUTTER_T:-11.75} --cam-shutter --lens 28 --dof 5.6 --obstacles 7.8,11.4 --alley-ref /content/out/alley_parry_slim.npz" ;;
  # the ending (shutter.py): a still camera past the closing shutter, half speed; the robot's base
  # passes under it at 11.75 s, the diving drone hits the curtain half a second later
  shutter) ARGS="--every 4 --range 545,649 --shutter 11.75 --cam-shutter --lens 28 --dof 5.6" ;;
  *) echo "unknown shot $S"; exit 1 ;;
esac
# (10/28 evening: one G4 came back as an A100) the run and the shutter, not yet begun, at 128 samples
case $S in run|shutter|runweave|shutterweave) SHOT_SPP=128 ;; *) SHOT_SPP=192 ;; esac
ARGS="$ARGS --samples ${SPP:-$SHOT_SPP}"
if [ -n "${3:-}" ]; then ARGS=$(echo "$ARGS" | sed "s/--range [0-9]*,[0-9]*/--range $3/"); fi
if [ -n "${NOBULLET:-}" ]; then ARGS=$(echo "$ARGS" | sed "s/--bullet-time [^ ]*//"); fi  # a test still
/content/blender/blender -b --factory-startup --python /content/repo/film/blender/import_take.py -- \
  $COMMON $ARGS ${EXTRA:-} --out /content/out/$S.blend --movie "$OUT" > /content/out/${S}_${SHARD/\//of}_full.log 2>&1
CODE=$?
grep -E "\[render\]|\[photoreal\]|\[detail\]|\[props\]|FILM_|Traceback|Error:|rror:" /content/out/${S}_${SHARD/\//of}_full.log | tail -20
echo "SHOT_DONE $S $SHARD exit $CODE"
