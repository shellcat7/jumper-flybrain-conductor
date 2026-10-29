# Colab: after this machine's own shards (run_all_colab.sh), render whatever frames are still missing on
# Drive -- a machine that died or stalled leaves its shard to the others. Each shot (and then the aerial)
# that is short of frames is rendered from its last frame back, skipping frames already there, so a
# machine still working forward on it meets this one in the middle.
# Usage: [FORWARD=1] [ONLY="parry establish aerial"] bash rescue_colab.sh <run_all log>   (waits for AERIAL_DONE in it; FORWARD=1 on a
# second rescuing machine, so the two meet in the middle instead of rendering the same frames)
set -u
I=/content/drive/MyDrive/jumper-flybrain-colab
B=$I/film-2026-10-28c
F=$I/inputs/2026-10-28-film
until grep -q AERIAL_DONE "$1"; do sleep 30; done
DIR=--reverse; [ -n "${FORWARD:-}" ] && DIR=""
declare -A NEED=([establish]=78 [parry]=109 [volley]=81 [wound]=95 [run]=66 [shutter]=105)
for s in ${ONLY:-shutter run wound volley parry establish}; do
  [ "$s" = aerial ] && continue
  if [ "$(ls $B/$s 2>/dev/null | grep -cE '^frame_[0-9]{4}[.]png$')" -lt "${NEED[$s]}" ]; then
    cd /content/repo && unzip -oq $F/current.zip
    echo "== rescue $s $(date +%H:%M:%S)"
    EXTRA=$DIR bash /content/repo/film/previz/render_short.sh $s 0/1
  fi
done
if [ -z "${ONLY:-}" -o -n "$(echo " ${ONLY:-} " | grep " aerial ")" ] && [ "$(ls $B/aerial 2>/dev/null | grep -cE '^frame_[0-9]{4}[.]png$')" -lt 300 ]; then
  cd /content/repo && unzip -oq $F/current.zip
  echo "== rescue aerial $(date +%H:%M:%S)"
  /content/blender/blender -b --factory-startup --python /content/repo/film/blender/city.py -- \
    --plateau /content/plateau/x/13100_tokyo23-ku_2020_obj_3_op --keep-textures --aerial 300 \
    --out /content/out/aerial_rescue.blend --movie $B/aerial --shard 0/1 $DIR --res 1920x1080 --samples 128 \
    > /content/out/aerial_rescue.log 2>&1
fi
echo "RESCUE_DONE $(date +%H:%M:%S)"
