# Colab G4, from a fresh runtime (Drive mounted): everything one machine does for the round-1 short C.
# Sets up Blender 5.0.1, the official jumper repo at the pinned SHA, PLATEAU and the cached CC0 assets
# from Drive; renders its shard of the six alley shots, then its shard of the aerial; with CUT=1 it then
# waits for every frame and cuts the film (cut_short.sh).
# (10/28 afternoon: two containers were recreated mid-render after an unanswered "still there?" check;
# frames already on Drive are skipped, so this can simply be run again.)
# Usage: bash run_all_colab.sh <i/3>
set -u
I=/content/drive/MyDrive/jumper-flybrain-colab
B=$I/film-2026-10-28c
F=$I/inputs/2026-10-28-film
SH=$1
cd /content
date
if [ ! -x /content/blender/blender ]; then
  curl -sS -L -o blender.tar.xz https://download.blender.org/release/Blender5.0/blender-5.0.1-linux-x64.tar.xz \
    && tar -xf blender.tar.xz && mv blender-5.0.1-linux-x64 blender && rm blender.tar.xz
  apt-get -qq install -y libxi6 libxxf86vm1 libxfixes3 libxrender1 libgl1 libxkbcommon0 libsm6 > /dev/null 2>&1
fi
/content/blender/blender --version | head -1
[ -d jumper ] || git clone -q https://github.com/KingKongRobotics/jumper.git
git -C jumper checkout -q 61d065219fca767f3142c8f10aff59eae5a5a004 && echo JUMPER_OK
mkdir -p /content/out /content/up /content/plateau/x /content/repo
cp $I/takes/alley_parry_slim.npz $I/takes/alley_parry_t21_cord.npz /content/out/
cp $F/NotoSansJP-VF.ttf $F/node_positions.npz /content/up/
[ -d /content/ph_cache ] || tar -xf $F/caches.tar -C /content
if [ ! -d /content/plateau/x/13100_tokyo23-ku_2020_obj_3_op ]; then
  cp $I/inputs/plateau/plateau_scrubbed.tar /content/plateau_scrubbed.tar
  H=$(cut -d' ' -f1 $I/inputs/plateau/plateau_scrubbed.sha256); G=$(sha256sum /content/plateau_scrubbed.tar | cut -d' ' -f1)
  [ "$H" = "$G" ] && echo PLATEAU_SHA_OK || echo PLATEAU_SHA_BAD
  tar -xf /content/plateau_scrubbed.tar -C /content/plateau/x && rm /content/plateau_scrubbed.tar
fi
echo "SETUP_DONE $(date +%H:%M:%S)"
for s in establish parry volley wound run shutter; do
  cd /content/repo && unzip -oq $F/current.zip
  echo "== $s $(date +%H:%M:%S)"
  bash /content/repo/film/previz/render_short.sh $s $SH
done
echo "ALLEY_DONE $(date +%H:%M:%S)"
cd /content/repo && unzip -oq $F/current.zip
/content/blender/blender -b --factory-startup --python /content/repo/film/blender/city.py -- \
  --plateau /content/plateau/x/13100_tokyo23-ku_2020_obj_3_op --keep-textures --aerial 300 \
  --out /content/out/aerial_final.blend --movie $B/aerial --shard $SH --res 1920x1080 --samples 128 \
  > /content/out/aerial_final.log 2>&1
grep -E "aerial:|Traceback|rror" /content/out/aerial_final.log | tail -5
echo "AERIAL_DONE $(date +%H:%M:%S)"
if [ -n "${CUT:-}" ]; then bash /content/repo/film/previz/cut_short.sh; fi
