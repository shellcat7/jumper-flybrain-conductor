# Colab G4, fresh runtime with Drive mounted: finish the round-1 short C after the machines dropped out.
# Sets up (as run_all_colab.sh does), then renders every frame still missing on Drive -- backward, or
# forward with FORWARD=1 so two machines meet in the middle -- and with CUT=1 waits for all frames and
# cuts the film.
# (10/28 evening: all three runtimes were disconnected by Colab's idle timeout, with 122 of 834 frames
# left. Cells run from a script do not count as activity; only the user's own interaction does.)
# Usage: [FORWARD=1] [CUT=1] bash finish_colab.sh
set -u
I=/content/drive/MyDrive/jumper-flybrain-colab
F=$I/inputs/2026-10-28-film
cd /content
if [ ! -x /content/blender/blender ]; then
  curl -sS -L -o blender.tar.xz https://download.blender.org/release/Blender5.0/blender-5.0.1-linux-x64.tar.xz \
    && tar -xf blender.tar.xz && mv blender-5.0.1-linux-x64 blender && rm blender.tar.xz
  apt-get -qq install -y libxi6 libxxf86vm1 libxfixes3 libxrender1 libgl1 libxkbcommon0 libsm6 > /dev/null 2>&1
fi
nvidia-smi --query-gpu=name --format=csv,noheader
[ -d jumper ] || git clone -q https://github.com/KingKongRobotics/jumper.git
git -C jumper checkout -q 61d065219fca767f3142c8f10aff59eae5a5a004 && echo JUMPER_OK
mkdir -p /content/out /content/up /content/plateau/x /content/repo
cp $I/takes/alley_parry_slim.npz $I/takes/alley_parry_t21_cord.npz /content/out/
cp $F/NotoSansJP-VF.ttf $F/node_positions.npz /content/up/
[ -d /content/ph_cache ] || tar -xf $F/caches.tar -C /content
if [ ! -d /content/plateau/x/13100_tokyo23-ku_2020_obj_3_op ]; then
  cp $I/inputs/plateau/plateau_scrubbed.tar /content/plateau_scrubbed.tar
  tar -xf /content/plateau_scrubbed.tar -C /content/plateau/x && rm /content/plateau_scrubbed.tar
fi
cd /content/repo && unzip -oq $F/current.zip
echo "SETUP_DONE $(date +%H:%M:%S)"
echo "AERIAL_DONE (none to wait for)" > /content/finish_go.log
if [ -n "${CUT:-}" ]; then (bash /content/repo/film/previz/cut_short.sh > /content/cut.log 2>&1 &); fi
bash /content/repo/film/previz/rescue_colab.sh /content/finish_go.log
