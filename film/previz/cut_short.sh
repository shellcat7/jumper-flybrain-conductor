# Colab: cut the round-1 short once every shot's frames are on Drive (version C 10/28 afternoon, D in the
# evening: the speed ramps, the dressed streets, the weave). Waits for the frames, unpacks the overlays and
# sound prepped on the PC (post_<v>.zip), runs `edit_short.py cut` and copies the 1440p and 1080p files to
# Drive film-2026-10-28c/final/.
# Usage: [VER=D] bash cut_short.sh
set -u
# (count exact names only: two machines rendering one frame leave Drive's 'frame_0075 (1).png' copies)
I=/content/drive/MyDrive/jumper-flybrain-colab
B=$I/film-2026-10-28c
V=${VER:-D}
v=$(echo "$V" | tr 'A-Z' 'a-z')
declare -A NEED=([aerial]=300 [establish]=78 [parry]=109 [volley]=81 [wound]=95 [runweave]=101 [shutterweave]=81)
for s in "${!NEED[@]}"; do
  until [ "$(ls $B/$s 2>/dev/null | grep -cE '^frame_[0-9]{4}[.]png$')" -ge "${NEED[$s]}" ]; do sleep 30; done
  echo "frames ok: $s"
done
sleep 30  # the last files' writes settle on Drive
rm -rf /content/post_c /content/cut_c
mkdir -p /content/post_c /content/cut_c $B/final
cd /content/post_c && unzip -oq $B/post_$v.zip
cd /content/repo && unzip -oq $I/inputs/2026-10-28-film/current.zip
python3 /content/repo/film/post/edit_short.py cut --renders $B --post /content/post_c/post_$v --work /content/cut_c \
  --font /content/up/NotoSansJP-VF.ttf --out /content/cut_c/jumper_round1_short_$V.mp4 --force
cp /content/cut_c/jumper_round1_short_$V.mp4 /content/cut_c/jumper_round1_short_${V}_1080p.mp4 $B/final/
ffmpeg -v error -y -i $B/final/jumper_round1_short_${V}_1080p.mp4 -vf scale=1280:-2 -c:v libx264 -crf 26 -preset slow \
  -c:a aac -b:a 128k -movflags +faststart $B/final/jumper_round1_short_${V}_preview720.mp4
ffmpeg -v error -y -i $B/final/jumper_round1_short_${V}_1080p.mp4 -vf fps=2/3,scale=384:-2,tile=6x6 -frames:v 1 $B/final/${V}_sheet.jpg
ls -la $B/final
echo CUT_DONE
