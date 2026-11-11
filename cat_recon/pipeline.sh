#!/bin/bash
# usage: pipeline.sh <refs_dir> <out_dir> [faces]
# refs_dir: generated turnaround views on a plain backdrop (png/jpg/webp), named
#   front left back right            whole body ("left" = the cat faces the left edge of the image)
#   top                              optional, from above
#   head_front head_left head_right  optional close-ups of the head (used for the head's shape and texture)
# Writes <out_dir>/model.glb, <out_dir>/texture.png and <out_dir>/timing.tsv (stage, seconds, peak GPU MiB).
# See README.md for the reference-image prompt, the environment and how to check the result.
set -euo pipefail
W=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
[ -f "$W/env.sh" ] && source "$W/env.sh"
PY=${CATRECON_PYTHON:?set CATRECON_PYTHON (see env.example.sh)}
[ $# -ge 2 ] || { sed -n 2,8p "${BASH_SOURCE[0]}"; exit 2; }
refs=$(realpath "$1"); out=$(realpath -m "$2"); faces=${3:-300000}
for v in front left back right; do
    ls "$refs/$v".* > /dev/null 2>&1 || { echo "missing reference view: $refs/$v.*"; exit 2; }
done
mkdir -p "$out"/{cut,views,views_shape,views_head,body,head,merged,tex}
: > "$out/gpu.log"
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -lms 500 >> "$out/gpu.log" &
smi=$!; trap 'kill $smi 2>/dev/null || true' EXIT
: > "$out/timing.tsv"
stage() {   # stage <name> <command...>: run, record wall time and peak GPU memory
    local name=$1; shift
    local l0; l0=$(wc -l < "$out/gpu.log"); local t0; t0=$(date +%s.%N)
    "$@" > "$out/log_$name.txt" 2>&1 || { tail -20 "$out/log_$name.txt"; echo "stage $name failed, see $out/log_$name.txt"; exit 1; }
    local t1; t1=$(date +%s.%N)
    local peak; peak=$(tail -n +"$((l0 + 1))" "$out/gpu.log" | sort -n | tail -1)
    printf "%s\t%.1f\t%s\n" "$name" "$(echo "$t1 - $t0" | bc)" "${peak:-0}" | tee -a "$out/timing.tsv"
}
shape_inputs() {
    for v in front left back right top; do [ -f "$out/cut/$v.png" ] && cp "$out/cut/$v.png" "$out/views/$v.png"; done
    for v in front left back right; do "$PY" -I "$W/dewhisker_rgb.py" "$out/views/$v.png" "$out/views_shape/$v.png" 4; done
    if [ -f "$out/cut/head_front.png" ] && [ -f "$out/cut/head_left.png" ] && [ -f "$out/cut/head_right.png" ]; then
        for v in front left right; do cp "$out/cut/head_$v.png" "$out/views_head/$v.png"; done
        echo close > "$out/views_head/kind"
    else            # no close-ups: crop the head out of the whole-body views
        "$PY" -I - "$out" <<'Q'
import sys, numpy as np
from PIL import Image
out = sys.argv[1]
def crop(k, fx0, fx1, fy1):
    p = Image.open(f"{out}/views/{k}.png"); a = np.array(p)[..., 3] > 127; ys, xs = np.where(a)
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max(); W, H = x1 - x0, y1 - y0
    p.crop((int(x0 + fx0 * W), y0 - 4, int(x0 + fx1 * W), int(y0 + fy1 * H))).save(f"{out}/views_head/{k}.png")
crop("front", 0, 1, 0.45); crop("left", 0, 0.27, 0.45); crop("right", 0.73, 1, 0.45)
Q
        echo crop > "$out/views_head/kind"
    fi
}
cd "$out"       # the Hunyuan3D code writes scratch files to the working directory
stage cutout "$PY" -I "$W/prep_all.py" "$refs" "$out/cut"
stage shape_inputs shape_inputs
stage shapes "$PY" "$W/gen_shapes.py" 512 "$out/views_shape" "$out/body" "$out/views_head" "$out/head"
if [ "$(cat "$out/views_head/kind")" = close ]; then
    stage merge_head "$PY" "$W/merge_head.py" "$out/body" "$out/head" "$out/views" "$out/merged" --open 3 --head-crop 0.6
    stage texture "$PY" "$W/texture_photos.py" "$out/views" "$out/merged/shape_raw.glb" "$out/tex" --faces "$faces" --head-prompts "$out/views_head" --head-close "$out/views_head"
else
    stage merge_head "$PY" "$W/merge_head.py" "$out/body" "$out/head" "$out/views" "$out/merged" --open 3
    stage texture "$PY" "$W/texture_photos.py" "$out/views" "$out/merged/shape_raw.glb" "$out/tex" --faces "$faces" --head-prompts "$out/views_head"
fi
cp "$out/tex/textured.glb" "$out/model.glb"; cp "$out/tex/texture.png" "$out/texture.png"
grep "^\[lap\]" "$out/log_texture.txt" || true
awk -F'\t' '{s += $2} END {printf "total\t%.1f s\n", s}' "$out/timing.tsv"
echo "model: $out/model.glb   (check it: review.sh $out)"
