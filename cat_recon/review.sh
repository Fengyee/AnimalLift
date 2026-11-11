#!/bin/bash
# usage: review.sh <out_dir of pipeline.sh>
# Flat-lit orthographic renders for checking a result against its references, in <out_dir>/review/:
#   input_vs_mesh.jpg  references (top row) and the model from the same sides (bottom row)
#   body1.jpg body2.jpg  eight azimuths      tb.jpg  top and underside
#   head_front.jpg head_back.jpg  close-ups of the head at 1100 px per view
set -euo pipefail
W=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
[ -f "$W/env.sh" ] && source "$W/env.sh"
PY=${CATRECON_PYTHON:?set CATRECON_PYTHON}; B=${BLENDER:?set BLENDER}; H=${CATRECON_HDR:-none}
out=$(realpath "$1"); glb=$out/model.glb; r=$out/review; mkdir -p "$r/r"
"$B" -b --python-exit-code 1 --python "$W/render_glb.py" -- "$glb" "$r/r/f" "$H" 2000 16 flat az000,az045,az090,az135,az180,az225,az270,az315,top,bottom > "$r/blender_body.log" 2>&1
"$B" -b --python-exit-code 1 --python "$W/render_glb.py" -- "$glb" "$r/r/h" "$H" 1400 48 flat az000,az045,az315,az090,az270,az180,az135,az225,top head > "$r/blender_head.log" 2>&1
"$PY" -I - "$r" <<'Q'
import sys, numpy as np
from PIL import Image
r = sys.argv[1]
def crop(v):
    im = Image.open(f"{r}/r/f_{v}.png").convert("RGBA")
    ys, xs = np.where(np.asarray(im)[..., 3] > 20)
    c = Image.new("RGB", im.size, "white"); c.paste(im, (0, 0), im)
    return c.crop((xs.min() - 10, ys.min() - 10, xs.max() + 10, ys.max() + 10))
def sheet(views, name):
    ims = [crop(v) for v in views]
    s = Image.new("RGB", (sum(i.width for i in ims), max(i.height for i in ims)), "white"); x = 0
    for i in ims: s.paste(i, (x, 0)); x += i.width
    s.save(f"{r}/{name}.jpg", quality=90)
sheet(["az000", "az045", "az090", "az135"], "body1"); sheet(["az180", "az225", "az270", "az315"], "body2"); sheet(["top", "bottom"], "tb")
for name, V in [("head_front", ["az045", "az000", "az315", "az090", "az270"]), ("head_back", ["az135", "az180", "az225", "top"])]:
    s = Image.new("RGB", (len(V) * 1100, 1100), "white")
    for i, v in enumerate(V):
        im = Image.open(f"{r}/r/h_{v}.png").crop((150, 150, 1250, 1250)); s.paste(im, (i * 1100, 0), im)
    s.save(f"{r}/{name}.jpg", quality=90)
Q
"$PY" -I "$W/sheet_input_vs_mesh.py" "$out/views" "$r/r/f" "$r/input_vs_mesh.jpg"
echo "review sheets: $r"
