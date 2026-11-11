"""usage: python sheet_input_vs_mesh.py <inputs_dir> <render_prefix> <out.jpg>
Top row: reference photos. Bottom row: flat-lit orthographic renders of the mesh from the same sides (+ underside)."""
import os, sys
import numpy as np
from PIL import Image

inp, pfx, out = sys.argv[1:4]

def fit(im, size=640, rot=0):
    im = im.convert("RGBA").rotate(rot, expand=True)
    ys, xs = np.where(np.array(im)[..., 3] > 20)
    im = im.crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))
    sc = (size - 30) / max(im.size)
    im = im.resize((int(im.width * sc), int(im.height * sc)), Image.LANCZOS)
    c = Image.new("RGB", (size, size), "white")
    c.paste(im, ((size - im.width) // 2, (size - im.height) // 2), im)
    return c

pairs = [("front", "az000", 0), ("left", "az090", 0), ("back", "az180", 0), ("right", "az270", 0), ("top", "top", 180)]
s = Image.new("RGB", (6 * 640, 2 * 640), "white")
for i, (k, v, rot) in enumerate(pairs):
    if os.path.exists(f"{inp}/{k}.png"):
        s.paste(fit(Image.open(f"{inp}/{k}.png")), (i * 640, 0))
    s.paste(fit(Image.open(f"{pfx}_{v}.png"), rot=rot), (i * 640, 640))
s.paste(fit(Image.open(f"{pfx}_bottom.png")), (5 * 640, 640))
s.save(out, quality=90)
