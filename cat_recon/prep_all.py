"""Segment the animal in every reference image, crop square with margin, save RGBA.
usage: python prep_all.py <refs_dir> <out_dir> [model]
One process, one model load for all images (the per-image version spent most of its time starting up)."""
import os, sys, time
os.environ.setdefault("OMP_NUM_THREADS", str(len(os.sched_getaffinity(0))))
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from rembg import remove, new_session

src, dst = sys.argv[1], sys.argv[2]
model = sys.argv[3] if len(sys.argv) > 3 else "birefnet-general"
os.makedirs(dst, exist_ok=True)
t = time.time()
session = new_session(model)
print("model load s", round(time.time() - t, 1))
for name in sorted(os.listdir(src)):
    if not name.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
        continue
    t = time.time()
    im = Image.open(f"{src}/{name}").convert("RGB")
    im.thumbnail((2048, 2048))
    arr = np.array(remove(im, session=session, post_process_mask=True))
    lab, n = ndi.label(arr[..., 3] > 128)
    big = lab == (np.argmax(ndi.sum(arr[..., 3] > 128, lab, range(1, n + 1))) + 1)
    arr[..., 3] = np.where(big, arr[..., 3], 0)
    a = arr[..., 3]
    ys, xs = np.where(a > 128)
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    side = int(max(x1 - x0, y1 - y0) * 1.12)
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    canvas = Image.new("RGBA", (side, side), (0, 0, 0, 0))
    canvas.paste(Image.fromarray(arr), (side // 2 - cx, side // 2 - cy))
    canvas.thumbnail((1024, 1024))
    out = f"{dst}/{os.path.splitext(name)[0]}.png"
    canvas.save(out)
    print(out, canvas.size, "fg frac %.3f" % (a > 128).mean(), "s %.1f" % (time.time() - t), flush=True)
