"""Make geometry-only inputs: erase thin bright lines (whiskers) from the RGB with a
grayscale opening so the shape model does not turn them into mesh spikes.
usage: python dewhisker_rgb.py <src.png> <dst.png> [radius_px]"""
import sys
import numpy as np
from PIL import Image
from scipy import ndimage as ndi

src, dst = sys.argv[1], sys.argv[2]
r = int(sys.argv[3]) if len(sys.argv) > 3 else 4
im = np.array(Image.open(src).convert("RGBA"))
yy, xx = np.ogrid[-r:r + 1, -r:r + 1]
disc = xx * xx + yy * yy <= r * r
rgb = im[..., :3].copy()
rgb[im[..., 3] < 128] = 255  # so the background never darkens the rim
for c in range(3):
    im[..., c] = ndi.grey_opening(rgb[..., c], footprint=disc)
Image.fromarray(im).save(dst)
