"""Graft a separately generated, detailed head onto the whole-body shape, in implicit space.

usage: python merge_head.py <body_dir> <head_dir> <photos_dir> <out_dir> [--open R] [--res N]

body_dir / head_dir each hold grid.npy from gen_shape.py (occupancy logits on [-1.01, 1.01]^3,
positive inside; x = cat's left, y = up, z = forward). The whole-body run sees the face at
~100 px, so its head is a blob; the head run sees only head + neck and resolves eyes, nose, ears.

1. clean the head field: 3D opening removes whiskers / ear-hair spikes, keep the main component
2. place the head in the body frame: scale + translation (no rotation, both runs are fed exact
   front/side views) maximising silhouette overlap with the front/left/right photos
3. blend the two fields across an oblique plane through the neck and run marching cubes
"""
import argparse, os
import numpy as np, torch, torch.nn.functional as F, trimesh
from PIL import Image
from scipy import ndimage as ndi
from scipy.optimize import minimize
from skimage import measure

ap = argparse.ArgumentParser()
ap.add_argument("body"); ap.add_argument("head"); ap.add_argument("photos"); ap.add_argument("out")
ap.add_argument("--open", type=int, default=2, help="opening radius (head voxels) that removes whiskers")
ap.add_argument("--smooth-px", type=int, default=5, help="how far around removed hairs the head field is smoothed")
ap.add_argument("--smooth-sigma", type=float, default=1.5)
ap.add_argument("--res", type=int, default=1024, help="voxels per 2.02 body units in the merged grid")
ap.add_argument("--head-crop", type=float, default=0.45, help="fraction of the cat's height the head run was cropped to")
ap.add_argument("--cut", type=float, nargs=2, default=(0.46, 0.56),
                help="blend band: head where (dist behind nose + dist below ear tips) < cut[0]*height, body beyond cut[1]")
ap.add_argument("--no-hull", action="store_true", help="do not confine the head to the whole-body side silhouettes")
ap.add_argument("--hull-sliver", type=int, default=4, help="excess thinner than ~2x this (merged voxels) is registration noise and ignored")
ap.add_argument("--hull-reach", type=int, default=24, help="merged voxels over which the surface is handed back to the whole-body shape")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
dev = "cuda"
BOUND, N = 1.01, 512


def idx2c(i):
    return i / N * 2 * BOUND - BOUND


def sample(vol, x, y, z):
    """Trilinear lookup of vol[X,Y,Z] (torch, on its own [-1.01,1.01]^3 box) at points given in that box."""
    g = torch.stack([z, y, x], -1)[None, None] / BOUND                # grid_sample wants (W,H,D) = (Z,Y,X)
    return F.grid_sample(vol[None, None], g, mode="bilinear", padding_mode="border", align_corners=True)[0, 0, 0]


def slope(g):
    """Median |grad| (per voxel) of a logits grid near its zero level."""
    c = g[1:-1:2, 1:-1:2, 1:-1:2]
    near = np.abs(c) < 0.3
    gx = (g[2::2, 1:-1:2, 1:-1:2] - g[:-2:2, 1:-1:2, 1:-1:2]) / 2
    gy = (g[1:-1:2, 2::2, 1:-1:2] - g[1:-1:2, :-2:2, 1:-1:2]) / 2
    gz = (g[1:-1:2, 1:-1:2, 2::2] - g[1:-1:2, 1:-1:2, :-2:2]) / 2
    return float(np.median(np.sqrt(gx ** 2 + gy ** 2 + gz ** 2)[near]))


gb = np.load(f"{args.body}/grid.npy").astype(np.float32)
gh = np.load(f"{args.head}/grid.npy").astype(np.float32)

# ---------------------------------------------------------------- 1. clean the head field
occ = torch.from_numpy(gh > 0).to(dev)
o = occ[None, None].float()
r = args.open
er = 1 - F.max_pool3d(1 - o, 2 * r + 1, 1, r)
keep = (F.max_pool3d(er, 2 * r + 3, 1, r + 1)[0, 0] > 0.5) & occ
del o, er
keep = keep.cpu().numpy()
lab, n = ndi.label(keep)
if n > 1:
    keep = lab == (np.argmax(np.bincount(lab.ravel())[1:]) + 1)
del lab
occ = occ.cpu().numpy()
removed = occ & ~keep
print(f"head cleaning: removed {int(removed.sum())} of {int(occ.sum())} inside voxels")
gh[removed] = -1.0
# the cut leaves stubs where each hair met the skin: smooth the field around them
near = ndi.binary_dilation(removed, iterations=args.smooth_px)
gh[near] = ndi.gaussian_filter(gh, args.smooth_sigma)[near]
keep = gh > 0
del occ, removed, near

# ---------------------------------------------------------------- 2. register head -> body frame
ob = np.argwhere(gb > 0)
b_lo, b_hi = idx2c(ob.min(0)), idx2c(ob.max(0))
y_top, z_front, Hb = b_hi[1], b_hi[2], b_hi[1] - b_lo[1]
del ob
surf = keep & ~ndi.binary_erosion(keep)
P = idx2c(np.argwhere(surf).astype(np.float32))
del surf
P = P[np.random.default_rng(0).choice(len(P), min(len(P), 600000), replace=False)]
h_top, h_front, h_ext = P[:, 1].max(), P[:, 2].max(), P[:, 1].max() - P[:, 1].min()
s0 = args.head_crop * Hb / h_ext
x0 = np.array([np.log(s0), 0.0, y_top - s0 * h_top, z_front - s0 * h_front])

views = {}
for k in ("front", "left", "right"):
    m = np.asarray(Image.open(f"{args.photos}/{k}.png").convert("RGBA"))[..., 3] > 127
    ys, xs = np.where(m)
    top, hpx = ys.min(), ys.max() - ys.min()
    kpx = Hb / hpx                                       # body units per photo pixel
    if k == "front":
        head_rows = m[top:top + int(0.19 * hpx)]
        cols = np.where(head_rows.any(0))[0]
        views[k] = dict(m=m, top=top, kpx=kpx, ref=(cols.min() + cols.max()) / 2)
    else:
        views[k] = dict(m=m, top=top, kpx=kpx, ref=xs.min() if k == "left" else xs.max())


def project(k, X):
    """Body coords -> photo pixel coords of view k, plus the mask of points inside the head window."""
    v = views[k]
    py = v["top"] + (y_top - X[:, 1]) / v["kpx"]
    if k == "front":
        px = v["ref"] + X[:, 0] / v["kpx"]
        win = (y_top - X[:, 1]) < 0.19 * Hb
    else:
        back = (z_front - X[:, 2]) / v["kpx"]
        px = v["ref"] + back if k == "left" else v["ref"] - back
        win = (z_front - X[:, 2]) + (y_top - X[:, 1]) < 0.44 * Hb
    return px, py, win


def window_mask(k):
    v = views[k]; m = v["m"]
    yy, xx = np.mgrid[:m.shape[0], :m.shape[1]]
    down = (yy - v["top"]) * v["kpx"]
    if k == "front":
        return down < 0.19 * Hb
    back = (xx - v["ref"] if k == "left" else v["ref"] - xx) * v["kpx"]
    return back + down < 0.44 * Hb


WIN = {k: window_mask(k) for k in views}


def splat(k, X):
    px, py, win = project(k, X)
    m = views[k]["m"]
    h = np.zeros((m.shape[0] // 2 + 1, m.shape[1] // 2 + 1), bool)
    ix, iy = (px[win] / 2).astype(int), (py[win] / 2).astype(int)
    ok = (ix >= 0) & (iy >= 0) & (ix < h.shape[1]) & (iy < h.shape[0])
    h[iy[ok], ix[ok]] = True
    h = ndi.binary_fill_holes(ndi.binary_closing(h, iterations=2))
    return np.kron(h, np.ones((2, 2), bool))[:m.shape[0], :m.shape[1]] & WIN[k]


def ious(p):
    X = np.exp(p[0]) * P + p[1:]
    out = []
    for k in views:
        a, b = splat(k, X), views[k]["m"] & WIN[k]
        out.append((a & b).sum() / max((a | b).sum(), 1))
    return out


print("registration init  s=%.4f t=%s IoU=%s" % (np.exp(x0[0]), x0[1:].round(3), np.round(ious(x0), 3)))
res = minimize(lambda p: -float(np.mean(ious(p))), x0, method="Nelder-Mead",
               options=dict(initial_simplex=x0 + np.vstack([np.zeros(4), np.diag([0.05, 0.02, 0.02, 0.02])]),
                            xatol=1e-4, fatol=1e-5, maxiter=400))
s, t = float(np.exp(res.x[0])), res.x[1:]
print("registration final s=%.4f t=%s IoU=%s" % (s, t.round(3), np.round(ious(res.x), 3)))
np.save(f"{args.out}/head_to_body.npy", np.array([s, *t]))
for k in views:
    a, b = splat(k, s * P + t), views[k]["m"]
    dbg = np.full(b.shape + (3,), 255, np.uint8)
    dbg[b] = (200, 200, 200); dbg[a & b] = (90, 90, 90); dbg[a & ~b] = (255, 0, 0); dbg[b & WIN[k] & ~a] = (0, 200, 0)
    Image.fromarray(dbg).save(f"{args.out}/register_{k}.png")

# ---------------------------------------------------------------- 3. blend and extract
v = 2 * BOUND / args.res                                  # merged voxel size, body units
Xh = s * P + t
u = (z_front - Xh[:, 2]) + (y_top - Xh[:, 1])
Xh = Xh[u < args.cut[1] * Hb]                             # the part of the head run that will be used
lo = np.minimum(b_lo, Xh.min(0)) - 16 * v
hi = np.maximum(b_hi, Xh.max(0)) + 16 * v
dims = np.ceil((hi - lo) / v).astype(int) + 1
print("merged grid", dims, "voxel", round(v, 5))
ax = [torch.arange(d, device=dev, dtype=torch.float32) * v + float(l) for d, l in zip(dims, lo)]
tb, th = torch.from_numpy(gb).to(dev), torch.from_numpy(gh).to(dev)
kb = slope(gb) * (v / (2 * BOUND / N))                    # logits per merged voxel, body field
kh = slope(gh) * (v / (s * 2 * BOUND / N))                # ... head field
print("field slopes per merged voxel: body %.3f head %.3f" % (kb, kh))

E = np.empty(dims, np.float32)                            # signed distance in merged voxels, + inside
for i in range(dims[0]):
    yy, zz = torch.meshgrid(ax[1], ax[2], indexing="ij")
    E[i] = (sample(tb, torch.full_like(yy, float(ax[0][i])), yy, zz) / kb).cpu().numpy()

# box that contains the whole blend region
u_max = args.cut[1] * Hb + 8 * v
j0 = max(int((y_top - u_max - lo[1]) / v), 0)
k0 = max(int((z_front - u_max - lo[2]) / v), 0)
sub = (slice(None), slice(j0, dims[1]), slice(k0, dims[2]))
xx, yy, zz = torch.meshgrid(ax[0], ax[1][j0:], ax[2][k0:], indexing="ij")
Fb = torch.empty_like(xx); Fh = torch.empty_like(xx)
for i in range(xx.shape[0]):
    Fb[i] = sample(tb, xx[i], yy[i], zz[i])
    hx, hy, hz = (xx[i] - t[0]) / s, (yy[i] - t[1]) / s, (zz[i] - t[2]) / s
    inside = (hx.abs() < BOUND) & (hy.abs() < BOUND) & (hz.abs() < BOUND)
    Fh[i] = torch.where(inside, sample(th, hx, hy, hz), torch.full_like(hx, -1.0))
w = ((args.cut[1] * Hb - ((z_front - zz) + (y_top - yy))) / ((args.cut[1] - args.cut[0]) * Hb)).clamp(0, 1)
w = (w * w * (3 - 2 * w)).cpu().numpy()
del xx, yy, zz


def extended(Fl, k):
    """Logits saturate a few voxels from the surface; continue them with a distance transform so two
    surfaces that are far apart still blend into one smooth surface instead of a step."""
    f = Fl.cpu().numpy()
    inside = f > 0
    sd = np.where(inside, ndi.distance_transform_edt(inside) - 0.5, 0.5 - ndi.distance_transform_edt(~inside)).astype(np.float32)
    a = np.clip(1 - np.abs(f) / 0.9, 0, 1)
    a = a * a * (3 - 2 * a)
    return a * f / k + (1 - a) * sd


Eb, Eh = extended(Fb, kb), extended(Fh, kh)
# The head run never saw the neck: where its input was cut off it closes the shape with a rounded
# end, which shows as a hump on the back of the head. The whole-body side photos do show that
# outline (the whole-body run follows it). Where the head's side outline sticks out of the photos'
# (mean of the two side views, which never agree exactly) by more than a sliver, hand the surface
# back to the whole-body shape, with a soft transition. Carving the head to the outline instead
# would leave a flat shelf across the neck.
if not args.no_hull:
    yb, zb = ax[1][j0:].cpu().numpy(), ax[2][k0:].cpu().numpy()
    hull = 0
    for k in ("left", "right"):
        vw = views[k]
        sd = ndi.distance_transform_edt(vw["m"]) - ndi.distance_transform_edt(~vw["m"])        # px, + inside
        py = vw["top"] + (y_top - yb) / vw["kpx"]
        back = (z_front - zb) / vw["kpx"]
        px = vw["ref"] + back if k == "left" else vw["ref"] - back
        hull = hull + ndi.map_coordinates(sd, np.meshgrid(py, px, indexing="ij"), order=1, mode="nearest") * vw["kpx"] / v / 2
    out_ = (Eh > 0).any(0) & (hull < 0)                           # head outline beyond the photos', in the (y, z) plane
    r = args.hull_sliver
    thick = ndi.binary_dilation(ndi.binary_erosion(out_, iterations=r), iterations=r + 1) & out_
    D = ndi.gaussian_filter(ndi.binary_dilation(thick, iterations=args.hull_reach).astype(np.float32), args.hull_reach / 2)
    D = np.clip(D, 0, 1); D = D * D * (3 - 2 * D)
    print(f"side-photo outline: head sticks out on {int(out_.sum())} (y,z) cells, {int(thick.sum())} more than a sliver; "
          f"largest excess {float(-hull[thick].min()) if thick.any() else 0:.1f} voxels")
    Image.fromarray((np.stack([out_ * 255, thick * 255, D * 255], -1)).astype(np.uint8)[:, ::-1]).save(f"{args.out}/hull_excess.png")
    w = w * (1 - D)[None]
E[sub] = (1 - w) * Eb + w * Eh
del Eb, Eh, Fb, Fh, w
verts, faces, _, _ = measure.marching_cubes(E, 0.0, method="lewiner")
mesh = trimesh.Trimesh(verts * v + lo, faces[:, ::-1], process=False)
parts = mesh.split(only_watertight=False)
mesh = max(parts, key=lambda m: len(m.faces))
print("merged mesh", mesh.vertices.shape, mesh.faces.shape, "components dropped", len(parts) - 1)
mesh.export(f"{args.out}/shape_raw.glb")
