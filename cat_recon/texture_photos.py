"""Texture a Hunyuan3D mesh by projecting the real reference photos onto it.

usage: python texture_photos.py <views_dir> <shape_raw.glb> <out_dir> [options]

Whole body: views front/left/back/right(/top) are warped onto the mesh silhouette of the matching
orthographic view (bbox fit, then a smooth non-rigid warp driven by the silhouettes) and
back-projected into the UV atlas. The Hunyuan paint model's 512 px generated views are too soft
to carry the fur pattern; they only cover what no photo shows (underside, inner thighs), levelled
to the photos' colours, and surfaces nothing sees are filled from their 3D neighbours.

Tail: a tail whose outline is interior to a view (seen from behind) is a separate layer, placed by
its own shift + stretch from DINOv2 correspondences, so a photo with the tail in another pose does
not paint tail onto legs.

Head (--head-prompts DIR): matching outlines says nothing about where the eyes, nose and mouth
land. In every view but the back one the head is therefore re-aligned in a zoomed camera against a view the
paint model generates for the head alone -- that view is pixel-aligned with the geometry, so it
shows where the mesh has its eyes and nose. DINOv2 correspondences between photo and generated
view (same parts, whatever the fur pattern) drive a smooth warp of the photo, faded out towards
the neck.

Intermediate results (UV-unwrapped mesh, generated views) are cached in <out_dir>.
"""
import argparse, os, pickle, time
import numpy as np, torch, torch.nn.functional as F, trimesh
from PIL import Image
from scipy import ndimage as ndi

from hy3dgen.shapegen import FloaterRemover, DegenerateFaceRemover, FaceReducer
from hy3dgen.texgen import Hunyuan3DPaintPipeline
from hy3dgen.texgen.differentiable_renderer.mesh_render import MeshRender
from hy3dgen.texgen.differentiable_renderer.camera_utils import transform_pos, get_mv_matrix
from hy3dgen.texgen.utils.uv_warp_utils import mesh_uv_wrap

_t0 = [time.time()]


def lap(name):
    """Stage timing for the efficiency report: wall time since the last lap and peak torch memory so far."""
    now = time.time()
    print(f"[lap] {name}: {now - _t0[0]:.1f} s, peak torch mem {torch.cuda.max_memory_allocated() / 2 ** 30:.1f} GiB", flush=True)
    _t0[0] = now


def uv_unwrap(mesh, parts=None):
    """xatlas UV atlas, computed in parallel: xatlas charts each input mesh on its own thread, so the
    mesh is handed over as spatially compact pieces (the cuts only add chart borders to the few
    thousand there are anyway) and packed into one atlas. ~5x faster than one piece on 8 cores."""
    import xatlas
    from scipy.cluster.vq import kmeans2
    V, Fc = np.asarray(mesh.vertices, dtype=np.float32), np.asarray(mesh.faces)
    parts = parts or len(os.sched_getaffinity(0))
    _, lab = kmeans2(V[Fc].mean(1), parts, seed=0, minit="++")
    atlas, used = xatlas.Atlas(), []
    for k in range(parts):
        f = Fc[lab == k]
        if len(f) == 0:
            continue
        u, inv = np.unique(f, return_inverse=True)
        atlas.add_mesh(V[u], inv.reshape(-1, 3).astype(np.uint32)); used.append(u)
    atlas.generate()
    verts, faces, uvs, n = [], [], [], 0
    for i, u in enumerate(used):
        vmap, idx, uv = atlas[i]
        verts.append(V[u[vmap]]); faces.append(idx.astype(np.int64) + n); uvs.append(uv); n += len(vmap)
    out = trimesh.Trimesh(np.concatenate(verts), np.concatenate(faces), process=False)
    out.visual = trimesh.visual.TextureVisuals(uv=np.concatenate(uvs))
    return out


ap = argparse.ArgumentParser()
ap.add_argument("views"); ap.add_argument("shape"); ap.add_argument("out")
ap.add_argument("--faces", type=int, default=150000)
ap.add_argument("--tex", type=int, default=4096)
ap.add_argument("--exp", type=float, default=6.0, help="view blending sharpness (cos^exp)")
ap.add_argument("--rigid", action="store_true", help="bbox fit only, no non-rigid warp")
ap.add_argument("--no-top", action="store_true")
ap.add_argument("--mu", type=float, default=0.0, help="whole-body photometric (photo vs generated view) alignment weight")
ap.add_argument("--edge-px", type=int, default=10,
                help="band (px @1024) along interior occlusion edges that is checked for colours of the other depth layer")
ap.add_argument("--rim-px", type=int, default=2, help="photo pixels this close to the cut-out's edge are not used (matting fringe)")
ap.add_argument("--gap-jump", type=float, default=0.35, help="depth step (mesh units, body ~1) that counts as a gap")
ap.add_argument("--gap-px", type=int, default=56, help="far patches narrower than ~1.5x this (px @1024) are not painted from that view")
ap.add_argument("--gap-margin", type=int, default=8, help="margin (px @1024) dropped behind a gap edge on large far surfaces")
ap.add_argument("--res-pow", type=float, default=4.0,
                help="views whose photo has more pixels per unit of mesh get weight (relative resolution)^res_pow")
ap.add_argument("--edge-views", nargs="*", default=["front", "left", "back", "right", "top"],
                help="photos whose outline pixels are stretched over the ridges they see edge-on")
ap.add_argument("--edge-support", type=float, default=0.1, help="edge-on photo weight at which a texel no longer needs the generated views")
ap.add_argument("--horizon-px", type=int, default=16, help="px@1024: surface seen at > 60 deg this close to a farther layer is not painted")
ap.add_argument("--no-register", action="store_true", help="skip the cross-view registration of front/back/top to the side photos")
ap.add_argument("--reg-max", type=float, default=90.0, help="largest cross-view registration shift, px@1024")
ap.add_argument("--reg-lam", type=float, default=0.1, help="smoothness of the cross-view registration field")
ap.add_argument("--min-support", type=float, default=0.03,
                help="a texel counts as seen by a source once its summed cos^exp * confidence reaches this")
ap.add_argument("--feather", type=float, nargs=2, default=(0.01, 0.06),
                help="distance (mesh units, body ~1) from photo-painted surface over which the generated colours stop being levelled to the photos")
ap.add_argument("--gen-w", type=float, default=1.0, help="weight of the generated side/top views relative to the generated underside view")
ap.add_argument("--head-prompts", default="", help="dir with front/left/right head crops; enables head refinement")
ap.add_argument("--head-cut", type=float, nargs=2, default=(0.40, 0.58),
                help="head = (dist behind nose + dist below ear tips) < cut[0]*height, fading out by cut[1]")
ap.add_argument("--head-fade", type=float, default=1.0, help="width of the fade-out band around the head, in zoomed-frame widths")
ap.add_argument("--head-nu", type=float, default=1.0, help="weight of the semantic correspondences in the head alignment")
ap.add_argument("--head-lam", type=float, default=100.0, help="smoothness of the head warp")
ap.add_argument("--head-close", default="", help="dir with front/left/right close-up photos of the head (cut-outs); painted over the head at their own resolution")
ap.add_argument("--close-mu", type=float, default=1.0, help="photometric weight when aligning a close-up to the whole-body photo of the same view")
ap.add_argument("--tag", default="", help="suffix for output names, to keep variants side by side")
args = ap.parse_args()
out, dev = args.out, "cuda"
os.makedirs(f"{out}/dbg", exist_ok=True)
tag = f"_{args.tag}" if args.tag else ""

# view name -> (elevation, azimuth) of the renderer camera that sees the same side
CAMS = {"front": (0, 0), "left": (0, 90), "back": (0, 180), "right": (0, 270), "top": (90, 0)}
KEYS = ["front", "left", "back", "right"]
GEN = {"front": 0, "left": 1, "back": 2, "right": 3, "top": 4}
gen_elevs, gen_azims = [0, 0, 0, 0, 90, -90], [0, 90, 180, 270, 0, 180]
photos = {k: Image.open(f"{args.views}/{k}.png").convert("RGBA") for k in KEYS}
if not args.no_top and os.path.exists(f"{args.views}/top.png"):
    photos["top"] = Image.open(f"{args.views}/top.png").convert("RGBA")

# ---------------------------------------------------------------- mesh + UVs (cached)
mesh_cache = f"{out}/mesh_uv_{args.faces}.pkl"
if os.path.exists(mesh_cache):
    mesh = pickle.load(open(mesh_cache, "rb"))
else:
    t = time.time()
    mesh = trimesh.load(args.shape, force="mesh")
    mesh = FaceReducer()(DegenerateFaceRemover()(FloaterRemover()(mesh)), max_facenum=args.faces)
    mesh = uv_unwrap(mesh)
    pickle.dump(mesh, open(mesh_cache, "wb"))
    print("decimate + uv unwrap s", round(time.time() - t))
print("mesh faces", len(mesh.faces))

# Generation frame: x = cat's left, y = up, z = forward.
V0 = np.asarray(mesh.vertices, dtype=np.float32)
y_top, z_front, height = V0[:, 1].max(), V0[:, 2].max(), V0[:, 1].max() - V0[:, 1].min()
u_vert = (z_front - V0[:, 2]) + (y_top - V0[:, 1])       # small on the head, grows towards neck and body
head_on = bool(args.head_prompts)
if head_on:
    hv = V0[u_vert < args.head_cut[1] * height]
    c_head = (hv.max(0) + hv.min(0)) / 2
    k_head = 1.15 / (2 * np.linalg.norm(hv - c_head, axis=1).max())
    mesh_zoom = mesh.copy()
    mesh_zoom.vertices = (V0 - c_head) * k_head           # head fills the frame; same vertex order and UVs


def gen_views(paint, m, prompts, auto_center=True):
    paint.render.scale_factor = 1.15      # only set by load_mesh when it centres the mesh itself
    paint.render.load_mesh(m, auto_center=auto_center)
    normal_maps = paint.render_normal_multiview(gen_elevs, gen_azims, use_abs_coor=True)
    position_maps = paint.render_position_multiview(gen_elevs, gen_azims)
    cam = [(((az // 30) + 9) % 12) // {-20: 1, 0: 1, 20: 1, -90: 3, 90: 3}[el]
           + {-20: 0, 0: 12, 20: 24, -90: 36, 90: 40}[el] for az, el in zip(gen_azims, gen_elevs)]
    return paint.models["multiview_model"](prompts, normal_maps + position_maps, cam)


lap("load + decimate + UV unwrap")
# ---------------------------------------------------------------- generated views (cached)
gen_paths = [f"{out}/gen_{i}.png" for i in range(6)]
hgen_paths = [f"{out}/gen_head_{i}.png" for i in range(6)]
need_body = not all(os.path.exists(p) for p in gen_paths)
need_head = head_on and not all(os.path.exists(p) for p in hgen_paths)
if need_body or need_head:
    paint = Hunyuan3DPaintPipeline.from_pretrained("tencent/Hunyuan3D-2", subfolder="hunyuan3d-paint-v2-0")
    delight = lambda im: paint.models["delight_model"](paint.recenter_image(im))
    if need_body:
        for p, v in zip(gen_paths, gen_views(paint, mesh, [delight(photos[k]) for k in KEYS])):
            v.save(p)
    if need_head:
        faces = np.where((u_vert[mesh.faces] < args.head_cut[1] * height).all(1))[0]
        head_mesh = mesh_zoom.submesh([faces], append=True)
        prompts = [delight(Image.open(f"{args.head_prompts}/{k}.png").convert("RGBA")) for k in ("front", "left", "right")]
        for p, v in zip(hgen_paths, gen_views(paint, head_mesh, prompts, auto_center=False)):
            v.save(p)
    del paint
    torch.cuda.empty_cache()

lap("generated fallback views (paint model)")
# ---------------------------------------------------------------- helpers
R = 4096 if args.tex >= 4096 else 2048   # back-projection resolution (pixels are splatted to texels)
WORK = 1024                              # resolution the warps are optimised at
render = MeshRender(default_resolution=R, texture_size=args.tex)
render.load_mesh(mesh)
# The photos are colour-bled past their outline and aligned to a few pixels, so the default 16 px
# exclusion zone along silhouettes and depth edges only starves thin parts (ear rims) of texture.
render.bake_unreliable_kernel_size = 6


def raster(rd, elev, azim, res, attr=None):
    """Visibility mask and camera-space depth (or a per-vertex attribute) for one orthographic view."""
    mv = get_mv_matrix(elev=elev, azim=azim, camera_distance=rd.camera_distance, center=None)
    cam = transform_pos(mv, rd.vtx_pos, keepdim=True)
    rast, _ = rd.raster_rasterize(transform_pos(rd.camera_proj_mat, cam), rd.pos_idx, resolution=(res, res))
    if attr is None:
        attr = (cam[:, 2] / cam[:, 3]).reshape(-1, 1)
    val, _ = rd.raster_interpolate(attr.reshape(1, -1, attr.shape[-1]).contiguous(), rast, rd.pos_idx)
    return (rast[0, ..., -1] > 0).cpu().numpy(), val[0].squeeze(-1).cpu().numpy()


def ndc(rd, elev, azim):
    mv = get_mv_matrix(elev=elev, azim=azim, camera_distance=rd.camera_distance, center=None)
    clip = transform_pos(rd.camera_proj_mat, transform_pos(mv, rd.vtx_pos, keepdim=True)).reshape(-1, 4)
    return (clip[:, :2] / clip[:, 3:4]).cpu().numpy()


def calibrate():
    """How rasterised pixels relate to NDC (axis order and signs), found by matching a splat to the mask."""
    S, _ = raster(render, 0, 0, 512)
    n = ndc(render, 0, 0)
    best = None
    for swap in (False, True):
        for sx in (1, -1):
            for sy in (1, -1):
                a = n[:, ::-1] if swap else n
                col = ((sx * a[:, 0] * 0.5 + 0.5) * 512).astype(int).clip(0, 511)
                row = ((sy * a[:, 1] * 0.5 + 0.5) * 512).astype(int).clip(0, 511)
                m = np.zeros((512, 512), bool); m[row, col] = True
                m = ndi.binary_fill_holes(ndi.binary_closing(m, iterations=2))
                score = (m & S).sum() / (m | S).sum()
                if best is None or score > best[0]:
                    best = (score, swap, sx, sy)
    assert best[0] > 0.9, f"could not calibrate pixel convention: {best}"
    return best[1:]


def grid_coords(rd, elev, azim, conv):
    """Per-vertex position in grid_sample coordinates (x = column, y = row, both in [-1, 1])."""
    swap, sx, sy = conv
    n = ndc(rd, elev, azim)
    a = n[:, ::-1] if swap else n
    return np.stack([sx * a[:, 0], sy * a[:, 1]], 1)


def bbox(m):
    ys, xs = np.where(m)
    return xs.min(), ys.min(), xs.max() + 1, ys.max() + 1


def iou(a, b):
    return (a & b).sum() / (a | b).sum()


def bleed(rgb, mask):
    """Extend the colours inside mask outwards so no background leaks in at the rim."""
    idx = ndi.distance_transform_edt(~mask, return_distances=False, return_indices=True)
    return rgb[idx[0], idx[1]]


def sdf(mask, clamp):
    d = ndi.distance_transform_edt(~mask) - ndi.distance_transform_edt(mask)
    return np.clip(d, -clamp, clamp).astype(np.float32) / clamp


def occlusion_edges(mask, depth, jump=0.02):
    """Interior depth discontinuities of a view: where one part of the mesh hides another."""
    d = np.where(mask, depth, np.nan)
    e = np.zeros_like(mask)
    e[:, 1:] |= np.abs(d[:, 1:] - d[:, :-1]) > jump
    e[1:, :] |= np.abs(d[1:, :] - d[:-1, :]) > jump
    return e


def leak_conf(photo_lum, gen_lum, S, depth, band, reach, jump=0.02):
    """1 where the photo can be trusted, 0 where (inside `band`) it shows the other depth layer.

    The generated view follows the mesh but invents its own markings, so plain disagreement with
    it proves nothing (the real white collar, the real tabby thighs). A leak is the specific case
    where the photo differs from the generated view at its own position and matches what the
    generated view shows on the other side of a nearby depth edge."""
    p = blur(torch.from_numpy(photo_lum)[None, None].to(dev), 2.0)[0, 0]
    g = blur(torch.from_numpy(gen_lum)[None, None].to(dev), 2.0)[0, 0]
    z, m = torch.from_numpy(np.where(S, depth, 0).astype(np.float32)).to(dev), torch.from_numpy(S).to(dev)
    px = 1.2 / S.shape[0]                                 # mesh units per pixel (ortho_scale 1.2)
    other = torch.full_like(p, float("inf"))
    for dy in range(-reach, reach + 1, 3):
        for dx in range(-reach, reach + 1, 3):
            if dy * dy + dx * dx > reach * reach or (dy == 0 and dx == 0):
                continue
            gs, zs, ms = (torch.roll(t, (dy, dx), (0, 1)) for t in (g, z, m))
            layer = ms & ((zs - z).abs() > jump + 1.5 * px * (dy * dy + dx * dx) ** 0.5)   # beyond what slope explains
            other = torch.minimum(other, torch.where(layer, (p - gs).abs(), torch.full_like(p, float("inf"))))
    bad = ((p - g).abs() - other - 0.05) / 0.15
    bad = bad.clamp(0, 1) ** 2 * (3 - 2 * bad.clamp(0, 1))
    return np.where(band & S, 1 - bad.cpu().numpy(), 1.0).astype(np.float32)


def blur(t, sigma):
    r = int(3 * sigma)
    k = torch.exp(-torch.arange(-r, r + 1, device=t.device, dtype=torch.float32) ** 2 / (2 * sigma ** 2))
    k = (k / k.sum())[None, None]
    t = F.conv2d(F.pad(t, (r, r, 0, 0), mode="replicate"), k[:, :, None, :])
    return F.conv2d(F.pad(t, (0, 0, r, r), mode="replicate"), k[:, :, :, None])


def lum(rgb):
    return (np.asarray(rgb, dtype=np.float32) @ np.array([0.299, 0.587, 0.114], dtype=np.float32)) / 255


def lcn(l, sigma=24.0):
    """Local contrast normalisation, so a photo and a generated view compare by pattern, not exposure."""
    t = torch.from_numpy(l)[None, None].to(dev)
    d = t - blur(t, sigma)
    return (d / (torch.sqrt(blur(d * d, sigma)) + 0.05))[0, 0].cpu().numpy()


def smoothstep(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def base_grid(n):
    ys, xs = torch.meshgrid(torch.linspace(-1, 1, n, device=dev), torch.linspace(-1, 1, n, device=dev), indexing="ij")
    return torch.stack([xs, ys], -1)[None]


def fit_warp(src_mask, dst_mask, src_lum=None, dst_lum=None, mu=0.0, where=None, matches=None, nu=1.0, rest=None, rho=2.0,
             levels=((6, 16.0), (12, 12.0), (24, 8.0), (48, 6.0)), iters=250, clamp=48.0, lam=30.0):
    """Displacement field (1,2,H,W, normalised coords) so that src sampled at x+d(x) matches dst.

    The silhouettes drive the fit. With mu > 0 the low-pass luminance of the photo is also pulled
    towards a generated view of the same camera, which is pixel-aligned with the mesh.
    matches = (points, offsets) asks for d(point) ~ offset (Huber), e.g. semantic correspondences.
    rest = map of where the field has nothing to hold on to and should relax to zero rather than
    extrapolate (weight rho).
    """
    H = dst_mask.shape[0]
    s = torch.from_numpy(sdf(src_mask, clamp))[None, None].to(dev)
    d = torch.from_numpy(sdf(dst_mask, clamp))[None, None].to(dev)
    if mu > 0:
        sl = torch.from_numpy(src_lum)[None, None].to(dev)
        dl = torch.from_numpy(dst_lum)[None, None].to(dev)
        inside = ndi.binary_erosion(dst_mask, iterations=4).astype(np.float32)
        inside = torch.from_numpy(inside if where is None else inside * where)[None, None].to(dev)
    if rest is not None:
        rest = torch.from_numpy(np.ascontiguousarray(rest, dtype=np.float32))[None, None].to(dev)
    base = base_grid(H)
    disp = torch.zeros(1, 2, levels[0][0], levels[0][0], device=dev)
    for g, sigma in levels:
        disp = F.interpolate(disp, size=(g, g), mode="bicubic", align_corners=True).detach().requires_grad_(True)
        # L-BFGS rather than Adam: Adam's per-parameter step normalisation lets nodes that only feel
        # numerical noise (frame border, far from any constraint) drift by hundreds of pixels.
        opt = torch.optim.LBFGS([disp], max_iter=iters, history_size=20, line_search_fn="strong_wolfe",
                                tolerance_grad=1e-9, tolerance_change=1e-12)
        if mu > 0:
            sb, db = blur(sl, sigma), blur(dl, sigma)
        terms = {}

        def closure():
            full = F.interpolate(disp, size=(H, H), mode="bicubic", align_corners=True)
            grid = base + full.permute(0, 2, 3, 1)
            w = F.grid_sample(s, grid, padding_mode="border", align_corners=True)
            fit = ((w - d) ** 2).mean()
            photo = torch.zeros((), device=dev)
            if mu > 0:
                diff = F.grid_sample(sb, grid, padding_mode="border", align_corners=True) - db
                photo = (inside * torch.sqrt(diff ** 2 + 1e-3)).sum() / inside.sum().clamp(min=1)
            match = torch.zeros((), device=dev)
            if matches is not None:
                got = F.grid_sample(full, matches[0][None, None], align_corners=True)[0, :, 0].T
                r = (got - matches[1]).norm(dim=1) * (H / 2) / clamp               # in units of clamp px
                match = torch.where(r < 0.25, r ** 2, 0.5 * r - 0.0625).mean()     # Huber, knee at clamp/4
            pad = F.pad(disp, (1, 1, 1, 1), mode="replicate")      # so the border nodes are regularised too
            lap = pad[..., 2:, 1:-1] + pad[..., :-2, 1:-1] + pad[..., 1:-1, 2:] + pad[..., 1:-1, :-2] - 4 * disp
            still = (rest * full ** 2).mean() if rest is not None else 0.0
            # bending energy scaled so the penalty is comparable across grid levels
            loss = fit + mu * photo + nu * match + rho * still + lam * (lap ** 2).mean() * (g / 48) ** 2
            opt.zero_grad(); loss.backward()
            terms.update(fit=float(fit), photo=float(photo), match=float(match))
            return loss

        opt.step(closure)
        fit, photo, match = terms["fit"], terms["photo"], terms["match"]
        print(f"   grid {g}: silhouette {float(fit):.5f} photometric {float(photo):.4f} matches {float(match):.5f}")
    return F.interpolate(disp.detach(), size=(H, H), mode="bicubic", align_corners=True), base


_dino = None


def dino_feats(img_u8, P=73):
    """L2-normalised DINOv2 patch features [P,P,C] of an RGB image (resized to 14P x 14P)."""
    global _dino
    if _dino is None:
        from transformers import AutoModel
        _dino = AutoModel.from_pretrained("facebook/dinov2-large").to(dev).eval()
    x = torch.from_numpy(np.asarray(Image.fromarray(img_u8).resize((14 * P, 14 * P), Image.LANCZOS))).float().to(dev) / 255
    x = ((x - torch.tensor([0.485, 0.456, 0.406], device=dev)) / torch.tensor([0.229, 0.224, 0.225], device=dev))
    with torch.no_grad():
        f = _dino(pixel_values=x.permute(2, 0, 1)[None]).last_hidden_state[0, 1:]
    return F.normalize(f.reshape(P, P, -1), dim=-1)


def semantic_matches(src_u8, dst_u8, wgt, P=73, radius=16, robust=True):
    """Correspondences dst -> src from DINOv2 features: where in the photo (src) is the eye, nose
    corner or ear base that the generated view (dst) shows at each patch. Mutual nearest neighbours
    within `radius` patches, then outliers against a robust affine fit are dropped.
    Returns (points [N,2], offsets [N,2]) in grid_sample coordinates: src(x + offset) ~ dst(x)."""
    fs, fd = dino_feats(src_u8, P).reshape(P * P, -1), dino_feats(dst_u8, P).reshape(P * P, -1)
    ii, jj = torch.meshgrid(torch.arange(P, device=dev), torch.arange(P, device=dev), indexing="ij")
    pos = torch.stack([jj, ii], -1).reshape(-1, 2).float()                 # (col, row) of every patch
    near = (pos[:, None] - pos[None]).abs().amax(-1) <= radius
    sim = (fd @ fs.T).masked_fill(~near, -1)                               # [dst, src]
    d2s, s2d = sim.argmax(1), sim.argmax(0)
    wp = F.interpolate(torch.from_numpy(np.ascontiguousarray(wgt, dtype=np.float32))[None, None].to(dev), size=(P, P), mode="area").reshape(-1)
    keep = ((pos[s2d[d2s]] - pos).abs().amax(-1) <= 1) & (wp > 0.3) & (sim.max(1).values > 0.4)
    centre = lambda q: ((q + 0.5) * 14 - 0.5) / (14 * P - 1) * 2 - 1       # patch index -> normalised coord
    x, y = centre(pos[keep]), centre(pos[d2s[keep]])
    n0 = len(x)
    if not robust:
        return x, y - x
    wts = torch.ones(n0, device=dev)
    for _ in range(5):                                                     # robust affine, Tukey weights
        Xh = torch.cat([x, torch.ones(n0, 1, device=dev)], 1) * wts[:, None]
        A = torch.linalg.lstsq(Xh, y * wts[:, None]).solution
        r = (torch.cat([x, torch.ones(n0, 1, device=dev)], 1) @ A - y).norm(dim=1)
        c = 4.0 * 2 / P                                                    # 4 patches
        wts = ((1 - (r / c) ** 2).clamp(min=0)) ** 2
    ok = wts > 0
    print(f"   semantic matches: {n0} mutual, {int(ok.sum())} kept; affine scale x {float(A[0, 0]):.3f} y {float(A[1, 1]):.3f}"
          f" shift px@1024 x {float(A[2, 0]) * 512:.0f} y {float(A[2, 1]) * 512:.0f}")
    return x[ok], (y - x)[ok]


def place(img, src_box, dst_box, size, resample):
    """Paste the src_box crop of img so it fills dst_box in a size x size canvas."""
    x0, y0, x1, y1 = dst_box
    canvas = Image.new(img.mode, (size, size), 0)
    canvas.paste(img.crop(src_box).resize((x1 - x0, y1 - y0), resample), (x0, y0))
    return canvas


def sample_img(img_u8, grid):
    t = torch.from_numpy(img_u8).float().permute(2, 0, 1)[None].to(dev) / 255
    o = F.grid_sample(t, grid, mode="bicubic", padding_mode="border", align_corners=True)[0]
    return (o.permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)


def to_u8(x):
    return (np.clip(x, 0, 1) * 255).astype(np.uint8)


u_attr = torch.from_numpy(u_vert).to(dev).reshape(-1, 1)
conv = calibrate()


def tail_vertices(res=512, thin=0.06):
    """Per-vertex flag for the tail: the thin part of the side silhouette that reaches furthest back.
    (Thin = removed by a morphological opening with radius thin * silhouette height; legs and ears
    are thin too, the tail is the piece with the rearmost point.)"""
    S, zimg = raster(render, 0, 90, res, attr=torch.from_numpy(V0[:, 2:3].copy()).to(dev))
    b = bbox(S)
    zs = np.where(S, zimg, np.inf)
    piece = None
    for th in (thin, 0.09, 0.12, 0.16):      # a thick tail survives a small opening: widen until the piece holds the rearmost point
        r = max(2.0, th * (b[3] - b[1]))
        opened = ndi.distance_transform_edt(~(ndi.distance_transform_edt(S) > r)) <= r
        lab, n = ndi.label(S & ~opened)
        if n == 0:
            continue
        idx = np.arange(1, n + 1)
        big = ndi.sum(lab > 0, lab, idx) > r * r
        zmin = np.where(big, ndi.minimum(zs, lab, idx), np.inf)
        if zmin.min() <= zs.min() + 0.01:
            piece = ndi.binary_dilation(lab == idx[zmin.argmin()], iterations=2)
            print(f"tail: thin part of the side outline at opening radius {th} x height")
            break
    if piece is None:
        return np.zeros(len(V0), np.float32)
    Image.fromarray(to_u8(0.4 * S + 0.6 * piece)).save(f"{out}/dbg/tail_side.png")
    g = grid_coords(render, 0, 90, conv)
    col = np.round((g[:, 0] + 1) / 2 * (res - 1)).astype(int).clip(0, res - 1)
    row = np.round((g[:, 1] + 1) / 2 * (res - 1)).astype(int).clip(0, res - 1)
    return piece[row, col].astype(np.float32)


tail_attr = torch.from_numpy(tail_vertices()).to(dev).reshape(-1, 1)


def area_frac(mask, P=73):
    return F.interpolate(torch.from_numpy(np.ascontiguousarray(mask, dtype=np.float32))[None, None].to(dev), size=(P, P), mode="area").reshape(-1)


def tail_prototypes(thin=0.06):
    """DINOv2 features of tail fur and of body fur, taken from the side photos, where the tail is
    the thin rear piece of the cut-out. Used to find the tail in a photo that shows it in front of
    the body (the view from behind)."""
    tp, bp = [], []
    for k in ("left", "right"):
        if k not in photos:
            continue
        a = np.asarray(photos[k].resize((WORK, WORK), Image.LANCZOS))
        m = a[..., 3] > 127
        rgb = np.where(m[..., None], a[..., :3], 127).astype(np.uint8)
        b = bbox(m)
        xs = np.where(m, np.arange(WORK)[None], -1 if k == "left" else WORK)   # left photo: the cat faces image-left
        rear = xs.max() if k == "left" else -xs.min()
        t = None
        for th in (thin, 0.09, 0.12, 0.16):
            r = max(2.0, th * (b[3] - b[1]))
            opened = ndi.distance_transform_edt(~(ndi.distance_transform_edt(m) > r)) <= r
            lab, n = ndi.label(m & ~opened)
            if n == 0:
                continue
            idx = np.arange(1, n + 1)
            big = ndi.sum(lab > 0, lab, idx) > r * r
            ext = np.where(big, ndi.maximum(xs, lab, idx) if k == "left" else -ndi.minimum(xs, lab, idx), -np.inf)
            if ext.max() >= rear - 2:
                t = ndi.binary_dilation(lab == idx[ext.argmax()], iterations=2) & m
                break
        if t is None:
            continue
        tb = bbox(t)
        x0 = int(np.clip((tb[0] + tb[2]) // 2 - WORK // 4, 0, WORK // 2)); y0 = int(np.clip((tb[1] + tb[3]) // 2 - WORK // 4, 0, WORK // 2))
        crop = lambda im: np.ascontiguousarray(im[y0:y0 + WORK // 2, x0:x0 + WORK // 2])
        f = dino_feats(crop(rgb)).reshape(73 * 73, -1)
        tp.append(f[area_frac(crop(t)) > 0.6]); bp.append(f[area_frac(crop(m & ~t)) > 0.9])
        bp.append(dino_feats(rgb).reshape(73 * 73, -1)[area_frac(m & ~t) > 0.9])
    if not tp or sum(len(x) for x in tp) < 50:
        return None
    return torch.cat(tp), torch.cat(bp)


def photo_tail(img_u8, S, centre, protos):
    """Where a WORK x WORK photo (already warped onto the view) shows the tail: patches whose
    features are closer to tail fur than to body fur, in a 2x zoom around `centre`."""
    half = WORK // 4
    x0 = int(np.clip(centre[0] - half, 0, WORK - 2 * half)); y0 = int(np.clip(centre[1] - half, 0, WORK - 2 * half))
    f = dino_feats(np.ascontiguousarray(img_u8[y0:y0 + 2 * half, x0:x0 + 2 * half])).reshape(73 * 73, -1)
    score = ((f @ protos[0].T).topk(5, dim=1).values.mean(1) - (f @ protos[1].T).topk(5, dim=1).values.mean(1)).reshape(1, 1, 73, 73)
    score = F.interpolate(score, size=(2 * half, 2 * half), mode="bicubic", align_corners=False)[0, 0].cpu().numpy()
    P = np.zeros_like(S)
    P[y0:y0 + 2 * half, x0:x0 + 2 * half] = ndi.gaussian_filter(score, 4.0) > 0
    P = ndi.binary_fill_holes(ndi.binary_closing(P & S, iterations=6)) & S
    lab, n = ndi.label(P)
    return lab == 1 + int(np.argmax(ndi.sum(P, lab, range(1, n + 1)))) if n else P


tail_protos = tail_prototypes()
tail_side_px = int((raster(render, 0, 90, WORK, attr=tail_attr)[1] > 0.5).sum())


def uv_raster():
    """Rasterisation of the UV layout: which triangle covers every texel."""
    swap, sx, sy = conv
    uv = render.vtx_uv.float()
    a = torch.stack([sx * (2 * uv[:, 0] - 1), sy * (2 * uv[:, 1] - 1)], 1)      # texel (row, col) = (v, u) * size
    clip = torch.cat([a.flip(1) if swap else a, torch.zeros_like(a[:, :1]), torch.ones_like(a[:, :1])], 1)
    return render.raster_rasterize(clip, render.uv_idx, resolution=(args.tex, args.tex))[0]


def texel_attr(vattr):
    """A per-vertex attribute [N,C] interpolated to every texel [tex,tex,C]."""
    A = torch.zeros(len(render.vtx_uv), vattr.shape[1], device=dev)
    A[render.uv_idx.reshape(-1).long()] = vattr[render.pos_idx.reshape(-1).long()].float()
    return render.raster_interpolate(A[None].contiguous(), uv_rast, render.uv_idx)[0][0]


def cam_cos(elev, azim):
    """Per-vertex camera-space position and cosine between normal and viewing direction."""
    mv = get_mv_matrix(elev=elev, azim=azim, camera_distance=render.camera_distance, center=None)
    cam = transform_pos(mv, render.vtx_pos, keepdim=True)
    cam = cam[:, :3] / cam[:, 3:4]
    fn = F.normalize(torch.cross(cam[render.pos_idx[:, 1]] - cam[render.pos_idx[:, 0]], cam[render.pos_idx[:, 2]] - cam[render.pos_idx[:, 0]], dim=-1), dim=-1)
    vn = trimesh.geometry.mean_vertex_normals(len(cam), render.pos_idx.cpu().numpy(), fn.cpu().numpy())
    return cam, -torch.from_numpy(vn[:, 2:3]).float().to(dev)


def texel_view(elev, azim, idx, tol=0.02):
    """For the texels idx [N,2]: position in the view (grid_sample coords), cosine to the viewing
    direction, and whether the view sees them (not hidden behind another part of the mesh)."""
    cam, cosv = cam_cos(elev, azim)
    g = torch.from_numpy(grid_coords(render, elev, azim, conv).copy()).float().to(dev)
    ta = texel_attr(torch.cat([g, cam[:, 2:3], cosv], 1))[idx[:, 0], idx[:, 1]]
    Sv, dv = raster(render, elev, azim, 2048)
    dmap = torch.from_numpy(bleed(dv, Sv)).float().to(dev)[None, None]
    vis = (ta[:, 2] - F.grid_sample(dmap, ta[None, None, :, :2], align_corners=True)[0, 0, 0]).abs() < tol
    return ta[:, :2], ta[:, 3], vis


uv_rast = uv_raster()
on_mesh = uv_rast[0, ..., -1] > 0


gen = [Image.open(p).convert("RGB") for p in gen_paths]

lap("tail detection + DINOv2 prototypes")
# ---------------------------------------------------------------- cross-view registration
# Each photo was fitted to the mesh by its outline alone, so markings inside the outline can sit a
# little differently in two photos (a spot on a leg a few percent higher in the back view than in
# the side view) and show twice where the views meet. Surface that two photos both see well tells
# how far apart they are: shift the front / back / top photo by a smooth field so that it shows
# the same large-scale pattern as the side photos on that shared surface.
def blurred(img, sigma=8.0):
    t = torch.from_numpy(np.asarray(Image.fromarray(img).resize((WORK, WORK), Image.LANCZOS))).float().to(dev) / 255
    return blur(t.permute(2, 0, 1)[:, None], sigma).permute(1, 0, 2, 3)          # [1,3,H,W]


def register_to_sides(k, img, mask, n=400_000):
    """img: the photo warped onto view k (R x R), mask: its cut-out (WORK x WORK). Returns both, registered."""
    max_px = args.reg_max
    idx = torch.nonzero(on_mesh)
    idx = idx[torch.randperm(len(idx), device=dev)[:4_000_000]]
    gk, ck, vk = texel_view(*CAMS[k], idx)
    wk = ck.clamp(min=0) * vk
    G, Rf, Wt = [], [], []
    for s_ in ("left", "right"):
        if s_ not in aligned:
            continue
        gs, cs, vs = texel_view(*CAMS[s_], idx)
        ws = cs.clamp(min=0) * vs * F.grid_sample(torch.from_numpy(body_conf[s_])[None, None].to(dev), gs[None, None], align_corners=True)[0, 0, 0]
        w = torch.minimum(wk, ws)
        sel = torch.nonzero(w > 0.25)[:, 0][:n]
        G.append(gk[sel]); Wt.append(w[sel])
        Rf.append(F.grid_sample(blurred(aligned[s_]), gs[sel][None, None], align_corners=True)[0, :, 0].T)
    G, Rf, Wt = torch.cat(G), torch.cat(Rf), torch.cat(Wt)
    if len(G) < 20_000:
        print(f"   too little surface shared with the side photos ({len(G)} texels), not registered")
        return img, mask
    src = blurred(img)
    cost = lambda d: (((F.grid_sample(src, (G + d)[None, None], align_corners=True)[0, :, 0].T - Rf) ** 2).sum(1) + 1e-4).sqrt()
    # The side views share only the vertical axis with a front / back view, and across the view the
    # outline fit already pins the photo: the field is a vertical shift that varies with height,
    # zero at the top and the bottom of the outline (ear tips and paws stay where they are).
    Sk, _ = raster(render, *CAMS[k], WORK)
    b = bbox(Sk)
    y0, y1 = 2 * b[1] / (WORK - 1) - 1, 2 * (b[3] - 1) / (WORK - 1) - 1
    lim = max_px * 2 / WORK

    eff = lambda t: lim * torch.tanh(t / lim) * torch.cat([torch.zeros(1), torch.ones(t.shape[2] - 2), torch.zeros(1)]).to(dev)[None, None, :, None]

    def shift(t, y):
        """Vertical shift at image heights y (grid coords) for node values t [1,1,g,1]."""
        t = eff(t)
        v = ((y - y0) / (y1 - y0) * 2 - 1).clamp(-1, 1)
        return F.grid_sample(t, torch.stack([torch.zeros_like(v), v], -1)[None, None], mode="bicubic", padding_mode="border", align_corners=True)[0, 0, 0]

    move = lambda t: torch.stack([torch.zeros(len(G), device=dev), shift(t, G[:, 1])], 1)
    dy = torch.zeros(1, 1, 3, 1, device=dev)
    e0 = float((cost(0) * Wt).sum() / Wt.sum())
    for g in (5, 9, 17):
        dy = F.interpolate(dy, size=(g, 1), mode="bicubic", align_corners=True).detach().requires_grad_(True)
        opt = torch.optim.LBFGS([dy], max_iter=60, history_size=20, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            t = eff(dy)
            lap = t[:, :, 2:] - 2 * t[:, :, 1:-1] + t[:, :, :-2]
            loss = (cost(move(dy)) * Wt).sum() / Wt.sum() + args.reg_lam * ((lap * (g - 1) ** 2 / 4) ** 2).mean()
            loss.backward()
            return loss
        opt.step(closure)
    dy = dy.detach()
    e1 = float((cost(move(dy)) * Wt).sum() / Wt.sum())
    print(f"   registered to the side photos on {len(G)} shared texels; colour difference {e0:.4f} -> {e1:.4f}; "
          f"vertical shift px@1024 top to bottom", (shift(dy, torch.linspace(y0, y1, 9, device=dev)) * WORK / 2).cpu().numpy().round(1))
    if e1 > 0.97 * e0:
        print("   no gain, photo left as it is")
        return img, mask

    def full(m):
        f = torch.zeros(1, m, m, 2, device=dev)
        f[0, :, :, 1] = shift(dy, torch.linspace(-1, 1, m, device=dev))[:, None]
        return f

    before = np.asarray(Image.fromarray(img).resize((WORK, WORK), Image.LANCZOS))
    img = sample_img(img, base_grid(R) + full(R))
    mask = F.grid_sample(torch.from_numpy(mask.astype(np.float32))[None, None].to(dev), base_grid(WORK) + full(WORK), padding_mode="zeros", align_corners=True)[0, 0].cpu().numpy() > 0.5
    after = np.asarray(Image.fromarray(img).resize((WORK, WORK), Image.LANCZOS))
    Image.fromarray(np.hstack([before, after])).save(f"{out}/dbg/register_{k}{tag}.jpg", quality=90)
    return img, mask


# ---------------------------------------------------------------- whole-body alignment
aligned, density, body_conf, gen_drop = {}, {}, {}, {}
for k in sorted(photos, key=lambda k: k not in ("left", "right")):   # side views first: the others are registered to them
    photo = photos[k]
    elev, azim = CAMS[k]
    S, depth = raster(render, elev, azim, WORK)
    assert depth[S].max() < 0, "expected camera-space z < 0 in front of the camera"
    edges = occlusion_edges(S, depth)
    edge_dist = ndi.distance_transform_edt(~edges) if edges.any() else np.full(S.shape, 1e6)
    gy, gx = np.gradient(ndi.gaussian_filter(np.where(S, depth, depth[S].min()), 1.5))
    cosd = 1 / np.sqrt(1 + (gx ** 2 + gy ** 2) / (1.2 / WORK) ** 2)      # cosine between surface normal and view direction
    if k == "top":  # in-plane rotation of the top photo is unknown: take the best of the four
        cands = []
        for rot in (0, 90, 180, 270):
            p = photo.rotate(rot, expand=True)
            m = np.asarray(place(p.getchannel("A"), bbox(np.asarray(p)[..., 3] > 127), bbox(S), WORK, Image.BILINEAR)) > 127
            cands.append((iou(m, S), rot))
        print("top rotation IoUs", [(round(a, 3), r) for a, r in cands])
        photo = photo.rotate(max(cands)[1], expand=True)
    pm = np.asarray(photo)[..., 3] > 127
    # Colours come from the opaque core only: the cut-out's rim pixels are fur mixed with the white
    # studio backdrop, and that light fringe would end up on every surface seen edge-on.
    core = ndi.binary_erosion(np.asarray(photo)[..., 3] > 250, iterations=args.rim_px)
    rgb = Image.fromarray(bleed(np.asarray(photo)[..., :3], core))
    sb = bbox(pm)
    # rigid part: fit the photo's bbox to the silhouette's bbox, at both resolutions
    SR, _ = raster(render, elev, azim, R)
    m0 = np.asarray(place(photo.getchannel("A"), sb, bbox(S), WORK, Image.BILINEAR)) > 127
    rgbR = place(rgb, sb, bbox(SR), R, Image.LANCZOS)
    mR = np.asarray(place(Image.fromarray(core.astype(np.uint8) * 255), sb, bbox(SR), R, Image.BILINEAR)) > 127
    rgbR = bleed(np.asarray(rgbR), mR)
    gen_lum = lum(gen[GEN[k]].resize((WORK, WORK), Image.LANCZOS))
    iou0 = iou(m0, S)
    b = bbox(S)
    density[k] = float(np.sqrt((sb[2] - sb[0]) * (sb[3] - sb[1]) / ((b[2] - b[0]) * (b[3] - b[1]))))  # photo px per px @1024
    if args.rigid:
        warped, m1 = rgbR, m0
    else:
        print(k)
        src_lum = lum(Image.fromarray(rgbR).resize((WORK, WORK), Image.LANCZOS))
        disp, base = fit_warp(m0, S, src_lum, gen_lum, mu=args.mu)
        m1 = F.grid_sample(torch.from_numpy(m0.astype(np.float32))[None, None].to(dev),
                           base + disp.permute(0, 2, 3, 1), padding_mode="zeros", align_corners=True)[0, 0].cpu().numpy() > 0.5
        dispR = F.interpolate(disp, size=(R, R), mode="bicubic", align_corners=True)
        warped = sample_img(rgbR, base_grid(R) + dispR.permute(0, 2, 3, 1))
    print(k, "silhouette IoU bbox-fit", round(iou0, 3), "-> warped", round(iou(m1, S), 3))
    if k in ("front", "back", "top") and not args.no_register and not args.rigid:   # (top: its vertical axis is the body axis)
        warped, m1 = register_to_sides(k, warped, m1)
    # The tail is a separate layer in front of the body. Where its outline is interior to the view
    # (seen from behind) the silhouette warp cannot place it, and a photo with the tail in another
    # pose paints tail onto legs and legs onto tail. Give that layer its own affine placement from
    # semantic correspondences with the generated view, and do not paint the body from the part of
    # the photo that shows the tail.
    drop, tail_valid = np.zeros_like(S), None
    T = (raster(render, elev, azim, WORK, attr=tail_attr)[1] > 0.5) & S
    ring = T & ~ndi.binary_erosion(T)
    Pt = np.zeros_like(S)
    interior_tail = T.sum() > 200 and 1 - (ring & ndi.binary_dilation(~S, iterations=2)).sum() / ring.sum() > 0.5
    if interior_tail and tail_protos is not None:
        # Does the photo show the tail where the mesh has it? If the poses differ (a tail pointing
        # at the camera on the mesh, hanging down in the photo), no placement can map one onto the
        # other: this view then paints neither the mesh tail nor the body under the photo's tail.
        tb = bbox(T)
        small0 = np.asarray(Image.fromarray(warped).resize((WORK, WORK), Image.LANCZOS))
        Pt = photo_tail(np.where(S[..., None], small0, 127).astype(np.uint8), S, ((tb[0] + tb[2]) // 2, (tb[1] + tb[3]) // 2), tail_protos)
        agree = iou(Pt, T)
        print(f"   tail: photo shows it on {Pt.sum() / S.sum():.3f} of the view, mesh has it on {T.sum() / S.sum():.3f}, IoU {agree:.2f}")
        dbg = small0.copy(); dbg[Pt & ~T] = dbg[Pt & ~T] // 2 + np.array([127, 0, 0], np.uint8); dbg[T & ~Pt] = dbg[T & ~Pt] // 2 + np.array([0, 127, 0], np.uint8)
        dbg[T & Pt] = dbg[T & Pt] // 2 + np.array([127, 127, 0], np.uint8)
        Image.fromarray(dbg).save(f"{out}/dbg/tailseg_{k}{tag}.jpg", quality=90)
        # Nor is a tail that points at the camera painted from this view: it shows the tip only,
        # and the side views see all of it.
        end_on = T.sum() < 0.6 * tail_side_px
        both_end_on = end_on and 200 < Pt.sum() < 0.6 * tail_side_px and (Pt & T).sum() > 0.5 * min(Pt.sum(), T.sum())
        if both_end_on:
            # ... unless the photo shows the tail end-on at the same place: then it is the only
            # view of the tail's tip, and the horizon rule below keeps it off the tail's flanks.
            interior_tail = False
            # The outline fit and the registration to the side photos do not pin the tip, which is
            # interior to this view: slide the photo locally so its tail sits on the mesh tail.
            # The tip is the part of the mesh tail that faces the camera; the rest of its outline is
            # the tail's flank seen at a grazing angle, which the side and top photos show properly.
            cap = T & (cosd > 0.7)
            if cap.sum() > 100:
                (py_, px_), (ty_, tx_) = ndi.center_of_mass(Pt), ndi.center_of_mass(cap)
                pb, cb_ = bbox(Pt), bbox(cap)
                kk = float(np.clip((cb_[2] - cb_[0]) / max(pb[2] - pb[0], 1), 0.7, 1.5))        # mesh tip width / photo tip width
                win = smoothstep(1 - ndi.distance_transform_edt(~ndi.binary_dilation(T | Pt, iterations=4)) / 40).astype(np.float32)
                winR = F.interpolate(torch.from_numpy(win)[None, None].to(dev), size=(R, R), mode="bilinear", align_corners=True)[0, 0]
                g0 = base_grid(R)
                nrm = lambda px: px * 2 / (WORK - 1) - 1
                c_t = torch.tensor([nrm(tx_), nrm(ty_)], device=dev, dtype=torch.float32)
                c_p = torch.tensor([nrm(px_), nrm(py_)], device=dev, dtype=torch.float32)
                warped = sample_img(warped, g0 + winR[None, ..., None] * ((c_p - c_t) + (g0 - c_t) * (1 / kk - 1)))
                drop = T & ~ndi.binary_dilation(cap, iterations=2)
                print(f"   tail: end-on in both the photo and the mesh, tip painted from this photo (moved {px_ - tx_:.0f}, {py_ - ty_:.0f} px@1024, scaled {kk:.2f})")
            else:
                drop = ndi.binary_dilation(Pt | T, iterations=12) & S
        elif Pt.sum() > 200 and (agree < 0.5 or end_on):
            drop = ndi.binary_dilation(Pt | T, iterations=12) & S
            gen_drop[GEN[k]] = ndi.binary_dilation(drop, iterations=8)   # the generated view copies this photo there
            interior_tail = False
            print(f"   tail: {'seen end-on' if end_on else 'poses differ'}, {drop.sum() / S.sum():.3f} of the view not painted from this photo")
    if interior_tail:
        tb = bbox(T)
        half = int(min(max(tb[2] - tb[0], tb[3] - tb[1]), WORK // 2 - 1))
        x0 = int(np.clip((tb[0] + tb[2]) // 2 - half, 0, WORK - 2 * half))
        y0 = int(np.clip((tb[1] + tb[3]) // 2 - half, 0, WORK - 2 * half))
        crop = lambda im: np.ascontiguousarray(im[y0:y0 + 2 * half, x0:x0 + 2 * half])
        small0 = np.asarray(Image.fromarray(warped).resize((WORK, WORK), Image.LANCZOS))
        src = crop(np.where(S[..., None], small0, 127).astype(np.uint8))
        dst = crop(np.asarray(gen[GEN[k]].resize((WORK, WORK), Image.LANCZOS)))
        pts, off = semantic_matches(src, dst, crop(ndi.binary_erosion(T, iterations=2)).astype(np.float32), radius=24, robust=False)
        # Same animal at the same scale, so across the tail only a shift is allowed; along it the
        # photo's tail may be longer or shorter (another pose, foreshortening): shift + scale.
        # Robust fit (medians / Theil-Sen), since a thin striped part gives few and repetitive matches.
        ok, A = torch.zeros(len(pts), dtype=torch.bool, device=dev), None
        if len(pts) >= 20:
            long_ax = int(tb[3] - tb[1] > tb[2] - tb[0])            # 1: tail runs along image y
            u, v = pts[:, long_ax], (pts + off)[:, long_ax]
            du, dv = u[:, None] - u[None], v[:, None] - v[None]
            far_pairs = du.abs() > 0.25 * (u.max() - u.min())
            a_l = float((dv[far_pairs] / du[far_pairs]).median()) if far_pairs.any() else 1.0
            b_l = float((v - a_l * u).median())
            b_s = float(off[:, 1 - long_ax].median())
            tol = 3.0 * 2 / 73                                       # 3 patches
            ok = ((v - a_l * u - b_l).abs() < tol) & ((off[:, 1 - long_ax] - b_s).abs() < tol)
            A = torch.zeros(3, 2, device=dev)
            A[long_ax, long_ax], A[2, long_ax] = a_l, b_l
            A[1 - long_ax, 1 - long_ax], A[2, 1 - long_ax] = 1.0, b_s
            print(f"   tail layer: {len(pts)} matches, {int(ok.sum())} agree; along-tail scale {a_l:.2f}, "
                  f"shift px@1024 along {b_l * half:.0f} across {b_s * half:.0f}")
        if A is not None and int(ok.sum()) >= 15 and 0.5 < a_l < 2.5:
            sc = torch.tensor([(2 * half - 1) / (WORK - 1)] * 2, device=dev)                 # crop coords -> frame coords
            of = torch.tensor([(2 * x0 + 2 * half - 1) / (WORK - 1) - 1, (2 * y0 + 2 * half - 1) / (WORK - 1) - 1], device=dev)
            fwd = lambda f: (((f - of) / sc) @ A[:2] + A[2]) * sc + of                       # mesh tail point -> photo point
            inv = lambda f: ((((f - of) / sc) - A[2]) @ torch.linalg.inv(A[:2])) * sc + of
            tail_img = sample_img(warped, fwd(base_grid(R)))
            # hard switch along the tail's occlusion outline, soft where it grows out of the rump
            root = ndi.binary_dilation(T) & ~T & S & (edge_dist > 2)
            wT = T * smoothstep(ndi.distance_transform_edt(~root) / 24) if root.any() else T.astype(np.float32)
            wT = F.interpolate(torch.from_numpy(wT.astype(np.float32))[None, None].to(dev), size=(R, R), mode="bilinear", align_corners=True)
            TR = torch.from_numpy((raster(render, elev, azim, R, attr=tail_attr)[1] > 0.5) & SR).to(dev)
            wT = (wT[0, 0] * TR)[..., None].cpu().numpy()
            before = small0
            warped = (tail_img * wT + warped * (1 - wT)).round().astype(np.uint8)
            # where the photo shows its tail: the mesh tail mapped into the photo
            Tt = torch.from_numpy(T.astype(np.float32))[None, None].to(dev)
            shown = F.grid_sample(Tt, inv(base_grid(WORK)), padding_mode="zeros", align_corners=True)[0, 0].cpu().numpy() > 0.5
            drop = ndi.binary_dilation(shown, iterations=16) & ~T & S
            # ... except where the photo's tail looks like the body beside it anyway (tabby tail root
            # on the tabby rump): compare the tail's colour with the body colour a little further out
            # (well clear of the tail, whose placement is only approximate).
            l6 = ndi.gaussian_filter(lum(Image.fromarray(small0)), 6.0)
            core_t = ndi.binary_erosion(shown, iterations=6)
            ring = ndi.binary_dilation(shown, iterations=45) & ~ndi.binary_dilation(shown, iterations=25) & S & ~T
            if ring.any():
                tail_l = bleed(l6, core_t if core_t.any() else shown)
                drop &= ndi.gaussian_filter(np.abs(tail_l - bleed(l6, ring)), 6.0) > 0.2
            # ... and paint the tail only from photo pixels that lie inside the cut-out
            inside = torch.from_numpy(ndi.binary_erosion(m1, iterations=3).astype(np.float32))[None, None].to(dev)
            tail_valid = F.grid_sample(inside, fwd(base_grid(WORK)), padding_mode="zeros", align_corners=True)[0, 0].cpu().numpy() > 0.5
            print(f"   tail layer: body not painted on {drop.sum() / S.sum():.3f} of the view")
            after = np.asarray(Image.fromarray(warped).resize((WORK, WORK), Image.LANCZOS)).copy()
            after[drop] = after[drop] // 2 + np.array([127, 0, 0], np.uint8)
            Image.fromarray(np.hstack([crop(before), crop(after), dst])).save(f"{out}/dbg/tail_{k}{tag}.jpg", quality=90)
        else:
            drop = ndi.binary_dilation(T, iterations=4) & S
            if tail_protos is not None and Pt.sum() > 200:      # nor the body from where the photo shows its tail
                drop = ndi.binary_dilation(Pt | T, iterations=12) & S
                gen_drop[GEN[k]] = ndi.binary_dilation(drop, iterations=8)
            print(f"   tail layer: no reliable placement, {drop.sum() / S.sum():.3f} of the view not painted from this photo")
    # Even with matching outlines, a photo pixel next to an interior occlusion edge can sit on the
    # wrong depth layer. (a) Close to any such edge, drop photo pixels that look like the other
    # layer rather than their own, judged against the generated view, which follows the mesh
    # (see leak_conf). (b) Never project onto a narrow patch seen through
    # a gap far behind its occluder (far legs between near legs, front legs behind the tail): a few
    # pixels of misalignment put the occluder's colours there, and another view sees it directly.
    # A large far surface (the head above the back, seen from behind) only loses a thin margin.
    small = Image.fromarray(warped).resize((WORK, WORK), Image.LANCZOS)
    if tail_valid is None:
        conf = leak_conf(lum(small), gen_lum, S, depth, edge_dist < args.edge_px, args.edge_px + 6)
    else:
        # The tail's outline is handled by the layer logic above. On the tail itself the placement
        # is only approximate, so drop what is clearly not tail: much lighter or darker than the
        # generated tail once the rings are blurred away.
        e2 = edges & ~ndi.binary_dilation(T, iterations=3)
        band = (ndi.distance_transform_edt(~e2) < args.edge_px) & ~T if e2.any() else np.zeros_like(S)
        conf = leak_conf(lum(small), gen_lum, S, depth, band, args.edge_px + 6)
        d6 = (blur(torch.from_numpy(lum(small))[None, None].to(dev), 6.0)
              - blur(torch.from_numpy(gen_lum)[None, None].to(dev), 6.0))[0, 0].abs().cpu().numpy()
        conf[T] = (tail_valid * (1 - smoothstep((d6 - 0.2) / 0.15)))[T]
    strong = occlusion_edges(S, depth, jump=args.gap_jump)
    if strong.any():
        z = np.where(S, depth, -np.inf)                      # camera looks down -z: nearer = larger z
        far = lambda px: (ndi.maximum_filter(z, size=2 * px + 1) - z > args.gap_jump / 2) & S
        # cut the view along those edges: each piece is one depth-continuous patch of surface
        lab, n = ndi.label(S & ~ndi.binary_dilation(strong))
        idx = np.arange(1, n + 1)
        inradius = ndi.maximum(ndi.distance_transform_edt(lab > 0), lab, idx)
        rim = ndi.binary_dilation(strong, iterations=3) & (lab > 0)
        far_frac = ndi.sum(far(3) & rim, lab, idx) / np.maximum(ndi.sum(rim, lab, idx), 1)
        sliver = np.isin(lab, idx[(far_frac > 0.5) & (inradius < 0.75 * args.gap_px)])
        # ... and only where the photo does not show what the generated view has there (a white collar
        # right behind the rump's horizon is thinner than the margin, and both agree on it).
        differs = np.abs(ndi.gaussian_filter(lum(small), 3.0) - ndi.gaussian_filter(gen_lum, 3.0)) > 0.12
        behind = sliver | (far(args.gap_margin) & (ndi.distance_transform_edt(~strong) < args.gap_margin) & differs)
        conf[behind] = 0
        print("   dropped behind occluders:", round(float(behind.sum() / S.sum()), 3), "of the view")
    # (c) A surface that turns away just before an interior occlusion edge (the rump's horizon with
    # the neck behind it, seen from the back) spreads its last few photo pixels over a long stretch
    # of surface, and a misalignment of those few pixels paints the far layer (the white collar)
    # across it. The side views' outline pixels show that ridge better (see edge_on).
    zb = np.where(S, depth, np.inf)                          # camera looks down -z: farther = smaller z
    horizon = S & (cosd < 0.5) & (depth - ndi.minimum_filter(zb, size=2 * args.horizon_px + 1) > 0.15)
    conf[ndi.binary_dilation(horizon, iterations=2) & S] = 0
    print("   dropped at interior horizons:", round(float(horizon.sum() / S.sum()), 3), "of the view")
    Image.fromarray(to_u8(np.where(S, cosd, 0))).save(f"{out}/dbg/cos_{k}{tag}.jpg", quality=90)
    conf[drop] = 0
    # Smooth outline for the trusted region (shrunk a little, never grown): the generated fallback
    # has a tiny blend weight, so any trace of confidence wins and a hard mask edge shows as stairs.
    conf = smoothstep((ndi.gaussian_filter(ndi.minimum_filter(conf, size=5), 2.0) - 0.2) / 0.6).astype(np.float32)
    aligned[k], body_conf[k] = warped, conf
    # debug: photo over the silhouette, green = mesh only, red = photo only
    dbg = np.asarray(small).copy()
    dbg[S & ~m1] = (0, 255, 0); dbg[m1 & ~S] = (255, 0, 0); dbg[~S & ~m1] = 255
    Image.fromarray(dbg).save(f"{out}/dbg/align_{k}{tag}.jpg", quality=90)
    Image.fromarray(to_u8(np.where(S, conf, 0))).save(f"{out}/dbg/conf_{k}{tag}.jpg", quality=90)



def with_conf(rgb, conf):
    n = rgb.shape[0]
    c = np.asarray(Image.fromarray(to_u8(conf)).resize((n, n), Image.BILINEAR))
    return np.dstack([rgb, c])


lap("whole-body alignment (4 views)")
# ---------------------------------------------------------------- head refinement
if head_on:
    rz = MeshRender(default_resolution=WORK, texture_size=64)              # zoomed camera, used for rasterising only
    rz.set_mesh(np.asarray(mesh_zoom.vertices, dtype=np.float32), np.asarray(mesh.faces), auto_center=False)
    hgen = [Image.open(p).convert("RGB") for p in hgen_paths]
    in_head = u_vert < args.head_cut[1] * height
    zoom_map = {}
    for k in aligned:
        if k in ("back", "top"):   # the head-only view from behind looks into the open neck, and the generated
            continue               # view from above is a featureless blob: neither is a usable target
        elev, azim = CAMS[k]
        print("head", k)
        # zoomed frame -> whole-body frame is an exact per-axis scale + offset (same ortho direction)
        gz, gf = grid_coords(rz, elev, azim, conv)[in_head], grid_coords(render, elev, azim, conv)[in_head]
        A = [np.polyfit(gz[:, i], gf[:, i], 1) for i in range(2)]
        resid = max(np.abs(np.polyval(A[i], gz[:, i]) - gf[:, i]).max() for i in range(2))
        assert resid < 1e-3, f"zoom/body frames are not related by scale + offset ({resid})"
        scale = torch.tensor([A[0][0], A[1][0]], device=dev, dtype=torch.float32)
        offset = torch.tensor([A[0][1], A[1][1]], device=dev, dtype=torch.float32)
        zoom_map[k] = (scale, offset)
        Sz, uu = raster(rz, elev, azim, WORK, attr=u_attr)
        w_head = smoothstep((args.head_cut[1] * height - uu) / ((args.head_cut[1] - args.head_cut[0]) * height)) * Sz
        before = sample_img(aligned[k], base_grid(WORK) * scale + offset)   # body-aligned photo, zoomed in
        target = np.asarray(hgen[GEN[k]].resize((WORK, WORK), Image.LANCZOS))
        grey = np.where(Sz[..., None], before, 127).astype(np.uint8)       # same backdrop as the generated view
        core = (w_head * ndi.binary_erosion(Sz, iterations=6)).astype(np.float32)
        pts, off = semantic_matches(grey, target, core)
        away = smoothstep(ndi.distance_transform_edt(w_head < 0.5) / (0.15 * WORK) - 1)   # 0 near the head, 1 beyond
        disp, base = fit_warp(Sz, Sz, matches=(pts, off), nu=args.head_nu, rest=away,
                              levels=((4, 0), (8, 0), (12, 0)), lam=args.head_lam)
        got = F.grid_sample(disp, pts[None, None], align_corners=True)[0, :, 0].T
        print("   match residual px@1024: median %.1f  p90 %.1f;  max shift %.1f" % (
            float((got - off).norm(dim=1).median()) * 512, float((got - off).norm(dim=1).quantile(0.9)) * 512,
            float((disp.abs() * torch.from_numpy(w_head)[None, None].to(dev)).max()) * 512))
        # Fade the correction out over a wide band around the head (a short band would stretch the
        # neck fur into streaks) and apply it to the whole-body image, so head and body stay one photo.
        Sf, uf = raster(render, elev, azim, WORK, attr=u_attr)
        core_f = (uf < 0.5 * (args.head_cut[0] + args.head_cut[1]) * height) & Sf
        band = args.head_fade * WORK * float(scale.abs().mean())               # px @1024 of the whole-body view
        fade = smoothstep(1 - ndi.distance_transform_edt(~core_f) / band).astype(np.float32)
        fade = F.interpolate(blur(torch.from_numpy(fade)[None, None].to(dev), 3.0), size=(R, R), mode="bilinear", align_corners=True)
        full = base_grid(R)
        Dz = F.grid_sample(disp, (full - offset) / scale, padding_mode="border", align_corners=True) * fade   # zoom units
        aligned[k] = sample_img(aligned[k], full + Dz.permute(0, 2, 3, 1) * scale)
        if os.environ.get("HEAD_DEBUG"):
            g = (full + Dz.permute(0, 2, 3, 1) * scale)[0]                      # [R,R,2] sampling positions
            dx, dy = g[:, 1:] - g[:, :-1], g[1:] - g[:-1]
            det = (dx[:-1, :, 0] * dy[:, :-1, 1] - dx[:-1, :, 1] * dy[:, :-1, 0]) * (R - 1) ** 2 / 4
            print("   jacobian det min %.2f max %.2f, folded px %d; |disp| zoom px: max %.0f, at frame border max %.0f" % (
                float(det.min()), float(det.max()), int((det <= 0).sum()), float(disp.norm(dim=1).max()) * 512,
                float(torch.cat([disp[..., 0, :], disp[..., -1, :], disp[..., :, 0], disp[..., :, -1]], -1).norm(dim=1).max()) * 512))
            Image.fromarray(to_u8((disp.norm(dim=1)[0] * 512 / 150).cpu().numpy())).save(f"{out}/dbg/headdisp_{k}{tag}.png")
            Image.fromarray(to_u8(F.interpolate(fade, size=(WORK, WORK))[0, 0].cpu().numpy())).save(f"{out}/dbg/headfade_{k}{tag}.png")
        after = sample_img(aligned[k], base_grid(WORK) * scale + offset)
        Image.fromarray(np.hstack([before, after, target])).save(f"{out}/dbg/head_{k}{tag}.jpg", quality=90)


# ---------------------------------------------------------------- head close-ups
# A close-up shows the head at ~3x the resolution of the whole-body photo. The whole-body photo of
# the same view is already aligned with the head geometry (above) and shows the same cat from the
# same direction, so the close-up is registered to it: a similarity transform from DINOv2
# correspondences, then a smooth warp driven by correspondences and by the low-pass luminance.
# It is then one more photo of the head, with a weight that reflects its resolution, fading out
# towards the neck.
close_views = []
if head_on and args.head_close:
    rzR = MeshRender(default_resolution=R, texture_size=args.tex)
    rzR.load_mesh(mesh_zoom, auto_center=False)
    rzR.bake_unreliable_kernel_size = 6
    for k in ("front", "left", "right"):
        path = f"{args.head_close}/{k}.png"
        if k not in zoom_map or not os.path.exists(path):
            continue
        print("close-up", k)
        elev, azim = CAMS[k]
        scale, offset = zoom_map[k]
        Sz, uu = raster(rz, elev, azim, WORK, attr=u_attr)
        assert iou(raster(rzR, elev, azim, WORK)[0], Sz) > 0.99, "zoomed renderers disagree"
        w_head = (smoothstep((args.head_cut[1] * height - uu) / ((args.head_cut[1] - args.head_cut[0]) * height)) * Sz).astype(np.float32)
        ref = sample_img(aligned[k], base_grid(WORK) * scale + offset)                 # whole-body photo, zoomed in
        ref_grey = np.where(Sz[..., None], ref, 127).astype(np.uint8)
        a = np.asarray(Image.open(path).convert("RGBA").resize((WORK, WORK), Image.LANCZOS))
        full_res = np.asarray(Image.open(path).convert("RGBA"))
        pm = full_res[..., 3] > 127
        core_p = ndi.binary_erosion(full_res[..., 3] > 250, iterations=args.rim_px)
        rgb = np.ascontiguousarray(bleed(full_res[..., :3], core_p))
        src_grey = np.where(a[..., 3:] > 127, a[..., :3], 127).astype(np.uint8)
        core = (w_head * ndi.binary_erosion(Sz, iterations=6)).astype(np.float32)
        # 1. similarity: where each patch of the zoomed view is in the close-up, anywhere in the image
        pts, off = semantic_matches(src_grey, ref_grey, core, radius=73)
        if len(pts) < 30:
            print("   too few correspondences, close-up not used"); continue
        y = pts + off
        xc, yc = pts.mean(0), y.mean(0)
        s_fit = float(((pts - xc) * (y - yc)).sum() / ((pts - xc) ** 2).sum())
        t_fit = yc - s_fit * xc
        resid = ((s_fit * pts + t_fit) - y).norm(dim=1) * WORK / 2
        print(f"   similarity: close-up shows the head {1 / s_fit:.2f}x larger than the zoomed view; residual px median {float(resid.median()):.1f} p90 {float(resid.quantile(0.9)):.1f}")
        sim = lambda g: g * s_fit + t_fit                                               # zoomed-frame coords -> close-up coords
        placed = sample_img(rgb, sim(base_grid(WORK)))
        inside = lambda g: F.grid_sample(torch.from_numpy(core_p.astype(np.float32))[None, None].to(dev), g, padding_mode="zeros", align_corners=True)[0, 0]
        m_placed = inside(sim(base_grid(WORK))).cpu().numpy() > 0.5
        # 2. non-rigid: correspondences again (now local) + low-pass luminance against the whole-body photo
        placed_grey = np.where((m_placed & Sz)[..., None], placed, 127).astype(np.uint8)
        pts, off = semantic_matches(placed_grey, ref_grey, core * m_placed, radius=8)
        away = smoothstep(ndi.distance_transform_edt(w_head < 0.5) / (0.15 * WORK) - 1)
        disp, base = fit_warp(Sz, Sz, src_lum=lum(placed), dst_lum=lum(ref), mu=args.close_mu, where=core * m_placed,
                              matches=(pts, off), nu=args.head_nu, rest=away,
                              levels=((4, 16.0), (8, 12.0), (16, 8.0), (32, 5.0)), lam=args.head_lam)
        got = F.grid_sample(disp, pts[None, None], align_corners=True)[0, :, 0].T
        print("   match residual px@1024: median %.1f  p90 %.1f;  max shift %.1f" % (
            float((got - off).norm(dim=1).median()) * 512, float((got - off).norm(dim=1).quantile(0.9)) * 512,
            float((disp.abs() * torch.from_numpy(w_head)[None, None].to(dev)).max()) * 512))
        gridW = sim(base + disp.permute(0, 2, 3, 1))
        gridR = sim(base_grid(R) + F.interpolate(disp, size=(R, R), mode="bicubic", align_corners=True).permute(0, 2, 3, 1))
        closeR = sample_img(rgb, gridR)
        closeW = sample_img(rgb, gridW)
        # colour: bring the close-up's low frequencies onto the whole-body photo's (two renders of the
        # same cat differ a little in exposure and white balance), so the join does not show
        valid = torch.from_numpy((inside(gridW).cpu().numpy() > 0.5) & Sz).float().to(dev)[None, None]
        lowp = lambda im: blur(torch.from_numpy(im).float().permute(2, 0, 1)[:, None].to(dev) / 255 * valid, 24.0) / blur(valid, 24.0).clamp(min=1e-3)
        gain = (lowp(ref) / lowp(closeW).clamp(min=0.02)).clamp(0.8, 1.25).permute(1, 0, 2, 3)          # [1,3,W,W]
        print("   colour gain (min, median, max):", [round(float(v), 3) for v in (gain.min(), gain.median(), gain.max())])
        gainR = F.interpolate(gain, size=(R, R), mode="bilinear", align_corners=True)[0].permute(1, 2, 0).cpu().numpy()
        closeR = to_u8(closeR / 255.0 * gainR)
        # confidence: on the head only, inside the close-up's cut-out, and where the whole-body photo is trusted
        cb = F.grid_sample(torch.from_numpy(body_conf[k])[None, None].to(dev), base_grid(WORK) * scale + offset, align_corners=True)[0, 0].cpu().numpy()
        conf = w_head * cb * ndi.gaussian_filter(ndi.binary_erosion(valid[0, 0].cpu().numpy() > 0.5, iterations=6).astype(np.float32), 3.0)
        density[f"head_{k}"] = s_fit * full_res.shape[0] / WORK / float(scale.abs().mean())
        close_views.append((f"head_{k}", rzR, with_conf(closeR, conf), elev, azim))
        Image.fromarray(np.hstack([ref, to_u8(closeW / 255.0 * gain[0].permute(1, 2, 0).cpu().numpy()), to_u8(np.dstack([conf] * 3))])).save(f"{out}/dbg/close_{k}{tag}.jpg", quality=92)

view_w = {k: (density[k] / max(density[b] for b in aligned)) ** args.res_pow for k in density}
print("view weights", {k: round(v, 2) for k, v in view_w.items()})
views = [(render, with_conf(aligned[k], body_conf[k]), *CAMS[k], view_w[k]) for k in CAMS if k in aligned]
views += [(rd, img, elev, azim, view_w[name]) for name, rd, img, elev, azim in close_views]

lap("head refinement")
# ---------------------------------------------------------------- bake
ones = np.full((R, R, 1), 255, np.uint8)
genR = [np.dstack([np.asarray(g.resize((R, R), Image.LANCZOS)), ones]) for g in gen]
for i, d in gen_drop.items():
    keep = 1 - ndi.gaussian_filter(d.astype(np.float32), 3.0)
    genR[i][..., 3] = np.asarray(Image.fromarray(to_u8(keep)).resize((R, R), Image.BILINEAR))
views.append((render, genR[5], gen_elevs[5], gen_azims[5], 1.0))                        # underside
views += [(render, genR[i], gen_elevs[i], gen_azims[i], args.gen_w) for i in range(5)]  # used only where no photo is trusted
n_photo = len(aligned) + len(close_views)
acc = [torch.zeros(args.tex, args.tex, 3, device=dev) for _ in range(2)]        # photos, generated
wsum = [torch.zeros(args.tex, args.tex, 1, device=dev) for _ in range(2)]
support = [torch.zeros(args.tex, args.tex, device=dev) for _ in range(2)]       # how well any view of each kind sees a texel
for i, (rd, img, elev, azim, w) in enumerate(views):
    tex, cos, _ = rd.back_project(Image.fromarray(img), elev, azim)
    wgt = w * cos ** args.exp * tex[..., 3:]
    acc[i >= n_photo] += tex[..., :3] * wgt; wsum[i >= n_photo] += wgt
    support[i >= n_photo] += (cos ** args.exp * tex[..., 3:]).squeeze(-1)
    if os.environ.get("TEX_DEBUG"):                       # which view paints each texel
        if i == 0:
            who_w = torch.zeros(args.tex, args.tex, 1, device=dev); who = torch.zeros(args.tex, args.tex, 3, device=dev)
        if i < n_photo:
            code = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1), (1, .5, 0), (.5, 1, 0), (0, .5, 1)][i if i < len(aligned) else 5 + i - len(aligned)]
            who = torch.where(wgt > who_w, torch.tensor(code, device=dev, dtype=torch.float32), who); who_w = torch.maximum(who_w, wgt)
    del tex, cos
tp, tg = acc[0] / wsum[0].clamp(min=1e-8), acc[1] / wsum[1].clamp(min=1e-8)
# The paint model's views are "delighted" and their levels differ a little from the photos'. Bring them
# onto the photos' white and dark levels (one gain + offset per channel, from the texels both see well).
both = (support[0] > 0.2) & (support[1] > 0.2)
white = both & (tp.mean(-1) > 0.7) & (tg.mean(-1) > 0.7)
dark = both & (tp.mean(-1) < 0.4) & (tg.mean(-1) < 0.4)
if white.sum() > 10000 and dark.sum() > 10000:
    pw, gw, pd, gd = tp[white].median(0).values, tg[white].median(0).values, tp[dark].median(0).values, tg[dark].median(0).values
    gain = ((pw - pd) / (gw - gd)).clamp(0.8, 1.25)
    tg = ((tg - gw) * gain + pw).clamp(0, 1)
    print("generated views: white", gw.cpu().numpy().round(3), "->", pw.cpu().numpy().round(3),
          "dark", gd.cpu().numpy().round(3), "->", pd.cpu().numpy().round(3))


def edge_on(need, tol=0.08, cos0=-0.1, cos1=0.2, chunk=2_000_000):
    """Photo colours for texels the photos see (almost) edge-on, where forward projection gives up:
    the ridge of the back between the two side views. Each texel looks up its own position in the
    photo, so the photo's outermost pixels are stretched over the ridge -- along the direction the
    side views cannot resolve, sharp along the body. Returns (colour, support)."""
    acc, ws = torch.zeros(args.tex, args.tex, 3, device=dev), torch.zeros(args.tex, args.tex, device=dev)
    idx = torch.nonzero(need)
    # Only surfaces that face upwards: what faces down is in shadow in the photos' outermost pixels
    # (belly, paw pads), and the generated view from below shows it properly.
    up = (texel_attr(cam_cos(90, 0)[1])[idx[:, 0], idx[:, 1], 0] / 0.3).clamp(0, 1)
    for name, (rd, img, elev, azim, _) in zip([k for k in CAMS if k in aligned], views[:n_photo]):
        if name not in args.edge_views:
            continue
        cam, cosv = cam_cos(elev, azim)
        g = torch.from_numpy(grid_coords(render, elev, azim, conv).copy()).float().to(dev)
        ta = texel_attr(torch.cat([g, cam[:, 2:3], cosv], 1))[idx[:, 0], idx[:, 1]]
        Sv, dv = raster(render, elev, azim, 2048)
        dmap = torch.from_numpy(bleed(dv, Sv)).float().to(dev)[None, None]
        im = torch.from_numpy(img).to(dev).permute(2, 0, 1)[None].float() / 255
        conf = im[:, 3:]
        for c in range(0, len(idx), chunk):
            t = ta[c:c + chunk]
            gg = t[None, None, :, :2]
            vis = (t[:, 2] - F.grid_sample(dmap, gg, align_corners=True)[0, 0, 0]).abs() < tol
            w = ((t[:, 3] - cos0) / (cos1 - cos0)).clamp(0, 1) ** 2 * vis * F.grid_sample(conf, gg, align_corners=True)[0, 0, 0] * up[c:c + chunk]
            col = F.grid_sample(im[:, :3], gg, mode="bilinear", padding_mode="border", align_corners=True)[0, :, 0].T
            i = idx[c:c + chunk]
            acc[i[:, 0], i[:, 1]] += col * w[:, None]; ws[i[:, 0], i[:, 1]] += w
        del im, conf, dmap
    return acc / ws.clamp(min=1e-8)[..., None], ws


def from_neighbours(src_mask, dst_mask, values, n_anchor=200_000, k_smooth=64, k=8):
    """Smooth extension of `values` from the src texels to the dst texels through 3D space.

    A sparse set of src texels serves as anchors, each carrying the mean over its k_smooth nearest
    anchors (a low-pass of ~1.5 % of the body length: fur detail averages out, markings stay).
    Every dst texel blends its k nearest anchors. Returns (dst indices, values, mean distance)."""
    from scipy.spatial import cKDTree
    src, dst = torch.nonzero(src_mask), torch.nonzero(dst_mask)
    src = src[torch.randperm(len(src), device=dev)[:n_anchor]]
    pts = tpos[src[:, 0], src[:, 1]].cpu().numpy()
    tree = cKDTree(pts)
    _, j = tree.query(pts, k=min(k_smooth, len(pts)), workers=8)
    anchor = values[src[:, 0], src[:, 1]][torch.from_numpy(j).to(dev)].mean(1)   # [n_anchor, C]
    d, j = tree.query(tpos[dst[:, 0], dst[:, 1]].cpu().numpy(), k=k, workers=8)
    wk = torch.from_numpy(1.0 / (d ** 2 + d[:, :1] ** 2 + 1e-10)).float().to(dev)
    out = (anchor[torch.from_numpy(j).to(dev)] * wk[..., None]).sum(1) / wk.sum(1, keepdim=True)
    return dst, out, torch.from_numpy(d.mean(1)).float().to(dev)


# Compose the texture.
#  - Where a photo sees the surface well, the photos alone.
#  - Elsewhere the generated views, levelled to the photos: next to photo-painted surface their
#    colour is shifted by the difference the two sources show on that surface, fading out with
#    distance. This hides the join, and where the generated views invent a marking that the photos
#    contradict right beside it (white strips in folds of tabby fur), it follows the photos.
#  - Surfaces nothing sees well (inner faces of the legs, deep folds) get the colours of the
#    nearest well-seen surface in 3D; UV-space inpainting would drag colours across the atlas.
tpos = texel_attr(render.vtx_pos[:, :3])
painted = (wsum[0] + wsum[1]).squeeze(-1) > 1e-8
inside = float((painted & on_mesh).sum() / painted.sum())
assert inside > 0.9, f"UV rasterisation does not match the baked texels ({inside:.3f})"
a_photo = torch.from_numpy(smoothstep((support[0] / args.min_support).cpu().numpy())).to(dev)
a_gen = torch.from_numpy(smoothstep((support[1] / args.min_support).cpu().numpy())).to(dev)
if head_on:
    # The head has photos from every side; what they miss (under the chin, inside folds) is small
    # and the generator likes to draw a pink mouth there: fill it from its neighbours instead.
    u_tex = texel_attr(u_attr)[..., 0]
    a_gen = a_gen * (1 - torch.from_numpy(smoothstep(((args.head_cut[1] * height - u_tex) / ((args.head_cut[1] - args.head_cut[0]) * height)).cpu().numpy())).to(dev))
te, e_sup = edge_on((a_photo < 1) & on_mesh)
a_edge = (1 - a_photo) * torch.from_numpy(smoothstep((e_sup / args.edge_support).cpu().numpy())).to(dev)
w_p = a_photo
w_g = (1 - a_photo - a_edge) * a_gen
dst, off, dist = from_neighbours((a_photo >= 1) & (a_gen >= 1) & on_mesh, (w_g > 0) & on_mesh, tp - tg)
near = 1 - torch.from_numpy(smoothstep(((dist - args.feather[0]) / (args.feather[1] - args.feather[0])).cpu().numpy())).to(dev)
levelled = tg.clone()
# (brightness only: a colour offset taken where the generator drew a pink mouth turns its neighbours green)
levelled[dst[:, 0], dst[:, 1]] = (tg[dst[:, 0], dst[:, 1]] + near[:, None] * off.mean(1, keepdim=True)).clamp(0, 1)
texture = w_p[..., None] * tp + a_edge[..., None] * te + w_g[..., None] * levelled
seen = w_p + a_edge + w_g
if ((seen < 1) & on_mesh).any():
    texture = texture / seen.clamp(min=1e-6)[..., None]
    dst2, fill, _ = from_neighbours((seen >= 1) & on_mesh, (seen < 1) & on_mesh, texture)
    t = seen[dst2[:, 0], dst2[:, 1]][:, None]
    texture[dst2[:, 0], dst2[:, 1]] = t * texture[dst2[:, 0], dst2[:, 1]] + (1 - t) * fill
print(f"surface texels: photos {float((a_photo >= 1)[on_mesh].float().mean()):.3f}, photos edge-on {float(a_edge[on_mesh].mean()):.3f}, "
      f"generated {float(((a_photo < 1) & (a_gen >= 1))[on_mesh].float().mean()):.3f}, "
      f"filled from neighbours {float((seen < 1)[on_mesh].float().mean()):.3f}")
mask = on_mesh          # texels splatted just outside the charts carry stray colours: pad from the charts instead
texture = torch.tensor(render.uv_inpaint(texture, to_u8(mask.cpu().numpy())) / 255).float().to(dev)
render.set_texture(texture)
res = render.save_mesh()
res.export(f"{out}/textured{tag}.glb")
Image.fromarray(to_u8(texture.cpu().numpy())).save(f"{out}/texture{tag}.png")
lap("bake + compose + export")
print("done", f"{out}/textured{tag}.glb")
if os.environ.get("TEX_DEBUG"):
    who = torch.where((a_photo < 1)[..., None], torch.tensor((1.0, 1.0, 1.0), device=dev) * a_gen[..., None], who)
    who = torch.where((a_edge > 0.5)[..., None], torch.tensor((0.0, 1.0, 1.0), device=dev), who)   # cyan = photos, edge-on
    render.set_texture(who)                                               # white = generated, black = filled from neighbours
    render.save_mesh().export(f"{out}/who{tag}.glb")
