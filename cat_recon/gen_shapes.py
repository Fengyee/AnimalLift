"""Hunyuan3D-2mv geometry for several view sets with one model load.
usage: python gen_shapes.py <octree_res> <views_dir> <out_dir> [<views_dir> <out_dir> ...]
Each views_dir holds front/left/back/right RGBA (any subset). Saves <out_dir>/grid.npy (occupancy
logits) and shape_raw.glb, like gen_shape.py."""
import os, sys, time
import numpy as np, torch, torch.nn.functional as F
from PIL import Image
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

res = int(sys.argv[1])
jobs = list(zip(sys.argv[2::2], sys.argv[3::2]))
chunks = int(os.environ.get("NUM_CHUNKS", 100000))
t = time.time()
pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
    "tencent/Hunyuan3D-2mv", subfolder="hunyuan3d-dit-v2-mv", variant="fp16")
print("model load s", round(time.time() - t, 1), flush=True)
cur = {}


@torch.no_grad()
def sparse_decoder(latents, geo_decoder, bounds=1.01, num_chunks=100000, octree_resolution=512, **kw):
    """Same dense logits grid as the stock decoder near the surface, at a fraction of the queries.
    The field saturates at about +-1 a few voxels from the surface and only its unsaturated band
    (and its sign elsewhere) is used downstream. So: evaluate every 2nd grid point, then every
    point within one coarse cell of a coarse sample that is unsaturated or differs in sign from a
    neighbour; the rest is interpolated from the coarse samples (saturated, correct sign).
    Checked against dense grids: no unsaturated voxel and no sign is missed."""
    cur["t_dec"] = time.time()
    n = octree_resolution + 1
    ax = torch.from_numpy(np.linspace(-bounds, bounds, n, dtype=np.float32)).to(latents.device, dtype=latents.dtype)   # as the stock decoder

    def query(ix):                                   # ix [m,3] integer grid indices -> logits [m]
        out = []
        for s in range(0, len(ix), num_chunks):
            q = ax[ix[s:s + num_chunks]]
            out.append(geo_decoder(queries=q[None], latents=latents)[0, :, 0].float())
        return torch.cat(out)

    c = torch.arange(0, n, 2, device=latents.device)
    coarse = query(torch.cartesian_prod(c, c, c)).view(len(c), len(c), len(c))
    act = (coarse.abs() < 0.99).float()[None, None]
    inside = (coarse > 0).float()[None, None]
    act = torch.maximum(act, F.max_pool3d(inside, 3, 1, 1) + F.max_pool3d(-inside, 3, 1, 1))   # sign change nearby
    act = F.max_pool3d(act, 3, 1, 1)
    fine = F.interpolate(act, size=(n, n, n), mode="trilinear", align_corners=True)[0, 0] > 0
    grid = F.interpolate(coarse[None, None], size=(n, n, n), mode="trilinear", align_corners=True)[0, 0]
    ix = torch.nonzero(fine)
    grid[fine] = query(ix)
    cur["frac"] = len(c) ** 3 / n ** 3 + len(ix) / n ** 3
    np.save(f"{cur['out']}/grid.npy", grid.cpu().numpy().astype(np.float16))
    cur["t_done"] = time.time()
    return grid[None]


dense = pipe.vae.volume_decoder
def dense_hook(*a, **k):
    cur["t_dec"] = time.time()
    g = dense(*a, **k)
    np.save(f"{cur['out']}/grid.npy", g[0].float().cpu().numpy().astype(np.float16))
    cur["t_done"] = time.time(); cur["frac"] = 1.0
    return g
hook = dense_hook if os.environ.get("DENSE_DECODE") else sparse_decoder
if os.environ.get("CHECK_DECODE"):      # decode the same latents both ways and compare
    def hook(*a, **k):
        gs = sparse_decoder(*a, **k)
        gd = dense(*a, **k).float()
        band = gd.abs() < 0.9
        print("check: sign mismatches", int(((gs > 0) != (gd > 0)).sum()), "max |diff| in band", float((gs - gd).abs()[band].max()), flush=True)
        return gs
pipe.vae.volume_decoder = hook
for src, out in jobs:
    os.makedirs(out, exist_ok=True)
    images = {k: Image.open(f"{src}/{k}.png").convert("RGBA") for k in ("front", "left", "back", "right")
              if os.path.exists(f"{src}/{k}.png")}
    cur["out"] = out
    t = time.time()
    mesh = pipe(image=images, num_inference_steps=50, octree_resolution=res, num_chunks=chunks,
                generator=torch.manual_seed(12345), output_type="trimesh")[0]
    print(f"{out}: views {list(images)}; diffusion {cur['t_dec'] - t:.1f} s, decode {cur['t_done'] - cur['t_dec']:.1f} s, "
          f"surface {time.time() - cur['t_done']:.1f} s; queries {cur['frac']:.3f} of dense; peak mem {torch.cuda.max_memory_allocated() / 2 ** 30:.1f} GiB; "
          f"extents {mesh.extents.round(3)}", flush=True)
    mesh.export(f"{out}/shape_raw.glb")
