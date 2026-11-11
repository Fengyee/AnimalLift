# cat_recon: photos of one cat → textured 3D model

A reconstruction workflow that is separate from the AnimalLift model in the rest of this
repository. It turns a set of generated turnaround views of one cat into a textured mesh
(`model.glb`, 300k faces, 4096² texture):

1. **Reference views** – an image model (GPT Image) draws the cat from fixed directions, from a
   few real photos. This step is manual; the prompt is below.
2. **Shape** – Hunyuan3D-2mv from the four whole-body views, plus a second run on the head alone
   that is grafted onto the body (the whole-body run sees the face at ~100 px).
3. **Texture** – the reference views themselves are projected onto the mesh. The Hunyuan paint
   model only fills what no view shows (belly, inner legs).

About 12.5 minutes per cat on one RTX A5000 (24 GB); peak GPU memory about 15 GB.

Rigging and animation are not part of this workflow. The neutral standing pose that the prompt
asks for is the bind pose a later rigging step would need.

## 1. Reference views

In one conversation with the image model, attach 3–6 real photos of the cat and send the prompt
once per view, changing only the `View:` line. Generate LEFT first, then RIGHT, TOP, FRONT, BACK,
then the three head close-ups, so each image can be checked against the earlier ones.

```
Using the attached photos of my cat as the identity reference, create one image
for a 3D-modeling turnaround reference set. All images in this set must show the
SAME cat in the SAME pose with the SAME markings; only the camera changes.

View: [one of]
  FRONT   – camera exactly head-on, at the cat's mid-body height
  LEFT    – exact 90° profile, cat facing the LEFT edge of the image
  RIGHT   – exact 90° profile, cat facing the RIGHT edge of the image
  BACK    – camera exactly behind, at the cat's mid-body height
  TOP     – camera directly above, looking straight down, head toward the top
            edge of the image, spine vertical in the image
  HEAD FRONT / HEAD LEFT / HEAD RIGHT – same three directions as above, but
            framed on the head and neck only, the head filling ~80% of the image

Pose (identical in every view):
- Neutral standing pose, all four legs straight and vertical, paws side by side,
  head facing straight forward in line with the spine, ears upright, mouth closed.
- Tail held straight out behind the body, horizontal, in line with the spine –
  in the BACK view it therefore points at the camera and appears as a short
  round stub, NOT hanging down.

Consistency (check against the views already generated in this conversation):
- Same body proportions in every view: the legs take the same fraction of the
  cat's height in the BACK and FRONT views as in the side views.
- Same markings in every view: the boundary between white and tabby fur on the
  neck, chest, belly and each leg is at the same place; the back of the neck has
  the same color in the side, back and top views; any spot on a leg is at the
  same height in every view that shows it.
- Stocky, round-bodied adult cat; the body is as wide in the FRONT and BACK
  views as it is in the TOP view.

Rendering:
- Orthographic-style, no perspective distortion, whole subject in frame with
  margin, centered, square image.
- No whiskers, no long stray hairs; short, smooth, close-lying coat.
- Flat, even, shadowless studio lighting; no cast shadow; no highlights.
- Plain solid light-gray background (#BFBFBF), nothing else in the image.
```

Adapt the "Stocky, round-bodied" line to the cat. Save the images (png, jpg or webp, ≥1024 px)
in one directory under these names:

| file | required | used for |
|---|---|---|
| `front` `left` `back` `right` | yes | body shape and texture. `left` = cat faces the left image edge |
| `top` | recommended | texture of the back, the top of the head and the tail |
| `head_front` `head_left` `head_right` | recommended | head shape and face texture (all three, or none) |

Before running, look at the set side by side and regenerate a view if

- the pose differs (a leg mid-stride, head turned, tail hanging in one view),
- a marking moves or changes (white on the legs ends at a different height, a collar present in
  one view only),
- whiskers or a cast shadow are visible.

The pipeline corrects small disagreements (it registers the front, back and top views to the
side views), but it cannot invent agreement: where two views contradict each other the side
views win.

## 2. Environment

`pipeline.sh` runs everything with one Python interpreter that has Hunyuan3D-2 installed. This
is not the `animallift` environment of the main README.

- Hunyuan3D-2 (https://github.com/Tencent-Hunyuan/Hunyuan3D-2, tested at commit `f8db630`),
  installed with `pip install -e .`, with its `custom_rasterizer` and `differentiable_renderer`
  extensions built as its README describes.
- Tested with Python 3.10, torch 2.4.0+cu124, transformers 4.48.3, diffusers 0.35.2,
  rembg 2.0.69 (CPU onnxruntime 1.23.2), xatlas, trimesh 5.2, pymeshlab, scikit-image, scipy.
- Models are downloaded on first use into `HF_HOME` (`tencent/Hunyuan3D-2mv`,
  `tencent/Hunyuan3D-2`, `facebook/dinov2-large`) and `U2NET_HOME` (`birefnet-general`).
- Blender 4.2 is needed only by `review.sh`.

Copy `env.example.sh` to `env.sh` (untracked) and set the paths.

Hunyuan3D-2 is released under the Tencent Hunyuan 3D 2.0 Community License, which restricts the
territory of use and has additional terms for commercial use. Read it before using models from
this workflow in a product.

## 3. Run

```bash
cat_recon/pipeline.sh <refs_dir> <out_dir>      # -> <out_dir>/model.glb, texture.png, timing.tsv
cat_recon/review.sh <out_dir>                   # -> <out_dir>/review/*.jpg
```

Stages and their time on an RTX A5000 with 8 CPU cores (8 reference images):

| stage | script | time | peak GPU |
|---|---|---|---|
| cut-outs | `prep_all.py` (rembg, CPU) | ~110 s | – |
| body + head shape | `gen_shapes.py` | ~150 s | ~10 GB |
| graft head onto body | `merge_head.py` | ~65 s | 3 GB |
| texture | `texture_photos.py` | ~410 s | ~15 GB |

Each stage writes `log_<stage>.txt`; the texture stage also writes debug images to
`<out_dir>/tex/dbg/` (how each view was aligned, where it is trusted). Re-running
`texture_photos.py` on the same output directory reuses the unwrapped mesh and the generated
fallback views, so texture-only iterations take about 4.5 minutes. With `TEX_DEBUG=1` it also
writes `who.glb`, colour-coded by which view paints each part of the surface.

## 4. Check the result

Always look at the review sheets before using a model; thumbnails hide the defects that matter.

- `input_vs_mesh.jpg`: outline and markings against every reference view.
- `head_front.jpg`, `head_back.jpg` at full size: eyes, nose and mouth in the right place and
  the same from the front, three-quarter and side views; no hump or step where the head joins
  the neck. (The black disc in the rear head view is the tail, cut by the camera.)
- `body2.jpg`, `tb.jpg`: tail tip, hind legs from behind, spine, underside.

Lines in `log_texture.txt` worth reading when something looks wrong:

- `registered to the side photos ... vertical shift px@1024` – how far a view had to be moved to
  agree with the side views. More than ~40 px means that reference view is inconsistent.
- `tail:` – whether the tail in the back view was taken as end-on, or as a different pose (then
  that part of the back view is not used).
- `close-up ... similarity` and `match residual` – how well each head close-up was registered.
- `surface texels: photos ..., generated ..., filled ...` – typically ~0.8 from the references.

In `log_merge_head.txt`, `registration final ... IoU` should be above ~0.9 for all three views,
and `side-photo outline` reports where the head run stuck out of the body outline and was
handed back to the body shape.

## Known limits

- The underside (belly, inner legs, underside of the tail) has no reference view; it is
  generated and soft.
- Geometry of the head carries some fur relief from the close-ups, the body is smooth.
- Long whiskers and long fur are not reconstructed; the prompt asks the image model to leave
  them out.
- Tested on short-haired cats in the standing pose only.
