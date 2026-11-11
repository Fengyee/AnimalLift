# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

AnimalLift: single image → animal mesh (fixed reference topology), UV texture, and hair map. Python/PyTorch does training and inference; `load_blender.py` runs inside Blender 4.2 to turn the predictions into a hair-groomed `.blend` scene. This is a public release repo (README is user-facing docs).

## Commands

Environment: Python 3.10, pinned deps in `requirements.txt` (PyTorch 2.4.0+cu124). The README uses a conda env named `animallift`; on nexus, conda owns the env and `uv pip` installs into it.

```bash
# Checks (no torch/Blender/checkpoint needed for the unit tests)
python -m unittest discover -s tests -v
python -m unittest tests.test_portability.PortabilityTests.test_portable_defaults   # single test
python -m compileall -q .
python train.py --help; python test.py --help

# Inference (needs ckpt/latest.pt from `hf download Chunyi99/AnimalLift --local-dir ckpt`)
python test.py --max_images 1 --batch_size 1          # -> outputs/test/
python test.py --input_dir my_images --output_dir outputs/my_test --checkpoint path.pt

# Blender export (headless); one sample per invocation, auto-saves <results_dir>/blender/<sample>.blend
blender --background --python-exit-code 1 --python load_blender.py -- --results_dir outputs/test --sample animal_2

# Training (needs dataset in data/Animal_Textures/<animal>/dataset/, see docs/DATASET.md)
python train.py --animals dog wolf --batch_size 8
python train.py --no-all_animals --data_root data/Animal_Textures/dog/dataset
python train.py --resume outputs/train/multitask/checkpoints/latest.pt
```

## Architecture

- **Entry points are thin.** `train.py` and `test.py` only define argparse options and validate paths, then lazily import heavy modules (`animallift.model.train_multitask/infer_multitask`, `animallift.inference.run`). Keep torch imports out of the argument-parsing path: `tests/test_portability.py` imports `train`/`test` without torch installed.
- **`animallift/model.py`** (~2.7k lines) holds everything model-side: dataset (`GroupedImageTextureMeshHairDataset`), OBJ/hair I/O, the network, losses, and the training loop. The core `AnimalTextureMeshHairFlowModel`: DINOv2 (timm) encoder tokens + global color token + DINO intermediate tokens condition a UNet flow-matching model in the latent space of a **frozen pretrained SD VAE** (`stabilityai/sd-vae-ft-mse`, via diffusers), sampled with CFG. Mesh (`MeshTokenHead`, per-vertex offsets from a mean/template mesh) and hair (`HairUNetHead`, plus hair region length/curl heads) branch off the clean final DINO tokens, so CFG condition dropout doesn't affect geometry.
- **`animallift/inference.py`** loads a checkpoint and rebuilds the model from the checkpoint's saved `args` (`build_runtime_args`); only runtime options (paths, device, batch size, NFE, CFG scale, VAE loading) come from the CLI. It imports the training module by name or path (`--module_name`, default `animallift.model`), so model code changes must stay compatible with `restore_checkpoint_model_args` and existing checkpoints. Only the SD-VAE multitask checkpoint format is supported (`vae_config.type == "stable_diffusion_autoencoderkl"`); the VAE identity is validated against the checkpoint.
- **`animallift/config.py`** handles config for all entry points, including Blender. Precedence: CLI > `--config FILE` (or `config.local.json` if no `--config`) > `config.json`. Each script reads one section (`train`, `test`, `blender`, `dataset_generation`); `shared` is merged into `train`/`test`. Override files can only set keys that already exist in `config.json` (unknown keys raise an error), so **a new config key must be added to `config.json`** first, and to `PATH_KEYS` if it's a path. All relative paths, CLI ones included, resolve against the project root, not the CWD.
- **`load_blender.py`** runs under Blender's bundled Python (`bpy`, `bmesh`, `mathutils`). It decodes the predicted hair NPZ into curves on the mesh using `assets/hair_map_meta_uvanchor.npz` (UV face/barycentric anchors), adds Geometry Nodes groom modifiers from Blender's Essentials hair assets, builds materials and underhair, then saves with packed images. `tests/test_blender_save.py` stubs the Blender modules to test save orchestration.
- **Output contract** between `test.py` and Blender: `<results_dir>/{meshes/<s>_mesh.obj, textures/<s>_uv.png, hair/<s>_hair.npz, manifest.csv}`; `config.select_result_obj`/`result_paths` encode it.
- **Topology invariant:** `assets/reference.obj`, the training shapes, the checkpoint template, and the hair anchor metadata must all share the same vertex order, faces, and UVs. Matching vertex counts is not enough.
- `dataset_generation/gen_dog_dataset.py` (Qwen-Image-Edit UV texture edits) needs a separate environment and is not covered by the pinned requirements.

Dataset layout and pairing rules are in `docs/DATASET.md`. Controls and textures are paired **by sorted index, not by filename**, and `.pt` hair caches take precedence over `.npz`. Asset files are described in `docs/ASSETS.md`.

## cat_recon (separate workflow)

`cat_recon/` is a self-contained reconstruction workflow that does not use the AnimalLift model: generated turnaround views of one cat → Hunyuan3D-2mv shape (body + grafted head) → reference views projected as texture. `cat_recon/README.md` has the image prompt, environment, stage timings and the checks to run on a result. It needs its own environment (Hunyuan3D-2), configured in the untracked `cat_recon/env.sh`:

```bash
cat_recon/pipeline.sh <refs_dir> <out_dir>   # -> model.glb, texture.png, timing.tsv
cat_recon/review.sh <out_dir>                # flat-lit review sheets (needs Blender 4.2)
```

Always inspect the review sheets (head close-ups at full size) before reporting a model as good.
