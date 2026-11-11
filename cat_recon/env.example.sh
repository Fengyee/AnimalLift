# Copy to env.sh (not tracked) and adjust. pipeline.sh and review.sh source env.sh if it exists.
export CATRECON_PYTHON=/path/to/envs/hy3d/bin/python      # env with Hunyuan3D-2 installed (see README)
export HF_HOME=/path/to/hf_home                           # Hugging Face cache (models are ~25 GB)
export U2NET_HOME=/path/to/caches/u2net                   # rembg model cache
export BLENDER=/path/to/blender-4.2/blender               # only for review.sh
export CATRECON_HDR=                                      # optional .hdr for lit renders; flat renders do not need it
