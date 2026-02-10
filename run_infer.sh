export T2V_REWRITE_BASE_URL="<your_vllm_server_base_url>"
export T2V_REWRITE_MODEL_NAME="<your_model_name>"
export I2V_REWRITE_BASE_URL="<your_vllm_server_base_url>"
export I2V_REWRITE_MODEL_NAME="<your_model_name>"

# Add project root to PYTHONPATH so hyvideo module can be imported
export PYTHONPATH="/home/ruijyang/workrepo/HY-WorldPlay-rocm:${PYTHONPATH}"

# MIOpen Configuration for better performance
export MIOPEN_USER_DB_PATH="/home/ruijyang/workrepo/HY-WorldPlay/.miopen_cache"  # Cache directory for compiled kernels
export MIOPEN_CUSTOM_CACHE_DIR=/home/ruijyang/workrepo/HY-WorldPlay/.cache 
export MIOPEN_FIND_MODE=3                                               # Enable kernel search and caching (1=normal, 3=fast+cache)
# export MIOPEN_DEBUG_DISABLE_FIND_DB=0                                   # Enable find-db for kernel auto-tuning
export MIOPEN_FIND_ENFORCE=3                                            # Search for fastest kernel
# Optional: Disable logs if they're too verbose
# export MIOPEN_LOG_LEVEL=3                                             # 0=All, 3=Warning, 4=Error, 5=Fatal

# Driving scene prompt matching the input image
PROMPT='A car driving forward on a road. The camera moves smoothly forward and then turns left, capturing the scene from theperspective of a driver.'
#"A scene of snow in Nordic country with aurora and fox running in the snow."
#"A young woman is standing in a room with a large window. She is looking out the window at a beautiful garden. The garden is full of flowers and trees. The woman is wearing a dress and a hat. She is looking at the garden with a smile on her face."
#'A paved pathway leads towards a stone arch bridge spanning a calm body of water.  Lush green trees and foliage line the path and the far bank of the water. A traditional-style pavilion with a tiered, reddish-brown roof sits on the far shore. The water reflects the surrounding greenery and the sky.  The scene is bathed in soft, natural light, creating a tranquil and serene atmosphere. The pathway is composed of large, rectangular stones, and the bridge is constructed of light gray stone.  The overall composition emphasizes the peaceful and harmonious nature of the landscape.'

IMAGE_PATH=./assets/img/3.png # Now we only provide the i2v model, so the path cannot be None
SEED=1
ASPECT_RATIO=16:9
RESOLUTION=480p # Now we only provide the 480p model
OUTPUT_PATH=./eval_outputs/assets_3_trained_500steps/
MODEL_PATH=/data/ruijyang/pretrained_models/hunyuanwp/HunyuanVideo-1.5
TRAINED_CKPT=/data/ruijyang/training_output/hy_worldplay_vkitti_1.5k_converg/checkpoint-500/transformer/diffusion_pytorch_model.safetensors
# AR_ACTION_MODEL_PATH=/data/ruijyang/pretrained_models/hunyuanwp/HY-WorldPlay/ar_model/diffusion_pytorch_model.safetensors

# BI_ACTION_MODEL_PATH=/data/ruijyang/pretrained_models/hunyuanwp/HY-WorldPlay/bidirectional_model/diffusion_pytorch_model.safetensors
# AR_DISTILL_ACTION_MODEL_PATH=/data/ruijyang/pretrained_models/hunyuanwp/HY-WorldPlay/ar_distilled_action_model/diffusion_pytorch_model.safetensors
# POSE='w-31'                   # Camera trajectory: pose string (e.g., 'w-31' means generating [1 + 31] latents) or JSON file path
# Formula: NUM_FRAMES = (total_latents × 4) - 3
# For AR models with 8 GPUs: latents should be divisible by 4 (ideally 8)
POSE='w-10, left-12, w-10, left-12, w-19'  # 1 + 10 + 12 + 10 + 12 + 19 = 64 latents (divisible by 8)
NUM_FRAMES=253           # 64 × 4 - 3 = 253
WIDTH=832
HEIGHT=480

# Configuration for faster inference
# The maximum number recommended is 8.
N_INFERENCE_GPU=8 # Parallel inference GPU count.

# Configuration for better quality
REWRITE=false   # Enable prompt rewriting. Please ensure rewrite vLLM server is deployed and configured.
ENABLE_SR=false # Enable super resolution. When the NUM_FRAMES == 125, you can set it to true

# inference with bidirectional model
# torchrun --nproc_per_node=$N_INFERENCE_GPU hyvideo/generate.py  \
#   --prompt "$PROMPT" \
#   --image_path $IMAGE_PATH \
#   --resolution $RESOLUTION \
#   --aspect_ratio $ASPECT_RATIO \
#   --video_length $NUM_FRAMES \
#   --seed $SEED \
#   --rewrite $REWRITE \
#   --sr $ENABLE_SR --save_pre_sr_video \
#   --pose "$POSE" \
#   --output_path $OUTPUT_PATH \
#   --model_path $MODEL_PATH \
#   --action_ckpt $BI_ACTION_MODEL_PATH \
#   --few_step false \
#   --model_type 'bi'

# inference with autoregressive model
# torchrun --nproc_per_node=$N_INFERENCE_GPU hyvideo/generate.py  \
#   --prompt "$PROMPT" \
#   --image_path $IMAGE_PATH \
#   --resolution $RESOLUTION \
#   --aspect_ratio $ASPECT_RATIO \
#   --video_length $NUM_FRAMES \
#   --seed $SEED \
#   --rewrite $REWRITE \
#   --sr $ENABLE_SR --save_pre_sr_video \
#   --pose "$POSE" \
#   --output_path $OUTPUT_PATH \
#   --model_path $MODEL_PATH \
#   --action_ckpt $AR_ACTION_MODEL_PATH \
#   --few_step false \
#   --width $WIDTH \
#   --height $HEIGHT \
#   --model_type 'ar'

# inference with autoregressive distilled model
torchrun --nproc_per_node=$N_INFERENCE_GPU hyvideo/generate.py \
  --prompt "$PROMPT" \
  --image_path $IMAGE_PATH \
  --resolution $RESOLUTION \
  --aspect_ratio $ASPECT_RATIO \
  --video_length $NUM_FRAMES \
  --seed $SEED \
  --rewrite $REWRITE \
  --sr $ENABLE_SR --save_pre_sr_video \
  --pose "$POSE" \
  --output_path $OUTPUT_PATH \
  --model_path $MODEL_PATH \
  --action_ckpt $TRAINED_CKPT \
  --few_step true \
  --num_inference_steps 4 \
  --model_type 'ar' \
  --use_vae_parallel false \
  --use_sageattn false \
  --use_fp8_gemm false \
