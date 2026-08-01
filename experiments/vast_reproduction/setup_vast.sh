#!/usr/bin/env bash
set -euo pipefail

export DEBIAN_FRONTEND=noninteractive
workspace=/workspace
comfy="$workspace/ComfyUI"
models="$comfy/models"

apt-get update -qq
apt-get install -y -qq aria2 git libgl1 libglib2.0-0 tmux >/dev/null
python -m pip install -q --upgrade pip setuptools wheel

if [[ ! -d "$comfy" ]]; then
  git clone -q https://github.com/comfyanonymous/ComfyUI.git "$comfy"
fi
python -m pip install -q -r "$comfy/requirements.txt"
python -m pip install -q insightface onnxruntime-gpu opencv-python-headless

mkdir -p "$models/checkpoints" "$models/diffusion_models" \
  "$models/text_encoders" "$models/vae"

download() {
  local url="$1"
  local output="$2"
  [[ -s "$output" ]] && return 0
  aria2c -c -x8 -s8 -k1M --retry-wait=5 --max-tries=30 \
    --timeout=30 --connect-timeout=15 --console-log-level=warn \
    --auto-file-renaming=false --allow-overwrite=true \
    -d "$(dirname "$output")" -o "$(basename "$output")" "$url"
}

hf=https://huggingface.co
comfy_org="$hf/Comfy-Org"
download "$hf/SG161222/RealVisXL_V5.0/resolve/main/RealVisXL_V5.0_fp16.safetensors" \
  "$models/checkpoints/RealVisXL_V5.0_fp16.safetensors"
download "$comfy_org/z_image_turbo/resolve/main/split_files/diffusion_models/z_image_turbo_bf16.safetensors" \
  "$models/diffusion_models/z_image_turbo_bf16.safetensors"
download "$comfy_org/z_image_turbo/resolve/main/split_files/text_encoders/qwen_3_4b.safetensors" \
  "$models/text_encoders/qwen_3_4b.safetensors"
download "$comfy_org/z_image_turbo/resolve/main/split_files/vae/ae.safetensors" \
  "$models/vae/ae.safetensors"

pkill -f 'ComfyUI/main.py' 2>/dev/null || true
setsid python "$comfy/main.py" --listen 127.0.0.1 --port 8188 \
  > "$workspace/comfy.log" 2>&1 < /dev/null &

for _ in $(seq 1 80); do
  if curl -sf http://127.0.0.1:8188/system_stats >/dev/null; then
    echo "COMFY_READY"
    exit 0
  fi
  sleep 3
done

tail -n 100 "$workspace/comfy.log"
exit 1
