#!/usr/bin/env bash
cd /home/boyer/spy
source .yolo310/bin/activate
export LD_LIBRARY_PATH=/usr/local/cuda-12.6/targets/aarch64-linux/lib:/usr/lib/aarch64-linux-gnu:/usr/local/lib/ollama/cuda_jetpack6:/usr/local/lib/ollama/cuda_v12:/usr/local/lib/ollama/cuda_v13:$LD_LIBRARY_PATH

python - << 'EOF'
import torch
print("Torch:", torch.__version__)
print("CUDA:", torch.cuda.is_available())
print("Devices:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("Device 0:", torch.cuda.get_device_name(0))
EOF
EOF
