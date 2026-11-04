#!/bin/bash
# 安装实验依赖（PyTorch CUDA 版 + 绘图/工具包）
PY=/home/user/venvs/cvhw2/bin/python
export PIP_CONFIG_FILE=/dev/null
set -x
$PY -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
$PY -m pip install matplotlib numpy pillow tqdm pyyaml configargparse pytorch-ignite
$PY -c "import torch,torchvision,matplotlib,numpy;print('TORCH',torch.__version__,torchvision.__version__,'CUDA',torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
echo INSTALL_DONE
