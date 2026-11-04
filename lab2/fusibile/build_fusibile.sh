#!/usr/bin/env bash
# 用离线 CUDA 12.4 工具链编译 fusibile（编译不需要 GPU/驱动；运行时才需要）
# 前置：OpenCV C++ dev 头已装 —— 任选其一：
#   (a) sudo apt install -y libopencv-dev          # 系统级，cmake 自动找到
#   (b) conda install -n mvsnet3 -c conda-forge libopencv -y   # 装后请先 `conda activate mvsnet3` 再跑本脚本
# 用法：bash /home/user/rvclab2/fusibile/build_fusibile.sh
set -e

CUDA_ROOT=/home/user/cuda124/builds/cuda_nvcc
export CUDA_HOME=$CUDA_ROOT
export CUDAToolkit_ROOT=$CUDA_ROOT
export PATH=$CUDA_ROOT/bin:$PATH

# 若 mvsnet3 里装了 conda 版 libopencv，激活它让 cmake 找到 C++ 头
if [ -d /home/user/miniconda3/envs/mvsnet3 ]; then
  source /home/user/miniconda3/bin/activate mvsnet3 2>/dev/null || true
fi

cd /home/user/rvclab2/fusibile
rm -rf build && mkdir build && cd build

cmake .. -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=$CUDA_ROOT/bin/nvcc \
  -DCUDAToolkit_ROOT=$CUDA_ROOT
make -j"$(nproc)"

echo "=========================================="
echo "BUILD DONE -> $(pwd)/fusibile"
echo "=========================================="
echo "运行时需要把 CUDA 12.4 的 cudart/curand 放进 LD_LIBRARY_PATH："
echo "  export LD_LIBRARY_PATH=$CUDA_ROOT/targets/x86_64-linux/lib64:$CUDA_ROOT/lib64:\$LD_LIBRARY_PATH"
echo "然后跑：bash /home/user/rvclab2/fusibile/run_depthfusion.sh"
