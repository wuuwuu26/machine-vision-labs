#!/usr/bin/env bash
# 跑原版 MVSNet/depthfusion.py（只调外部 fusibile 二进制，不 import tensorflow）
# 需要：fusibile 已编译（build_fusibile.sh）；mvsnet3 含 cv2/numpy/matplotlib；本机有 NVIDIA 驱动
# 用法：bash /home/user/rvclab2/fusibile/run_depthfusion.sh
set -e

CUDA_ROOT=/home/user/cuda124/builds/cuda_nvcc
# fusibile 运行时依赖 cudart/curand（libcuda.so 由本机驱动提供）
export LD_LIBRARY_PATH=$CUDA_ROOT/targets/x86_64-linux/lib64:$CUDA_ROOT/lib64:${LD_LIBRARY_PATH:-}

source /home/user/miniconda3/bin/activate mvsnet3

cd /home/user/rvclab2/MVSNet
python mvsnet/depthfusion.py \
  --dense_folder /home/user/rvclab2/dtu_test_scan9/scan9 \
  --fusibile_exe_path /home/user/rvclab2/fusibile/build/fusibile \
  --prob_threshold 0.8

echo "=========================================="
echo "fusion 完成。原版流程只产出 prob_filtered + fused 两档点云："
echo "  /home/user/rvclab2/dtu_test_scan9/scan9/points_mvsnet/"
echo "（注意：原版不含 naive 档；报告图3/§3.3 的 naive 对比需另行 reconcile）"
echo "=========================================="
