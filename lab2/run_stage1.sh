#!/usr/bin/env bash
# 阶段 1：MVSNet 多视图深度估计（对应 PPT 步骤1）
# 用 nvidia-tensorflow 1.15.5（CUDA 12，支持 RTX 4070 的 sm_89）在 49 视角上推理深度图 + 概率图。
# 需 GPU，在**有 RTX 4070 的宿主机**终端运行： bash /home/user/rvclab2/run_stage1.sh
# 输出：dtu_test_scan9/scan9/depths_mvsnet/（*.pfm 深度图 + 概率图，分辨率 288×216 = 输入的 1/4）
set -u

source /home/user/miniconda3/bin/activate mvsnet3
cd /home/user/rvclab2/MVSNet

DENSE=/home/user/rvclab2/dtu_test_scan9/scan9
CKPT=/home/user/rvclab2/tf_model_blendedmvs_200301/tf_model_blendedmvs/3DCNNs/model.ckpt

echo "=== 阶段1 深度估计：dense_folder=$DENSE  max_d=128  interval_scale=1.06 ==="
python mvsnet/test.py \
  --dense_folder "$DENSE" \
  --regularization '3DCNNs' \
  --max_w 1152 --max_h 864 --max_d 128 \
  --interval_scale 1.06 --gpu_memory_fraction 0.9 \
  --pretrained_model_ckpt_path "$CKPT" \
  --ckpt_step 150000

echo "=== 阶段1完成：深度图/概率图已写入 $DENSE/depths_mvsnet/ ==="
echo "提示：受 8 GB 显存限制 max_d 由 192 降到 128（可恢复深度上限约 764 mm）；如需重跑请确认 depths_mvsnet/ 可覆盖。"
