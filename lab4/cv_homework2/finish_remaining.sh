#!/bin/bash
# 补齐剩余实验：交通标志 v1/v5 → CIFAR ResNet50 剩余 epoch → 课程原脚本完整 50 epoch
cd "$(dirname "$0")"
export PY=${PY:-/home/user/venvs/cvhw2/bin/python}
rm -f results/warmtest*
set -x

# 1) 交通标志 v1（此前漏传 --dataset 而失败）与 v5（OOM，改为 96px/bs16）
$PY run_all.py traffic

# 2) CIFAR-100 ResNet50：从 ep34 的最佳权重热启动，续训 35~50
if [ ! -f results/cifar100_resnet50_sgd_lr0.1_schedstep_summary.json ]; then
  $PY train.py --name cifar100_resnet50_sgd_lr0.1_schedstep \
    --data-root ./data --workers 4 --bsize 128 --epochs 50 --amp \
    --arch resnet50 --opt sgd --lr 0.1 --sched step \
    --warm-start results/cifar100_resnet50_sgd_lr0.1_schedstep_best.pth --start-epoch 35
fi

# 3) 课程原脚本完整 50 epoch（要求1：原代码本身的精度）
$PY run_all.py 0f

# 4) 重新出图 / 混淆矩阵 / 报告
$PY make_figures.py
BEST=$($PY - <<'PYEOF'
import json, glob
best, name = -1, None
for p in glob.glob('results/traffic_*_summary.json'):
    s = json.load(open(p))
    if s.get('best_acc', -1) > best:
        best, name = s['best_acc'], s['name']
print(name or '')
PYEOF
)
if [ -n "$BEST" ]; then
  ARCH=resnet18; case "$BEST" in *resnet50*) ARCH=resnet50;; esac
  $PY eval_traffic.py "$BEST" --arch "$ARCH"
fi
$PY make_report.py
echo FINISH_DONE
