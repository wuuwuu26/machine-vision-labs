#!/bin/bash
# 一键完成全部实验 → 混淆矩阵 → 出图 → 出报告
cd "$(dirname "$0")"
export PY=${PY:-/home/user/venvs/cvhw2/bin/python}
mkdir -p results

echo "======== [1/4] 实验矩阵 ========"
$PY run_all.py 0 1 2 3 4 5 traffic

echo "======== [2/4] 交通标志混淆矩阵（自动选最佳 run）========="
BEST=$($PY - <<'PYEOF'
import json, glob, os
best, name = -1, None
for p in glob.glob('results/traffic_*_summary.json'):
    s = json.load(open(p))
    if s.get('best_acc', -1) > best:
        best, name = s['best_acc'], s['name']
print(name or '')
PYEOF
)
echo "best traffic run = $BEST"
if [ -n "$BEST" ]; then
  ARCH=resnet18
  case "$BEST" in *resnet50*) ARCH=resnet50;; esac
  $PY eval_traffic.py "$BEST" --arch "$ARCH" --img-size 128
fi

echo "======== [3/4] 生成图表 ========"
$PY make_figures.py

echo "======== [4/4] 生成报告 ========"
$PY make_report.py

echo PIPELINE_DONE
