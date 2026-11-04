#!/bin/bash
# 冒烟测试：1 epoch 快速验证 CIFAR-100 与交通标志两条数据/训练链路是否通畅
cd "$(dirname "$0")"
PY=${PY:-/home/user/venvs/cvhw2/bin/python}
set -e
$PY -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
$PY env_info.py
echo "--- smoke: cifar100 resnet10 1ep"
$PY train.py --name smoke_cifar --arch resnet10 --opt sgd --lr 0.1 --sched step \
  --data-root ./data --bsize 256 --epochs 1 --workers 4 --amp
echo "--- smoke: traffic resnet18 1ep (aug+balance+mixup)"
$PY train.py --name smoke_traffic --dataset traffic --data-root ./data/Traffic_sign \
  --arch resnet18 --img-size 128 --bsize 64 --epochs 1 --workers 4 --amp \
  --opt sgd --lr 0.05 --sched cosine --strong-aug --label-smoothing 0.1 --balance --mixup 0.2
echo "--- smoke: traffic resnet18 pretrained 1ep"
$PY train.py --name smoke_traffic_pt --dataset traffic --data-root ./data/Traffic_sign \
  --arch resnet18 --pretrained --img-size 128 --bsize 64 --epochs 1 --workers 4 --amp \
  --opt sgd --lr 0.005 --sched cosine --strong-aug --label-smoothing 0.1 --balance
echo "--- smoke: cifar100 resnet50 1ep"
$PY train.py --name smoke_r50 --arch resnet50 --opt sgd --lr 0.1 --sched step \
  --data-root ./data --bsize 128 --epochs 1 --workers 4 --amp
echo "SMOKE_OK"
