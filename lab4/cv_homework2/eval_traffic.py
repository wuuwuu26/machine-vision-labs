"""计算并保存交通标志最佳模型的混淆矩阵与各类准确率。

用法: python eval_traffic.py [run_name] [--arch resnet18] [--img-size 128]
"""
import os
import sys
import json
import argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import build_model, TrafficSignDataset, compute_stats  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, 'results')
DATA = os.path.join(HERE, 'data', 'Traffic_sign')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('name', nargs='?', default='traffic_resnet18_v6_pretrained')
    p.add_argument('--arch', default='resnet18')
    p.add_argument('--img-size', type=int, default=128)
    p.add_argument('--bsize', type=int, default=256)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--out', default=None, help='输出 json 路径（默认 results/traffic_confusion.json）')
    p.add_argument('--pretrained', action='store_true', help='标准 ImageNet 结构（v6 预训练微调）')
    args = p.parse_args()

    # 根据 run 名自动推断结构/分辨率，避免用错网络定义导致 state_dict 不匹配
    if 'pretrained' in args.name:
        args.pretrained = True
    if '_v5_' in args.name:
        args.img_size = 96

    ckpt = os.path.join(RES, args.name + '_best.pth')
    out = args.out or os.path.join(RES, 'traffic_confusion.json')
    if not os.path.exists(ckpt):
        print(f'[WARN] 缺少权重 {ckpt}')
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    stats = compute_stats(DATA, args.img_size)
    ds = TrafficSignDataset(DATA, 'test', args.img_size, stats=stats)
    loader = torch.utils.data.DataLoader(ds, batch_size=args.bsize, shuffle=False,
                                        num_workers=args.workers,
                                        pin_memory=(device.type == 'cuda'))
    model = build_model(args.arch, len(ds.classes), pretrained=args.pretrained).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    n = len(ds.classes)
    M = np.zeros((n, n), dtype=int)
    with torch.no_grad():
        for x, y in loader:
            pred = model(x.to(device)).argmax(1).cpu().numpy()
            for yi, pi in zip(y.numpy(), pred):
                M[yi, pi] += 1

    per_class = (M.diagonal() / (M.sum(1) + 1e-8) * 100).round(1).tolist()
    overall = round(M.diagonal().sum() / M.sum() * 100, 2)
    json.dump({'run': args.name, 'classes': ds.classes, 'matrix': M.tolist(),
               'per_class_acc': per_class, 'overall': overall},
              open(out, 'w'), indent=2, ensure_ascii=False)
    print('overall', overall)
    print('per_class', dict(zip(ds.classes, per_class)))


if __name__ == '__main__':
    main()
