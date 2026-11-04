#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""实验课2 可视化：深度图/概率图、朴素点云 vs 后处理点云、参数对比"""
from __future__ import print_function
import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa

plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'Noto Sans CJK JP', 'Noto Serif CJK SC']
plt.rcParams['axes.unicode_minus'] = False

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp2_depthfusion import read_pfm, read_cam, read_pair, backproject  # noqa


def colorize_depth(d, vmin=None, vmax=None):
    d = np.asarray(d, np.float64)
    m = d > 0
    if vmin is None:
        vmin = np.percentile(d[m], 2) if m.any() else 0
    if vmax is None:
        vmax = np.percentile(d[m], 98) if m.any() else 1
    out = np.clip((d - vmin) / max(vmax - vmin, 1e-6), 0, 1)
    out[~m] = 0
    return plt.cm.rainbow_r(out)[:, :, :3]


def load_ply_pts(path):
    """读取二进制 PLY 顶点坐标"""
    with open(path, 'rb') as f:
        header = b''
        while b'end_header' not in header:
            header += f.readline()
        n = int([l for l in header.split(b'\n') if b'element vertex' in l][0].split()[2])
        has_color = b'red' in header
        dt = np.dtype([('x', '<f8'), ('y', '<f8'), ('z', '<f8')] +
                      ([('r', 'u1'), ('g', 'u1'), ('b', 'u1')] if has_color else []))
        arr = np.fromfile(f, dtype=dt, count=n)
    return np.stack([arr['x'], arr['y'], arr['z']], axis=1)


def fig_inputs(dense, out_dir, view=0):
    df = os.path.join(dense, 'depths_mvsnet')
    d = read_pfm(os.path.join(df, '%08d_init.pfm' % view))
    p = read_pfm(os.path.join(df, '%08d_prob.pfm' % view))
    img_path = os.path.join(df, '%08d.jpg' % view)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    if os.path.isfile(img_path):
        try:
            import cv2
            img = cv2.imread(img_path)[:, :, ::-1]
            axes[0].imshow(img)
        except Exception:
            axes[0].axis('off')
    else:
        axes[0].axis('off')
    axes[0].set_title('参考图像 view %d' % view, fontsize=12)
    axes[0].axis('off')
    m = d > 0
    axes[1].imshow(colorize_depth(d))
    axes[1].set_title('深度图 (%.0f~%.0f mm)' % (d[m].min(), d[m].max()), fontsize=12)
    axes[1].axis('off')
    im = axes[2].imshow(p, cmap='jet', vmin=0, vmax=1)
    axes[2].set_title('概率图 (置信度)', fontsize=12)
    axes[2].axis('off')
    plt.colorbar(im, ax=axes[2], fraction=0.046)
    plt.tight_layout()
    out = os.path.join(out_dir, 'fig_inputs_depth_prob.png')
    plt.savefig(out, dpi=130)
    plt.close()
    print('saved', out)


def fig_pcd_compare(naive_path, fused_path, out_dir, n_sample=25000):
    A = load_ply_pts(naive_path)
    B = load_ply_pts(fused_path)
    rng = np.random.RandomState(0)
    def sub(P):
        if len(P) > n_sample:
            return P[rng.choice(len(P), n_sample, replace=False)]
        return P
    a, b = sub(A), sub(B)

    fig = plt.figure(figsize=(14, 9))
    for row, (P, name) in enumerate([(a, '朴素点云（未滤波，N=%d）' % len(A)),
                                     (b, '后处理点云（滤波+融合，N=%d）' % len(B))]):
        ax = fig.add_subplot(2, 2, row * 2 + 1, projection='3d')
        ax.scatter(P[:, 0], P[:, 1], P[:, 2], s=0.4, c=P[:, 2], cmap='viridis', alpha=0.5)
        ax.set_title(name, fontsize=11)
        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
        ax.view_init(elev=20, azim=-60)
        ax2 = fig.add_subplot(2, 2, row * 2 + 2)
        ax2.scatter(P[:, 0], P[:, 2], s=0.3, c=P[:, 2], cmap='viridis', alpha=0.5)
        ax2.set_title(name + ' — XZ 侧视', fontsize=11)
        ax2.set_xlabel('X'); ax2.set_ylabel('Z')
        ax2.set_aspect('equal')
    plt.tight_layout()
    out = os.path.join(out_dir, 'fig_pointcloud_compare.png')
    plt.savefig(out, dpi=130)
    plt.close()
    print('saved', out)

    # 定量统计
    def stats(P):
        c = P.mean(0)
        dist = np.linalg.norm(P - c, axis=1)
        med = np.median(dist)
        mad = np.median(np.abs(dist - med))
        bbox = P.max(0) - P.min(0)
        return dict(n=len(P), bbox=bbox, outlier=float((dist > med + 5 * (mad + 1e-6)).mean() * 100),
                    std=float(dist.std()))
    sa, sb = stats(A), stats(B)
    print('朴素点云   点数=%d 包围盒=%s 离群比例=%.2f%% 距离std=%.1f'
          % (sa['n'], np.round(sa['bbox'], 1), sa['outlier'], sa['std']))
    print('后处理点云 点数=%d 包围盒=%s 离群比例=%.2f%% 距离std=%.1f'
          % (sb['n'], np.round(sb['bbox'], 1), sb['outlier'], sb['std']))
    return sa, sb


def fig_param(base, out_dir, view=0):
    """不同 interval_scale 的深度图对比"""
    tags = [('0.8', 'interval_scale=0.8'), ('1.06', 'interval_scale=1.06'), ('1.3', 'interval_scale=1.3')]
    imgs, hists = [], []
    for tag, _ in tags:
        p = os.path.join(base, 'result_is_%s' % tag, '%08d_init.pfm' % view)
        if not os.path.isfile(p):
            print('missing', p)
            return
        d = read_pfm(p)
        imgs.append(colorize_depth(d))
        m = d > 0
        hists.append(d[m] if m.any() else np.array([0]))
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for i, (tag, label) in enumerate(tags):
        axes[0, i].imshow(imgs[i])
        axes[0, i].set_title(label, fontsize=12)
        axes[0, i].axis('off')
        axes[1, i].hist(hists[i], bins=80, color='steelblue')
        axes[1, i].set_title('深度分布 (%.0f~%.0f mm)' % (hists[i].min(), hists[i].max()), fontsize=10)
        axes[1, i].set_xlabel('depth (mm)')
        axes[1, i].set_xlim(400, 1100)
    plt.tight_layout()
    out = os.path.join(out_dir, 'fig_param_interval_scale.png')
    plt.savefig(out, dpi=130)
    plt.close()
    print('saved', out)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dense', default='/home/user/rvclab2/dtu_test_scan9/scan9')
    ap.add_argument('--out', default='/home/user/rvclab2/report_figures')
    ap.add_argument('--base', default='/home/user/rvclab2/dtu_test_scan9')
    ap.add_argument('--view', type=int, default=0)
    ap.add_argument('--param', action='store_true')
    args = ap.parse_args()
    if not os.path.isdir(args.out):
        os.makedirs(args.out)
    fig_inputs(args.dense, args.out, args.view)
    pn = os.path.join(args.dense, 'points_mvsnet', 'point_cloud_naive.ply')
    pf = os.path.join(args.dense, 'points_mvsnet', 'point_cloud_fused.ply')
    if os.path.isfile(pn) and os.path.isfile(pf):
        fig_pcd_compare(pn, pf, args.out)
    if args.param:
        fig_param(args.base, args.out, args.view)
