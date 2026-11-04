#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""扫描点云融合参数 num_consistent（几何一致性检查所需一致视图数），
量化其对融合结果（点数 / 候选保留率 / 包围盒）的影响，并生成对比图。
输出：
  dtu_test_scan9/scan9/points_param_sweep/nc{N}/  各档点云与 stats.npy
  report_figures/param_nc_results.json            数值结果
  report_figures/fig_param_num_consistent.png     柱状图
"""
import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import exp2_depthfusion as ef

BASE = '/home/user/rvclab2/dtu_test_scan9/scan9'
OUT_BASE = os.path.join(BASE, 'points_param_sweep')
FIG = '/home/user/rvclab2/report_figures'
NC_VALUES = [2, 3, 4, 5]


def read_ply_points(path):
    with open(path, 'rb') as f:
        lines = []
        while True:
            line = f.readline()
            if line.startswith(b'end_header'):
                break
            lines.append(line.decode('utf-8'))
        n = 0
        props = []
        for l in lines:
            if l.startswith('element vertex'):
                n = int(l.split()[2])
            if l.startswith('property'):
                props.append(l.split()[2])
        dtype = []
        for p in props:
            if p in ('x', 'y', 'z'):
                dtype.append((p, '<f8'))
            elif p in ('red', 'green', 'blue'):
                dtype.append((p, 'u1'))
        arr = np.fromfile(f, dtype=np.dtype(dtype), count=n)
        return arr


def main():
    rows = []
    for nc in NC_VALUES:
        out = os.path.join(OUT_BASE, 'nc%d' % nc)
        sys.argv = ['exp2_depthfusion.py',
                    '--dense_folder', BASE,
                    '--prob_threshold', '0.8',
                    '--disp_thresh', '1.0',
                    '--num_consistent', str(nc),
                    '--out_folder', out]
        ef.main()
        stats = np.load(os.path.join(out, 'stats.npy'))   # [naive, prob, fused, tot_cand, tot_keep]
        fused_n = int(stats[2])
        keep_ratio = float(stats[4]) / max(1, int(stats[3]))
        pts = read_ply_points(os.path.join(out, 'point_cloud_fused.ply'))
        bbox = pts['x'].max() - pts['x'].min(), \
               pts['y'].max() - pts['y'].min(), \
               pts['z'].max() - pts['z'].min()
        rows.append({'num_consistent': nc, 'fused_n': fused_n,
                     'keep_ratio': keep_ratio, 'bbox': bbox})
        print('nc=%d -> fused=%d  keep=%.1f%%  bbox=(%.0f,%.0f,%.0f)'
              % (nc, fused_n, 100 * keep_ratio, bbox[0], bbox[1], bbox[2]))

    with open(os.path.join(FIG, 'param_nc_results.json'), 'w') as f:
        json.dump(rows, f, indent=2)

    # ---------------- 柱状图：融合点数 vs num_consistent ----------------
    xs = [r['num_consistent'] for r in rows]
    ys = [r['fused_n'] for r in rows]
    plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    fig, ax = plt.subplots(figsize=(5.2, 3.0), dpi=150)
    bars = ax.bar([str(x) for x in xs], ys, color='#3b6ea5')
    for b, y in zip(bars, ys):
        ax.text(b.get_x() + b.get_width() / 2, y, '%d' % y,
                ha='center', va='bottom', fontsize=8)
    ax.set_xlabel('num_consistent (几何一致性所需一致视图数)', fontsize=9)
    ax.set_ylabel('融合后点数', fontsize=9)
    ax.set_title('图 4　num_consistent 对融合点云完整性的影响', fontsize=10)
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'fig_param_num_consistent.png'))
    print('saved figure:', os.path.join(FIG, 'fig_param_num_consistent.png'))


if __name__ == '__main__':
    main()
