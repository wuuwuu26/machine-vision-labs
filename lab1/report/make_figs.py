# -*- coding: utf-8 -*-
"""生成实验报告所需的三张插图。

输入全部来自 results/ 下的已有结果，不做任何重新推理。
输出写到 report/figs/。
"""
import json
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
RES = os.path.join(ROOT, 'results', 'results')
RGB = os.path.join(ROOT, 'nyu_data', 'rgb')
OUT = os.path.join(ROOT, 'report', 'figs')
os.makedirs(OUT, exist_ok=True)

FONT_PATH = r'C:\Windows\Fonts\msyh.ttc'


def font(size):
    return ImageFont.truetype(FONT_PATH, size)


def center_text(draw, cx, cy, text, size, fill=(20, 20, 20)):
    f = font(size)
    box = draw.textbbox((0, 0), text, font=f)
    draw.text((cx - (box[2] - box[0]) / 2 - box[0],
               cy - (box[3] - box[1]) / 2 - box[1]), text, font=f, fill=fill)


# ---------------------------------------------------------------- 图 1
# RGB / 518 / 280 三栏，两行对应两个典型场景
def fig_resolution_compare(rows=('nyu_1201', 'nyu_0201'), W=560, H=420):
    cols = [('RGB 输入', RGB),
            ('input-size = 518', os.path.join(RES, 'depth_518')),
            ('input-size = 280', os.path.join(RES, 'depth_280'))]
    head, lab = 46, 40
    sheet = Image.new('RGB', (W * len(cols), head + (H + lab) * len(rows)), (255, 255, 255))
    d = ImageDraw.Draw(sheet)

    for c, (title, _) in enumerate(cols):
        center_text(d, (c + 0.5) * W, head / 2, title, 24)

    for r, sid in enumerate(rows):
        y = head + r * (H + lab)
        # 左侧场景名条形
        d.rectangle([0, y, W * len(cols), y + lab], fill=(242, 242, 244))
        center_text(d, W * len(cols) / 2, y + lab / 2, sid, 22)
        for c, (_, folder) in enumerate(cols):
            im = Image.open(os.path.join(folder, sid + '.png')).convert('RGB').resize((W, H), Image.LANCZOS)
            sheet.paste(im, (c * W, y + lab))
        d.line([(0, y + lab), (W * len(cols), y + lab)], fill=(200, 200, 200), width=1)
        for c in range(1, len(cols)):
            d.line([(c * W, y), (c * W, y + lab + H)], fill=(200, 200, 200), width=1)

    p = os.path.join(OUT, 'fig1_resolution.png')
    sheet.save(p)
    print('fig1 ->', p, sheet.size)


# ---------------------------------------------------------------- 图 2
# 从已有的点云三方对比大图中裁出需要的场景行，保留表头
def fig_pointcloud(sids=('nyu_0201', 'nyu_0401', 'nyu_1201')):
    order = ['nyu_0001', 'nyu_0101', 'nyu_0201', 'nyu_0401',
             'nyu_0601', 'nyu_0801', 'nyu_1001', 'nyu_1201']
    src = os.path.join(RES, '点云对比', '点云三方差比.png')
    im = Image.open(src).convert('RGB')
    W = im.width
    row_h = round((im.height - 98) / len(order))          # 大图布局：54 标题 + 44 列名
    bands = [im.crop((0, 0, W, 98))]
    for sid in sids:
        i = order.index(sid)
        bands.append(im.crop((0, 98 + i * row_h, W, 98 + (i + 1) * row_h)))
    h = sum(b.height for b in bands)
    sheet = Image.new('RGB', (W, h), (255, 255, 255))
    y = 0
    for b in bands:
        sheet.paste(b, (0, y))
        y += b.height
    p = os.path.join(OUT, 'fig2_pointcloud.png')
    sheet.save(p)
    print('fig2 ->', p, sheet.size)


# ---------------------------------------------------------------- 图 3
# 三组指标的五种设置对比。全部数值直接读自 metrics_metric_vs_relative.csv
def fig_metrics():
    import csv
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    matplotlib.rcParams['axes.unicode_minus'] = False

    rows = list(csv.DictReader(open(os.path.join(RES, 'metrics_metric_vs_relative.csv'), encoding='utf-8')))
    mean = lambda k: float(np.mean([float(r[k]) for r in rows]))

    sets = [
        ('相对深度 518\n(仿射对齐)', 'absrel_518', 'depth_rmse_518', 'xyz_rmse_518', '#c0392b'),
        ('相对深度 280\n(仿射对齐)', 'absrel_280', 'depth_rmse_280', 'xyz_rmse_280', '#e08b7f'),
        ('Metric 518\n(零拟合)', 'absrel_metric_zero', 'depth_rmse_metric_zero', 'xyz_rmse_metric_zero', '#7f8c8d'),
        ('Metric 518\n(池化尺度)', 'absrel_metric_pooled', 'depth_rmse_metric_pooled', 'xyz_rmse_metric_pooled', '#5b8fb9'),
        ('Metric 518\n(逐图尺度)', 'absrel_metric_perimg', 'depth_rmse_metric_perimg', 'xyz_rmse_metric_perimg', '#2471a3'),
    ]
    titles = ['AbsRel（无量纲，越小越好）', 'Depth RMSE（米，越小越好）', 'XYZ RMSE（米，越小越好）']

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.5))
    for ax, title, idx in zip(axes, titles, range(3)):
        vals = [mean(s[1 + idx]) for s in sets]
        cols = [s[4] for s in sets]
        bars = ax.bar(range(len(sets)), vals, color=cols, width=0.62)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + max(vals) * 0.03,
                    f'{v:.4f}' if idx == 0 else f'{v:.4f}', ha='center', fontsize=9)
        ax.set_title(title, fontsize=11)
        ax.set_xticks(range(len(sets)))
        ax.set_xticklabels([s[0] for s in sets], fontsize=8)
        ax.set_ylim(0, max(vals) * 1.22)
        ax.grid(axis='y', alpha=0.25, linestyle='--')
        ax.set_axisbelow(True)
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)
    fig.tight_layout()
    p = os.path.join(OUT, 'fig3_metrics.png')
    fig.savefig(p, dpi=200)
    print('fig3 ->', p)


if __name__ == '__main__':
    fig_resolution_compare()
    fig_pointcloud()
    fig_metrics()
