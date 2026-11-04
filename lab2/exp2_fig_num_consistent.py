#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从 param_nc_results.json 重画 num_consistent 参数影响柱状图（显式注册 CJK 字体）"""
import os
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

FIG = '/home/user/rvclab2/report_figures'
FONT = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
fm.fontManager.addfont(FONT)
family = fm.FontProperties(fname=FONT).get_name()
plt.rcParams['font.family'] = family
plt.rcParams['axes.unicode_minus'] = False
print('using font:', family)

with open(os.path.join(FIG, 'param_nc_results.json')) as f:
    rows = json.load(f)

xs = [str(r['num_consistent']) for r in rows]
ys = [r['fused_n'] for r in rows]
vols = [r['bbox'][0] * r['bbox'][1] * r['bbox'][2] / 1e6 for r in rows]  # mm^3 -> x1e6
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(5.8, 2.5), dpi=150)
bars = ax.bar(xs, ys, color='#3b6ea5', width=0.55)
for b, y in zip(bars, ys):
    ax.text(b.get_x() + b.get_width() / 2, y + 12000, '{:,}'.format(y),
            ha='center', va='bottom', fontsize=6.5)
ax.set_xlabel('num_consistent', fontsize=9)
ax.set_ylabel('融合后点数（完整性）', fontsize=9)
ax.set_ylim(0, max(ys) * 1.14)
ax.tick_params(labelsize=8)
bars2 = ax2.bar(xs, vols, color='#a53b3b', width=0.55)
for b, v in zip(bars2, vols):
    ax2.text(b.get_x() + b.get_width() / 2, v + 1.2, '%.0f' % v,
             ha='center', va='bottom', fontsize=6.5)
ax2.set_xlabel('num_consistent', fontsize=9)
ax2.set_ylabel(r'包围盒体积 ($\times 10^6\,$mm$^3$，紧凑度)', fontsize=9)
ax2.set_ylim(0, max(vols) * 1.14)
ax2.tick_params(labelsize=8)
fig.tight_layout()
out = os.path.join(FIG, 'fig_param_num_consistent.png')
fig.savefig(out)
print('saved', out)
