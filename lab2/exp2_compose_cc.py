#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 生成图3：融合前后两档点云对比（CloudCompare/ccViewer 截图合成，1×2）
# 左：B 概率滤波后（融合前，白色单色渲染突出噪点）；右：C 几何一致性+融合（RGB 颜色）
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
import matplotlib.image as mpimg

_CJK = '/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc'
_fp = fm.FontProperties(fname=_CJK)
plt.rcParams['axes.unicode_minus'] = False

FIG = '/home/user/rvclab2/report_figures'
panels = [
    ('cc_prob_filtered.png',  'B  概率滤波后（融合前，2,032,121 点）'),
    ('cc_final3d_model.png',  'C  几何一致性 + 融合（1,709,940 点，原版 final3d_model.ply）'),
]

fig, axes = plt.subplots(1, 2, figsize=(13.2, 3.9))
for ax, (fn, title) in zip(axes, panels):
    img = mpimg.imread(os.path.join(FIG, fn))
    ax.imshow(img)
    ax.set_axis_off()
    ax.set_title(title, fontproperties=_fp, fontsize=11, pad=6)
plt.tight_layout()
out = os.path.join(FIG, 'fig_pointcloud_cc.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
print('saved', out)
