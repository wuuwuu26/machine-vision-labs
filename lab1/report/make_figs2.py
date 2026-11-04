# -*- coding: utf-8 -*-
"""补充两张插图：
   fig0_depth_error.png  RGB / 对齐后深度 / 绝对误差图
   fig2_pointcloud.png   点云三方对比（重裁为两个场景）

误差图直接由已有的 *_depth_m.npy 与 reference/*.npz 现算，不重新推理。
"""
import os

import matplotlib
matplotlib.use('Agg')
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
RES = os.path.join(ROOT, 'results', 'results')
RGB = os.path.join(ROOT, 'nyu_data', 'rgb')
REF = os.path.join(ROOT, 'nyu_data', 'reference')
OUT = os.path.join(ROOT, 'report', 'figs')
os.makedirs(OUT, exist_ok=True)

FONT_PATH = r'C:\Windows\Fonts\msyh.ttc'
font = lambda s: ImageFont.truetype(FONT_PATH, s)


def center_text(draw, cx, cy, text, size, fill=(20, 20, 20)):
    f = font(size)
    box = draw.textbbox((0, 0), text, font=f)
    draw.text((cx - (box[2] - box[0]) / 2 - box[0],
               cy - (box[3] - box[1]) / 2 - box[1]), text, font=f, fill=fill)


def error_rgb(sid, W=560, H=420):
    """返回误差图的 PIL 图与色标上限。色标上限取该图 2×Depth RMSE。"""
    z = np.load(os.path.join(RES, 'depth_518', sid + '_depth_m.npy')).astype(float)
    d = np.load(os.path.join(REF, sid + '.npz'))
    g, valid = d['depth_m'], d['valid']
    assert z.shape == g.shape and valid.any()

    e = np.abs(z - g)
    rmse = float(np.sqrt(np.mean(e[valid] ** 2)))
    vmax = 2 * rmse
    e = np.clip(e / vmax, 0, 1)

    cmap = matplotlib.colormaps['magma']
    rgb = (cmap(e)[..., :3] * 255).astype(np.uint8)
    rgb[~valid] = 168                                   # 无效像素统一置灰

    im = Image.fromarray(rgb).resize((W, H), Image.NEAREST)
    d2 = ImageDraw.Draw(im)
    # 右下角画一条小色标
    bw, bh, mx, my = 16, 110, 14, 14
    x0, y0 = W - bw - mx - 68, H - bh - my
    for i in range(bh):
        v = 1 - i / (bh - 1)
        d2.rectangle([x0, y0 + i, x0 + bw, y0 + i + 1],
                     fill=tuple(int(c) for c in (np.array(cmap(v))[:3] * 255)))
    d2.rectangle([x0, y0, x0 + bw, y0 + bh], outline=(255, 255, 255), width=2)
    d2.text((x0 + bw + 6, y0 - 6), f'{vmax:.2f} m', font=font(18), fill=(255, 255, 255))
    d2.text((x0 + bw + 6, y0 + bh - 12), '0', font=font(18), fill=(255, 255, 255))
    return im, vmax


def fig_depth_error(rows=('nyu_0001', 'nyu_0201'), W=560, H=420):
    cols = ['RGB 输入', '对齐后预测深度', '绝对深度误差 |ẑ − z|']
    head, lab = 46, 40
    sheet = Image.new('RGB', (W * 3, head + (H + lab) * len(rows)), (255, 255, 255))
    d = ImageDraw.Draw(sheet)
    for c, t in enumerate(cols):
        center_text(d, (c + 0.5) * W, head / 2, t, 24)

    vmaxes = []
    for r, sid in enumerate(rows):
        y = head + r * (H + lab)
        d.rectangle([0, y, W * 3, y + lab], fill=(242, 242, 244))
        center_text(d, W * 3 / 2, y + lab / 2, sid, 22)
        for c, path in enumerate([os.path.join(RGB, sid + '.png'),
                                  os.path.join(RES, 'depth_518', sid + '.png')]):
            sheet.paste(Image.open(path).convert('RGB').resize((W, H), Image.LANCZOS),
                        (c * W, y + lab))
        err, vm = error_rgb(sid, W, H)
        vmaxes.append(vm)
        sheet.paste(err, (2 * W, y + lab))
        for c in range(1, 3):
            d.line([(c * W, y), (c * W, y + lab + H)], fill=(200, 200, 200), width=1)
        d.line([(0, y + lab), (W * 3, y + lab)], fill=(200, 200, 200), width=1)

    p = os.path.join(OUT, 'fig0_depth_error.png')
    sheet.save(p)
    print('fig0 ->', p, sheet.size, 'vmax(99%):', [f'{v:.3f}' for v in vmaxes])


def fig_pointcloud(sids=('nyu_0401', 'nyu_1201')):
    order = ['nyu_0001', 'nyu_0101', 'nyu_0201', 'nyu_0401',
             'nyu_0601', 'nyu_0801', 'nyu_1001', 'nyu_1201']
    im = Image.open(os.path.join(RES, '点云对比', '点云三方差比.png')).convert('RGB')
    W = im.width
    row_h = round((im.height - 98) / len(order))
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


if __name__ == '__main__':
    fig_depth_error()
    fig_pointcloud()
