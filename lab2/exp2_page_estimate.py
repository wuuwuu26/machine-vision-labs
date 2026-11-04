#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""估算 docx 页数：忠实读取行距/段距，按文本行数 + 图片高度 + 表格（逐单元格）累加。

说明：本估算以"物理排版高度"为基准，仍需乘以校准系数 K 才接近 MS Word 真实页数
（沙箱无 LibreOffice/可用无头渲染器，无法真渲染；K 由已知锚点标定：
 行距=1.0 版本实测 6.0 页、本估算给出 3.9 页 → K≈1.538）。
"""
import sys
from docx import Document
from docx.shared import Emu
from docx.oxml.ns import qn

EMU_CM = 360000.0
USABLE_W = 16.0          # A4 21cm - 左右边距各 2.5cm
USABLE_H = 24.7          # A4 29.7cm - 上下边距各 2.5cm
K = 1.538                # 真实页数校准系数（锚点：行距1.0版 实测6页/估3.9页）
PT2CM = 0.0353           # 1pt = 1/72 inch = 0.0353 cm


def eff_line_spacing(p):
    v = p.paragraph_format.line_spacing
    if v is not None:
        return float(v)
    st = p.style
    if st is not None and st.paragraph_format.line_spacing is not None:
        return float(st.paragraph_format.line_spacing)
    return 1.0


def eff_space(p, attr):
    def clean(v):
        if v is None:
            return None
        v = float(v)
        if v > 200:        # 钳制 python-docx 对 Pt() 的 EMU 误读（如 25400pt）
            return 0.0
        return v
    v = clean(getattr(p.paragraph_format, attr))
    if v is not None:
        return v
    st = p.style
    if st is not None:
        v2 = clean(getattr(st.paragraph_format, attr))
        if v2 is not None:
            return v2
    return 0.0


def text_width_pt(txt, fs):
    w = 0.0
    for ch in txt:
        w += fs * (1.0 if ord(ch) > 0x2000 else 0.5)
    return w


def para_lines(p, fs):
    w = text_width_pt(p.text, fs)
    max_w_pt = USABLE_W * 28.35
    return max(1, int(w / max_w_pt) + (1 if w % max_w_pt else 0))


def para_height(p):
    txt = p.text
    sizes = [r.font.size.pt for r in p.runs if r.font.size] or [10.5]
    fs = max(sizes)
    style_name = p.style.name
    if style_name.startswith('Heading'):
        fs = {1: 15, 2: 12.5, 3: 11.5}.get(int(style_name[-1]), 11)
    ls = eff_line_spacing(p)
    n = para_lines(p, fs)
    line_h = fs * ls * 1.45 * PT2CM
    sa = eff_space(p, 'space_after')
    sb = eff_space(p, 'space_before')
    h = n * line_h + (sa + sb) * PT2CM
    if style_name.startswith('Heading'):
        h += fs * 0.4 * PT2CM
    return h


def cell_height(cell):
    h = 0.0
    for pp in cell.paragraphs:
        h += para_height(pp)
    return h + 0.18  # 上下内边距


def table_height(t):
    total = 0.0
    for row in t.rows:
        total += max(cell_height(c) for c in row.cells) + 0.06
    return total + 0.2


def main(path):
    doc = Document(path)
    total = 0.0
    detail = []
    body = doc.element.body
    for child in body.iterchildren():
        if child.tag == qn('w:p'):
            from docx.text.paragraph import Paragraph
            p = Paragraph(child, doc)
            h = para_height(p)
            for r in p.runs:
                for dr in r._element.findall(qn('w:drawing')):
                    ext = dr.find('.//' + qn('wp:extent'))
                    if ext is not None:
                        h_img = int(ext.get('cy')) / EMU_CM
                        total += h_img
                        detail.append(('IMG', round(h_img, 2), ''))
            total += h
            if p.text.strip():
                detail.append(('TXT', round(h, 2), p.text[:34]))
        elif child.tag == qn('w:tbl'):
            from docx.table import Table
            t = Table(child, doc)
            h = table_height(t)
            total += h
            detail.append(('TBL', round(h, 2), '%d 行' % len(t.rows)))
    from collections import defaultdict
    s = defaultdict(float)
    for k, h, _ in detail:
        s[k] += h
    print('分类: ' + ', '.join('%s=%.1fcm' % (k, v) for k, v in s.items()))
    est = total / USABLE_H
    print('物理高度估算 %.1f cm → %.2f 页' % (total, est))
    print('×校准系数 K=%.3f → 真实页数约 %.2f 页 (目标 ≤5)' % (K, est * K))
    return est * K


if __name__ == '__main__':
    main(sys.argv[1])
