# -*- coding: utf-8 -*-
"""直接解析生成的 docx：校验结构（标题层级、真实表格、图片），并估算页数。

估算模型（读取真实版式，不再是固定 A4 假设）：
  - 页面尺寸/页边距：从 sectPr 读（A4 / Letter 都支持）
  - 正文 10.5 pt 宋体，行高按 fontSize × 1.25 × 行距倍数
  - 段落的 w:before / w:after 间距计入高度
  - 用 PIL 加载同款字体做逐行折行测量；表格按单元格折行后取最大行数
"""
import os
import re
import sys
import zipfile

from docx import Document
from docx.oxml.ns import qn
from PIL import Image, ImageFont

DOCX = sys.argv[1]
FONT = r'C:\Windows\Fonts\simsun.ttc'

doc = Document(DOCX)
sec = doc.sections[0]
TW, PT_PER_IN = 20.0, 72.0


def tw2pt(v):
    return (v.twips / TW) if v is not None else None


PW = tw2pt(sec.page_width) or 612.0
PH = tw2pt(sec.page_height) or 792.0
LM = tw2pt(sec.left_margin) or 62.0
RM = tw2pt(sec.right_margin) or 62.0
TM = tw2pt(sec.top_margin) or 62.0
BM = tw2pt(sec.bottom_margin) or 62.0
TEXT_W = PW - LM - RM
TEXT_H = PH - TM - BM

# 默认正文字号：从 docDefaults 读
s = zipfile.ZipFile(DOCX).read('word/styles.xml').decode('utf-8')
i = s.find('docDefaults')
m = re.search(r'<w:rPrDefault>.*?<w:sz w:val="(\d+)"', s[i:i + 800], re.S)
BODY_PT = (int(m.group(1)) / 2.0) if m else 10.5

_mult = 1.0
mm = re.search(r'w:line="(\d+)"', s[i:i + 800])
if mm:
    _mult = int(mm.group(1)) / 240.0

LINE_H = BODY_PT * 1.25 * _mult
TABLE_LINE_H = LINE_H + 1.5
TITLE_PT = 22.0
H2_PT = 16.0
TITLE_H = TITLE_PT * 1.25 * _mult + 10
H2_H = H2_PT * 1.25 * _mult + 10

pagesize = 'A4' if abs(PW - 595.3) < 2 else ('Letter' if abs(PW - 612.0) < 2 else '%.0fx%.0f' % (PW, PH))
print('页面 = %s，正文 %g pt，行距倍数 %.3f -> 行高 %.2f pt' % (pagesize, BODY_PT, _mult, LINE_H))
print('版心 = %.1f x %.1f pt（页边距 T/B/L/R = %.0f/%.0f/%.0f/%.0f）' % (TEXT_W, TEXT_H, TM, BM, LM, RM))


def px(pt):
    return pt * 96 / 72


f_cache = {}


def font(pt):
    key = round(px(pt))
    if key not in f_cache:
        f_cache[key] = ImageFont.truetype(FONT, key)
    return f_cache[key]


def text_width(st, pt):
    if not st:
        return 0.0
    return font(pt).getlength(st) / 96 * 72


def wrapped_lines(st, pt, avail_pt):
    if not st.strip():
        return 1
    lines, cur = 1, 0.0
    for ch in st:
        w = text_width(ch, pt)
        if cur + w > avail_pt and cur > 0:
            lines += 1
            cur = w
        else:
            cur += w
    return lines


def para_spacing_pt(p):
    pPr = p._p.pPr
    if pPr is None:
        return 0.0
    sp = pPr.find(qn('w:spacing'))
    if sp is None:
        return 0.0
    tot = 0.0
    b = sp.get(qn('w:before'))
    a = sp.get(qn('w:after'))
    if b:
        tot += int(b) / TW
    if a:
        tot += int(a) / TW
    return tot


def outline_level(p):
    pPr = p._p.pPr
    if pPr is None:
        return None
    ol = pPr.find(qn('w:outlineLvl'))
    return ol.get(qn('w:val')) if ol is not None else None


# ---------------------------------------------------------------- 结构校验
tables = len(doc.tables)
imgs = sum(1 for r in doc.part.rels.values() if 'image' in r.reltype)
print('段落数 =', len(doc.paragraphs), ' 表格数 =', tables, ' 内嵌图片数 =', imgs)

# ---------------------------------------------------------------- 页数估算
body = doc.element.body
total = 0.0
heights = []
for child in body.iterchildren():
    tag = child.tag.split('}')[-1]
    if tag == 'sectPr':
        continue
    if tag == 'tbl':
        from docx.table import Table
        t = Table(child, doc)
        ncols = len(t.columns)
        avail = TEXT_W / ncols - 6
        h = 0.0
        for row in t.rows:
            mx = 1
            for c in row.cells:
                mx = max(mx, wrapped_lines(c.text, 9.5, avail))
            h += mx * TABLE_LINE_H + 2
        total += h
        heights.append(('表格 %d行x%d列' % (len(t.rows), ncols), h))
        continue

    from docx.text.paragraph import Paragraph
    p = Paragraph(child, doc)
    if 'graphicData' in child.xml or 'w:drawing' in child.xml:
        cy = 0
        for ext in child.iter():
            if ext.tag.split('}')[-1] == 'extent':
                cy = int(ext.get('cy', '0'))
        hpt = cy / 914400 * 72
        total += hpt + para_spacing_pt(p) + 2
        heights.append(('图片', hpt))
        continue

    txt = p.text
    sp = para_spacing_pt(p)
    lvl = outline_level(p)
    if lvl == '0':
        total += TITLE_H + sp
        heights.append((txt[:28], TITLE_H + sp))
        continue
    if lvl == '1':
        total += H2_H + sp
        heights.append((txt[:28], H2_H + sp))
        continue
    h = wrapped_lines(txt, BODY_PT, TEXT_W) * LINE_H + sp
    total += h
    if txt.strip():
        heights.append((txt[:28], h))

print()
print('--- 各块估算高度（pt）---')
for name, h in heights:
    print('   %7.1f  %s' % (h, name))

print()
print('估算总高度 = %.1f pt，每页版心 %.1f pt' % (total, TEXT_H))
print('估算页数 ≈ %.2f' % (total / TEXT_H))
