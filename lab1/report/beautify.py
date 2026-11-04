# -*- coding: utf-8 -*-
"""对已生成的实验报告 docx 做格式润色（不改正文，不影响页数估算）：
   1) 标题居中 + 深蓝 + 下边框；作者行居中
   2) 一级节标题深色 + 下边框 + 加大段前间距，视觉上与正文分离
   3) 表题 / 图题居中
   4) 表格统一 100% 宽度；表头浅蓝底纹；均值/合计行浅灰底纹
"""
import os
import shutil
import sys
from docx import Document
from docx.shared import Emu
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

SRC = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', '单目深度与点云重建-实验报告.docx')
DST = sys.argv[2] if len(sys.argv) > 2 else SRC

HEADING_COLOR = '1F4E79'   # 深蓝
TITLE_COLOR = '1F4E79'
HEADER_FILL = 'DDEBF7'     # 表头浅蓝
SUMMARY_FILL = 'F2F2F2'    # 均值/合计行浅灰


def set_jc(p, val):
    pPr = p._p.get_or_add_pPr()
    jc = pPr.find(qn('w:jc'))
    if jc is None:
        jc = OxmlElement('w:jc')
        pPr.append(jc)
    jc.set(qn('w:val'), val)


def set_run_color(p, val):
    for r in p.runs:
        rPr = r._r.get_or_add_rPr()
        color = rPr.find(qn('w:color'))
        if color is None:
            color = OxmlElement('w:color')
            rPr.append(color)
        color.set(qn('w:val'), val)


def add_bottom_border(p, color=HEADING_COLOR, sz='6', space='6'):
    pPr = p._p.get_or_add_pPr()
    pBdr = pPr.find(qn('w:pBdr'))
    if pBdr is None:
        pBdr = OxmlElement('w:pBdr')
        pPr.append(pBdr)
    b = pBdr.find(qn('w:bottom'))
    if b is None:
        b = OxmlElement('w:bottom')
        pBdr.append(b)
    b.set(qn('w:val'), 'single')
    b.set(qn('w:sz'), sz)
    b.set(qn('w:space'), space)
    b.set(qn('w:color'), color)


def set_spacing(p, before=None, after=None):
    pPr = p._p.get_or_add_pPr()
    sp = pPr.find(qn('w:spacing'))
    if sp is None:
        sp = OxmlElement('w:spacing')
        pPr.append(sp)
    if before is not None:
        sp.set(qn('w:before'), str(before))
    if after is not None:
        sp.set(qn('w:after'), str(after))


def outline_level(p):
    pPr = p._p.pPr
    if pPr is None:
        return None
    ol = pPr.find(qn('w:outlineLvl'))
    return ol.get(qn('w:val')) if ol is not None else None


def shade_cell(cell, fill):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = tcPr.find(qn('w:shd'))
    if shd is None:
        shd = OxmlElement('w:shd')
        tcPr.append(shd)
    shd.set(qn('w:val'), 'clear')
    shd.set(qn('w:color'), 'auto')
    shd.set(qn('w:fill'), fill)


def set_table_full_width(table):
    tblPr = table._tbl.tblPr
    tblW = tblPr.find(qn('w:tblW'))
    if tblW is None:
        tblW = OxmlElement('w:tblW')
        tblPr.append(tblW)
    tblW.set(qn('w:w'), '5000')
    tblW.set(qn('w:type'), 'pct')


def set_page_a4(doc, margin_pt=54):
    """改成 A4（11906x16838 twips）+ 指定页边距，避免被默认 Letter 撑过 5 页。"""
    sec = doc.sections[0]
    T = 635  # 1 twip = 635 EMU
    sec.page_width = Emu(11906 * T)
    sec.page_height = Emu(16838 * T)
    m = Emu(int(margin_pt) * 20 * T)
    sec.top_margin = m
    sec.bottom_margin = m
    sec.left_margin = m
    sec.right_margin = m


def set_image_width(doc, target_px=460):
    """把所有显示宽度超过 target_px 的插图等比缩到 target_px（仅改显示尺寸）。"""
    EMU_PER_PX = 9525
    body = doc.element.body
    for ext in body.iter(qn('wp:extent')):
        cx = int(ext.get('cx') or 0)
        cy = int(ext.get('cy') or 0)
        wpx = cx / EMU_PER_PX
        if wpx > target_px:
            scale = target_px / wpx
            ncx = int(target_px * EMU_PER_PX)
            ncy = int(cy * scale)
            ext.set('cx', str(ncx))
            ext.set('cy', str(ncy))
            parent = ext.getparent()
            for aext in parent.iter(qn('a:ext')):
                aext.set('cx', str(ncx))
                aext.set('cy', str(ncy))


def main():
    # 备份
    if DST == SRC:
        bak = SRC[:-5] + '.bak.docx'
        if not os.path.exists(bak):
            shutil.copy(SRC, bak)
            print('   备份 ->', bak)
    doc = Document(SRC)

    paras = doc.paragraphs
    # 1) 标题（首个 outlineLvl=0）与作者行（紧随其后）
    for i, p in enumerate(paras):
        lvl = outline_level(p)
        if lvl == '0':
            set_jc(p, 'center')
            set_run_color(p, TITLE_COLOR)
            add_bottom_border(p, TITLE_COLOR, '12', '6')
            set_spacing(p, after=80)
            # 作者行：下一个段落
            if i + 1 < len(paras):
                set_jc(paras[i + 1], 'center')
                set_spacing(paras[i + 1], after=160)
            break

    # 2) 一级节标题 outlineLvl=1
    for p in paras:
        if outline_level(p) == '1':
            set_run_color(p, HEADING_COLOR)
            add_bottom_border(p, HEADING_COLOR, '6', '4')
            set_spacing(p, before=160, after=120)

    # 3) 表题 / 图题居中
    for p in paras:
        t = p.text.strip()
        if t.startswith('表 ') or t.startswith('图 '):
            set_jc(p, 'center')
            set_spacing(p, after=60)

    # 4) 表格：100% 宽 + 表头底纹 + 均值/合计行底纹
    for table in doc.tables:
        set_table_full_width(table)
        # 表头
        for cell in table.rows[0].cells:
            shade_cell(cell, HEADER_FILL)
        # 汇总行（首单元格含「均值」或「合计」，且非表头）
        for row in table.rows[1:]:
            first = row.cells[0].text.strip()
            if '均值' in first or '合计' in first:
                for cell in row.cells:
                    shade_cell(cell, SUMMARY_FILL)

    # 5) 插图显示宽度收到 460px（仅改显示尺寸，省版面）
    set_image_width(doc, target_px=460)

    # 6) 页面：A4 + 54pt 页边距（避免默认 Letter 超出 5 页）
    set_page_a4(doc, margin_pt=54)

    doc.save(DST)
    print('   已润色 ->', DST)


if __name__ == '__main__':
    main()
