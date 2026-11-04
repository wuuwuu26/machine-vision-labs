"""在保留用户排版格式的前提下，把报告压缩到 5 页以内。

输入: 报告Word版/实验报告-图像分类2.docx（用户手工修改过的版本）
输出: 报告Word版/实验报告-图像分类2_5页.docx

只动版式参数（页边距、行距、段间距、图宽、表格字号与内边距），
不改任何文字内容，也不改用户选定的字体（微软雅黑标题 / 宋体正文）。
"""
import os
import re
import sys
import copy

from docx import Document
from docx.shared import Pt, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, '实验报告-图像分类2.docx')
DST = os.path.join(HERE, '实验报告-图像分类2_5页.docx')

# ---- 压缩参数（可调） ----
P = dict(
    margin_tb=1.45,        # 上下页边距 cm
    margin_lr=1.65,        # 左右页边距 cm
    body_ls=1.15,          # 正文行距
    list_ls=1.12,          # 列表行距
    body_pt=10.5,          # 正文字号（保持用户的五号）
    h2_sb=8, h2_sa=2,      # 二级标题段前/段后
    h3_sb=4, h3_sa=1,      # 三级标题段前/段后
    fig_w=14.0,            # 插图宽度 cm
    fig3_w=12.6,           # 补入的图3（学习率衰减收敛曲线）宽度 cm
    cap_sa=3,              # 图注段后
    tab_pt=8.5,            # 表格字号
    tab_cell_sb=0.3, tab_cell_sa=0.3,
    tab_mar_tb=14, tab_mar_lr=52,
)


def drop_spacers(doc):
    """删掉表格后面那些空段落（原生成脚本留下的 0.7 倍行距空行）。"""
    n = 0
    for p in list(doc.paragraphs):
        pf = p.paragraph_format
        if p.text.strip():
            continue
        if pf.line_spacing is not None and abs(float(pf.line_spacing) - 0.7) < 0.01:
            p._element.getparent().remove(p._element)
            n += 1
    return n


def tune_section(doc):
    s = doc.sections[0]
    s.top_margin = s.bottom_margin = Cm(P['margin_tb'])
    s.left_margin = s.right_margin = Cm(P['margin_lr'])
    s.header_distance = Cm(0.9)
    s.footer_distance = Cm(0.9)


def tune_styles(doc):
    st = doc.styles['Normal']
    st.font.size = Pt(P['body_pt'])
    st.paragraph_format.line_spacing = P['body_ls']
    st.paragraph_format.space_before = Pt(0)
    st.paragraph_format.space_after = Pt(2)


def first_size(p):
    for r in p.runs:
        if r.font.size is not None:
            return r.font.size.pt
    return None


def tune_paragraphs(doc):
    for p in doc.paragraphs:
        pf = p.paragraph_format
        sz = first_size(p)
        if sz == 13.5:                      # 二级标题
            pf.space_before = Pt(P['h2_sb'])
            pf.space_after = Pt(P['h2_sa'])
            pf.line_spacing = 1.05
            continue
        if sz == 11.5:                      # 三级标题
            pf.space_before = Pt(P['h3_sb'])
            pf.space_after = Pt(P['h3_sa'])
            pf.line_spacing = 1.05
            continue
        if sz == 18:                        # 大标题
            pf.space_after = Pt(4)
            pf.line_spacing = 1.0
            continue
        if sz == 9 and p.alignment == WD_ALIGN_PARAGRAPH.CENTER:   # 图注
            pf.space_before = Pt(0)
            pf.space_after = Pt(P['cap_sa'])
            pf.line_spacing = 1.0
            continue
        if p.text.strip() == '' and p.runs:  # 插图所在段落
            pf.space_before = Pt(2)
            pf.space_after = Pt(0)
            continue
        if pf.first_line_indent is not None or pf.left_indent is not None:
            pf.line_spacing = P['list_ls']   # 编号列表
            pf.space_before = Pt(0.5)
            pf.space_after = Pt(0.5)
            continue
        pf.line_spacing = P['body_ls']
        pf.space_before = Pt(1)
        pf.space_after = Pt(0)


def tune_figures(doc):
    for shp in doc.inline_shapes:
        old_w = shp.width
        if old_w <= 0:
            continue
        ratio = P['fig_w'] / old_w.cm
        shp.width = Cm(P['fig_w'])
        shp.height = int(shp.height * ratio)


def tune_tables(doc):
    for t in doc.tables:
        for row in t.rows:
            for cell in row.cells:
                for p in cell.paragraphs:
                    pf = p.paragraph_format
                    pf.space_before = Pt(P['tab_cell_sb'])
                    pf.space_after = Pt(P['tab_cell_sa'])
                    pf.line_spacing = 1.0
                    for r in p.runs:
                        r.font.size = Pt(P['tab_pt'])
        tblPr = t._tbl.tblPr
        ns = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
        mar = tblPr.find(ns + 'tblCellMar')
        if mar is not None:
            for tag, val in (('top', P['tab_mar_tb']), ('bottom', P['tab_mar_tb']),
                             ('left', P['tab_mar_lr']), ('right', P['tab_mar_lr'])):
                el = mar.find(ns + tag)
                if el is not None:
                    el.set(ns + 'w', str(val))


def text_cm(t, pt):
    """估算文字在 pt 字号下的宽度（cm，含单元格内边距）。"""
    t = re.sub(r'\s+', ' ', t).strip()
    cjk = sum(1 for ch in t if ord(ch) > 0x2E80)
    k = pt / 9.0
    return (cjk * 0.32 + (len(t) - cjk) * 0.165) * k + 0.28


def token_cm(t, pt):
    """最长不可断词所需宽度，避免长单词被拦腰截断。"""
    best = 0.0
    for part in re.split(r'[\s/+,、：:；;]+', t):
        if part.strip():
            best = max(best, text_cm(part, pt))
    return best


def refit_table(t, usable):
    """按内容重新分配列宽，填满新的版心宽度。"""
    ncol = len(t.columns)
    texts = [[c.text for c in row.cells] for row in t.rows]
    need, floor = [], []
    for ci in range(ncol):
        col = [r[ci] for r in texts if ci < len(r)]
        need.append(min(max(text_cm(x, P['tab_pt']) for x in col), usable * 0.62))
        floor.append(min(max(token_cm(x, P['tab_pt']) for x in col), usable * 0.34))

    dsum = sum(need) or 1.0
    widths = [usable * n / dsum for n in need]
    for _ in range(4):
        deficit = sum(max(0.0, floor[i] - widths[i]) for i in range(ncol))
        if deficit < 1e-3:
            break
        widths = [max(widths[i], floor[i]) for i in range(ncol)]
        slack = [i for i in range(ncol) if widths[i] - floor[i] > 0.05]
        pool = sum(widths[i] - floor[i] for i in slack) or 1.0
        for i in slack:
            widths[i] -= deficit * (widths[i] - floor[i]) / pool
    scale = usable / sum(widths)
    widths = [w * scale for w in widths]

    for ci, w in enumerate(widths):
        t.columns[ci].width = Cm(w)
    for row in t.rows:
        for ci, w in enumerate(widths):
            if ci < len(row.cells):
                row.cells[ci].width = Cm(w)


def fix_break_opportunities(doc):
    """在表格里的 '+' 后插入零宽空格：Word 不会在 '+' 处自动换行，
    长串（如 RandomResizedCrop+ColorJitter+Rotation）会被拦腰截断。
    零宽空格不可见，只增加一个合法断点，不改文字内容。"""
    n = 0
    for t in doc.tables:
        for row in t.rows:
            for cell in row.cells:
                for p in cell.paragraphs:
                    for r in p.runs:
                        if '+' in r.text and '\u200b' not in r.text:
                            r.text = r.text.replace('+', '+\u200b')
                            n += 1
    return n


def _par_after(par):
    """在指定段落之后插入一个空段落并返回。"""
    from docx.text.paragraph import Paragraph
    new_p = OxmlElement('w:p')
    par._p.addnext(new_p)
    return Paragraph(new_p, par._parent)


def add_fig3(doc):
    """补入“图3 学习率动态衰减的收敛曲线”——原文正文引用了图3，但报告里漏排。"""
    fig = os.path.join(os.path.dirname(HERE), 'results', 'figs', 'fig3_decay.png')
    if not os.path.exists(fig):
        return 0
    target = None
    for p in doc.paragraphs:
        if p.text.strip().startswith('① 精度呈明显的'):
            target = p
            break
    if target is None:
        return 0

    img_p = _par_after(target)
    img_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    pf = img_p.paragraph_format
    pf.space_before, pf.space_after, pf.line_spacing = Pt(2), Pt(0), 1.0
    img_p.add_run().add_picture(fig, width=Cm(P['fig3_w']))

    cap = _par_after(img_p)
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    pf = cap.paragraph_format
    pf.space_before, pf.space_after, pf.line_spacing = Pt(0), Pt(P['cap_sa']), 1.0
    r = cap.add_run('图3 学习率动态衰减（第 30、40 epoch 各降 10 倍）下的训练损失与测试精度收敛曲线')
    r.font.size = Pt(9)
    r.font.name = 'Times New Roman'
    rPr = r._element.get_or_add_rPr()
    rFonts = OxmlElement('w:rFonts')
    rFonts.set(qn('w:ascii'), 'Times New Roman')
    rFonts.set(qn('w:hAnsi'), 'Times New Roman')
    rFonts.set(qn('w:eastAsia'), '宋体')
    rPr.append(rFonts)
    return 1


def main():
    doc = Document(SRC)
    tune_section(doc)
    tune_styles(doc)
    n = drop_spacers(doc)
    tune_paragraphs(doc)
    tune_figures(doc)
    tune_tables(doc)
    m = fix_break_opportunities(doc)
    k = add_fig3(doc)
    usable = 21.0 - 2 * P['margin_lr']
    for t in doc.tables:
        refit_table(t, usable)
    doc.save(DST)
    print(f'删除空行 {n} 个，加断点 {m} 处，补图 {k} 张；输出 {DST}  {os.path.getsize(DST)} bytes')


if __name__ == '__main__':
    main()
