"""把 report/report.html 转成可编辑的 Word 文档（.docx），与 PDF 并存。

用法:  python make_docx.py
输入:  <项目>/report/report.html（由 make_report.py 生成，图片以 base64 内嵌）
输出:  本脚本所在目录下的 实验报告-图像分类2.docx

说明: 标题、表格、图片均为原生 Word 对象，可直接在 Word/WPS 中修改；
      PDF 仍由 make_report.py 生成，两者内容同源（同一份 HTML）。
"""
import os
import re
import base64
import io

from bs4 import BeautifulSoup, NavigableString, Tag
from docx import Document
from docx.shared import Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_ALIGN_VERTICAL
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, '实验报告-图像分类2.docx')

BODY_EA, BODY_ASCII = '宋体', 'Times New Roman'
HEAD_EA, HEAD_ASCII = '黑体', 'Arial'
MONO = 'Consolas'
BODY_PT = 10.5          # 五号
TAB_PT = 9.0
C_DARK = RGBColor(0x1A, 0x1A, 0x1A)
C_H2 = RGBColor(0x1F, 0x38, 0x64)
C_H3 = RGBColor(0x2B, 0x4A, 0x6F)
C_GRAY = RGBColor(0x44, 0x44, 0x44)


def find_project_root():
    """脚本可能放在子目录里，向上找到含 make_report.py 的项目根。"""
    d = HERE
    for _ in range(4):
        if os.path.exists(os.path.join(d, 'make_report.py')):
            return d
        d = os.path.dirname(d)
    return HERE


# ------------------------------------------------------------------ 基础工具
def style_run(run, ea=BODY_EA, ascii_=BODY_ASCII, size=BODY_PT,
              bold=False, color=None, mono=False):
    if mono:
        ea, ascii_ = MONO, MONO
    run.font.name = ascii_
    run.font.size = Pt(size)
    run.bold = bold
    if color is not None:
        run.font.color.rgb = color
    rPr = run._element.get_or_add_rPr()
    rFonts = rPr.find(qn('w:rFonts'))
    if rFonts is None:
        rFonts = OxmlElement('w:rFonts')
        rPr.append(rFonts)
    rFonts.set(qn('w:ascii'), ascii_)
    rFonts.set(qn('w:hAnsi'), ascii_)
    rFonts.set(qn('w:eastAsia'), ea)


def set_normal_style(doc):
    st = doc.styles['Normal']
    st.font.name = BODY_ASCII
    st.font.size = Pt(BODY_PT)
    st.element.rPr.rFonts.set(qn('w:eastAsia'), BODY_EA)
    pf = st.paragraph_format
    pf.line_spacing = 1.35
    pf.space_before = Pt(0)
    pf.space_after = Pt(3)


def bottom_border(par, color='2B6CB0', sz=8):
    pPr = par._p.get_or_add_pPr()
    pBdr = OxmlElement('w:pBdr')
    bottom = OxmlElement('w:bottom')
    bottom.set(qn('w:val'), 'single')
    bottom.set(qn('w:sz'), str(sz))
    bottom.set(qn('w:space'), '2')
    bottom.set(qn('w:color'), color)
    pBdr.append(bottom)
    pPr.append(pBdr)


def shade_cell(cell, hex_fill):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement('w:shd')
    shd.set(qn('w:val'), 'clear')
    shd.set(qn('w:color'), 'auto')
    shd.set(qn('w:fill'), hex_fill)
    tcPr.append(shd)


def text_cm(text):
    """粗估一段文字在 9pt 下的宽度（cm，含单元格内边距）。"""
    t = re.sub(r'\s+', ' ', text).strip()
    cjk = sum(1 for ch in t if ord(ch) > 0x2E80)
    return cjk * 0.32 + (len(t) - cjk) * 0.165 + 0.30


def token_cm(text):
    """最长不可断词（如 WeightedRandomSampler）所需宽度，用来避免单词被拦腰截断。"""
    best = 0.0
    for part in re.split(r'[\s/+,、：:；;]+', text):
        if part.strip():
            best = max(best, text_cm(part))
    return best


def compute_widths(rows, ncol, usable=17.0, min_cm=1.6):
    """按各列内容分配列宽；HTML 里写了 width:x% 的列优先遵循。"""
    explicit = {}
    head_cells = rows[0].find_all(['td', 'th'], recursive=False)
    for ci, cell in enumerate(head_cells[:ncol]):
        m = re.search(r'width:\s*([\d.]+)%', cell.get('style') or '')
        if m:
            explicit[ci] = float(m.group(1)) / 100.0 * usable

    need, floor = [], []
    for ci in range(ncol):
        want, must = 0.0, min_cm
        for tr in rows:
            cells = tr.find_all(['td', 'th'], recursive=False)
            if ci < len(cells):
                txt = cells[ci].get_text()
                want = max(want, text_cm(txt))
                must = max(must, token_cm(txt))
        need.append(min(want, usable * 0.62))
        floor.append(min(must, usable * 0.34))

    free = [ci for ci in range(ncol) if ci not in explicit]
    rest = usable - sum(explicit.values())
    dsum = sum(need[ci] for ci in free) or 1.0
    widths = [explicit.get(ci, rest * need[ci] / dsum) for ci in range(ncol)]

    # 补足“不可断词”所需的最小宽度，差额从有余量的列里按比例扣
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
    return [w * scale for w in widths]


def keep_together(table):
    """让行数不多的表格整体不跨页。"""
    for row in table.rows[:-1]:
        for cell in row.cells:
            for par in cell.paragraphs:
                par.paragraph_format.keep_with_next = True


def table_layout(table, cell_mar=(28, 60, 28, 60)):
    """表格宽度 100%，固定列宽（由 compute_widths 计算），并收紧单元格内边距。"""
    tblPr = table._tbl.tblPr
    tblW = OxmlElement('w:tblW')
    tblW.set(qn('w:type'), 'pct')
    tblW.set(qn('w:w'), '5000')
    tblPr.append(tblW)
    layout = OxmlElement('w:tblLayout')
    layout.set(qn('w:type'), 'fixed')
    tblPr.append(layout)
    mar = OxmlElement('w:tblCellMar')
    for tag, val in zip(('top', 'left', 'bottom', 'right'), cell_mar):
        el = OxmlElement(f'w:{tag}')
        el.set(qn('w:w'), str(val))
        el.set(qn('w:type'), 'dxa')
        mar.append(el)
    tblPr.append(mar)


def row_no_split(row, header=False):
    trPr = row._tr.get_or_add_trPr()
    trPr.append(OxmlElement('w:cantSplit'))
    if header:
        trPr.append(OxmlElement('w:tblHeader'))


def add_page_footer(section):
    par = section.footer.paragraphs[0]
    par.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = par.add_run()
    style_run(run, size=9, color=C_GRAY)
    fld_begin = OxmlElement('w:fldChar')
    fld_begin.set(qn('w:fldCharType'), 'begin')
    instr = OxmlElement('w:instrText')
    instr.set(qn('xml:space'), 'preserve')
    instr.text = 'PAGE'
    fld_end = OxmlElement('w:fldChar')
    fld_end.set(qn('w:fldCharType'), 'end')
    for el in (fld_begin, instr, fld_end):
        run._r.append(el)


# ------------------------------------------------------------------ 行内渲染
def segments(node, bold=False, code=False, out=None):
    """把行内结构摊平成 (文本, 加粗, 等宽, 是否换行) 序列。"""
    if out is None:
        out = []
    for child in node.children:
        if isinstance(child, NavigableString):
            out.append((str(child), bold, code, False))
        elif isinstance(child, Tag):
            if child.name == 'br':
                out.append(('', bold, code, True))
            elif child.name in ('b', 'strong'):
                segments(child, True, code, out)
            elif child.name in ('code', 'tt', 'kbd'):
                segments(child, True, True, out)
            else:
                segments(child, bold, code, out)
    return out


def render_inline(par, node, size=BODY_PT, bold=False, color=None,
                  ea=BODY_EA, ascii_=BODY_ASCII):
    segs = segments(node)
    # HTML 源码里的换行缩进压成单个空格
    norm = [(re.sub(r'\s+', ' ', t) if not br else t, b, c, br)
            for t, b, c, br in segs]
    for i, (t, b, c, br) in enumerate(norm):
        if i == 0:
            t = t.lstrip()
        if i == len(norm) - 1:
            t = t.rstrip()
        norm[i] = (t, b, c, br)

    for text, is_bold, is_code, br in norm:
        if br:
            par.add_run().add_break()
            continue
        if not text:
            continue
        run = par.add_run(text)
        style_run(run, ea=(MONO if is_code else ea),
                  ascii_=(MONO if is_code else ascii_),
                  size=(size - 0.5 if is_code else size),
                  bold=(bold or is_bold), color=color, mono=is_code)


# ------------------------------------------------------------------ 块级渲染
def add_heading(doc, node, level):
    par = doc.add_paragraph()
    pf = par.paragraph_format
    if level == 1:
        par.alignment = WD_ALIGN_PARAGRAPH.CENTER
        pf.space_before = Pt(0)
        pf.space_after = Pt(6)
        render_inline(par, node, size=18, bold=True, color=C_DARK,
                      ea=HEAD_EA, ascii_=HEAD_ASCII)
    elif level == 2:
        pf.space_before = Pt(14)
        pf.space_after = Pt(5)
        pf.keep_with_next = True
        render_inline(par, node, size=13.5, bold=True, color=C_H2,
                      ea=HEAD_EA, ascii_=HEAD_ASCII)
        bottom_border(par)
    else:
        pf.space_before = Pt(9)
        pf.space_after = Pt(3)
        pf.keep_with_next = True
        render_inline(par, node, size=11.5, bold=True, color=C_H3,
                      ea=HEAD_EA, ascii_=HEAD_ASCII)
    return par


def add_meta(doc, node):
    par = doc.add_paragraph()
    par.alignment = WD_ALIGN_PARAGRAPH.CENTER
    par.paragraph_format.space_after = Pt(10)
    render_inline(par, node, size=9, color=C_GRAY)


def add_para(doc, node, size=BODY_PT, color=C_DARK):
    par = doc.add_paragraph()
    pf = par.paragraph_format
    pf.space_before = Pt(2)
    pf.space_after = Pt(3)
    pf.line_spacing = 1.35
    cls = node.get('class') or []
    if 'small' in cls or 'note' in cls:
        size, color = 9.5, C_GRAY
    render_inline(par, node, size=size, color=color)
    return par


def add_list(doc, node, ordered=True):
    for idx, li in enumerate(node.find_all('li', recursive=False), 1):
        par = doc.add_paragraph()
        pf = par.paragraph_format
        pf.left_indent = Cm(0.78)
        pf.first_line_indent = Cm(-0.5)
        pf.space_before = Pt(1)
        pf.space_after = Pt(1)
        pf.line_spacing = 1.3
        run = par.add_run(f'{idx}. ' if ordered else '· ')
        style_run(run, size=BODY_PT, color=C_DARK)
        render_inline(par, li, size=BODY_PT, color=C_DARK)


def add_figure(doc, node):
    img = node.find('img')
    cap = node.find(class_='figcap')
    if img is not None and img.get('src', '').startswith('data:image'):
        raw = img['src'].split(',', 1)[1]
        par = doc.add_paragraph()
        par.alignment = WD_ALIGN_PARAGRAPH.CENTER
        par.paragraph_format.space_before = Pt(6)
        par.paragraph_format.space_after = Pt(2)
        par.add_run().add_picture(io.BytesIO(base64.b64decode(raw)),
                                  width=Cm(15.4))
    if cap is not None:
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_after = Pt(8)
        render_inline(p, cap, size=9, color=C_GRAY)


def add_table(doc, node):
    rows = node.find_all('tr')
    if not rows:
        return
    ncol = max(len(r.find_all(['td', 'th'], recursive=False)) for r in rows)
    table = doc.add_table(rows=0, cols=ncol)
    table.style = 'Table Grid'
    table.autofit = False
    table_layout(table)
    widths = compute_widths(rows, ncol)
    for ci, w in enumerate(widths):
        table.columns[ci].width = Cm(w)

    for ri, tr in enumerate(rows):
        cells = tr.find_all(['td', 'th'], recursive=False)
        head_row = all(c.name == 'th' for c in cells)
        row = table.add_row()
        row_no_split(row, header=(ri == 0 and head_row))
        for ci in range(ncol):
            cell = row.cells[ci]
            cell.width = Cm(widths[ci])
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            par = cell.paragraphs[0]
            par.paragraph_format.space_before = Pt(1)
            par.paragraph_format.space_after = Pt(1)
            par.paragraph_format.line_spacing = 1.15
            if ci >= len(cells):
                continue
            src = cells[ci]
            left = 'l' in (src.get('class') or [])
            par.alignment = (WD_ALIGN_PARAGRAPH.LEFT if left
                             else WD_ALIGN_PARAGRAPH.CENTER)
            render_inline(par, src, size=TAB_PT,
                          bold=(src.name == 'th'), color=C_DARK,
                          ea=(HEAD_EA if src.name == 'th' else BODY_EA),
                          ascii_=(HEAD_ASCII if src.name == 'th' else BODY_ASCII))
            if src.name == 'th':
                shade_cell(cell, 'EEF3F8')
    if len(table.rows) <= 9:
        keep_together(table)
    # 表后留一点空隙
    tail = doc.add_paragraph()
    tail.paragraph_format.space_before = Pt(0)
    tail.paragraph_format.space_after = Pt(0)
    tail.paragraph_format.line_spacing = 0.7


# ------------------------------------------------------------------ 主流程
def build():
    html_path = os.path.join(find_project_root(), 'report', 'report.html')
    if not os.path.exists(html_path):
        raise SystemExit(f'找不到 {html_path}，请先运行 make_report.py')

    soup = BeautifulSoup(open(html_path, encoding='utf-8').read(), 'lxml')
    doc = Document()
    set_normal_style(doc)

    sec = doc.sections[0]
    sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)
    sec.top_margin = sec.bottom_margin = Cm(2.0)
    sec.left_margin = sec.right_margin = Cm(2.0)
    add_page_footer(sec)

    n_tab = n_fig = 0
    for el in soup.body.children:
        if not isinstance(el, Tag):
            continue
        name, cls = el.name, (el.get('class') or [])
        if name == 'h1':
            add_heading(doc, el, 1)
        elif name == 'h2':
            add_heading(doc, el, 2)
        elif name == 'h3':
            add_heading(doc, el, 3)
        elif name == 'div' and 'meta' in cls:
            add_meta(doc, el)
        elif name == 'div' and 'fig' in cls:
            add_figure(doc, el)
            n_fig += 1
        elif name == 'p':
            add_para(doc, el)
        elif name == 'ol':
            add_list(doc, el, ordered=True)
        elif name == 'ul':
            add_list(doc, el, ordered=False)
        elif name == 'table':
            add_table(doc, el)
            n_tab += 1

    doc.core_properties.title = '基于卷积神经网络的图像分类 实验报告'
    doc.core_properties.subject = '计算机视觉 实验二'
    doc.save(OUT)
    print(f'DOCX written: {OUT}  {os.path.getsize(OUT)} bytes')
    print(f'表格 {n_tab} 个，插图 {n_fig} 张，段落 {len(doc.paragraphs)} 个')


if __name__ == '__main__':
    build()
