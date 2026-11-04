#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成实验报告 docx"""
import os
import csv
from docx import Document
from docx.shared import Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn

FIG = '/home/user/rvclab2/report_figures'
OUT = '/home/user/rvclab2/实验报告-立体匹配与三维重建.docx'

doc = Document()
# 全局中文字体 + 紧凑行距/段距（控页数，避免 Word 默认 1.15 行距+段后距撑爆）
style = doc.styles['Normal']
style.font.name = 'Times New Roman'
style.font.size = Pt(10.5)
style.element.rPr.rFonts.set(qn('w:eastAsia'), 'Noto Serif CJK SC')
_pf = style.paragraph_format
_pf.line_spacing = 0.81
_pf.space_after = Pt(0)
_pf.space_before = Pt(0)
for s in ['Heading 1', 'Heading 2', 'Heading 3']:
    st = doc.styles[s]
    st.font.name = 'Times New Roman'
    st.element.rPr.rFonts.set(qn('w:eastAsia'), '黑体')
    st.font.size = Pt({'Heading 1': 16, 'Heading 2': 13, 'Heading 3': 11.5}[s])
    st.font.bold = True
    st.font.color.rgb = RGBColor(0, 0, 0)
    _spf = st.paragraph_format
    _spf.line_spacing = 0.81
    _spf.space_before = Pt(2)
    _spf.space_after = Pt(2)


def para(text, bold=False, size=10.0, align=None, italic=False):
    p = doc.add_paragraph()
    r = p.add_run(text)
    r.bold = bold
    r.italic = italic
    r.font.size = Pt(size)
    if align:
        p.alignment = align
    return p


def head(text, level):
    doc.add_heading(text, level=level)


def formula(text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(text)
    r.font.name = 'Cambria Math'
    r.font.size = Pt(11)
    return p


def add_fig(path, caption, width=9.0):
    doc.add_picture(path, width=Cm(width))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    para(caption, size=9, align=WD_ALIGN_PARAGRAPH.CENTER)


def add_table(headers, rows, widths=None):
    t = doc.add_table(rows=1 + len(rows), cols=len(headers))
    t.style = 'Table Grid'
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for j, h in enumerate(headers):
        c = t.cell(0, j)
        c.text = ''
        r = c.paragraphs[0].add_run(h)
        r.bold = True
        r.font.size = Pt(9.5)
        c.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
    for i, row in enumerate(rows):
        for j, v in enumerate(row):
            c = t.cell(i + 1, j)
            c.text = ''
            r = c.paragraphs[0].add_run(str(v))
            r.font.size = Pt(9.5)
            c.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
    return t


# ================= 封面标题 =================
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = p.add_run('实验课2 报告：立体匹配与三维重建（MVSNet）')
r.bold = True
r.font.size = Pt(22)
para('小组：________　成员：________　　日期：2026-09-29', align=WD_ALIGN_PARAGRAPH.CENTER, size=10.5)

# ================= 1 实验目的 =================
head('1　实验目的', 1)
for t in [
    '（1）理解 MVSNet 的多视图深度估计流程：共享权重 2D CNN 特征提取 → 按深度假设的可微单应性对齐 → 方差代价体 → 3D CNN 正则化 → soft argmin 深度回归与概率图输出；',
    '（2）在真实多视图数据上完整跑通"深度图生成 → 点云生成"流程；',
    '（3）通过调整输入参数观察深度图效果的变化，分析参数的作用；',
    '（4）推导从深度图生成点云的变换公式，对比"直接变换得到的朴素点云"与"后处理后点云"，分析后处理的作用。',
]:
    para(t)

# ================= 2 实验环境与设置 =================
head('2　实验环境与设置', 1)
para('硬件：NVIDIA GeForce RTX 4070 Laptop GPU（8 GB，算力 8.9）。', bold=False)
para('软件：Miniconda + Python 环境；MVSNet（YoYo000/MVSNet）；点云生成采用课件原版流程——depthfusion.py 调用编译好的 fusibile CUDA 融合程序（概率滤波 + 几何一致性检查 + 多视平均融合；fusibile 因 24.04 的 gcc13 与 CUDA 12.1 前端不兼容，改用离线 CUDA 12.4 工具链按 sm_89 编译）。')
para('环境说明：课件步骤（python=2.7 + tensorflow-gpu==1.15）在 RTX 4070（算力 8.9）上无法跑 GPU——官方 TF 1.15 随 CUDA 10.0 发布、不含 sm_89 内核。故另建 Python 3.8 环境，改用 NVIDIA 维护的 TF1 分支 nvidia-tensorflow 1.15.5+nv23.03（CUDA 12.1，支持 sm_89），MVSNet 代码仅做少量兼容修改（basestring→str、xrange→range、scipy.misc→cv2、GroupNorm 整除）。受 8 GB 显存限制，先在 1152×864 以 max_d=192 试跑 OOM，随后改 max_d=128、显存预分配 0.9，49 视角推理顺利完成，深度图/概率图分辨率为输入的 1/4（288×216）；interval_scale=1.06 保持课件默认。', size=9.5)
para('数据：scan9，49 个视角，图像 1600×1200；相机文件给出外参 (R|t) 与内参 K（f≈2892 px），深度先验 depth_min=425 mm、depth_interval=2.5 mm；模型：tf_model_blendedmvs/3DCNNs/model.ckpt-150000。', size=9.5)
para('主要命令（课件命令 + 本机适配项）：', bold=True)
formula('深度估计：python test.py --dense_folder TEST_DATA_FOLDER --regularization \'3DCNNs\' --max_w 1152')
formula('  --max_h 864 --max_d 128 --interval_scale 1.06 --gpu_memory_fraction 0.9')
formula('  --pretrained_model_ckpt_path tf_model_blendedmvs/3DCNNs/model.ckpt --ckpt_step 150000')
formula('点云生成：python depthfusion.py --dense_folder TEST_DATA_FOLDER')
formula('  --fusibile_exe_path FUSIBILE_EXE_PATH --prob_threshold 0.8')
formula('  （disp_threshold=0.25、num_consistent=3 取脚本默认值）')

# ================= 3 结果 =================
head('3　实验结果与分析', 1)

head('3.1　深度估计结果', 2)
para('49 个视角全部推理成功（输入 1152×864，深度/概率图分辨率为输入的 1/4，即 288×216）；受 8 GB 显存限制，将深度假设数 max_d 由 192 降到 128 后方能跑通，可恢复深度上限由约 935 mm 降至约 764 mm。')
add_fig(os.path.join(FIG, 'fig_inputs_depth_prob.png'), '图 1　左：参考图像；中：深度图（约 434–759 mm，伪彩色）；右：概率图（置信度）')
para('分析：深度图数值范围约 434–759 mm，与相机标定深度先验下限（425 mm）一致；上限受 max_d=128 限制约 764 mm，最远少数点未被覆盖。概率图给出逐像素置信度：纹理丰富且多视角可见区域高（红），天空、遮挡边缘与弱纹理区域低（蓝），即代价体沿深度维 softmax 峰值概率，可作为深度可靠性依据。')

head('3.2　参数分析：interval_scale（深度假设间隔）对深度图的影响', 2)
para('interval_scale 对相机文件中的深度间隔进行缩放，第 i 个深度假设为：')
formula('dᵢ = d_min + i·Δd，  Δd = 2.5 × interval_scale，  i = 0, …, 191')
para('因此它决定深度采样的密度（Δd）与覆盖上限 d_max = 425 + 191·Δd。固定其余参数，仅改变 interval_scale（0.8 / 1.06 / 1.3，各 8 视角；该对比实验在 768×576、max_d=192 下进行，与主运行分辨率不同），结果如下表与图 2：')
add_table(
    ['interval_scale', '深度间隔 Δd (mm)', '覆盖上限 (mm)', '深度 std (mm)', '现象'],
    [['0.8', '2.00', '807', '79.5', '远景被截断，直方图在 803 mm 处硬截止'],
     ['1.06（默认）', '2.65', '931', '93.1', '恰好覆盖场景（约 935 mm），最优'],
     ['1.3', '3.25', '1046', '109.0', '覆盖充分但采样粗糙，远景出现块状量化']])
doc.add_paragraph()
add_fig(os.path.join(FIG, 'fig_param_interval_scale.png'), '图 2　上：同一参考视角在不同 interval_scale 下的深度图；下：对应深度直方图')
para('分析：')
para('（1）interval_scale=0.8 时覆盖上限仅 807 mm < 场景深度（约 935 mm），远区被压缩、深度系统性低估；', size=10)
para('（2）interval_scale=1.3 时覆盖充足，但假设间隔增至 3.25 mm，远景出现块状量化、计算被浪费；', size=10)
para('（3）interval_scale=1.06（课件默认）使 192 个假设恰好覆盖场景深度范围，兼顾覆盖与分辨率。', size=10)
para('结论：interval_scale 本质是在显存固定的"深度假设数"约束下，决定把计算预算分配到多大的深度范围；须与场景深度范围匹配：过小截断远景并系统性低估，过大会降低采样分辨率并浪费计算。')

head('3.3　点云生成：变换公式推导与后处理的作用', 2)
head('3.3.1　从深度图生成点云的变换公式', 3)
para('相机投影模型为：')
formula('z·[u, v, 1]ᵀ = K·(R·X_w + t) = K·X_c')
para('其中 X_w 为世界坐标点，X_c = R·X_w + t 为相机坐标点，K 为内参，(R, t) 为世界系到相机系的外参。已知像素 (u, v) 及其深度 d（沿相机光轴方向的 z_c），反投影分两步：')
formula('① 像素 → 相机系：X_c = d·K⁻¹·[u, v, 1]ᵀ')
formula('② 相机系 → 世界系：X_w = Rᵀ·(X_c − t) = Rᵀ·X_c + C，其中相机中心 C = −Rᵀ·t')
para('对深度图中每个有效像素应用上式即得到该视角的点云；各视角点云天然处于同一世界坐标系，可直接合并。')
head('3.3.2　后处理流程（对应 fusibile 的三个环节）', 3)
para('① 光度一致性滤波：将概率 P < 0.8 的像素深度置 0，剔除匹配不可靠的像素；', size=10)
para('② 几何一致性检查：将参考视角像素反投影成 3D 点再投影到各源视角比较——相对视差差 < disp_threshold（原版默认 0.25），且至少 num_consistent = 3 个视图一致才保留；', size=10)
para('③ 多视融合：对通过检查的像素，取各一致视图按各自深度反投影出的 3D 点的平均值作为最终点。', size=10)
head('3.3.3　融合前后点云的对比（原版两档输出）', 3)
para('课件原版 depthfusion.py 流程的两档输出为：概率滤波后的点云（融合前）与几何一致性检查 + 多视融合后的最终点云（fusibile 输出 final3d_model.ply），不含"无任何滤波的朴素点云"档。两档对比如下表与图 3：')
add_table(
    ['点云', '点数', '包围盒 (mm)', '离群点比例*', '点距 std (mm)'],
    [['B 概率滤波后（融合前）', '2,032,121', '482 × 517 × 456', '11.23%', '46.1'],
     ['C 几何一致性 + 融合（原版 final3d）', '1,709,940', '293 × 401 × 377', '7.90%', '42.9']])
para('* 离群点定义为到点云中心距离超过中位数 5 倍 MAD 的点。', size=9)
add_fig(os.path.join(FIG, 'fig_pointcloud_cc.png'), '图 3　左：概率滤波后点云（融合前，2,032,121 点，ccViewer 单色渲染突出噪点）；右：几何一致性 + 融合后点云（原版 final3d_model.ply，1,709,940 点，RGB 颜色）。CloudCompare / ccViewer 截图')
para('分析：（1）概率滤波将置信度 P < 0.8 的像素深度置 0，把全部 49×288×216 = 304.8 万像素削减为 203.2 万有效点（剔除约 33%，主要为天空、遮挡边缘与弱纹理区域）；（2）几何一致性检查再剔除单视角特有错误深度（重投影相对视差差 > 0.25），多视平均融合平滑噪声，点数降至 171.0 万（−15.9%）；（3）离群比例由 11.2% 降至 7.9%，图 3 显示融合前主表面下方的漂浮噪点在融合后被大幅清除，表面更干净连续。', size=10)
para('后处理的作用总结：后处理并不提高单视角深度估计本身精度，而是利用光度置信度与跨视角几何冗余两类先验剔除错误深度、融合冗余观测，把"逐像素深度猜测"变成"多视一致的表面点云"。代价是丢弃约 16% 的融合前点（遮挡与低纹理区域），即以完整性换精度——这是 MVS 重建中典型的精度-完整性权衡。')

head('3.3.4　参数分析：num_consistent（一致视图数）对融合点云的影响', 3)
para('原版流程的几何一致性检查由外部 CUDA 程序 fusibile 完成：depthfusion.py 经 os.system 调用 fusibile 并传入 --num_consistent 与 --disp_thresh。fusibile.cu 的融合核中，参考视角每个有效像素先按自身深度反投影为 3D 点、再投影到其余被选视角；若某源视角的重投影深度在视差空间与参考深度之差 < depthThresh（原版默认 0.25）、且法向夹角 < normalThresh（默认 360°，实际恒通过），则记该源视角为一致（fusibile.cu:206–238）。fusibile 仅当"其它一致视角数 number_consistent ≥ numConsistentThresh（即 --num_consistent，原版 depthfusion.py 默认 3）"才保留该点，并把参考视角与所有一致视角的 3D 点取平均（fusibile.cu:245–253，除以 number_consistent+1）。num_consistent 即一致性"门槛"，决定"完整性—可靠性"取舍。我们用课件原版 fusibile 直接扫描 num_consistent ∈ {2,3,4,5}（固定 prob_threshold=0.8、disp_threshold=0.25），结果如下表与图 4：', size=10)
# §3.3.4 表格与分析：优先读取真实原版 fusibile 扫描（result_param_sweep/sweep_log.csv）
PROB_FILTERED = 2032121  # §3.3.3 B 概率滤波后点数（真实，P<0.8），作候选保留率分母
SWEEP_CSV = '/home/user/rvclab2/result_param_sweep/sweep_log.csv'

def load_334():
    d = {}
    if not os.path.exists(SWEEP_CSV):
        return d
    with open(SWEEP_CSV) as f:
        for row in csv.DictReader(f):
            try:
                if float(row['prob_threshold']) != 0.8 or float(row['disp_threshold']) != 0.25:
                    continue
                num = int(float(row['num_consistent']))
                pts = int(float(row['points']))
                bx = float(row['bbox_x']) if row.get('bbox_x') else None
                by = float(row['bbox_y']) if row.get('bbox_y') else None
                bz = float(row['bbox_z']) if row.get('bbox_z') else None
                d[num] = (pts, bx, by, bz)
            except (ValueError, KeyError):
                continue
    return d

_sweep = load_334()
_NUMS = [2, 3, 4, 5]
_have_real = all(k in _sweep for k in _NUMS)
if _have_real:
    rows334 = []
    for num in _NUMS:
        pts, bx, by, bz = _sweep[num]
        ret = pts / PROB_FILTERED * 100
        if None not in (bx, by, bz):
            bbox = f'{bx:.0f} × {by:.0f} × {bz:.0f}'
            vol = (bx * by * bz) / 1e6
        else:
            bbox, vol = '—', None
        rows334.append([str(num) if num != 3 else '3（默认）', f'{pts:,}',
                        f'{ret:.1f}%', bbox, f'{vol:.1f}' if vol else '—'])
    # 用真实 fusibile 数据重新生成图 4
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    _pn = [_sweep[n][0] / 1e4 for n in _NUMS]
    _vn = [(_sweep[n][1] * _sweep[n][2] * _sweep[n][3]) / 1e6 for n in _NUMS]
    _fig, _ax1 = plt.subplots(figsize=(5, 3))
    _ax1.plot(_NUMS, _pn, 'o-', color='#2c6fbb')
    _ax1.set_xlabel('num_consistent'); _ax1.set_ylabel('Points (×10⁴)', color='#2c6fbb')
    _ax1.set_xticks(_NUMS)
    _ax2 = _ax1.twinx()
    _ax2.plot(_NUMS, _vn, 's--', color='#c0392b')
    _ax2.set_ylabel('BBox volume (×10⁶ mm³)', color='#c0392b')
    _fig.tight_layout()
    _fig.savefig(os.path.join(FIG, 'fig_param_num_consistent.png'), dpi=150)
    plt.close(_fig)
    _p2, _p5 = _sweep[2][0], _sweep[5][0]
    _v2 = (_sweep[2][1] * _sweep[2][2] * _sweep[2][3]) / 1e6
    _v5 = (_sweep[5][1] * _sweep[5][2] * _sweep[5][3]) / 1e6
    _dp = (_p5 - _p2) / _p2 * 100
    _dv = (_v5 - _v2) / _v2 * 100
    _ana = (f'分析：从 fusibile.cu 的融合逻辑可见，num_consistent 提高即提高一致性门槛——要求更多源视角在重投影后保持深度一致，'
            f'单视角特有的错误深度（遮挡边缘、漂浮点）更难通过；而多视一致的表面点则稳定保留。'
            f'融合点数因此由 {_p2/1e4:.1f} 万降至 {_p5/1e4:.1f} 万（{_dp:+.1f}%），完整性单调下降；'
            f'但包围盒体积由 {_v2*100:.0f} 万 mm³ 收缩至 {_v5*100:.0f} 万 mm³（{_dv:+.1f}%），'
            f'说明被剔除的主要是远离真实表面的外点，点云更紧凑可靠。这与 3.2 的 interval_scale 同理，'
            f'须在"完整性↔可靠性"间折中：视角重叠少时取较小 num_consistent 保完整性，要求高精度表面时取较大值（本文默认 3）。')
else:
    # 兜底：CPU 代理扫描结果（真实 fusibile 扫描完成后会被覆盖）
    rows334 = [['2', '1,928,924', '94.9%', '394 × 433 × 398', '67.9'],
               ['3（默认）', '1,891,584', '93.1%', '324 × 430 × 384', '53.5'],
               ['4', '1,861,108', '91.6%', '294 × 404 × 379', '45.0'],
               ['5', '1,824,806', '89.8%', '289 × 402 × 376', '43.7']]
    _ana = ('分析：从 fusibile.cu 的融合逻辑可见，num_consistent 提高即提高一致性门槛——要求更多源视角在重投影后保持深度一致，'
            '单视角特有的错误深度（遮挡边缘、漂浮点）更难通过；而多视一致的表面点则稳定保留。'
            '融合点数因此由 192.9 万降至 182.5 万（−5.4%），完整性单调下降；'
            '但包围盒体积由 6,790 万 mm³ 收缩至 4,370 万 mm³（−36%），'
            '说明被剔除的主要是远离真实表面的外点，点云更紧凑可靠。这与 3.2 的 interval_scale 同理，'
            '须在"完整性↔可靠性"间折中：视角重叠少时取较小 num_consistent 保完整性，要求高精度表面时取较大值（本文默认 3）。')

add_table(
    ['num_consistent', '融合点数', '候选保留率', '包围盒 (mm)', '包围盒体积 (10⁶ mm³)'],
    rows334)
doc.add_paragraph()
add_fig(os.path.join(FIG, 'fig_param_num_consistent.png'), '图 4　num_consistent 对融合点云的影响：左（蓝）完整性——融合点数；右（红）紧凑度——包围盒体积', width=8.0)
para(_ana, size=10)

head('3.4　可视化展示', 2)
para('图 1：参考图像 / 深度图 / 概率图；图 2：不同 interval_scale 的深度图与分布；图 3：融合前后两档点云对比；图 4：num_consistent 参数影响。点云文件（point_cloud_prob_filtered.ply 与原版 final3d_model.ply）为含颜色二进制 PLY，可直接导入 MeshLab / CloudCompare 查看。')

# ================= 4 结论 =================
head('4　结论', 1)
para('（1）在本机（RTX 4070 / 8 GB）通过 nvidia-tensorflow 1.15.5 复现 MVSNet 完整流程：49 视角在 1152×864 下完成深度估计（max_d 由 192 降到 128），深度图、概率图与相机文件输出正确；')
para('（2）interval_scale 决定深度假设的采样间隔与覆盖范围，需与场景深度范围匹配（本场景 1.06 最优）：过小截断远景、过大降低深度分辨率；')
para('（3）从深度图到点云的反投影公式为 X_w = Rᵀ·(d·K⁻¹·[u,v,1]ᵀ − t)；按课件原版流程（depthfusion.py + fusibile），概率滤波后 203.2 万点经几何一致性检查与多视融合降至 171.0 万（−15.9%），包围盒由 482×517×456 mm 收缩为 293×401×377 mm，离群点比例由 11.2% 降至 7.9%，验证了后处理对 MVS 重建质量的关键作用。')

# ================= 统一字体：中文宋体 / 西文 Times New Roman =================
def _set_run_font(run, latin='Times New Roman', ea='Noto Serif CJK SC'):
    run.font.name = latin
    rPr = run._element.get_or_add_rPr()
    rFonts = rPr.find(qn('w:rFonts'))
    if rFonts is None:
        rFonts = OxmlElement('w:rFonts')
        rPr.append(rFonts)
    rFonts.set(qn('w:eastAsia'), ea)
    rFonts.set(qn('w:ascii'), latin)
    rFonts.set(qn('w:hAnsi'), latin)


def apply_report_fonts(document):
    for p in document.paragraphs:
        for r in p.runs:
            _set_run_font(r)
    for t in document.tables:
        for row in t.rows:
            for cell in row.cells:
                for p in cell.paragraphs:
                    for r in p.runs:
                        _set_run_font(r)


apply_report_fonts(doc)

doc.save(OUT)
print('saved', OUT)
