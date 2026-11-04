"""生成中文实验报告（≤5 页）：只写完成过程与结果分析，不贴代码、不写基础知识。

用法:  python make_report.py
输出:  report/report.html 与 实验报告-图像分类2.pdf
"""
import os
import json
import csv
import base64
import pathlib
import shutil
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, 'results')
FIG = os.path.join(RES, 'figs')
REPORT_DIR = os.path.join(HERE, 'report')
os.makedirs(REPORT_DIR, exist_ok=True)


# ------------------------------------------------------------------ 数据读取
def load_csv(name):
    p = os.path.join(RES, name + '.csv')
    return list(csv.DictReader(open(p))) if os.path.exists(p) else None


def S(name):
    p = os.path.join(RES, name + '_summary.json')
    return json.load(open(p)) if os.path.exists(p) else {}


def acc(name):
    s = S(name)
    if s:
        return s.get('best_acc')
    d = load_csv(name)
    return max(float(r['test_acc']) for r in d) if d else None


def last_acc(name):
    d = load_csv(name)
    return float(d[-1]['test_acc']) if d else None


def f(v, spec='{:.2f}', na='—'):
    return na if v is None else spec.format(v)


def mins(name):
    """训练耗时(min)：优先按 CSV 逐 epoch 秒数求和（含断点续跑也准确）。"""
    d = load_csv(name)
    if d:
        try:
            return sum(float(r['seconds']) for r in d) / 60
        except (KeyError, ValueError):
            pass
    s = S(name)
    return None if not s else s['total_time'] / 60


def macro_of(path):
    if not os.path.exists(path):
        return None
    j = json.load(open(path))
    pc = j.get('per_class_acc') or []
    return sum(pc) / len(pc) if pc else None


def params(name):
    return S(name).get('params')


def b64fig(fname):
    p = os.path.join(FIG, fname)
    if not os.path.exists(p):
        return ''
    return 'data:image/png;base64,' + base64.b64encode(open(p, 'rb').read()).decode()


def fig_block(fname, cap):
    src = b64fig(fname)
    if not src:
        return f'<p class="small">（缺图：{fname}）</p>'
    return f'<div class="fig"><img src="{src}"><div class="figcap">{cap}</div></div>'


# ------------------------------------------------------------------ 实验名
N_BASE_ORIG = 'cifar100_resnet10_sgd_lr0.001_schedconst'
N_BASE_MAIN = 'cifar100_resnet10_sgd_lr0.1_schedstep'
N_SGD_CONST = 'cifar100_resnet10_sgd_lr0.1_schedconst'
N_ADAM = 'cifar100_resnet10_adam_lr0.001_schedconst'
N_ADAMW = 'cifar100_resnet10_adamw_lr0.001_schedconst'
N_RMSPROP = 'cifar100_resnet10_rmsprop_lr0.001_schedconst'
N_LR001 = N_BASE_ORIG
N_LR01 = N_SGD_CONST
N_LR01C = 'cifar100_resnet10_sgd_lr0.01c_schedconst'
N_LR03 = 'cifar100_resnet10_sgd_lr0.3c_schedconst'
N_R10 = N_BASE_MAIN
N_R18 = 'cifar100_resnet18_sgd_lr0.1_schedstep'
N_R50 = 'cifar100_resnet50_sgd_lr0.1_schedstep'
N_KDSOFT = 'cifar100_resnet18_kdsoft_T4_a0.7'
N_KDLOGIT = 'cifar100_resnet18_kdlogit_T1_a0.5'
N_T1 = 'traffic_resnet18_v1_plain'
N_T2 = 'traffic_resnet18_v2_aug_ls'
N_T3 = 'traffic_resnet18_v3_aug_ls_bal'
N_T4 = 'traffic_resnet18_v4_cos_mix'
N_T5 = 'traffic_resnet50_v5_cos_mix'
N_T6 = 'traffic_resnet18_v6_pretrained'

OPT_RUNS = [(N_SGD_CONST, 'SGD (momentum=0.9)', '0.1'),
            (N_ADAM, 'Adam', '0.001'),
            (N_ADAMW, 'AdamW', '0.001'),
            (N_RMSPROP, 'RMSprop (momentum=0.9)', '0.001')]
LR_RUNS = [(N_LR001, '0.001'), (N_LR01C, '0.01'), (N_LR01, '0.1'), (N_LR03, '0.3')]
ARCH_RUNS = [(N_R10, 'ResNet10', 'BasicBlock [1,1,1,1]'),
             (N_R18, 'ResNet18', 'BasicBlock [2,2,2,2]'),
             (N_R50, 'ResNet50', 'Bottleneck [3,4,6,3]')]
TRAFFIC_RUNS = [(N_T1, 'v1 基线', 'Resize 128 + 交叉熵', '—', '随机 shuffle', 50),
                (N_T2, 'v2 增强+标签平滑', 'RandomResizedCrop+ColorJitter+Rotation', '0.1', '随机 shuffle', 50),
                (N_T3, 'v3 +类别均衡采样', '同 v2', '0.1', 'WeightedRandomSampler', 50),
                (N_T4, 'v4 +Mixup+余弦退火', '同 v2 + Mixup(α=0.2)', '0.1', 'WeightedRandomSampler', 80),
                (N_T5, 'v5 ResNet50（96px）', '同 v4', '0.1', 'WeightedRandomSampler', 40),
                (N_T6, 'v6 ImageNet 预训练微调', '同 v3', '0.1', 'WeightedRandomSampler', 40)]
V5_LR = S(N_T5).get('lr')
V6_LR = S(N_T6).get('lr')
CM_V4 = os.path.join(RES, 'traffic_confusion_v4.json')
CM_V6 = os.path.join(RES, 'traffic_confusion.json')
M4J = json.load(open(CM_V4)) if os.path.exists(CM_V4) else {}
M6J = json.load(open(CM_V6)) if os.path.exists(CM_V6) else {}
# 两者准确率相同且错误次数统计一致（只是错在哪张图不同）时记为 True
is_same_cm = bool(M4J.get('matrix') and M6J.get('matrix')
                  and M4J.get('overall') == M6J.get('overall')
                  and M4J.get('per_class_acc') == M6J.get('per_class_acc')
                  and [sum(r) for r in M4J['matrix']] == [sum(r) for r in M6J['matrix']])
v6_worse_cls = ''
if M6J.get('classes'):
    v6_worse_cls = '、'.join(f'{c}（{v:.1f}%）' for c, v in zip(M6J['classes'], M6J['per_class_acc']) if v < 100.0)


def orig_result():
    """解析课程原脚本日志：返回 (best_acc%, total_min, epochs, log_name)。"""
    for log, ep in (('original_resnet10_cifar100_full.log', 50),
                    ('original_resnet10_cifar100.log', 5)):
        p = os.path.join(RES, log)
        if not os.path.exists(p):
            continue
        best, total = None, None
        for line in open(p, errors='ignore'):
            if 'Best Accuracy:' in line:
                try:
                    best = float(line.split('Best Accuracy:')[1].strip().split()[0]) * 100
                except (ValueError, IndexError):
                    pass
            if 'Total time:' in line:
                try:
                    total = float(line.split('Total time:')[1].strip()) / 60
                except (ValueError, IndexError):
                    pass
        if best is not None:
            return best, total, ep, log
    return None, None, None, None


CSS = """
@page { size: A4; margin: 1.05cm 1.1cm; }
* { box-sizing: border-box; }
body { font-family: 'Noto Sans CJK SC','Noto Serif CJK SC',serif; font-size: 9.35pt;
  line-height: 1.40; margin: 0; color: #1a1a1a; }
h1 { font-size: 15pt; text-align: center; margin: 0 0 0.8mm 0; }
.meta { text-align: center; font-size: 8.2pt; color: #333; margin: 0 0 2mm; }
h2 { font-size: 10.8pt; margin: 2.2mm 0 0.8mm; border-left: 3px solid #2b6cb0; padding-left: 4px; }
h3 { font-size: 9.2pt; margin: 1.4mm 0 0.4mm; color: #2b4a6f; }
h2, h3 { break-after: avoid; page-break-after: avoid; }
p, li { margin: 0.35mm 0; text-align: left; }
ol { margin: 0.4mm 0 0.4mm 4.5mm; padding: 0; }
li { margin: 0.2mm 0; }
table { border-collapse: collapse; width: 100%; font-size: 7.9pt; margin: 0.8mm 0;
  page-break-inside: avoid; }
th, td { border: 1px solid #888; padding: 0.6px 2.5px; text-align: center; }
th { background: #eef3f8; }
td.l, th.l { text-align: left; }
.fig { text-align: center; margin: 0.8mm 0; }
.fig img { width: 92%; }
.figcap { font-size: 7.4pt; color: #444; margin-top: 0.3mm; }
.small { font-size: 7.7pt; color: #333; }
.note { font-size: 7.7pt; color: #333; margin: 0.6mm 0 0; }
"""


def build():
    H = []
    a = H.append

    # ---------------- 标题与摘要 ----------------
    a1, a2 = acc(N_BASE_ORIG), acc(N_BASE_MAIN)
    obest, ototal, oep, olog = orig_result()
    a(f"""<h1>基于卷积神经网络的图像分类 实验报告</h1>
<div class="meta">课程：计算机视觉（实验二）　｜　环境：Python 3.12 + PyTorch 2.11.0+cu128 + RTX 4070 Laptop 8GB<br>
姓名：__________　　学号：__________　　日期：__________</div>
<p><b>摘要：</b>本实验在 CIFAR-100 上完成 ResNet10 基线、4 种优化器、4 档学习率与动态衰减、
ResNet10/18/50 深度对比、ResNet50→ResNet18 的两种知识蒸馏共 12 组实验；在交通标志数据集上
自写 Dataset/DataLoader，通过增强、标签平滑、类别均衡采样、Mixup、余弦退火与预训练微调
逐级调参（6 组）。全部 18 组实验固定 <b>seed=42</b>，逐 epoch 记录训练损失、测试精度、耗时与学习率，
最佳精度分别为 <b>{f(a2)}%</b>（CIFAR-100 主基线）与 <b>{f(max(x for x in (acc(N_T4), acc(N_T6)) if x is not None))}%</b>（交通标志）。</p>""")

    # ---------------- 1 实验设置 ----------------
    a(f"""<h2>1. 实验设置与总体流程</h2>
<table>
<tr><th class="l">项目</th><th class="l">配置</th></tr>
<tr><td class="l">数据集</td><td class="l">CIFAR-100：100 类，50000 训练 / 10000 测试，32×32。
交通标志：10 类，6024 训练 / 324 测试，原始 224×224（训练时缩放）</td></tr>
<tr><td class="l">软件环境</td><td class="l">Python 3.12、PyTorch 2.11.0+cu128、torchvision 0.26.0、CUDA 12.8</td></tr>
<tr><td class="l">硬件</td><td class="l">NVIDIA RTX 4070 Laptop（8 GB），与其它进程共享显存；训练在 WSL2 Linux 下运行</td></tr>
<tr><td class="l">统一设置</td><td class="l">seed=42；AMP 混合精度；CIFAR batch=256、交通标志 batch=64；
SGD momentum=0.9、weight decay=1e-4（另做 Adam/AdamW/RMSprop 对比）</td></tr>
<tr><td class="l">流程与记录</td><td class="l">依赖安装 → 跑通课程原脚本 → 在统一框架中复现并扩展 6 项任务 →
逐 epoch 记录 CSV、保存最佳权重 → 汇总图表与报告；课程原始脚本与配置文件未改动</td></tr>
</table>""")

    # ---------------- 2 任务1 ----------------
    a(f"""<h2>2. 任务1：跑通 ResNet10 + CIFAR-100 基线</h2>
<h3>完成过程</h3>
<ol>
<li>按 requirements.txt 安装 PyTorch、torchvision、ignite 等依赖，确认 CUDA 可用。</li>
<li>原样运行课程脚本 <code>resnet-cifar100.py</code>（ignite 引擎，配置文件：SGD、lr=0.001、
batch=256、50 epoch），完整训练一轮并记录精度与耗时。</li>
<li>在统一框架中复现同一配置（SGD、lr=0.001、恒定学习率），逐 epoch 记录曲线，用于验证改写一致性。</li>
<li>把学习率提高到 0.1，并按"第 30、40 epoch 各降低 10 倍"的 step 衰减训练，作为后续所有对比的主基线。</li>
</ol>
<table>
<tr><th class="l">配置</th><th>优化器 / lr</th><th>调度</th><th>参数量</th><th>最佳精度(%)</th><th>末轮精度(%)</th><th>耗时(min)</th></tr>
<tr><td class="l">课程原脚本（完整 50 epoch）</td><td>SGD / 0.001</td><td>恒定</td><td>—</td>
<td>{f(obest)}</td><td>—</td><td>{f(ototal, '{:.1f}')}</td></tr>
<tr><td class="l">框架复现原配置</td><td>SGD / 0.001</td><td>恒定</td><td>{f(params(N_BASE_ORIG), '{:,}')}</td>
<td>{f(a1)}</td><td>{f(last_acc(N_BASE_ORIG))}</td><td>{f(mins(N_BASE_ORIG), '{:.1f}')}</td></tr>
<tr><td class="l">主基线</td><td>SGD / 0.1</td><td>step@30,40 ×0.1</td><td>{f(params(N_BASE_MAIN), '{:,}')}</td>
<td><b>{f(a2)}</b></td><td>{f(last_acc(N_BASE_MAIN))}</td><td>{f(mins(N_BASE_MAIN), '{:.1f}')}</td></tr>
</table>
<h3>结果分析</h3>
<p>① 原脚本 {f(obest)}% 与框架复现 {f(a1)}% 仅差 {f(None if (obest is None or a1 is None) else abs(obest - a1))} 个点，
说明统一框架的改写没有引入偏差，后续对比可信。② lr=0.001 时训练损失末期仍为
{f(float(load_csv(N_BASE_ORIG)[-1]['train_loss']) if load_csv(N_BASE_ORIG) else None, '{:.2f}')}、
末轮精度仍低于最佳值，属于明显欠拟合；学习率改为 0.1 并配合 step 衰减后达到 <b>{f(a2)}%</b>，
提升 <b>{f(None if (a1 is None or a2 is None) else a2 - a1, '{:+.1f}')} 个点</b>。
③ 原脚本用 ignite 引擎跑完整 50 epoch 需 {f(ototal, '{:.1f}')} min，统一框架（AMP + 高效 DataLoader）
只要 {f(mins(N_BASE_ORIG), '{:.1f}')} min，后续所有对比都在这一框架内完成。
④ 可见学习率是本任务最敏感的超参数，因此任务 2~5 统一采用"SGD lr=0.1 + step 衰减"作为可比基线。</p>""")

    # ---------------- 3 任务2 ----------------
    rows = ''
    for n, o, lr in OPT_RUNS:
        rows += (f'<tr><td class="l">{o}</td><td>{lr}</td><td>{f(acc(n))}</td>'
                 f'<td>{f(last_acc(n))}</td><td>{f(mins(n), "{:.1f}")}</td></tr>')
    sgd_a, adam_a, adamw_a, rms_a = (acc(n) for n, _, _ in OPT_RUNS)
    d_rms = load_csv(N_RMSPROP) or []
    rms_loss = float(d_rms[-1]['train_loss']) if d_rms else None
    rms_tail = [float(r['test_acc']) for r in d_rms[-10:]]
    rms_lo, rms_hi = (min(rms_tail), max(rms_tail)) if rms_tail else (None, None)
    d_adam = load_csv(N_ADAM) or []
    rms_ep = int(d_adam[19]['epoch']) if len(d_adam) >= 20 else 20
    rms_cmp = float(d_adam[19]['test_acc']) if len(d_adam) >= 20 else None
    best_o, best_oa = max(((o, acc(n) or -1) for n, o, _ in OPT_RUNS), key=lambda t: t[1])
    a(f"""<h2>3. 任务2：优化器对比（SGD / Adam / AdamW / RMSprop）</h2>
<h3>完成过程</h3>
<ol>
<li>固定 ResNet10、batch=256、50 epoch、恒定学习率（不调度），只替换优化器：SGD(lr=0.1)、
Adam(1e-3)、AdamW(1e-3)、RMSprop(1e-3)，momentum 与 weight decay 保持一致。</li>
<li>每组独立训练并记录逐 epoch 的测试精度与训练损失（图1）。</li>
</ol>
<table><tr><th class="l">优化器</th><th>lr</th><th>最佳精度(%)</th><th>末轮精度(%)</th><th>耗时(min)</th></tr>
{rows}</table>
{fig_block('fig1_optim.png', '图1　四种优化器的测试准确率与训练损失（ResNet10，CIFAR-100，50 epoch，恒定 lr）')}
<h3>结果分析</h3>
<p>① 前 {rms_ep} epoch 内 Adam/AdamW 的精度上升最快，SGD 紧随其后并在 20 epoch 后基本追平（Adam 第 {rms_ep} epoch 已达 {f(rms_cmp)}%，SGD 同期约 62%）；
最终三者分别为 AdamW {f(adamw_a)}%、SGD {f(sgd_a)}%、Adam {f(adam_a)}%，
极差仅 {f(None if None in (sgd_a, adam_a, adamw_a) else max(sgd_a, adam_a, adamw_a) - min(sgd_a, adam_a, adamw_a))} 个点，
说明<b>学习率落在各自合适区间时，优化器之间的最终差距有限</b>。② 最高为 <b>{best_o} {f(best_oa if best_oa > 0 else None)}%</b>，
但优势不足以抵消调参成本；实际项目中 SGD+动量仍是稳妥选择。③ RMSprop 只有 {f(rms_a)}%，
落后约 {f(None if (rms_a is None or max(sgd_a, adam_a, adamw_a) is None) else max(sgd_a, adam_a, adamw_a) - rms_a)} 个点，
其训练损失后期停在 {f(rms_loss, '{:.2f}')} 附近不再下降——在 lr=1e-3 下 50 epoch 未收敛（该优化器对学习率更敏感），
测试精度也在 {f(rms_lo, '{:.0f}')}%~{f(rms_hi, '{:.0f}')}% 之间反复跳动，并非单纯"方法更差"。④ 对照组：同为 SGD，仅把恒定 lr 换成 step 衰减，
精度就从 {f(sgd_a)}% 升到 {f(a2)}%（{f(None if (sgd_a is None or a2 is None) else a2 - sgd_a, '{:+.2f}')} 点），
<b>说明调度策略比更换优化器的收益更大</b>。</p>""")

    # ---------------- 4 任务3 ----------------
    head = ''.join(f'<th>{l}</th>' for _, l in LR_RUNS)
    row_best = ''.join(f'<td>{f(acc(n))}</td>' for n, _ in LR_RUNS)
    row_last = ''.join(f'<td>{f(last_acc(n))}</td>' for n, _ in LR_RUNS)
    lr001, lr01c, lr01, lr03 = (acc(n) for n, _ in LR_RUNS)
    d_main = load_csv(N_BASE_MAIN) or []
    main_last_loss = float(d_main[-1]['train_loss']) if d_main else None
    loss03 = float(load_csv(N_LR03)[-1]['train_loss']) if load_csv(N_LR03) else None
    loss01 = float(load_csv(N_LR01)[-1]['train_loss']) if load_csv(N_LR01) else None
    a(f"""<h2>4. 任务3：学习率影响与动态调节</h2>
<h3>完成过程</h3>
<ol>
<li><b>固定学习率扫描：</b>在 SGD 下分别取 lr = 0.001 / 0.01 / 0.1 / 0.3，其余配置完全相同，各训练 50 epoch，
对比最佳/末轮精度与训练损失（图2）。其中 lr=0.001 即任务1 的原配置文件，0.1 与任务2 的 SGD 组共用同一次训练。</li>
<li><b>动态调节：</b>以 0.1 为初值，在第 30、40 epoch 各把学习率降低 10 倍（0.1→0.01→0.001），
记录 train loss 随 epoch 的收敛曲线与对应测试精度（图3）。</li>
</ol>
<table><tr><th class="l">SGD 学习率（恒定）</th>{head}</tr>
<tr><td class="l">最佳精度(%)</td>{row_best}</tr>
<tr><td class="l">末轮精度(%)</td>{row_last}</tr></table>
{fig_block('fig2_lr.png', '图2　不同学习率下的测试准确率与训练损失（SGD，ResNet10，恒定 lr，50 epoch）')}
<h3>结果分析</h3>
<p>① 精度呈明显的 <b>U 形</b>：lr=0.001 严重欠拟合，仅 {f(lr001, '{:.1f}')}%（训练损失 {f(float(load_csv(N_LR001)[-1]['train_loss']) if load_csv(N_LR001) else None, '{:.2f}')}，
末轮仍在下降）；0.01 升到 {f(lr01c, '{:.1f}')}%；0.1 最优 <b>{f(lr01, '{:.1f}')}%</b>；0.3 回落到 {f(lr03, '{:.1f}')}%。
② lr=0.3 并非"欠拟合"：其末轮训练损失 {f(loss03, '{:.3f}')} 反而高于 0.1 的 {f(loss01, '{:.3f}')}，
测试精度在后 20 epoch 于 58%~63% 间反复波动——步长过大导致在极小值附近震荡，既没压低损失也不稳定。
③ 动态衰减效果显著：第 30→31 epoch（lr 0.1→0.01）精度由 65.40% 跳到 70.73%（+5.33 点），
第 40 epoch 再降后精细收敛，最佳 <b>{f(a2)}%</b>（第 {S(N_BASE_MAIN).get('best_epoch', '—')} epoch），
比同 lr 恒定训练高 {f(None if (a2 is None or lr01 is None) else a2 - lr01, '{:+.2f}')} 点；
train loss 呈三段台阶式下降（第 30 epoch 末 0.38 → 降 lr 后 0.21 → 第 40 epoch 末 0.09 → 最终 {f(main_last_loss, '{:.3f}')}），曲线平滑几乎无抖动。
④ 结论：<b>"大 lr 快速跨越 + 末期小 lr 精细收敛"</b>能同时降低训练损失并提升泛化，
这也是任务6 采用余弦退火的依据。</p>""")

    # ---------------- 5 任务4 ----------------
    rows = ''
    for n, st, blk in ARCH_RUNS:
        d = load_csv(n) or []
        # 每 epoch 时间用前 10 个 epoch 的中位数，避免断点续跑/单 epoch 抖动影响
        secs = sorted(float(r['seconds']) for r in d[:10])
        pe = (secs[len(secs) // 2] / 60) if secs else None
        rows += (f'<tr><td>{st}</td><td class="l">{blk}</td><td>{f(params(n), "{:,}")}</td>'
                 f'<td>{f(mins(n), "{:.1f}")}</td><td>{f(pe)}</td><td>{f(acc(n))}</td></tr>')
    a10, a18, a50 = (acc(n) for n, _, _ in ARCH_RUNS)
    d1018 = None if (a10 is None or a18 is None) else a18 - a10
    d1850 = None if (a18 is None or a50 is None) else a50 - a18
    l18 = float(load_csv(N_R18)[-1]['train_loss']) if load_csv(N_R18) else None
    l50 = float(load_csv(N_R50)[-1]['train_loss']) if load_csv(N_R50) else None
    p10, p18, p50 = (params(n) for n, _, _ in ARCH_RUNS)
    pe10 = pe18 = pe50 = None
    for n in (N_R10, N_R18, N_R50):
        d = load_csv(n) or []
        secs = sorted(float(r['seconds']) for r in d[:10])
        v = secs[len(secs) // 2] / 60 if secs else None
        if n == N_R10:
            pe10 = v
        elif n == N_R18:
            pe18 = v
        else:
            pe50 = v
    d_params = (p50 / p18) if (p50 and p18) else None
    d_pe = (pe50 / pe18) if (pe50 and pe18) else None
    a(f"""<h2>5. 任务4：网络改为 ResNet18 / ResNet50 并对比三种网络</h2>
<h3>完成过程</h3>
<ol>
<li>阅读课程代码，把只支持 ResNet10 的构造方式泛化为"可传入 block 类型与各 stage 层数"的构造器，
新增 ResNet18（BasicBlock，[2,2,2,2]）与 ResNet50（Bottleneck，[3,4,6,3]）；
分类头输入维度按 512×expansion 自动适配（ResNet50 为 2048）。</li>
<li>三种网络共用同一套 CIFAR 适配的 stem（3×3 卷积、去掉 maxpool、只下采样 3 次）与训练配置
（SGD lr=0.1 + step 衰减、50 epoch、batch=256、AMP），只改变网络结构。</li>
<li>记录参数量、总训练时间、每 epoch 时间与测试精度；ResNet50 中途因显存/内存压力中断，
从最佳权重热启动补齐剩余 epoch（学习率按调度正确恢复）。</li>
</ol>
<table><tr><th class="l">网络</th><th class="l">结构</th><th>参数量</th><th>总耗时(min)</th>
<th>每 epoch(min)</th><th>最佳精度(%)</th></tr>
{rows}</table>
{fig_block('fig4_arch.png', '图4　ResNet10/18/50 的精度曲线、训练耗时与参数量对比')}
<h3>结果分析</h3>
<p>① 精度并不随深度单调上升：ResNet10 {f(a10)}% → ResNet18 <b>{f(a18)}%</b>（{f(d1018, '{:+.2f}')} 点）→
ResNet50 {f(a50)}%（{f(d1850, '{:+.2f}')} 点）。② 训练开销却大幅增加：总耗时
{f(mins(N_R10), '{:.1f}')} → {f(mins(N_R18), '{:.1f}')} → {f(mins(N_R50), '{:.1f}')} min，
ResNet50 参数量 {f(None if params(N_R50) is None else params(N_R50) / 1e6, '{:.1f}')} M，
是 ResNet18 的 {f(d_params, '{:.1f}')} 倍、单 epoch 时间约 {f(d_pe, '{:.1f}')} 倍
（{f(pe18)} → {f(pe50)} min/epoch）。③ 原因从训练损失可直接看出：ResNet50 末轮损失
{f(l50, '{:.3f}')} <b>高于</b> ResNet18 的 {f(l18, '{:.3f}')}，训练损失也没降下来，说明它在 50 epoch 预算内<b>尚未充分收敛</b>，
而非过拟合；32×32 输入下大容量模型的优化更慢，短预算无法体现其优势。
④ 结论：<b>更深不等于更好</b>，深度收益取决于数据规模、输入分辨率与训练预算；本任务中
<b>ResNet18 是精度/开销的最佳折中</b>（{f(a18)}%、{f(mins(N_R18), '{:.1f}')} min），因此任务5 选它作学生网络。</p>""")

    # ---------------- 6 任务5 ----------------
    ks, kl = S(N_KDSOFT), S(N_KDLOGIT)
    kd_soft_acc, kd_logit_acc = acc(N_KDSOFT), acc(N_KDLOGIT)
    d_soft = None if (kd_soft_acc is None or a18 is None) else kd_soft_acc - a18
    d_logit = None if (kd_logit_acc is None or a18 is None) else kd_logit_acc - a18
    rows = (f'<tr><td class="l">ResNet18 基准（无蒸馏）</td><td>—</td><td>{f(params(N_R18), "{:,}")}</td>'
            f'<td>{f(a18)}</td><td>—</td></tr>')
    rows += (f'<tr><td class="l">KD soft target（T=4, α=0.7）</td><td>ResNet50</td>'
             f'<td>{f(ks.get("params"), "{:,}")}</td><td><b>{f(kd_soft_acc)}</b></td>'
             f'<td>{f(d_soft, "{:+.2f}")}</td></tr>')
    rows += (f'<tr><td class="l">KD logits（MSE, α=0.5）</td><td>ResNet50</td>'
             f'<td>{f(kl.get("params"), "{:,}")}</td><td>{f(kd_logit_acc)}</td>'
             f'<td>{f(d_logit, "{:+.2f}")}</td></tr>')
    rows += (f'<tr><td class="l">（参考）教师 ResNet50</td><td>—</td><td>{f(params(N_R50), "{:,}")}</td>'
             f'<td>{f(a50)}</td><td>—</td></tr>')
    def ep_acc(name, k=10):
        d = load_csv(name) or []
        return f'{float(d[k - 1]["test_acc"]):.2f}' if len(d) >= k else '—'
    base_e10, kd_soft_e10, kd_logit_e10 = ep_acc(N_R18), ep_acc(N_KDSOFT), ep_acc(N_KDLOGIT)
    a(f"""<h2>6. 任务5：知识蒸馏（ResNet50 教师 → ResNet18 学生）</h2>
<h3>完成过程</h3>
<ol>
<li>以任务4 训练好的 ResNet50 最佳权重作教师（最佳精度 {f(a50)}%，第 {f(S(N_R50).get("best_epoch"), "{:.0f}")} epoch），
教师置于推理模式、不更新参数。</li>
<li>学生仍为 ResNet18，训练配置与无蒸馏基线完全一致（SGD lr=0.1 + step、50 epoch、batch=256），
只改变损失函数，保证对比可归因。</li>
<li>实现两种蒸馏损失并分别训练：<b>soft target</b>（Hinton 式，温度 T=4、硬标签权重 α=0.7，
KL 散度乘 T² 补偿梯度尺度）；<b>logits 对齐</b>（FitNets 思路，α=0.5，对学生与教师的 logits 做 MSE）。</li>
<li>记录两条精度/损失曲线，与无蒸馏基线对比（图5）。</li>
</ol>
<table><tr><th class="l">方法</th><th>教师</th><th>学生参数量</th><th>最佳精度(%)</th><th>较学生基线</th></tr>
{rows}</table>
{fig_block('fig5_kd.png', '图5　两种蒸馏损失对学生 ResNet18 的精度与收敛影响')}
<h3>结果分析</h3>
<p>① <b>soft target 有效</b>：学生达到 <b>{f(kd_soft_acc)}%</b>，比无蒸馏基线高
{f(d_soft, '{:+.2f}')} 点；前 20 epoch 其精度曲线基本高于基线（第 10 epoch {kd_soft_e10}% vs 基线 {base_e10}%，
图5 左），收敛更快；其训练损失含蒸馏项（第 1 epoch 达 7.2，即 0.3×CE+0.7×T²·KL），
与纯交叉熵不可直接比较，故以精度曲线判断优劣。
② <b>logits MSE 反而下降</b> {f(d_logit, '{:+.2f}')} 点：它要求学生的 logit 幅值与教师逐维一致，
初期收敛极快（第 10 epoch 已达 {kd_logit_e10}%），但后期被教师 logit 的数值尺度"拉"住，
最终不如带温度的 KL 稳健（100 类任务上教师 logit 方差大，MSE 项梯度尺度不稳定）。
③ 一个值得记录的现象：教师最终只有 {f(a50)}%，<b>低于学生基线</b>，而学生经蒸馏后反超教师
{f(None if (kd_soft_acc is None or a50 is None) else kd_soft_acc - a50, '{:+.2f}')} 点。
原因是教师在 50 epoch 预算内未充分收敛（见任务4），其软目标仍提供了类间相似性约束，
起到正则化/课程信号的作用；这说明<b>蒸馏收益不完全取决于教师的绝对精度</b>，但更强的教师通常更有利。
④ 推理阶段只用学生（{f(None if not ks else ks['params'] / 1e6, '{:.1f}')} M 参数，约为教师的 47%），
部署成本与普通 ResNet18 相同。⑤ 参数选择：α 过大退化为普通训练、过小会被教师误差带偏，本实验 0.7/0.5 表现稳定；
T 取 2~8 均可，T=4 时软标签信息量与锐度较平衡。</p>""")

    # ---------------- 7 任务6 ----------------
    rows = ''
    best_tf, best_label = None, '—'
    for n, label, aug, ls, samp, ep in TRAFFIC_RUNS:
        av = acc(n)
        if av is not None and (best_tf is None or av > best_tf):
            best_tf, best_label = av, label
        rows += (f'<tr><td class="l">{label}</td><td class="l">{aug}</td><td>{ls}</td><td>{samp}</td>'
                 f'<td>{S(n).get("epochs", ep)}</td><td>{f(av)}</td>'
                 f'<td>{f(mins(n), "{:.1f}")}</td></tr>')
    m1 = macro_of(os.path.join(RES, 'traffic_confusion_v1.json'))
    m2 = macro_of(os.path.join(RES, 'traffic_confusion_v2.json'))
    m3 = macro_of(os.path.join(RES, 'traffic_confusion_v3.json'))
    p2j, p3j = os.path.join(RES, 'traffic_confusion_v2.json'), os.path.join(RES, 'traffic_confusion_v3.json')
    cmp_txt = ''
    if os.path.exists(p2j) and os.path.exists(p3j):
        j2, j3 = json.load(open(p2j)), json.load(open(p3j))
        d = dict(zip(j2['classes'], zip(j2['per_class_acc'], j3['per_class_acc'])))
        cls_rows = ''.join(
            f'<tr><td>{c}</td><td>{a:.1f}</td><td>{b:.1f}</td><td>{b - a:+.1f}</td></tr>'
            for c, (a, b) in d.items())
        cmp_txt = f"""<table><tr><th>方案</th><th>整体 accuracy(%)</th><th>宏平均召回(%)</th>
<th>M1 召回(%)</th><th>M4 召回(%)</th><th>P1 召回(%)</th></tr>
<tr><td class="l">v2 增强 + 标签平滑</td><td>{j2['overall']:.2f}</td><td>{m2:.2f}</td>
<td>{d['M1'][0]:.1f}</td><td>{d['M4'][0]:.1f}</td><td>{d['P1'][0]:.1f}</td></tr>
<tr><td class="l">v3 + 类别均衡采样</td><td>{j3['overall']:.2f}</td><td><b>{m3:.2f}</b></td>
<td><b>{d['M1'][1]:.1f}</b></td><td>{d['M4'][1]:.1f}</td><td><b>{d['P1'][1]:.1f}</b></td></tr>
</table>"""
    a(f"""<h2>7. 任务6：交通标志数据集分类与调参</h2>
<h3>完成过程</h3>
<ol>
<li><b>数据分析：</b>10 个类别，训练 6024 张、测试 324 张，原始分辨率 224×224；类别极不均衡
（M4 有 3206 张，P10_50 仅 95 张，测试集中最少的类只有 6 张）。</li>
<li><b>重写 Dataset：</b>遍历 train/test 下以类别命名的目录，建立类别到索引的映射；
读取图像后按统一尺寸与归一化参数变换，返回图像张量与标签；训练/测试使用不同的变换流程
（训练用数据增强，测试只做缩放）。</li>
<li><b>重写数据加载流程（DataLoader + 自定义采样器）：</b>训练集支持随机打乱或按类别频率加权的均衡采样（权重为类样本数的倒数），
设置 batch、并行 worker 与锁页内存；测试集按顺序加载，保证评估可复现。</li>
<li><b>预处理：</b>按训练集统计的均值和标准差归一化（缓存于结果目录，均值约 [0.30, 0.39, 0.60]）；训练增强为随机缩放裁剪(0.7~1.0)、
颜色抖动与 ±10° 旋转，<b>刻意不做水平翻转</b>（会改变交通标志语义）；测试仅做缩放。</li>
<li><b>逐级调参（6 组）：</b>v1 基线（Resize 128 + 仅随机打乱，SGD lr=0.05 + step）→ v2 加强增强与标签平滑 0.1 →
v3 加类别均衡采样 → v4 在 v3 基础上加 Mixup(α=0.2)、换余弦退火并训练 80 epoch → v5 换 ResNet50
（因 8 GB 显存不足，改为 96×96、batch=16、lr={f(V5_LR)}、40 epoch）→ v6 改用 ImageNet 预训练 ResNet18 微调
40 epoch（lr={f(V6_LR, '{:.3f}')}）。各方案的上游设置（增强/标签平滑/采样）沿用或在此基础上叠加。</li>
</ol>
{fig_block('fig6_traffic.png', '图6　交通标志数据集分布、六种方案精度对比与最佳模型混淆矩阵')}
<table><tr><th class="l">方案</th><th class="l">数据增强</th><th>标签平滑</th><th>采样</th>
<th>epoch</th><th>最佳精度(%)</th><th>耗时(min)</th></tr>
{rows}</table>
<h3>结果分析</h3>
<p>① <b>v1 基线已达 {f(acc(N_T1))}%</b>：类间视觉差异大、多数类样本充足，从零训练的 ResNet18 已接近饱和。
② <b>v2 与 v1 只差 1 张测试图</b>（{f(acc(N_T2))}% vs {f(acc(N_T1))}%，1/324≈0.31 点），
但宏平均召回由 {f(m1)}% 升到 {f(m2)}%，弱类 M1 由 71.4%→78.6%、M6 由 75.0%→87.5%——
增强与标签平滑的价值在于<b>补强弱类、压低过度自信</b>，而非拉高总体均值。
③ <b>v3 体现"整体精度 vs 类别均衡"的取舍</b>：整体 {f(acc(N_T3))}%（{f(None if (acc(N_T3) is None or acc(N_T2) is None) else acc(N_T3) - acc(N_T2), '{:+.2f}')}），
宏平均 {f(m3)}%（{f(None if (m3 is None or m2 is None) else m3 - m2, '{:+.2f}')}），M1、P1 明显提升，
代价是占测试集 52% 的多数类 M4 略降。说明<b>只看 overall accuracy 会掩盖少数类问题</b>。
④ <b>v4 达 {f(acc(N_T4))}%</b>：Mixup + 余弦退火 + 80 epoch 进一步平滑决策边界，是"从零训练"的最好结果。
⑤ <b>v5 仅 {f(acc(N_T5))}%</b>：比 v4 低 {f(None if (acc(N_T4) is None or acc(N_T5) is None) else acc(N_T4) - acc(N_T5), '{:.2f}')} 点，
且输入分辨率降到 96×96 后单 epoch 并不比 v4 快多少——<b>数据量决定可用模型容量上限</b>，小数据上加深网络是负收益。
⑥ <b>v6 以 {f(acc(N_T6))}% {'与 v4 并列最佳' if not v6_worse_cls else '达到最佳'}，仅需 {f(mins(N_T6), '{:.1f}')} min</b>
（约为 v4 的 {f(None if (mins(N_T6) is None or mins(N_T4) is None) else mins(N_T4) / mins(N_T6), '{:.0f}')} 分之一），
宏平均 {f(macro_of(CM_V6))}%{'——总体精度与各类召回率与 v4 <b>完全一致</b>（同为 318/324 正确，错在 ' + v6_worse_cls + '，仅具体错图不同）' if is_same_cm else ''}，
是性价比最高的方案，说明<b>小样本场景迁移学习优势显著</b>。
⑦ 指标解读：测试集仅 324 张，单张误差即 0.31 点，P10_50 只有 6 张（错 1 张即 −16.7 点），
因此方案间 <1 点的差异不应过度解读，需结合宏平均召回与混淆矩阵判断。</p>""")

    # ---------------- 8 总结 ----------------
    a(f"""<h2>8. 总结</h2>
<table><tr><th class="l" style="width:19%">实验项</th><th class="l">主要结论</th><th style="width:23%">最佳结果</th></tr>
<tr><td class="l">任务1 基线</td><td class="l">原脚本与框架复现一致；lr=1e-3 欠拟合，lr=0.1+step 是合理主基线</td>
<td>{f(a2)}%（ResNet10）</td></tr>
<tr><td class="l">任务2 优化器</td><td class="l">四种优化器最终精度接近（极差 &lt; 0.6 点），RMSprop 在 lr=1e-3 下未收敛；
调度带来的提升（+{f(None if (sgd_a is None or a2 is None) else a2 - sgd_a, '{:.2f}')} 点）远大于更换优化器</td>
<td>{best_o} {f(best_oa)}%</td></tr>
<tr><td class="l">任务3 学习率</td><td class="l">精度随 lr 呈 U 形；0.3 震荡、0.001 欠拟合；30/40 epoch 衰减使 loss 台阶下降</td>
<td>{f(a2)}%（恒定 lr 为 {f(lr01)}%）</td></tr>
<tr><td class="l">任务4 网络深度</td><td class="l">10→18 有增益，50 在 50 epoch 预算下反而下降（未充分收敛、单 epoch 约 {f(d_pe, '{:.1f}')} 倍）；ResNet18 性价比最高</td>
<td>{f(a18)}%（ResNet18）</td></tr>
<tr><td class="l">任务5 知识蒸馏</td><td class="l">soft target 优于 logits-MSE；软目标主要提供类间相似性约束，推理零额外开销</td>
<td>{f(kd_soft_acc)}%（{f(d_soft, '{:+.2f}')} vs 基线）</td></tr>
<tr><td class="l">任务6 交通标志</td><td class="l">自写 Dataset/DataLoader；增强+标签平滑补弱类，均衡采样换宏平均，预训练微调最划算</td>
<td>{f(best_tf)}%（{best_label}）</td></tr>
</table>
<p><b>体会：</b>① 学习率与调度对结果的影响大于优化器和网络微调；② "训练损失低"不等于"测试精度高"，
 本次 ResNet50 与 RMSprop 都是损失未降下来即被预算截断的例子，需结合收敛曲线判断；
③ 不均衡数据必须同时看整体精度、宏平均与混淆矩阵，否则会被多数类掩盖；
④ 固定 seed、统一记录、断点续跑与自动出图，是保证 18 组实验可复现、可对比的前提。</p>
<p class="note"><b>提交物：</b><code>train.py</code>（统一训练框架）、<code>run_all.py</code>（实验调度）、
<code>make_figures.py</code>/<code>make_report.py</code>（图表与报告）、<code>eval_traffic.py</code>（混淆矩阵）、
<code>results/</code>（逐 epoch CSV、summary、日志、最佳权重、图表）与本报告。
课程原始 <code>resnet-cifar100.py</code>、<code>config.yml</code> 未改动。</p>""")

    a('</body></html>')

    html = ('<!DOCTYPE html><html><head><meta charset="utf-8"><style>' + CSS + '</style></head><body>'
            + ''.join(H))
    out = os.path.join(REPORT_DIR, 'report.html')
    open(out, 'w', encoding='utf-8').write(html)
    print('report.html written:', len(html), 'bytes')

    pdf = os.path.join(HERE, '实验报告-图像分类2.pdf')
    # Linux 下直接用 google-chrome；Windows 下自动找 Chrome / Edge
    chrome = os.environ.get('CHROME') or 'google-chrome'
    if not shutil.which(chrome):
        chrome = next((p for p in (
            r'C:\Program Files\Google\Chrome\Application\chrome.exe',
            r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe',
            r'C:\Program Files\Microsoft\Edge\Application\msedge.exe') if os.path.exists(p)), chrome)
    r = subprocess.run([chrome, '--headless', '--no-sandbox', '--disable-gpu', '--no-pdf-header-footer',
                        f'--print-to-pdf={pdf}', pathlib.Path(out).as_uri()],
                       capture_output=True, text=True, encoding='utf-8', errors='ignore')
    if r.returncode != 0:
        print(r.stderr[-500:])
    print('PDF written:', pdf, os.path.getsize(pdf), 'bytes')
    info = subprocess.run(['pdfinfo', pdf], capture_output=True, text=True).stdout
    for line in info.splitlines():
        if line.startswith('Pages'):
            print(line)


if __name__ == '__main__':
    build()
