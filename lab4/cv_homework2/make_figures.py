"""读取 results/*.csv 与 *_summary.json，生成分析图（results/figs/）。"""
import os
import json
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, 'results')
FIG = os.path.join(RES, 'figs')
os.makedirs(FIG, exist_ok=True)

for p in ['/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
          '/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc']:
    if os.path.exists(p):
        fm.fontManager.addfont(p)
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'Noto Serif CJK SC']
plt.rcParams['axes.unicode_minus'] = False
BLUE, ORANGE, GREEN, RED, PURPLE, BROWN = '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'


def load_csv(name):
    path = os.path.join(RES, name + '.csv')
    if not os.path.exists(path):
        return None
    ep, loss, acc, lr, secs = [], [], [], [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            ep.append(int(row['epoch'])); loss.append(float(row['train_loss']))
            acc.append(float(row['test_acc'])); lr.append(float(row['lr']))
            secs.append(float(row['seconds']))
    return {'ep': ep, 'loss': loss, 'acc': acc, 'lr': lr, 'secs': secs}


def load_sum(name):
    path = os.path.join(RES, name + '_summary.json')
    return json.load(open(path)) if os.path.exists(path) else None


def save(fig, fname):
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, fname), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('wrote', fname)


def two_panel(runs, title_l, title_r, fname, ylog=False):
    """左：测试准确率；右：训练损失。runs=[(name,label,color)]"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.8, 2.6))
    got = False
    for name, label, c in runs:
        d = load_csv(name)
        if not d:
            continue
        got = True
        ax1.plot(d['ep'], d['acc'], label=label, color=c, lw=1.3)
        ax2.plot(d['ep'], d['loss'], label=label, color=c, lw=1.3)
    if not got:
        plt.close(fig)
        print('skip (no data)', fname)
        return
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('测试准确率 (%)'); ax1.set_title(title_l, fontsize=9)
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('训练损失'); ax2.set_title(title_r, fontsize=9)
    if ylog:
        ax2.set_yscale('log')
    for ax in (ax1, ax2):
        ax.legend(fontsize=6.5); ax.grid(alpha=0.3)
    save(fig, fname)


def fig_optim():
    two_panel([('cifar100_resnet10_sgd_lr0.1_schedconst', 'SGD(lr=0.1)', BLUE),
               ('cifar100_resnet10_adam_lr0.001_schedconst', 'Adam(lr=1e-3)', ORANGE),
               ('cifar100_resnet10_adamw_lr0.001_schedconst', 'AdamW(lr=1e-3)', GREEN),
               ('cifar100_resnet10_rmsprop_lr0.001_schedconst', 'RMSprop(lr=1e-3)', RED)],
              '测试准确率', '训练损失', 'fig1_optim.png')


def fig_lr():
    two_panel([('cifar100_resnet10_sgd_lr0.001_schedconst', 'lr=0.001', BLUE),
               ('cifar100_resnet10_sgd_lr0.01c_schedconst', 'lr=0.01', ORANGE),
               ('cifar100_resnet10_sgd_lr0.1_schedconst', 'lr=0.1', GREEN),
               ('cifar100_resnet10_sgd_lr0.3c_schedconst', 'lr=0.3', RED)],
              '不同学习率的测试准确率', '不同学习率的训练损失', 'fig2_lr.png')


def fig_decay():
    d = load_csv('cifar100_resnet10_sgd_lr0.1_schedstep')
    if not d:
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.8, 2.6))
    ax1.plot(d['ep'], d['loss'], color=RED, lw=1.4)
    for m in (30, 40):
        ax1.axvline(m, color='gray', ls=':', lw=0.8)
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('训练损失')
    ax1.set_title('动态学习率下的收敛曲线 (Train Loss)', fontsize=9); ax1.grid(alpha=0.3)
    ax2.plot(d['ep'], d['acc'], color=BLUE, lw=1.4, label='测试准确率')
    ax2_ = ax2.twinx()
    ax2_.plot(d['ep'], d['lr'], color=GREEN, lw=1.0, ls='--', label='学习率(右轴)')
    ax2_.set_ylabel('学习率', color=GREEN)
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('测试准确率 (%)')
    ax2.set_title('Val Acc + LR (step@30,40, ×0.1)', fontsize=9); ax2.grid(alpha=0.3)
    save(fig, 'fig3_decay.png')


def fig_arch():
    runs = [('cifar100_resnet10_sgd_lr0.1_schedstep', 'ResNet10', BLUE),
            ('cifar100_resnet18_sgd_lr0.1_schedstep', 'ResNet18', ORANGE),
            ('cifar100_resnet50_sgd_lr0.1_schedstep', 'ResNet50', GREEN)]
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(9.2, 2.5))
    labels, accs, times, params, colors = [], [], [], [], []
    for name, label, c in runs:
        d, s = load_csv(name), load_sum(name)
        if d:
            ax1.plot(d['ep'], d['acc'], label=label, color=c, lw=1.3)
        if s:
            labels.append(label); accs.append(s['best_acc']); colors.append(c)
            # 耗时优先用 CSV 逐 epoch 秒数求和（续跑/中断后更准确）
            times.append(sum(d['secs']) / 60 if d else s['total_time'] / 60)
            params.append(s['params'] / 1e6)
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('测试准确率 (%)')
    ax1.set_title('精度曲线', fontsize=9); ax1.legend(fontsize=7); ax1.grid(alpha=0.3)
    ax2.bar(labels, times, color=colors); ax2.set_ylabel('总训练时间 (min)')
    ax2.set_title('训练耗时', fontsize=9)
    for i, v in enumerate(times):
        ax2.text(i, v, f'{v:.0f}', ha='center', va='bottom', fontsize=7)
    ax3.bar(labels, params, color=colors); ax3.set_ylabel('参数量 (M)')
    ax3.set_title('模型规模', fontsize=9)
    for i, v in enumerate(params):
        ax3.text(i, v, f'{v:.1f}', ha='center', va='bottom', fontsize=7)
    save(fig, 'fig4_arch.png')


def fig_kd():
    two_panel([('cifar100_resnet18_sgd_lr0.1_schedstep', 'ResNet18 基准', BLUE),
               ('cifar100_resnet18_kdsoft_T4_a0.7', 'KD-soft (T=4,α=0.7)', ORANGE),
               ('cifar100_resnet18_kdlogit_T1_a0.5', 'KD-logit (MSE,α=0.5)', GREEN)],
              '知识蒸馏测试准确率', '知识蒸馏训练损失', 'fig5_kd.png')


def fig_traffic():
    runs = [('traffic_resnet18_v1_plain', 'v1 基线', BLUE),
            ('traffic_resnet18_v2_aug_ls', 'v2 增强+LS', ORANGE),
            ('traffic_resnet18_v3_aug_ls_bal', 'v3 +均衡', GREEN),
            ('traffic_resnet18_v4_cos_mix', 'v4 +Mixup/余弦', RED),
            ('traffic_resnet50_v5_cos_mix', 'v5 ResNet50', PURPLE),
            ('traffic_resnet18_v6_pretrained', 'v6 预训练', BROWN)]
    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(9.4, 2.8),
                                       gridspec_kw={'width_ratios': [1, 1.15, 1.05]})

    # 左：数据集类别分布（训练/测试）
    import numpy as np
    root = os.path.join(HERE, 'data', 'Traffic_sign')
    if os.path.exists(root):
        classes = sorted(os.listdir(os.path.join(root, 'train_dataset')))
        tr = [len(os.listdir(os.path.join(root, 'train_dataset', c))) for c in classes]
        te = [len(os.listdir(os.path.join(root, 'test_dataset', c))) for c in classes]
        x = np.arange(len(classes))
        ax0.bar(x - 0.2, tr, 0.4, label='训练', color=BLUE)
        ax0.bar(x + 0.2, te, 0.4, label='测试', color=ORANGE)
        ax0.set_yscale('log'); ax0.set_xticks(x)
        ax0.set_xticklabels(classes, rotation=45, fontsize=5, ha='right')
        ax0.set_ylabel('样本数 (log)', fontsize=7)
        ax0.set_title('数据集类别分布（严重不均衡）', fontsize=8)
        ax0.legend(fontsize=5.5); ax0.grid(alpha=0.3, axis='y')

    labels, accs, colors = [], [], []
    for name, label, c in runs:
        s = load_sum(name)
        if not s:
            continue
        labels.append(label); accs.append(s['best_acc']); colors.append(c)
    if accs:
        ax1.bar(labels, accs, color=colors)
        ax1.set_ylabel('测试准确率 (%)'); ax1.set_title('交通标志调参对比', fontsize=9)
        ax1.tick_params(axis='x', labelsize=6, rotation=20)
        for i, v in enumerate(accs):
            ax1.text(i, v, f'{v:.1f}', ha='center', va='bottom', fontsize=6)
        ax1.set_ylim(max(0, min(accs) - 8), 100)
        ax1.grid(alpha=0.3, axis='y')
    else:
        ax1.set_title('（无数据）')

    cm = os.path.join(RES, 'traffic_confusion.json')
    if os.path.exists(cm):
        data = json.load(open(cm))
        M = np.array(data['matrix'], dtype=float)
        classes = data['classes']
        M = M / (M.sum(1, keepdims=True) + 1e-8)
        ax2.imshow(M, cmap='Blues', vmin=0, vmax=1)
        ax2.set_xticks(range(len(classes))); ax2.set_yticks(range(len(classes)))
        ax2.set_xticklabels(classes, rotation=90, fontsize=5)
        ax2.set_yticklabels(classes, fontsize=5)
        ax2.set_title('混淆矩阵（最佳模型，按行归一化）', fontsize=9)
    save(fig, 'fig6_traffic.png')


def fig_traffic_curves():
    runs = [('traffic_resnet18_v1_plain', 'v1 基线', BLUE),
            ('traffic_resnet18_v3_aug_ls_bal', 'v3 增强+均衡', GREEN),
            ('traffic_resnet18_v4_cos_mix', 'v4 Mixup+余弦', RED),
            ('traffic_resnet18_v6_pretrained', 'v6 预训练', BROWN)]
    two_panel(runs, '交通标志测试准确率', '交通标志训练损失', 'fig7_traffic_curves.png')


def fig_dataset():
    """交通标志数据集类别分布（训练/测试）。"""
    root = os.path.join(HERE, 'data', 'Traffic_sign')
    if not os.path.exists(root):
        return
    import numpy as np
    fig, ax = plt.subplots(figsize=(3.6, 2.4))
    classes = sorted(os.listdir(os.path.join(root, 'train_dataset')))
    tr = [len(os.listdir(os.path.join(root, 'train_dataset', c))) for c in classes]
    te = [len(os.listdir(os.path.join(root, 'test_dataset', c))) for c in classes]
    x = np.arange(len(classes))
    ax.bar(x - 0.2, tr, 0.4, label='训练', color=BLUE)
    ax.bar(x + 0.2, te, 0.4, label='测试', color=ORANGE)
    ax.set_yscale('log'); ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45, fontsize=6, ha='right')
    ax.set_ylabel('样本数 (log)'); ax.set_title('交通标志数据集类别分布（严重不均衡）', fontsize=8)
    ax.legend(fontsize=7); ax.grid(alpha=0.3, axis='y')
    save(fig, 'fig0_dataset.png')


if __name__ == '__main__':
    fig_dataset(); fig_optim(); fig_lr(); fig_decay(); fig_arch(); fig_kd()
    fig_traffic(); fig_traffic_curves()
    print('ALL FIGURES DONE')
