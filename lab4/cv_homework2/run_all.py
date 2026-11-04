"""实验矩阵调度：顺序运行 train.py 的各组实验，已完成的自动跳过。

用法:
    python run_all.py                 # 运行全部阶段
    python run_all.py 1 2 3 4         # 只运行 CIFAR-100 的任务 1~4
    python run_all.py traffic         # 只运行交通标志调参
    python run_all.py 5               # 只运行知识蒸馏（需先有 ResNet50 教师权重）
    PY=/path/to/python python run_all.py   # 指定解释器
"""
import subprocess
import sys
import os

PY = os.environ.get('PY') or sys.executable
HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, 'results')
os.makedirs(RESULTS, exist_ok=True)

CIFAR_ROOT = './data'
TRAFFIC_ROOT = './data/Traffic_sign'
# 8GB 显存（且与其他进程共享）下的批大小：CIFAR 32×32 用 256，交通标志 128×128 用 64
CIFAR_BS, TRAFFIC_BS = 256, 64


def run(name, extra, epochs=50, data_root=CIFAR_ROOT, bsize=CIFAR_BS, workers=6):
    csv = os.path.join(RESULTS, name + '.csv')
    summary = os.path.join(RESULTS, name + '_summary.json')
    if os.path.exists(summary) and os.path.exists(csv) and sum(1 for _ in open(csv)) >= epochs + 1:
        print(f"[SKIP] {name}", flush=True)
        return True
    log = os.path.join(RESULTS, name + '.log')
    # 若 CSV 已存在但不完整（例如中途卡死/被杀），带 --resume 续跑，避免重头再来
    resume = []
    if os.path.exists(csv) and sum(1 for _ in open(csv)) > 1:
        resume = ['--resume']
        print(f"[RESUME] {name} 检测到未完成的 CSV，将从断点继续", flush=True)
    cmd = [PY, 'train.py', '--name', name, '--data-root', data_root,
           '--workers', str(workers), '--bsize', str(bsize),
           '--epochs', str(epochs), '--amp'] + resume + extra
    print(f"\n>>> START {name}\n    {' '.join(cmd)}", flush=True)
    with open(log, 'w') as lf:
        r = subprocess.run(cmd, cwd=HERE, stdout=lf, stderr=subprocess.STDOUT)
    if r.returncode != 0:
        print(f"[FAIL] {name} -> see {log}", flush=True)
        return False
    print(f"[OK] {name}", flush=True)
    return True


TEACHER = os.path.join(RESULTS, 'cifar100_resnet50_sgd_lr0.1_schedstep_best.pth')
CIFAR_COMMON = ['--opt', 'sgd', '--lr', '0.1', '--sched', 'step']
TRAFFIC_COMMON = ['--dataset', 'traffic', '--img-size', '128',
                  '--strong-aug', '--label-smoothing', '0.1']

# stage -> [(name, extra, epochs, data_root, bsize)]
STAGES = {
    # ---- 任务1(前置)：直接运行课程原脚本，验证依赖安装成功、代码跑通 ----
    '0': 'original',
    # ---- 任务1(补充)：原脚本按 config.yml 完整跑 50 epoch，得到原脚本精度 ----
    '0f': 'original_full',
    # ---- 任务1：跑通基线 ----
    '1': [
        # 原 config.yml 设置（lr=1e-3，恒定）：说明学习率过小时的欠拟合
        ('cifar100_resnet10_sgd_lr0.001_schedconst',
         ['--arch', 'resnet10', '--opt', 'sgd', '--lr', '0.001', '--sched', 'const'],
         50, CIFAR_ROOT, CIFAR_BS),
        # 主基线：lr=0.1 + 30/40 epoch 各降 10 倍
        ('cifar100_resnet10_sgd_lr0.1_schedstep',
         ['--arch', 'resnet10'] + CIFAR_COMMON, 50, CIFAR_ROOT, CIFAR_BS),
    ],
    # ---- 任务2：优化器对比（恒定 lr，其余一致）----
    '2': [
        ('cifar100_resnet10_sgd_lr0.1_schedconst',
         ['--arch', 'resnet10', '--opt', 'sgd', '--lr', '0.1', '--sched', 'const'],
         50, CIFAR_ROOT, CIFAR_BS),
        ('cifar100_resnet10_adam_lr0.001_schedconst',
         ['--arch', 'resnet10', '--opt', 'adam', '--lr', '0.001', '--sched', 'const'],
         50, CIFAR_ROOT, CIFAR_BS),
        ('cifar100_resnet10_adamw_lr0.001_schedconst',
         ['--arch', 'resnet10', '--opt', 'adamw', '--lr', '0.001', '--sched', 'const'],
         50, CIFAR_ROOT, CIFAR_BS),
        ('cifar100_resnet10_rmsprop_lr0.001_schedconst',
         ['--arch', 'resnet10', '--opt', 'rmsprop', '--lr', '0.001', '--sched', 'const'],
         50, CIFAR_ROOT, CIFAR_BS),
    ],
    # ---- 任务3：学习率影响（0.001 与 0.1 复用任务1/2 的结果）----
    '3': [
        ('cifar100_resnet10_sgd_lr0.01c_schedconst',
         ['--arch', 'resnet10', '--opt', 'sgd', '--lr', '0.01', '--sched', 'const'],
         50, CIFAR_ROOT, CIFAR_BS),
        ('cifar100_resnet10_sgd_lr0.3c_schedconst',
         ['--arch', 'resnet10', '--opt', 'sgd', '--lr', '0.3', '--sched', 'const'],
         50, CIFAR_ROOT, CIFAR_BS),
    ],
    # ---- 任务4：网络深度对比 ----
    '4': [
        ('cifar100_resnet18_sgd_lr0.1_schedstep',
         ['--arch', 'resnet18'] + CIFAR_COMMON, 50, CIFAR_ROOT, CIFAR_BS),
        ('cifar100_resnet50_sgd_lr0.1_schedstep',
         ['--arch', 'resnet50'] + CIFAR_COMMON, 50, CIFAR_ROOT, 128),
    ],
    # ---- 任务5：知识蒸馏（teacher=ResNet50 → student=ResNet18）----
    '5': [
        ('cifar100_resnet18_kdsoft_T4_a0.7',
         ['--arch', 'resnet18'] + CIFAR_COMMON +
         ['--kd', '--kd-mode', 'soft', '--T', '4', '--alpha', '0.7', '--teacher', TEACHER],
         50, CIFAR_ROOT, 128),
        ('cifar100_resnet18_kdlogit_T1_a0.5',
         ['--arch', 'resnet18'] + CIFAR_COMMON +
         ['--kd', '--kd-mode', 'logit', '--T', '1', '--alpha', '0.5', '--teacher', TEACHER],
         50, CIFAR_ROOT, 128),
    ],
    # ---- 任务6：交通标志（自写 Dataset/DataLoader）逐步调参 ----
    'traffic': [
        ('traffic_resnet18_v1_plain',
         ['--dataset', 'traffic', '--arch', 'resnet18', '--opt', 'sgd', '--lr', '0.05',
          '--sched', 'step', '--img-size', '128'], 50, TRAFFIC_ROOT, TRAFFIC_BS),
        ('traffic_resnet18_v2_aug_ls',
         ['--arch', 'resnet18', '--opt', 'sgd', '--lr', '0.05', '--sched', 'step'] + TRAFFIC_COMMON,
         50, TRAFFIC_ROOT, TRAFFIC_BS),
        ('traffic_resnet18_v3_aug_ls_bal',
         ['--arch', 'resnet18', '--opt', 'sgd', '--lr', '0.05', '--sched', 'step',
          '--balance'] + TRAFFIC_COMMON, 50, TRAFFIC_ROOT, TRAFFIC_BS),
        ('traffic_resnet18_v4_cos_mix',
         ['--arch', 'resnet18', '--opt', 'sgd', '--lr', '0.05', '--sched', 'cosine',
          '--balance', '--mixup', '0.2'] + TRAFFIC_COMMON, 80, TRAFFIC_ROOT, TRAFFIC_BS),
        ('traffic_resnet50_v5_cos_mix',
         ['--arch', 'resnet50', '--opt', 'sgd', '--lr', '0.03', '--sched', 'cosine',
          '--balance', '--mixup', '0.2',
          '--dataset', 'traffic', '--img-size', '96',
          '--strong-aug', '--label-smoothing', '0.1'], 40, TRAFFIC_ROOT, 16),
        ('traffic_resnet18_v6_pretrained',
         ['--arch', 'resnet18', '--pretrained', '--opt', 'sgd', '--lr', '0.005',
          '--sched', 'cosine', '--balance'] + TRAFFIC_COMMON, 40, TRAFFIC_ROOT, TRAFFIC_BS),
    ],
}

ORDER = ['0', '1', '2', '3', '4', '5', 'traffic']
ORIG_LOG = os.path.join(RESULTS, 'original_resnet10_cifar100.log')
ORIG_FULL_LOG = os.path.join(RESULTS, 'original_resnet10_cifar100_full.log')


def run_original(epochs=5, log=None):
    """直接运行课程提供的 resnet-cifar100.py，验证依赖安装成功、代码跑通。

    epochs=5 为快速跑通验证；epochs=None 表示按 config.yml 完整跑 50 epoch，
    得到课程原脚本的精度（写入 *_full.log）。
    """
    log = log or ORIG_LOG
    if os.path.exists(log) and 'Best Accuracy' in open(log, errors='ignore').read():
        print(f'[SKIP] original resnet-cifar100.py ({os.path.basename(log)})', flush=True)
        return True
    cmd = [PY, 'resnet-cifar100.py', '-c', 'config.yml', '--device', '0', '--workers', '8']
    if epochs:
        cmd += ['--epochs', str(epochs)]
    print(f"\n>>> START original\n    {' '.join(cmd)}", flush=True)
    with open(log, 'w') as lf:
        r = subprocess.run(cmd, cwd=HERE, stdout=lf, stderr=subprocess.STDOUT)
    print(('[OK] original' if r.returncode == 0 else f'[FAIL] original -> {log}'), flush=True)
    return r.returncode == 0


def main():
    want = sys.argv[1:] or ORDER
    for st in want:
        if st not in STAGES:
            print(f'[WARN] unknown stage {st}', flush=True)
            continue
        print(f'\n######## STAGE {st} ########', flush=True)
        if STAGES[st] == 'original':
            run_original()
            continue
        if STAGES[st] == 'original_full':
            run_original(epochs=None, log=ORIG_FULL_LOG)
            continue
        for name, extra, epochs, root, bs in STAGES[st]:
            if st == '5' and not os.path.exists(TEACHER):
                print(f'[WARN] teacher ckpt 缺失: {TEACHER}，跳过蒸馏', flush=True)
                break
            run(name, extra, epochs, root, bs)
    print('\n=== ALL DONE ===', flush=True)


if __name__ == '__main__':
    main()
