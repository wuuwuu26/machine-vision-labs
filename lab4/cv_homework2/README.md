# 基于卷积网络的图像分类（实验二）

CIFAR-100（ResNet10/18/50、优化器、学习率、知识蒸馏）+ 交通标志数据集（自写 Dataset/DataLoader 并调参）。

> 仓库说明：课程提供的交通标志图像已包含在 `data/Traffic_sign/`；CIFAR-100 与 `results/*.pth` 权重未提交，获取/放置方式见根目录 `README.zh-CN.md` 的“未纳入仓库的大文件”一节。

## 1. 环境

```bash
python3 -m venv venv && source venv/bin/activate      # Python 3.12
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
python env_info.py                                     # 把版本/GPU 信息写入 results/env.json
```

实测环境：Python 3.12.3 + PyTorch 2.11.0+cu128 / torchvision 0.26.0，NVIDIA RTX 4070 Laptop 8GB。

## 2. 文件说明

| 文件 | 说明 |
| --- | --- |
| `resnet-cifar100.py` / `config.yml` | 课程原始示例与配置（**未修改**） |
| `train.py` | 统一训练框架：ResNet10/18/50、SGD/Adam/AdamW/RMSprop、const/step/cosine 调度、知识蒸馏（soft & logits）、交通标志 Dataset/DataLoader、AMP、Mixup、标签平滑、类别均衡采样、ImageNet 预训练微调 |
| `run_all.py` | 实验矩阵调度（分阶段、断点跳过），`PY=` 可指定解释器 |
| `smoke_test.sh` | 1-epoch 冒烟测试，快速验证两条数据链路 |
| `run_pipeline.sh` | 一键：全部实验 → 混淆矩阵 → 出图 → 出报告 |
| `eval_traffic.py` | 交通标志最佳模型的混淆矩阵与各类召回率 |
| `make_figures.py` | 由 `results/*.csv` 生成 8 张分析图到 `results/figs/` |
| `make_report.py` | 生成中文实验报告 `report/report.html` 与 `实验报告-图像分类2.pdf` |
| `env_info.py` | 记录 Python/PyTorch/CUDA/GPU 信息 |
| `results/` | 每个实验的逐 epoch CSV、`*_summary.json`、日志、图表（最佳权重 `.pth` 未提交） |

## 3. 复现

```bash
bash smoke_test.sh          # 先冒烟测试（约 3 分钟）
bash run_pipeline.sh        # 全部实验 + 图表 + 报告（GPU 上数小时）

# 或分阶段：
python run_all.py 0         # 跑通课程原脚本 resnet-cifar100.py
python run_all.py 1 2 3 4   # 任务1~4（CIFAR-100）
python run_all.py traffic   # 任务6（交通标志调参）
python run_all.py 5         # 任务5（知识蒸馏，需先有 ResNet50 教师权重）
python make_figures.py && python make_report.py
```

所有实验固定 `seed=42`；`results/<name>_summary.json` 已存在则自动跳过，可断点续跑。

## 4. 实验矩阵（对应 PDF 的六项要求）

| 要求 | 实验 | 关键设置 |
| --- | --- | --- |
| 1 跑通基线 | `original_*` + `cifar100_resnet10_sgd_lr0.001_schedconst` | 原脚本验证 + SGD lr=1e-3 复现；主基线 lr=0.1+step |
| 2 优化器 | `cifar100_resnet10_{sgd,adam,adamw,rmsprop}_*_schedconst` | 恒定 lr，ResNet10，其余一致 |
| 3 学习率 | `..._lr{0.001,0.01,0.1,0.3}c_schedconst` + `..._lr0.1_schedstep` | 0.1 初值，30/40 epoch 各 ×0.1，看 train loss 收敛 |
| 4 网络深度 | `cifar100_resnet{10,18,50}_sgd_lr0.1_schedstep` | 同配置只改结构，记录精度/参数量/耗时 |
| 5 知识蒸馏 | `cifar100_resnet18_kdsoft_T4_a0.7`、`..._kdlogit_T1_a0.5` | teacher=ResNet50，student=ResNet18 |
| 6 交通标志 | `traffic_resnet18_v1…v6` | 自写 Dataset/DataLoader，逐步叠加增强/LS/均衡采样/Mixup/余弦/预训练 |

## 5. 结果

最终精度（CIFAR-100 测试集 10000 张；交通标志测试集 324 张）：

| 任务 | 实验 | 最佳精度 | 最佳 epoch | 耗时 |
| --- | --- | --- | --- | --- |
| 1 基线 | 课程原脚本（lr=1e-3, 50ep） | 37.21% | — | 13.1 min |
| 1 基线 | 框架复现 lr=1e-3 | 36.86% | 49 | 6.5 min |
| 1 主基线 | ResNet10, lr=0.1+step | **71.38%** | 45 | 6.5 min |
| 2 优化器 | SGD / Adam / AdamW / RMSprop | 66.44 / 66.16 / **66.70** / 51.70% | — | 6.5~6.6 min |
| 3 学习率 | lr=0.001 / 0.01 / 0.1 / 0.3 | 36.86 / 63.60 / **66.44** / 63.47% | — | ~7.3 min |
| 4 深度 | ResNet10 / ResNet18 / ResNet50 | 71.38 / **72.73** / 70.16% | — | 6.5 / 12.6 / 45.1 min |
| 5 蒸馏 | ResNet18 基线 / KD-soft / KD-logit | 72.73 / **73.46** / 71.97% | — | 12.6 / ~26 / ~24 min |
| 6 交通标志 | v1 基线 | 97.84% | 30 | 21.6 min |
| 6 | v2 强增强+标签平滑 | 97.53% | 28 | 25.8 min |
| 6 | v3 +类别均衡采样 | 96.91% | 42 | 24.3 min |
| 6 | **v4 +Mixup+余弦（80ep）** | **98.15%** | 70 | 36.2 min |
| 6 | v5 ResNet50（96px, 40ep） | 94.75% | 32 | 31.0 min |
| 6 | **v6 ImageNet 预训练微调（40ep）** | **98.15%** | 22 | 2.0 min |

- 实验报告：`报告LaTeX版/lab4.pdf`（PDF）、`报告Word版/实验报告-图像分类2.docx`（Word）、`report/report.html`（HTML）
- 图表：`results/figs/*.png`
- 逐 epoch 数据：`results/*.csv`（列：epoch, train_loss, test_acc, seconds, lr）
- 混淆矩阵：`results/traffic_confusion{,_v1,_v2,_v3,_v4}.json`
