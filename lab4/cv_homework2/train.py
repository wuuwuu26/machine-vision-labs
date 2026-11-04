"""统一训练框架：CIFAR-100 分类 / 优化器对比 / 学习率分析 /
网络深度对比 / 知识蒸馏 / 交通标志数据集。

在原 resnet-cifar100.py 思路基础上扩展，不修改原文件。
所有 run 固定 seed=42，结果写入 results/<name>.csv 与 _summary.json。
"""
import argparse
import os
import json
import time
import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
import torchvision
from torchvision import transforms
from torchvision.models.resnet import BasicBlock, Bottleneck, ResNet
from PIL import Image


# ----------------------------------------------------------------------------
# 网络：CIFAR 适配版 ResNet（stem 3x3/s1，仅 3 次下采样）
# ----------------------------------------------------------------------------
class ResNetCustom(ResNet):
    def __init__(self, block, layers, num_classes=1000):
        super(ResNet, self).__init__()
        self._norm_layer = nn.BatchNorm2d
        self.inplanes = 64
        self.dilation = 1
        self.groups = 1
        self.base_width = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)
        return self.fc(x)


_ARCH = {
    'resnet10': (BasicBlock, [1, 1, 1, 1]),
    'resnet18': (BasicBlock, [2, 2, 2, 2]),
    'resnet50': (Bottleneck, [3, 4, 6, 3]),
}


def build_model(arch, num_classes, pretrained=False):
    if pretrained:
        from torchvision.models import resnet18, resnet50
        fn = {'resnet18': resnet18, 'resnet50': resnet50}[arch]
        model = fn(weights='IMAGENET1K_V1')
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    block, layers = _ARCH[arch]
    return ResNetCustom(block, layers, num_classes=num_classes)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


# ----------------------------------------------------------------------------
# 交通标志数据集（自写 Dataset + DataLoader）
# ----------------------------------------------------------------------------
def _list_samples(root, split):
    base = os.path.join(root, 'train_dataset' if split == 'train' else 'test_dataset')
    classes = sorted(d for d in os.listdir(base)
                     if os.path.isdir(os.path.join(base, d)))
    samples, class_to_idx = [], {c: i for i, c in enumerate(classes)}
    for c in classes:
        d = os.path.join(base, c)
        for f in sorted(os.listdir(d)):
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                samples.append((os.path.join(d, f), class_to_idx[c]))
    return samples, classes


def compute_stats(root, img_size=96, cache=None):
    os.makedirs('results', exist_ok=True)
    cache = cache or f'results/traffic_stats_{img_size}.json'
    if os.path.exists(cache):
        return json.load(open(cache))
    samples, _ = _list_samples(root, 'train')
    mean = torch.zeros(3)
    var = torch.zeros(3)
    n = 0
    for path, _ in samples:
        im = Image.open(path).convert('RGB').resize((img_size, img_size))
        t = transforms.ToTensor()(im)
        mean += t.mean(dim=(1, 2))
        var += t.pow(2).mean(dim=(1, 2))
        n += 1
    mean /= n
    var /= n
    std = (var - mean ** 2).sqrt()
    stats = {'mean': mean.tolist(), 'std': std.tolist()}
    json.dump(stats, open(cache, 'w'))
    return stats


class TrafficSignDataset(torch.utils.data.Dataset):
    def __init__(self, root, split, img_size=96, strong_aug=False, stats=None):
        self.samples, self.classes = _list_samples(root, split)
        self.img_size = img_size
        if stats is None:
            stats = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}
        self.stats = stats
        if split == 'train' and strong_aug:
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
                transforms.ColorJitter(0.3, 0.3, 0.3, 0.1),
                transforms.RandomRotation(10),
                transforms.ToTensor(),
                transforms.Normalize(stats['mean'], stats['std']),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(stats['mean'], stats['std']),
            ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        return self.transform(img), label


def traffic_weights(root, split):
    samples, classes = _list_samples(root, split)
    counts = [0] * len(classes)
    for _, y in samples:
        counts[y] += 1
    w = [1.0 / counts[y] for _, y in samples]
    return w


# ----------------------------------------------------------------------------
# 优化器 / 学习率调度
# ----------------------------------------------------------------------------
def build_optimizer(model, opt, lr, wd, momentum=0.9):
    if opt == 'sgd':
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=wd)
    if opt == 'adam':
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    if opt == 'adamw':
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    if opt == 'rmsprop':
        return torch.optim.RMSprop(model.parameters(), lr=lr, momentum=momentum, weight_decay=wd)
    raise ValueError(opt)


def build_scheduler(optimizer, sched, lr, epochs=50):
    if sched == 'const':
        return None
    if sched == 'step':
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 40], gamma=0.1)
    if sched == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    raise ValueError(sched)


# ----------------------------------------------------------------------------
# 数据加载
# ----------------------------------------------------------------------------
def _loader(ds, args, device, shuffle=False, sampler=None):
    return DataLoader(ds, batch_size=args.bsize, shuffle=shuffle, sampler=sampler,
                      num_workers=args.workers, pin_memory=(device.type == 'cuda'),
                      persistent_workers=args.workers > 0,
                      prefetch_factor=4 if args.workers > 0 else None,
                      drop_last=False,
                      timeout=600 if args.workers > 0 else 0)


def get_loaders(args, device):
    if args.dataset == 'cifar100':
        norm = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        train_tf = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(), norm])
        test_tf = transforms.Compose([transforms.ToTensor(), norm])
        train_set = torchvision.datasets.CIFAR100(root=args.data_root, train=True, transform=train_tf)
        test_set = torchvision.datasets.CIFAR100(root=args.data_root, train=False, transform=test_tf)
        num_classes = 100
    else:
        stats = compute_stats(args.data_root, args.img_size)
        strong = getattr(args, 'strong_aug', False)
        train_set = TrafficSignDataset(args.data_root, 'train', args.img_size, strong_aug=strong, stats=stats)
        test_set = TrafficSignDataset(args.data_root, 'test', args.img_size, strong_aug=False, stats=stats)
        num_classes = len(train_set.classes)
        if getattr(args, 'balance', False):
            weights = traffic_weights(args.data_root, 'train')
            sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
            return (_loader(train_set, args, device, sampler=sampler),
                    _loader(test_set, args, device), num_classes)

    return (_loader(train_set, args, device, shuffle=True),
            _loader(test_set, args, device), num_classes)


# ----------------------------------------------------------------------------
# 蒸馏
# ----------------------------------------------------------------------------
def kd_loss(student_logits, teacher_logits, labels, T, alpha, mode):
    student_logits = student_logits.float()
    teacher_logits = teacher_logits.float()
    ce = F.cross_entropy(student_logits, labels)
    if mode == 'soft':
        s = F.log_softmax(student_logits / T, dim=1)
        t = F.softmax(teacher_logits / T, dim=1)
        kl = F.kl_div(s, t, reduction='batchmean') * (T * T)
        return alpha * ce + (1 - alpha) * kl
    else:  # logits (FitNets 风格)
        mse = F.mse_loss(student_logits, teacher_logits.detach())
        return alpha * ce + (1 - alpha) * mse


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return 100.0 * correct / total


# ----------------------------------------------------------------------------
# 主训练
# ----------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--name', type=str, default=None)
    p.add_argument('--arch', type=str, default='resnet10')
    p.add_argument('--opt', type=str, default='sgd')
    p.add_argument('--lr', type=float, default=0.1)
    p.add_argument('--wd', type=float, default=1e-4)
    p.add_argument('--momentum', type=float, default=0.9)
    p.add_argument('--sched', type=str, default='const')
    p.add_argument('--dataset', type=str, default='cifar100')
    p.add_argument('--data-root', type=str, default='./data')
    p.add_argument('--img-size', type=int, default=96)
    p.add_argument('--bsize', type=int, default=256)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--kd', action='store_true')
    p.add_argument('--teacher', type=str, default=None)
    p.add_argument('--T', type=float, default=4.0)
    p.add_argument('--alpha', type=float, default=0.7)
    p.add_argument('--kd-mode', type=str, default='soft')
    p.add_argument('--strong-aug', action='store_true')
    p.add_argument('--balance', action='store_true')
    p.add_argument('--label-smoothing', type=float, default=0.0)
    p.add_argument('--amp', action='store_true', help='mixed precision training')
    p.add_argument('--mixup', type=float, default=0.0, help='mixup alpha, 0 disables')
    p.add_argument('--pretrained', action='store_true', help='use ImageNet pretrained weights')
    p.add_argument('--resume', action='store_true',
                   help='从 results/<name>_last.pth 断点续跑（也可用于中断后补齐）')
    p.add_argument('--warm-start', type=str, default=None,
                   help='从指定权重热启动（配合 --start-epoch 追加训练剩余 epoch）')
    p.add_argument('--start-epoch', type=int, default=0,
                   help='热启动时的起始 epoch（1-based），调度器会先推进到该 epoch')
    args = p.parse_args()

    if args.name is None:
        args.name = f"{args.dataset}_{args.arch}_{args.opt}_lr{args.lr}_sched{args.sched}"
        if args.kd:
            args.name += f"_kd{args.kd_mode}_T{args.T}_a{args.alpha}"
    out_csv = os.path.join('results', args.name + '.csv')
    out_json = os.path.join('results', args.name + '_summary.json')
    os.makedirs('results', exist_ok=True)
    if not args.resume and os.path.exists(out_csv) and sum(1 for _ in open(out_csv)) >= args.epochs + 1:
        print(f"[SKIP] {args.name} 已完成")
        return

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True

    train_loader, test_loader, num_classes = get_loaders(args, device)
    model = build_model(args.arch, num_classes, pretrained=args.pretrained).to(device)

    teacher = None
    if args.kd:
        assert args.teacher and os.path.exists(args.teacher), "need --teacher ckpt"
        teacher = build_model('resnet50', num_classes)
        teacher.load_state_dict(torch.load(args.teacher, map_location=device))
        teacher.eval().to(device)

    optimizer = build_optimizer(model, args.opt, args.lr, args.wd, args.momentum)
    scheduler = build_scheduler(optimizer, args.sched, args.lr, args.epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    use_amp = args.amp and device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    # ---- 断点续跑：results/<name>_last.pth 保存 model/opt/sched/epoch/best ----
    n_params = count_params(model)
    best_acc, best_ep = 0.0, 0
    start_epoch = 1
    elapsed_before = 0.0
    last_p = os.path.join('results', args.name + '_last.pth')
    has_ckpt = args.resume and os.path.exists(last_p)
    if has_ckpt:
        ck = torch.load(last_p, map_location=device)
        model.load_state_dict(ck['model'])
        optimizer.load_state_dict(ck['opt'])
        if scheduler and ck.get('sched'):
            scheduler.load_state_dict(ck['sched'])
        if ck.get('scaler'):
            scaler.load_state_dict(ck['scaler'])
        start_epoch = ck['epoch'] + 1
        best_acc, best_ep = ck.get('best_acc', 0.0), ck.get('best_ep', 0)
        elapsed_before = ck.get('elapsed', 0.0)
        print(f"[RESUME] {args.name} 从 epoch {start_epoch} 继续（已完成 {start_epoch - 1}，"
              f"best={best_acc:.2f}%）", flush=True)

    # ---- 热启动：从已有权重继续训练剩余 epoch（用于补齐被中断的实验）----
    warm = args.warm_start and os.path.exists(args.warm_start) and args.start_epoch > 1
    if warm:
        model.load_state_dict(torch.load(args.warm_start, map_location=device))
        start_epoch = args.start_epoch
        for _ in range(start_epoch - 1):
            if scheduler:
                scheduler.step()
        f = open(out_csv, 'a')
        prev = [ln.strip().split(',') for ln in open(out_csv) if ln[:1].isdigit()]
        if prev:
            best_acc = max(float(r[2]) for r in prev)
            best_ep = int(max(prev, key=lambda r: float(r[2]))[0])
        print(f"[WARM] {args.name} 从 {args.warm_start} 热启动，续训 epoch "
              f"{start_epoch}~{args.epochs}（已有 best={best_acc:.2f}%）", flush=True)
    elif has_ckpt and os.path.exists(out_csv):
        f = open(out_csv, 'a')
        done = sum(1 for _ in open(out_csv)) - 1
        start_epoch = max(start_epoch, done + 1)
        print(f"[RESUME] CSV 已有 {done} 个 epoch，从 {start_epoch} 继续", flush=True)
    else:
        # 无可用断点：从头训练，CSV 截断重写，避免误跳过 epoch
        f = open(out_csv, 'w')
        f.write('epoch,train_loss,test_acc,seconds,lr\n')
    t0 = time.time()
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        running, ns = 0.0, 0
        e0 = time.time()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            if args.mixup > 0:
                lam = float(torch.distributions.Beta(args.mixup, args.mixup).sample())
                idx = torch.randperm(x.size(0), device=device)
                x = lam * x + (1 - lam) * x[idx]
                y_b = y[idx]
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(x)
                if args.kd:
                    with torch.no_grad():
                        tlogits = teacher(x)
                    loss = kd_loss(logits, tlogits, y, args.T, args.alpha, args.kd_mode)
                elif args.mixup > 0:
                    loss = lam * criterion(logits, y) + (1 - lam) * criterion(logits, y_b)
                else:
                    loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += loss.item() * x.size(0)
            ns += x.size(0)
        train_loss = running / ns
        acc = evaluate(model, test_loader, device)
        lr_now = optimizer.param_groups[0]['lr']
        dt = time.time() - e0
        if scheduler:
            scheduler.step()
        if acc > best_acc:
            best_acc, best_ep = acc, epoch
            torch.save(model.state_dict(), os.path.join('results', args.name + '_best.pth'))
        f.write(f"{epoch},{train_loss:.4f},{acc:.2f},{dt:.1f},{lr_now:.6f}\n")
        f.flush()
        torch.save({'model': model.state_dict(), 'opt': optimizer.state_dict(),
                    'sched': scheduler.state_dict() if scheduler else None,
                    'scaler': scaler.state_dict(), 'epoch': epoch,
                    'best_acc': best_acc, 'best_ep': best_ep,
                    'elapsed': elapsed_before + (time.time() - t0)},
                   last_p)
        print(f"[{args.name}] ep{epoch:02d} loss={train_loss:.4f} acc={acc:.2f} "
              f"lr={lr_now:.5f} t={dt:.1f}s", flush=True)
    total = elapsed_before + (time.time() - t0)
    summary = {'name': args.name, 'arch': args.arch, 'opt': args.opt, 'lr': args.lr,
               'sched': args.sched, 'dataset': args.dataset, 'epochs': args.epochs,
               'params': n_params, 'best_acc': best_acc, 'best_epoch': best_ep,
               'total_time': total, 'device': str(device), 'seed': args.seed,
               'pretrained': args.pretrained, 'amp': args.amp}
    json.dump(summary, open(out_json, 'w'), indent=2)
    f.close()
    print(f"[DONE] {args.name} best_acc={best_acc:.2f}@{best_ep} total={total/60:.1f}min params={n_params}")


if __name__ == '__main__':
    main()
