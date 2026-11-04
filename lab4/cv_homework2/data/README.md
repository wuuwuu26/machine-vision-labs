# Datasets

This folder contains the datasets used by Lab 4.

## CIFAR-100

Not tracked. It is downloaded automatically by `torchvision.datasets.CIFAR100(root='./data', download=True)` on first run. Expected layout:

```text
lab4/cv_homework2/data/cifar-100-python/
```

## Traffic signs

**Included in this repository** (about 25 MB, 6,348 JPG images). Layout:

```text
lab4/cv_homework2/data/Traffic_sign/
├── train_dataset/          # 6,024 images
│   ├── GuideSign/
│   ├── M1/
│   ├── M4/
│   ├── M5/
│   ├── M6/
│   ├── M7/
│   ├── P1/
│   ├── P10_50/
│   ├── P12/
│   └── W1/
└── test_dataset/           # 324 images, same class folders
```

The dataset has 10 classes in total. `train.py` reads this layout through the custom `TrafficSignDataset`. It is the course-provided release and is kept here because it is not available from a public download page.
