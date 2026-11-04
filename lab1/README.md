# Lab 1 — Monocular Depth Estimation and Point-Cloud Reconstruction

This experiment evaluates **Depth Anything V2 ViT-S** on 8 indoor images from **NYU Depth V2**, reconstructs metric depth and point clouds, and compares input resolutions and the metric-depth checkpoint.

## Folder contents

| Path | Description |
| --- | --- |
| `nyu_data/rgb/` | 8 RGB images (640×480) |
| `nyu_data/reference/` | Reference depth, valid mask, and camera intrinsics (`.npz`) |
| `nyu_data/metadata.json` | Dataset metadata |
| `output/` | Detailed analysis notes (Chinese) |
| `report/` | Report text parts, figures, and document builders |
| `results/*.csv` | Per-image metrics for resolution and metric-depth comparisons |
| `results/depth_*/` | Pseudo-colour depth visualisations (PNG) |
| `results/pointcloud_compare/` | Point-cloud comparison figure and PDF |

Large `.npy` depth arrays and `.ply` point clouds are not tracked; they can be regenerated from the depth pipeline.

## Setup

- Python 3.12, PyTorch 2.7.1 (CUDA 12.8), Open3D 0.19
- Weights: `depth_anything_v2_vits.pth` and `depth_anything_v2_metric_hypersim_vits.pth` from the [Depth Anything V2 repository](https://github.com/DepthAnything/Depth-Anything-V2)

The original inference was run in the course GPU environment. This repository keeps the data, outputs, metrics, and figures; the scripts under `report/` rebuild the DOCX report and depend on the Windows/WorkBuddy document SDK used at the time.

## Key results

| Setting | AbsRel | Depth RMSE (m) | XYZ RMSE (m) | Ground-truth DOF used |
| --- | --- | --- | --- | --- |
| Relative depth, 518 (affine alignment) | **0.0755** | **0.3738** | **0.3986** | 2 per image |
| Relative depth, 280 (affine alignment) | 0.0804 | 0.4019 | 0.4299 | 2 per image |
| Metric depth, 518 (zero fitting) | 0.2222 | 0.7055 | 0.7533 | 0 |
| Metric depth, 518 (pooled scale) | 0.1497 | 0.5431 | 0.5764 | 1 for all 8 |
| Metric depth, 518 (per-image scale) | 0.0947 | 0.3743 | 0.3987 | 1 per image |

- 518 → 280 is **2.41× faster**, with 6–8 % mean accuracy loss and about 10 % less depth-gradient detail.
- The metric checkpoint overestimates distance on NYU by about **16 %** on average, with a per-image scale between 0.711 and 1.046; a single pooled scale cannot correct it.
- Errors concentrate at depth discontinuities and around Kinect missing regions; XYZ RMSE is only about 1.07× the depth RMSE, so lateral error is small.

## Report

- [lab1.pdf](lab1.pdf) / [lab1.docx](lab1.docx)
- Assignment: [实验课1-单目深度与点云重建.pdf](实验课1-单目深度与点云重建.pdf)

For the full analysis in Chinese, see `output/01_深度预测与精度评估.md`, `output/02_输入分辨率对比_518_vs_280.md`, and `output/03_Metric_Depth版本对比.md`.
