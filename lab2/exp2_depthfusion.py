#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实验课2 - 立体匹配与三维重建
深度图 -> 点云：
  (A) 朴素点云   : 直接用变换公式反投影全部深度像素，不做任何滤波
  (B) 概率滤波   : 滤除光度一致性(置信度)低于阈值的像素
  (C) 几何一致性 + 多视融合 : 跨视角重投影检查后再融合平均
用法:
  python exp2_depthfusion.py --dense_folder <scan9路径> \
      --prob_threshold 0.8 --disp_thresh 1.0 --num_consistent 3
"""
from __future__ import print_function

import os
import re
import sys
import argparse
import numpy as np


# ---------------------------------------------------------------- PFM 读写 ---
def read_pfm(path):
    with open(path, 'rb') as f:
        header = f.readline().decode('utf-8').rstrip()
        if header not in ('PF', 'Pf'):
            raise Exception('Not a PFM file: %s' % path)
        dim_match = re.match(r'^(\d+)\s(\d+)\s$', f.readline().decode('utf-8'))
        if not dim_match:
            raise Exception('Malformed PFM header: %s' % path)
        width, height = map(int, dim_match.groups())
        scale = float(f.readline().decode('utf-8').rstrip())
        dtype = '<f' if scale < 0 else '>f'
        data = np.fromfile(f, dtype)
        shape = (height, width) if header == 'Pf' else (height, width, 3)
        return np.flipud(np.reshape(data, shape))


def write_pfm(path, image, scale=1):
    image = np.asarray(image, dtype=np.float32)
    with open(path, 'wb') as f:
        f.write(b'Pf\n')
        f.write(('%d %d\n' % (image.shape[1], image.shape[0])).encode())
        endian = image.dtype.byteorder
        s = -scale if (endian == '<' or (endian == '=' and sys.byteorder == 'little')) else scale
        f.write(('%f\n' % s).encode())
        f.write(np.flipud(image).tobytes())


def write_ply(path, points, colors=None):
    points = np.asarray(points, dtype=np.float64)
    n = points.shape[0]
    has_color = colors is not None and len(colors) == n
    with open(path, 'wb') as f:
        f.write(b'ply\nformat binary_little_endian 1.0\n')
        f.write(('element vertex %d\n' % n).encode())
        f.write(b'property double x\nproperty double y\nproperty double z\n')
        if has_color:
            f.write(b'property uchar red\nproperty uchar green\nproperty uchar blue\n')
        f.write(b'end_header\n')
        if has_color:
            dt = np.dtype([('x', '<f8'), ('y', '<f8'), ('z', '<f8'),
                           ('r', 'u1'), ('g', 'u1'), ('b', 'u1')])
            arr = np.empty(n, dtype=dt)
            arr['x'], arr['y'], arr['z'] = points[:, 0], points[:, 1], points[:, 2]
            arr['r'], arr['g'], arr['b'] = colors[:, 0], colors[:, 1], colors[:, 2]
        else:
            dt = np.dtype([('x', '<f8'), ('y', '<f8'), ('z', '<f8')])
            arr = np.empty(n, dtype=dt)
            arr['x'], arr['y'], arr['z'] = points[:, 0], points[:, 1], points[:, 2]
        f.write(arr.tobytes())
    return n


# ------------------------------------------------------------- 相机与变换 ---
# MVSNet cam 格式:  cam[0]=外参  X_cam = R X_world + t  (相机中心 C = -R^T t)
#                   cam[1]=内参 K
def read_cam(path):
    with open(path, 'r') as f:
        words = f.read().split()
    cam = np.zeros((2, 4, 4), dtype=np.float64)
    for i in range(4):
        for j in range(4):
            cam[0, i, j] = float(words[4 * i + j + 1])
    for i in range(3):
        for j in range(3):
            cam[1, i, j] = float(words[3 * i + j + 18])
    return cam


def cam_R(cam):
    return cam[0, 0:3, 0:3]


def cam_t(cam):
    return cam[0, 0:3, 3]


def cam_K(cam):
    return cam[1, 0:3, 0:3]


def cam_center(cam):
    R = cam_R(cam)
    return -cam_t(cam).dot(R)          # C = -R^T t （行向量写法）


def backproject(depth, cam):
    """整张深度图 -> 世界坐标 (H,W,3)
       X_cam = d * K^{-1}[u,v,1]^T ;  X_world = R^T (X_cam - t)
    """
    R, t, K = cam_R(cam), cam_t(cam), cam_K(cam)
    C = cam_center(cam)
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w, dtype=np.float64) + 0.5,
                       np.arange(h, dtype=np.float64) + 0.5)
    rays = np.stack([u, v, np.ones_like(u)], axis=-1).dot(np.linalg.inv(K).T)
    X_cam = rays * depth[..., None]
    return X_cam.dot(R) + C[None, None, :]


def backproject_pixels(u, v, d, cam):
    """单个像素集合 -> 世界坐标 (N,3)"""
    R, t, K = cam_R(cam), cam_t(cam), cam_K(cam)
    pts = np.stack([np.asarray(u, np.float64) + 0.5,
                    np.asarray(v, np.float64) + 0.5,
                    np.ones(len(u))], axis=-1)
    X_cam = pts.dot(np.linalg.inv(K).T) * np.asarray(d, np.float64)[:, None]
    return (X_cam - t[None, :]).dot(R)


def project_world(X, cam):
    """世界坐标 (N,3) -> (像素 uv, 相机系深度 z, 有效性)"""
    R, t, K = cam_R(cam), cam_t(cam), cam_K(cam)
    X_cam = X.dot(R.T) + t[None, :]
    z = X_cam[:, 2]
    proj = X_cam.dot(K.T)
    valid = z > 1e-6
    uv = np.zeros_like(proj[:, :2])
    uv[valid] = proj[valid, :2] / z[valid][:, None]
    return uv, z, valid


def read_pair(pair_path):
    with open(pair_path, 'r') as f:
        tokens = f.read().split()
    n = int(tokens[0])
    pairs, pos = {}, 1
    for _ in range(n):
        ref = int(tokens[pos]); pos += 1
        cnt = int(tokens[pos]); pos += 1
        pairs[ref] = [int(tokens[pos + 2 * i]) for i in range(cnt)]
        pos += 2 * cnt
    return pairs


# ------------------------------------------------------------------- 主流程 ---
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dense_folder', default='/home/user/rvclab2/dtu_test_scan9/scan9')
    ap.add_argument('--out_folder', default='')
    ap.add_argument('--prob_threshold', type=float, default=0.8)
    ap.add_argument('--disp_thresh', type=float, default=1.0)
    ap.add_argument('--num_consistent', type=int, default=3)
    ap.add_argument('--rel_depth_thresh', type=float, default=0.01)
    ap.add_argument('--max_views', type=int, default=0, help='>0 时只处理前 N 个视图')
    args = ap.parse_args()

    dense = args.dense_folder
    depth_folder = os.path.join(dense, 'depths_mvsnet')
    out_folder = args.out_folder or os.path.join(dense, 'points_mvsnet')
    if not os.path.isdir(out_folder):
        os.makedirs(out_folder)

    pairs = read_pair(os.path.join(dense, 'pair.txt'))
    view_ids = sorted(int(f.split('_')[0])
                      for f in os.listdir(depth_folder) if f.endswith('_init.pfm'))
    if args.max_views > 0:
        view_ids = view_ids[:args.max_views]
    print('参考视图数: %d' % len(view_ids))

    depths, probs, cams = {}, {}, {}
    for vid in view_ids:
        depths[vid] = read_pfm(os.path.join(depth_folder, '%08d_init.pfm' % vid))
        probs[vid] = read_pfm(os.path.join(depth_folder, '%08d_prob.pfm' % vid))
        cams[vid] = read_cam(os.path.join(depth_folder, '%08d.txt' % vid))
    d0 = depths[view_ids[0]]
    print('深度图尺寸: %s' % str(d0.shape))

    # ---------------- (A) 朴素点云：无滤波，全部直接反投影 ----------------
    naive = []
    for vid in view_ids:
        d = depths[vid]
        m = d > 0
        if m.any():
            naive.append(backproject(d, cams[vid])[m].reshape(-1, 3))
    naive = np.concatenate(naive, axis=0) if naive else np.zeros((0, 3))
    write_ply(os.path.join(out_folder, 'point_cloud_naive.ply'), naive)
    print('[A 朴素] 点数 = %d' % len(naive))

    # ---------------- (B) 光度(概率)滤波 ----------------
    filt = {}
    for vid in view_ids:
        d = depths[vid].copy()
        d[probs[vid] < args.prob_threshold] = 0
        filt[vid] = d
    prob_pts = []
    for vid in view_ids:
        d = filt[vid]
        m = d > 0
        if m.any():
            prob_pts.append(backproject(d, cams[vid])[m].reshape(-1, 3))
    prob_pts = np.concatenate(prob_pts, axis=0) if prob_pts else np.zeros((0, 3))
    write_ply(os.path.join(out_folder, 'point_cloud_prob_filtered.ply'), prob_pts)
    print('[B 概率滤波 p>=%.2f] 点数 = %d (占朴素 %.1f%%)'
          % (args.prob_threshold, len(prob_pts), 100.0 * len(prob_pts) / max(1, len(naive))))

    # ---------------- (C) 几何一致性 + 多视融合 ----------------
    fused_pts, fused_cols = [], []
    tot_cand = tot_keep = 0
    for vid in view_ids:
        d_ref = filt[vid]
        cam_ref = cams[vid]
        h, w = d_ref.shape
        yy, xx = np.nonzero(d_ref > 0)
        if len(yy) == 0:
            continue
        dvals = d_ref[yy, xx]
        X_ref = backproject_pixels(xx, yy, dvals, cam_ref)       # (N,3) 参考视图点云
        sum_pts = X_ref.copy()                                   # 累加各一致视图的 3D 点
        cnt = np.ones(len(dvals), dtype=np.int32)                # 自身算 1 个视图

        for sid in pairs.get(vid, []):
            if sid not in cams:
                continue
            uv, z, valid = project_world(X_ref, cams[sid])
            u = np.round(uv[:, 0] - 0.5).astype(np.int64)
            v = np.round(uv[:, 1] - 0.5).astype(np.int64)
            inside = valid & (u >= 0) & (u < w) & (v >= 0) & (v < h)
            idx = np.nonzero(inside)[0]
            if len(idx) == 0:
                continue
            d_src = filt[sid][v[idx], u[idx]]
            zs = z[idx]
            rel = np.abs(d_src - zs) / np.maximum(zs, 1e-6)
            px = np.hypot(uv[idx, 0] - (u[idx] + 0.5), uv[idx, 1] - (v[idx] + 0.5))
            good = (d_src > 0) & (rel < args.rel_depth_thresh) & (px < args.disp_thresh)
            gi = idx[good]
            if len(gi) == 0:
                continue
            # 用源视图自身深度反投影出该视图的 3D 点，再累加求平均
            X_src = backproject_pixels(u[gi], v[gi], d_src[good], cams[sid])
            sum_pts[gi] += X_src
            cnt[gi] += 1

        keep = cnt >= args.num_consistent
        tot_cand += len(dvals)
        tot_keep += int(keep.sum())
        if keep.any():
            fused_pts.append(sum_pts[keep] / cnt[keep][:, None].astype(np.float64))
            img_path = os.path.join(depth_folder, '%08d.jpg' % vid)
            if os.path.isfile(img_path):
                try:
                    import cv2
                    img = cv2.imread(img_path)
                    if img is not None:
                        sh = max(1, round(h / img.shape[0])) if img.shape[0] != h else 1
                        sy = img.shape[0] / float(h)
                        sx = img.shape[1] / float(w)
                        iy = np.clip((yy[keep] * sy).astype(int), 0, img.shape[0] - 1)
                        ix = np.clip((xx[keep] * sx).astype(int), 0, img.shape[1] - 1)
                        fused_cols.append(img[iy, ix][:, ::-1])
                except Exception:
                    pass
    fused = np.concatenate(fused_pts, axis=0) if fused_pts else np.zeros((0, 3))
    cols = np.concatenate(fused_cols, axis=0) if (fused_cols and len(fused) == sum(len(c) for c in fused_cols)) else None
    write_ply(os.path.join(out_folder, 'point_cloud_fused.ply'), fused, cols)
    print('[C 几何一致性+融合] 点数 = %d (候选 %d, 保留 %.1f%%)'
          % (len(fused), tot_cand, 100.0 * tot_keep / max(1, tot_cand)))

    np.save(os.path.join(out_folder, 'stats.npy'),
            np.array([len(naive), len(prob_pts), len(fused), tot_cand, tot_keep]))
    print('输出目录:', out_folder)


if __name__ == '__main__':
    main()
