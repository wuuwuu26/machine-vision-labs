#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 解析二进制 PLY（兼容 float/uchar 混合属性），计算两档点云的真实指标：
#   点数 / 包围盒 / 质心距 std / 离群比例(>5xMAD)
import sys, json
import numpy as np

TYPEMAP = {'float':'f4','float32':'f4','double':'f8','float64':'f8',
           'uchar':'u1','uint8':'u1','char':'i1','int':'i4','int32':'i4'}

def read_ply(path):
    with open(path,'rb') as f:
        assert f.readline().strip()==b'ply'
        fmt=None; n=0; props=[]
        while True:
            line=f.readline()
            s=line.strip().decode('utf-8')
            if s.startswith('format'): fmt=s.split()[1]
            elif s.startswith('element vertex'): n=int(s.split()[2])
            elif s.startswith('property'):
                p=s.split(); props.append((p[2], TYPEMAP[p[1]]))
            elif s=='end_header': break
        assert fmt in ('binary_little_endian','binary_big_endian')
        endian='<' if 'little' in fmt else '>'
        dtype=np.dtype({'names':[p[0] for p in props],
                        'formats':[endian+t[1] for t in props]})
        arr=np.fromfile(f, dtype=dtype, count=n)
    xyz=np.column_stack([arr['x'],arr['y'],arr['z']]).astype(np.float64)
    return xyz

def metrics(xyz):
    n=len(xyz)
    bbox_min=xyz.min(0); bbox_max=xyz.max(0)
    size=bbox_max-bbox_min
    centroid=xyz.mean(0)
    d=np.linalg.norm(xyz-centroid,axis=1)
    med=np.median(d); mad=np.median(np.abs(d-med))
    thr=5*mad
    outlier=float((d>thr).sum())/n*100
    return dict(n=int(n),
                bbox=[round(float(v),1) for v in size],
                centroid=[round(float(v),1) for v in centroid],
                dist_std=round(float(d.std()),1),
                dist_median=round(float(med),1),
                outlier_pct=round(outlier,3))

B='/home/user/rvclab2/dtu_test_scan9/scan9/points_mvsnet/point_cloud_prob_filtered.ply'
C='/home/user/rvclab2/dtu_test_scan9/scan9/points_mvsnet/consistencyCheck-20261005-214159/final3d_model.ply'

mb=metrics(read_ply(B)); mc=metrics(read_ply(C))
out={'B_prob_filtered':mb,'C_fused_original':mc,
     'B_to_C_reduction_pct':round((mb['n']-mc['n'])/mb['n']*100,1)}
print(json.dumps(out,ensure_ascii=False,indent=2))
