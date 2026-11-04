#!/usr/bin/env bash
# 原版 MVSNet/depthfusion.py 参数扫描 —— num_consistent 扫描（对应报告 §3.3.4）
# 每次跑完把 fusibile 生成的 final3d_model.ply 改名存出，并记录点数与包围盒。
# fusibile 输出嵌套在 points_mvsnet/consistencyCheck-<ts>/final3d_model.ply，
# 故取该目录下“最新修改”的 final3d_model.ply 作为本次结果。
# 用法（本机终端，需 GPU）： bash /home/user/rvclab2/fusibile/run_param_sweep.sh
set -u

CUDA_ROOT=/home/user/cuda124/builds/cuda_nvcc
export LD_LIBRARY_PATH=$CUDA_ROOT/targets/x86_64-linux/lib64:$CUDA_ROOT/lib64:${LD_LIBRARY_PATH:-}
source /home/user/miniconda3/bin/activate mvsnet3
cd /home/user/rvclab2/MVSNet

DENSE=/home/user/rvclab2/dtu_test_scan9/scan9
FUSIBILE=/home/user/rvclab2/fusibile/build/fusibile
PTS=$DENSE/points_mvsnet
SWEEP=/home/user/rvclab2/result_param_sweep
mkdir -p "$SWEEP"
LOG=$SWEEP/sweep_log.csv
echo "tag,prob_threshold,disp_threshold,num_consistent,points,bbox_x,bbox_y,bbox_z" > "$LOG"

# 解析 PLY：点数 + 包围盒（按真实 property 布局读取 xyz，稳健）
analyze_ply () {
  /home/user/miniconda3/envs/mvsnet3/bin/python - "$1" <<'PY'
import sys, numpy as np
p=sys.argv[1]
n=None; props=[]
with open(p,'rb') as f:
    for _ in range(200):
        line=f.readline()
        if not line: break
        s=line.split()
        if s[:2]==[b'element',b'vertex']: n=int(s[2])
        if s[:1]==[b'property']: props.append((s[2].decode(), s[1].decode()))
        if line.strip()==b'end_header': break
    blob=f.read()
ts={'float':4,'uchar':1,'int':4,'uint':4,'short':2,'ushort':2,'double':8}
np_t={'float':'<f4','uchar':'u1','int':'<i4','uint':'<u4','short':'<i2','ushort':'<u2','double':'<f8'}
stride=sum(ts[t] for _,t in props)
dt=[(nm,np_t[t]) for nm,t in props]
try:
    arr=np.frombuffer(blob,dtype=np.dtype(dt,align=False))
    xs=arr['x'].astype(float); ys=arr['y'].astype(float); zs=arr['z'].astype(float)
    mn=[xs.min(),ys.min(),zs.min()]; mx=[xs.max(),ys.max(),zs.max()]
    print(f"{n},{mx[0]-mn[0]:.0f},{mx[1]-mn[1]:.0f},{mx[2]-mn[2]:.0f}")
except Exception:
    print(f"{n},,,,")
PY
}

latest_ply () {
  # 取 points_mvsnet 下最新修改的 final3d_model.ply
  find "$PTS" -name final3d_model.ply -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-
}

run_one () {
  local tag=$1 prob=$2 disp=$3 num=$4
  echo "=== [$tag] prob=$prob disp=$disp num=$num ==="
  python mvsnet/depthfusion.py --dense_folder "$DENSE" --fusibile_exe_path "$FUSIBILE" \
    --prob_threshold "$prob" --disp_threshold "$disp" --num_consistent "$num"
  local ply
  ply=$(latest_ply)
  if [ -z "$ply" ] || [ ! -f "$ply" ]; then
    echo "!! fusibile 未生成 final3d_model.ply（检查 GPU / LD_LIBRARY_PATH / fusibile 二进制），本组跳过"
    return 1
  fi
  cp "$ply" "$SWEEP/${tag}.ply"
  local info
  info=$(analyze_ply "$SWEEP/${tag}.ply")
  echo "$tag,$prob,$disp,$num,$info" >> "$LOG"
  echo "   -> $SWEEP/${tag}.ply  $info"
}

# §3.3.4 所需：num_consistent 扫描（固定 prob_threshold=0.8, disp_threshold=0.25）
run_one nc2 0.8 0.25 2
run_one nc3 0.8 0.25 3
run_one nc4 0.8 0.25 4
run_one nc5 0.8 0.25 5

# （可选）其它参数扫描：取消注释即可，产出可供后续分析
# run_one prob0.4 0.4 0.25 3
# run_one prob0.6 0.6 0.25 3
# run_one prob0.9 0.9 0.25 3
# run_one disp0.1 0.8 0.10 3
# run_one disp0.4 0.8 0.40 3

echo "==== 完成，结果见 $LOG ===="
echo "回填报告：在沙箱或本机重跑  python /home/user/rvclab2/exp2_make_docx.py  （脚本会自动读取 $LOG 写入 §3.3.4）"
