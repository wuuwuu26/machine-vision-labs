#!/usr/bin/env python
# 用 open3d 查看 / 对比 fusibile 融合点云（PPT 步骤3：观察参数对点云的影响）
# 注意：交互窗口需要本机有 X 显示器；用 --render 可离屏渲染成 PNG（无需 GUI）。
# 用法示例：
#   python view_pointcloud.py --file result_param_sweep/prob0.8.ply
#   python view_pointcloud.py --dir result_param_sweep --all
#   python view_pointcloud.py --dir result_param_sweep --all --render out.png
import os, sys, argparse
import open3d as o3d

def load_ply(path):
    pc = o3d.io.read_point_cloud(path)
    if not pc.has_points():
        print("WARN: 读取为空", path)
    return pc

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default='/home/user/rvclab2/result_param_sweep',
                    help='含多个 .ply 的目录（配合 --all）')
    ap.add_argument('--file', default=None, help='单个 .ply 路径')
    ap.add_argument('--all', action='store_true', help='把 --dir 下全部 .ply 同屏显示')
    ap.add_argument('--render', default=None, help='离屏渲染成 PNG 后退出（无需 GUI）')
    args = ap.parse_args()

    pcs = []
    if args.file:
        pcs.append(load_ply(args.file))
    elif args.all:
        fs = sorted(f for f in os.listdir(args.dir) if f.lower().endswith('.ply'))
        colors = [[0.85,0.35,0.30],[0.30,0.65,0.85],[0.40,0.80,0.40],
                  [0.80,0.65,0.25],[0.60,0.40,0.80],[0.25,0.80,0.70],
                  [0.85,0.45,0.70],[0.55,0.55,0.55]]
        for i, f in enumerate(fs):
            pc = load_ply(os.path.join(args.dir, f))
            pc.paint_uniform_color(colors[i % len(colors)])
            pcs.append(pc)
        print("加载 %d 个点云：" % len(fs), fs)
    else:
        # 默认：最新的 final3d_model.ply
        import glob
        cand = sorted(glob.glob('/home/user/rvclab2/dtu_test_scan9/scan9/points_mvsnet/**/final3d_model.ply', recursive=True)
                      or glob.glob('/home/user/rvclab2/dtu_test_scan9/scan9/points_mvsnet/*.ply'),
                      key=os.path.getmtime, reverse=True)
        if not cand:
            print("没找到任何 ply，请用 --file / --all 指定"); sys.exit(1)
        pcs.append(load_ply(cand[0]))
        print("默认打开：", cand[0])

    if args.render:
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False)
        for pc in pcs:
            vis.add_geometry(pc)
        vis.poll_events(); vis.update_renderer()
        vis.capture_screen_image(args.render)
        vis.destroy_window()
        print("已离屏渲染 ->", args.render)
        return

    o3d.visualization.draw_geometries(pcs,
        window_name="fusibile point cloud",
        mesh_show_wireframe=False, point_show_normal=False)

if __name__ == '__main__':
    main()
