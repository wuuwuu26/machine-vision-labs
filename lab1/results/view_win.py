"""查看点云（WSL 内无 /dev/dri，open3d 开不了窗口，所以放 Windows 侧跑）。

放在哪个 results 目录下，就默认读哪个目录，不用再写 --root。
目录名两套写法自动互认：results_518 = depth_518，out280 = depth_280，
results_metric_518 = depth_metric_518。

注意：8 张图是 8 个彼此独立的场景，一个窗口默认只显示其中一张。
想看全部 8 张用 --all，会把它们排成网格。

取景不用手动缩放：脚本把点云投影到屏幕上量出实际占比，再迭代逼近合适的
zoom，所以场景多大、窗口多大都能自动装满。

用法（Windows 终端里）：
    python view_win.py                              # 单块：默认 nyu_0001
    python view_win.py --all                        # 8 张图排成网格
    python view_win.py --all --cols 3               # 指定列数（不指定则按窗口比例自动选）
    python view_win.py --sid nyu_0201               # 换一张图
    python view_win.py --gt                         # 看真值点云
    python view_win.py --folder depth_280           # 换目录
    python view_win.py --compare                    # 并排对比（自动包含所有存在的目录）
    python view_win.py --compare depth_518:pred depth_518:gt
                                                    # 目录名后可加 :gt / :pred 指定看哪份
    python view_win.py --shot 图.png                # 不交互，直接存一张截图
    python view_win.py --check                      # 只加载不弹窗，验证路径通不通
"""
import argparse
import os
import sys

import numpy as np
import open3d as o3d

# 两套目录名的互认表
ALIAS = {
    'results_518': 'depth_518', 'depth_518': 'results_518',
    'out280': 'depth_280', 'depth_280': 'out280',
    'results_metric_518': 'depth_metric_518',
    'depth_metric_518': 'results_metric_518',
}
CANON = ['results_518', 'out280', 'results_metric_518']
GAP = 0.3       # 网格/并排时各块之间留的空隙（米）

ap = argparse.ArgumentParser()
ap.add_argument('--sid', default='nyu_0001', help='图片编号，如 nyu_0001')
ap.add_argument('--all', action='store_true', help='把该目录下所有图片排成网格显示')
ap.add_argument('--cols', type=int, default=None,
                help='网格列数。不指定则自动选一个最贴合窗口长宽比的')
ap.add_argument('--folder', default=None, help='点云所在目录，如 depth_518 / results_518')
ap.add_argument('--gt', action='store_true', help='改看真值点云')
ap.add_argument('--compare', nargs='*', default=None,
                help='并排显示。不带值 = 自动包含所有存在的目录；'
                     '也可显式给列表，目录名后可用 :gt / :pred 指定，'
                     '如 --compare depth_518:pred depth_518:gt')
ap.add_argument('--root', default=None, help='结果目录，默认 = 本脚本所在目录')
ap.add_argument('--shot', default=None, help='渲染一张截图存到该路径，然后直接退出')
ap.add_argument('--width', type=int, default=1600, help='窗口/截图宽度，默认 1600')
ap.add_argument('--height', type=int, default=900, help='窗口/截图高度，默认 900')
ap.add_argument('--fill', type=float, default=0.90,
                help='画面填充比例，默认 0.90（留 10%% 边距）')
ap.add_argument('--check', action='store_true', help='只加载不弹窗')
args = ap.parse_args()

if args.root is None:
    args.root = os.path.dirname(os.path.abspath(__file__))


def resolve(folder):
    """把 results_518 / depth_518 这类别名解析成本目录下真实存在的名字。"""
    if os.path.isdir(os.path.join(args.root, folder)):
        return folder
    alt = ALIAS.get(folder)
    if alt and os.path.isdir(os.path.join(args.root, alt)):
        return alt
    return folder


def list_sids(folder, suffix):
    d = os.path.join(args.root, resolve(folder))
    tail = suffix + '.ply'
    return sorted(f[:-len(tail)] for f in os.listdir(d) if f.endswith(tail))


def load(spec, sid=None, suffix=None):
    """spec 形如 'depth_518' 或 'depth_518:gt'。"""
    folder, _, hinted = spec.partition(':')
    if hinted:
        suffix = '_' + hinted
    elif suffix is None:
        suffix = '_gt' if args.gt else '_pred'
    sid = sid or args.sid

    folder = resolve(folder)
    path = os.path.join(args.root, folder, sid + suffix + '.ply')
    if not os.path.isfile(path):
        here = sorted(d for d in os.listdir(args.root)
                      if os.path.isdir(os.path.join(args.root, d)))
        sys.exit('找不到文件：%s\n该目录下现有子目录：%s' % (path, ', '.join(here) or '(空)'))
    p = o3d.io.read_point_cloud(path)
    print('已加载 %-46s %d 点'
          % (folder + '\\' + sid + suffix + '.ply', len(p.points)))
    return p


def existing():
    return [f for f in (resolve(c) for c in CANON)
            if os.path.isdir(os.path.join(args.root, f))]


def extent(p):
    pts = np.asarray(p.points)
    return (pts[:, 0].max() - pts[:, 0].min(),
            pts[:, 1].max() - pts[:, 1].min())


# ---------------------------------------------------------------- 取景

def _screen_frac(vc, pts):
    """把世界坐标点投到屏幕，返回占画面的最大比例（宽/高取大者）。

    有点落在相机后方时返回 None，交给调用方判断是"太近了"。
    """
    p = vc.convert_to_pinhole_camera_parameters()
    K = np.asarray(p.intrinsic.intrinsic_matrix)
    cam = (np.asarray(p.extrinsic) @ np.c_[pts, np.ones(len(pts))].T).T[:, :3]
    front = cam[:, 2] > 1e-6
    if front.sum() < len(pts) * 0.99:
        return None
    cam = cam[front]
    u = K[0, 0] * cam[:, 0] / cam[:, 2] + K[0, 2]
    v = K[1, 1] * cam[:, 1] / cam[:, 2] + K[1, 2]
    return max((u.max() - u.min()) / p.intrinsic.width,
               (v.max() - v.min()) / p.intrinsic.height)


def frame_scene(vis, clouds, fill):
    """把整个场景装进画面。

    要点有两个：

    1. 量的是**实际的点**，不是包围盒的角点。点云在包围盒里是松的（相机
       前方整个深度范围都被框住），照盒子取景会缩得很小。
    2. zoom 得迭代逼近，不能一步算出来。Open3D 的 set_zoom 在改视野的
       同时还会动相机距离，实测占比并不与 1/zoom 成正比。
    """
    vis.reset_view_point(True)
    vc = vis.get_view_control()

    lo = np.min([np.asarray(c.get_min_bound()) for c in clouds], axis=0)
    hi = np.max([np.asarray(c.get_max_bound()) for c in clouds], axis=0)

    pts = np.vstack([np.asarray(c.points) for c in clouds])
    if len(pts) > 40000:
        idx = np.random.default_rng(0).choice(len(pts), 40000, replace=False)
        pts = pts[idx]

    vc.set_front([0.0, 0.0, -1.0])      # 相机看向 -z
    vc.set_up([0.0, -1.0, 0.0])         # 图像 y 轴向下，所以 up 取 -y
    vc.set_lookat(((lo + hi) / 2).tolist())

    z = 1.0
    for _ in range(16):
        vc.set_zoom(z)
        vis.poll_events()
        vis.update_renderer()
        frac = _screen_frac(vc, pts)
        if frac is None:                # 点跑到相机后方了，说明太近
            z *= 1.4
            continue
        if abs(frac - fill) < 0.008:
            break
        z *= (frac / fill) ** 0.7       # 按比例修正并加阻尼


# ---------------------------------------------------------------- 排布

if args.all:
    folder = args.folder or (existing() or ['results_518'])[0]
    suffix = '_gt' if args.gt else '_pred'
    sids = list_sids(folder, suffix)
    if not sids:
        sys.exit('目录里没有 %s 结尾的点云：%s' % (suffix + '.ply', folder))

    clouds = [load(folder, sid=s, suffix=suffix) for s in sids]
    exts = [extent(p) for p in clouds]
    W = max(e[0] for e in exts)
    H = max(e[1] for e in exts)

    if args.cols:                       # 指定了就用指定的
        cols = args.cols
    else:                               # 否则挑一个最贴合窗口长宽比的列数
        want = args.width / args.height
        cols = min(range(1, len(clouds) + 1),
                   key=lambda c: abs(np.log(
                       (c * (W + GAP)) / (-(-len(clouds) // c) * (H + GAP)) / want)))
        print('自动选定列数：%d' % cols)

    for i, p in enumerate(clouds):
        row, col = divmod(i, cols)
        c = p.get_center()
        # y 轴向下为正（up 取 -y），所以第 0 行放 y=0，往下递增
        p.translate((col * (W + GAP) - c[0], row * (H + GAP) - c[1], 0))

    title = '%s  %d 张  %s' % (resolve(folder), len(sids), suffix[1:])

elif args.compare is not None:
    # 依次沿 x 排开，偏移量按各块自身宽度累加（点云有 4~5 米宽，
    # 用固定偏移量会互相穿插）
    specs = args.compare or existing() or ['results_518', 'out280']
    clouds = []
    cursor = 0.0
    for spec in specs:
        p = load(spec)
        w, _ = extent(p)
        c = p.get_center()
        p.translate((cursor + w / 2 - c[0], 0, 0))
        cursor += w + GAP
        clouds.append(p)
    title = args.sid + '：' + ' | '.join(specs)

else:
    folder = args.folder or (existing() or ['results_518'])[0]
    clouds = [load(folder)]
    title = args.sid + ' @ ' + resolve(folder)

if args.check:
    for c in clouds:
        pts = np.asarray(c.points)
        print('  x[%.2f, %.2f] y[%.2f, %.2f] z[%.2f, %.2f]  有颜色=%s'
              % (pts[:, 0].min(), pts[:, 0].max(),
                 pts[:, 1].min(), pts[:, 1].max(),
                 pts[:, 2].min(), pts[:, 2].max(),
                 c.has_colors()))
    print('共 %d 块点云，加载正常。去掉 --check 即可弹窗。' % len(clouds))
    sys.exit(0)

# ---------------------------------------------------------------- 渲染

vis = o3d.visualization.Visualizer()
if not vis.create_window(window_name=title, width=args.width, height=args.height):
    sys.exit('创建窗口失败')
for c in clouds:
    vis.add_geometry(c)

opt = vis.get_render_option()
opt.background_color = np.array([1.0, 1.0, 1.0])
opt.point_size = 1.5

frame_scene(vis, clouds, args.fill)

for _ in range(5):                      # 先渲染几帧，否则可能截到空白
    vis.poll_events()
    vis.update_renderer()

if args.shot:
    img = np.asarray(vis.capture_screen_float_buffer(False))
    # 用 open3d 自带的图像写入，避免依赖 matplotlib
    o3d.io.write_image(args.shot, o3d.geometry.Image((img * 255).astype(np.uint8)))
    print('已保存截图：' + args.shot)
    vis.destroy_window()
    sys.exit(0)

vis.run()                               # 交互：拖动旋转、滚轮缩放、Q 退出
vis.destroy_window()
