"""每个场景出一张三方点云对比图（518 | 280 | Metric），8 张拼成一个文件。

在 Windows 侧跑（WSL 里 open3d 开不了窗口）：

    D:\\codetool\\miniconda\\envs\\open3d_win\\python.exe shot_compare.py

产出（都放在 results/ 下）：
    点云三方差比.png        8 个场景竖着拼成一张，可直接插进报告
    点云对比/nyu_XXXX.png   每个场景单独一张
    点云三方差比.pdf        同一份内容的多页 PDF，每页一个场景

关于取景，有两个坑值得记一笔：

1. 三栏必须共用同一个相机。如果像早先那样把三块并排塞进一个窗口，
   透视投影会让左右两栏被斜着看，同一个场景的三栏形态就不一样了，
   读者分不清是模型差异还是视角差异。所以这里逐栏渲染，但把第一栏
   标定好的相机参数读出来、原样套给另外两栏。
   本机不支持 OffscreenRenderer（EGL Headless not supported），
   也拿不到正交投影（ViewControl 没有 set_projection_type），
   所以走"单窗口 + 复制相机"这条路。

2. 缩放不能拍脑袋写死，也不能拿包围盒的角点去算——点云在包围盒里是松
   的，照盒子取景会缩得很小。这里直接投影实际的点量出画面占比，再迭代
   逼近合适的 zoom（Open3D 的 set_zoom 同时会改相机距离，占比并不严格
   与 1/zoom 成正比，所以得量几次）。
"""
import argparse
import os
import sys

import numpy as np
import open3d as o3d
from PIL import Image, ImageDraw, ImageFont

ap = argparse.ArgumentParser()
ap.add_argument('--root', default=r'D:\大学文件\wps文档\大三上\机器视觉\results')
ap.add_argument('--out', default=None, help='输出目录，默认 = --root')
ap.add_argument('--with-gt', action='store_true', help='每行再加一栏真值')
ap.add_argument('--cell', type=int, default=620, help='单栏宽度（像素），默认 620')
ap.add_argument('--cell-h', type=int, default=470, help='单栏高度（像素），默认 470')
ap.add_argument('--point-size', type=float, default=1.5)
ap.add_argument('--fill', type=float, default=0.92, help='画面填充比例')
args = ap.parse_args()

OUT = args.out or args.root
PER_SCENE = os.path.join(OUT, '点云对比')

SIDS = ['nyu_0001', 'nyu_0101', 'nyu_0201', 'nyu_0401',
        'nyu_0601', 'nyu_0801', 'nyu_1001', 'nyu_1201']

# 三栏的顺序就是报告里三个任务的顺序
PANELS = [('depth_518', '_pred', '518'),
          ('depth_280', '_pred', '280'),
          ('depth_metric_518', '_pred', 'Metric')]
if args.with_gt:
    PANELS.insert(0, ('depth_518', '_gt', '真值'))


def load(folder, sid, suffix):
    path = os.path.join(args.root, folder, sid + suffix + '.ply')
    if not os.path.isfile(path):
        sys.exit('找不到文件：' + path)
    return o3d.io.read_point_cloud(path)


def union_bounds(clouds):
    lo = np.min([np.asarray(c.get_min_bound()) for c in clouds], axis=0)
    hi = np.max([np.asarray(c.get_max_bound()) for c in clouds], axis=0)
    return lo, hi


def scene_sample(clouds, n=40000, seed=0):
    """从整个场景里抽一批点，用来量画面占比。

    不能拿包围盒的 8 个角点去量：点云在包围盒里是松的（相机前方一整个
    深度范围都被框进去），角点撑满了盒子，实际的点只占其中一小块，
    照盒子取景会缩得很小。直接投影实际的点才准。
    """
    pts = np.vstack([np.asarray(c.points) for c in clouds])
    if len(pts) > n:
        idx = np.random.default_rng(seed).choice(len(pts), n, replace=False)
        pts = pts[idx]
    return pts


def screen_frac(vc, pts):
    """把世界坐标点投到屏幕，返回占画面的最大比例（宽/高取大者）。

    有点落在相机后方时返回 None，交给调用方判断是"太近了"。
    """
    p = vc.convert_to_pinhole_camera_parameters()
    K = np.asarray(p.intrinsic.intrinsic_matrix)
    cam = (np.asarray(p.extrinsic) @ np.c_[pts, np.ones(len(pts))].T).T[:, :3]
    front = cam[:, 2] > 1e-6
    if front.sum() < len(pts) * 0.99:        # 超过 1% 的点跑到相机后方
        return None
    cam = cam[front]
    u = K[0, 0] * cam[:, 0] / cam[:, 2] + K[0, 2]
    v = K[1, 1] * cam[:, 1] / cam[:, 2] + K[1, 2]
    return max((u.max() - u.min()) / p.intrinsic.width,
               (v.max() - v.min()) / p.intrinsic.height)


def fit_zoom(vis, vc, pts, fill):
    """迭代找出让包围盒刚好装满画面的 zoom。

    不能像一开始那样一步算出来。Open3D 的 set_zoom 在改视野的同时还会动
    相机距离，实测占比并不与 1/zoom 成正比（本机 620x470、nyu_0001 上，
    zoom=1.0 占比 0.86，zoom=3.0 占比 0.21，远不到 3 倍），而且 zoom 小到
    一定程度相机会钻进包围盒里。所以老老实实量几次逼近。

    占比随 zoom 单调递减，按比例修正并加阻尼即可稳定收敛。
    """
    z = 1.0
    for _ in range(16):
        vc.set_zoom(z)
        vis.poll_events()
        vis.update_renderer()
        frac = screen_frac(vc, pts)
        if frac is None:                    # 点跑到相机后方了，说明太近
            z *= 1.4
            continue
        if abs(frac - fill) < 0.008:
            break
        z *= (frac / fill) ** 0.7
    return z, frac


def render_scene(vis, clouds, lo, hi):
    """按公共包围盒取景，逐栏渲染。返回每栏一张 uint8 图。

    相机在渲染第一栏时定下来，之后原样复制给其余各栏，保证三栏视角完全
    一致——否则透视会让左右两栏被斜着看，读者分不清差异来自模型还是视角。
    """
    sample = scene_sample(clouds)
    shots = []
    cam_params = None

    for i, cloud in enumerate(clouds):
        vis.clear_geometries()
        vis.add_geometry(cloud)

        vc = vis.get_view_control()
        if cam_params is None:
            vis.reset_view_point(True)      # 让 Open3D 先把相机拉到合适距离
            vc.set_front([0.0, 0.0, -1.0])  # 相机看向 -z
            vc.set_up([0.0, -1.0, 0.0])     # 图像 y 轴向下，所以 up 取 -y
            vc.set_lookat(((lo + hi) / 2).tolist())
            zoom, frac = fit_zoom(vis, vc, sample, args.fill)
            print('    zoom=%.3f 占比=%.3f' % (zoom, frac))
            cam_params = vc.convert_to_pinhole_camera_parameters()
        else:
            vc.convert_from_pinhole_camera_parameters(cam_params, False)

        for _ in range(6):
            vis.poll_events()
            vis.update_renderer()
        img = np.asarray(vis.capture_screen_float_buffer(False))
        shots.append((img * 255).astype(np.uint8))

    vis.clear_geometries()
    return shots


def font(size):
    for name in ('msyh.ttc', 'msyhl.ttc', 'simhei.ttf'):
        try:
            return ImageFont.truetype(os.path.join(r'C:\Windows\Fonts', name), size)
        except OSError:
            continue
    return ImageFont.load_default()


def band(width, height, texts, size, color=(255, 255, 255)):
    """画一条横标签栏。texts 是 (文字, 中心x) 的列表。"""
    bar = Image.new('RGB', (width, height), color)
    d = ImageDraw.Draw(bar)
    f = font(size)
    for text, cx in texts:
        box = d.textbbox((0, 0), text, font=f)
        d.text((cx - (box[2] - box[0]) / 2, (height - (box[3] - box[1])) / 2 - box[1]),
               text, font=f, fill=(20, 20, 20))
    return np.asarray(bar)


def tag(shot, sid):
    """把场景名贴在左上角，加个白底方框免得压在点云上看不清。"""
    im = Image.fromarray(shot)
    d = ImageDraw.Draw(im)
    f = font(26)
    d.rectangle([8, 8, 8 + d.textlength(sid, font=f) + 18, 48], fill=(255, 255, 255))
    d.text((17, 14), sid, font=f, fill=(20, 20, 20))
    return im


def main():
    os.makedirs(PER_SCENE, exist_ok=True)
    strip_w = args.cell * len(PANELS)

    vis = o3d.visualization.Visualizer()
    if not vis.create_window(window_name='render', width=args.cell, height=args.cell_h):
        sys.exit('创建窗口失败')
    opt = vis.get_render_option()
    opt.background_color = np.array([1.0, 1.0, 1.0])
    opt.point_size = args.point_size

    rows = []
    pages = []
    for sid in SIDS:
        clouds = [load(folder, sid, suffix) for folder, suffix, _ in PANELS]
        lo, hi = union_bounds(clouds)
        shots = render_scene(vis, clouds, lo, hi)

        strip = np.hstack(shots)
        Image.fromarray(strip).save(os.path.join(PER_SCENE, sid + '.png'))
        rows.append(np.hstack([np.asarray(tag(s, sid)) if i == 0 else s
                               for i, s in enumerate(shots)]))
        pages.append(tag(shots[0], sid))
        print('  %s 已渲染' % sid)

    vis.destroy_window()

    head = band(strip_w, 54, [('点云三方对比', strip_w / 2)], 30)
    cols = [(name, (i + 0.5) * args.cell) for i, (_, _, name) in enumerate(PANELS)]
    colbar = band(strip_w, 44, cols, 26)

    total = np.vstack([head, colbar] + rows)
    big = os.path.join(OUT, '点云三方差比.png')
    Image.fromarray(total).save(big)
    print('\n合成图：%s  (%d x %d)' % (big, total.shape[1], total.shape[0]))

    pdf = os.path.join(OUT, '点云三方差比.pdf')
    pages[0].save(pdf, save_all=True, append_images=pages[1:], resolution=150)
    print('多页 PDF：%s' % pdf)
    print('单场景图：%s  (%d 张)' % (PER_SCENE, len(SIDS)))


main()
