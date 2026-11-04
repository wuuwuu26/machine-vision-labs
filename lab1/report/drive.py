# -*- coding: utf-8 -*-
"""按顺序把报告各片段与插图写入 editor_sdk 中新建的文档。"""
import json
import os
import subprocess
import sys

SKILL = r'C:\Program Files\WorkBuddy\resources\app.asar.unpacked\resources\plugins\workbuddy-builtin\skills\tencent-local-office-edit'
PY = r'C:\Users\user\.workbuddy\binaries\python\versions\3.13.12\python.exe'
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
PARTS = os.path.join(ROOT, 'report', 'parts')
FIGS = os.path.join(ROOT, 'report', 'figs')
FILE_ID = sys.argv[1]


def call(tool, **kw):
    cmd = [PY, 'edsdk.py', 'call', tool]
    for k, v in kw.items():
        cmd.append('%s=%s' % (k, v))
    r = subprocess.run(cmd, cwd=SKILL, capture_output=True, text=True, encoding='utf-8')
    out = (r.stdout or '').strip()
    if not out:
        raise SystemExit('空输出 %s: %s' % (tool, r.stderr))
    # 取最后一行 JSON
    line = [l for l in out.splitlines() if l.strip().startswith('{')]
    d = json.loads(line[-1]) if line else {'raw': out}
    if 'error' in d or d.get('success') is False:
        raise SystemExit('%s 失败: %s' % (tool, out[:500]))
    return d


def end_pos():
    d = call('doc_get_last_operable_pos', file_id=FILE_ID)
    return int(d['position'])


def md(name):
    p = os.path.join(PARTS, name + '.md')
    d = call('doc_insert_markdown', file_id=FILE_ID, idx=end_pos(),
             markdown='file://' + p.replace('\\', '/'))
    print('  md  %-9s -> pos %s' % (name, d.get('position')))
    return d


def img(png, w):
    """插入图片；先把源图缩到 2 倍显示宽度再嵌，控制 docx 体积。"""
    from PIL import Image
    src = os.path.join(FIGS, png)
    im = Image.open(src)
    h = int(round(w * im.height / im.width))
    opt_dir = os.path.join(ROOT, 'report', 'figs_opt')
    os.makedirs(opt_dir, exist_ok=True)
    tw = min(im.width, w * 2)
    im2 = im.resize((tw, int(round(im.height * tw / im.width))), Image.LANCZOS) if tw < im.width else im
    dst = os.path.join(opt_dir, png)
    im2.save(dst, optimize=True)
    print('     (%d KB -> %d KB)' % (os.path.getsize(src) // 1024, os.path.getsize(dst) // 1024))
    d = call('doc_insert_image', file_id=FILE_ID, idx=end_pos(),
             image_path=dst.replace('\\', '/'), w=w, h=h)
    print('  img %-24s -> %d x %d' % (png, w, h))
    return d


STEPS = [
    ('md', 'part1'),
    ('md', 'part2'),
    ('img', 'fig0_depth_error.png', 550),
    ('md', 'fig1_cap'),
    ('md', 'part3'),
    ('md', 'part4'),
    ('img', 'fig1_resolution.png', 550),
    ('md', 'fig2_cap'),
    ('md', 'part5'),
    ('md', 'part6'),
    ('img', 'fig2_pointcloud.png', 550),
    ('md', 'fig3_cap'),
    ('md', 'part7'),
]

for s in STEPS:
    if s[0] == 'md':
        md(s[1])
    else:
        img(s[1], s[2])

print('doc length pos =', end_pos())
