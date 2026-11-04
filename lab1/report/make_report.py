# -*- coding: utf-8 -*-
"""一键重建报告：生成片段 -> 新建文档 -> 写入内容与插图 -> 设样式 -> 保存 -> 估算页数。"""
import json
import os
import re
import subprocess
import sys
import zipfile

PY = r'C:\Users\user\.workbuddy\binaries\python\versions\3.13.12\python.exe'
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..'))
SKILL = r'C:\Program Files\WorkBuddy\resources\app.asar.unpacked\resources\plugins\workbuddy-builtin\skills\tencent-local-office-edit'

LINE_SPACING = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
OUT = os.path.join(HERE, 'preview', 'report.docx')
FINAL = os.path.join(ROOT, '单目深度与点云重建-实验报告.docx')


def sh(args):
    r = subprocess.run(args, capture_output=True, text=True, encoding='utf-8')
    return ((r.stdout or '') + (r.stderr or '')).strip()


def edsdk_raw(args):
    r = subprocess.run(args, cwd=SKILL, capture_output=True, text=True, encoding='utf-8')
    return ((r.stdout or '') + (r.stderr or '')).strip()


def edsdk(tool, **kw):
    cmd = [PY, 'edsdk.py', 'call', tool] + ['%s=%s' % (k, v) for k, v in kw.items()]
    r = subprocess.run(cmd, cwd=SKILL, capture_output=True, text=True, encoding='utf-8')
    out = (r.stdout or '').strip()
    if 'error' in out.lower() and 'JSON-RPC' in out:
        raise SystemExit('%s -> %s' % (tool, out[:400]))
    return out


os.makedirs(os.path.join(HERE, 'preview'), exist_ok=True)

print('== 1 生成正文片段 ==')
print(sh([PY, os.path.join(HERE, 'build_parts.py')]).splitlines()[-1])

print('== 2 新建文档 ==')
out = edsdk('create_doc')
fid = re.search(r'file_id=(\S+?),', out).group(1)
print('   file_id =', fid)

print('== 3 写入正文与插图 ==')
print(sh([PY, os.path.join(HERE, 'drive.py'), fid]))

print('== 4 设置文档样式（行距 %.2f）==' % LINE_SPACING)
args = {'file_id': fid,
        'default_text_style': {'font_family': 'SimSun', 'font_size': 10.5},
        'default_paragraph_style': {'line_spacing': LINE_SPACING, 'line_spacing_rule': 1},
        'page_style': {'top_margin': 54, 'bottom_margin': 54,
                       'left_margin': 54, 'right_margin': 54}}
# 注：editor_sdk 默认页面是 Letter；A4 由 beautify.py 的 set_page_a4() 在后处理改
p = os.path.join(HERE, 'style_args.json')
json.dump(args, open(p, 'w', encoding='utf-8'), ensure_ascii=False)
print('  ', edsdk_raw([PY, 'edsdk.py', 'call', 'doc_set_document_style',
                      '--json-file', p]))

print('== 5 保存 ==')
print('  ', edsdk('save_file', file_id=fid,
                  file_path=OUT.replace('\\', '/'))[:120])
print('  ', edsdk('save_file', file_id=fid,
                  file_path=FINAL.replace('\\', '/'))[:120])
for f in (OUT, FINAL):
    print('   %8.2f MB  %s' % (os.path.getsize(f) / 1048576, f))

print('== 5.5 格式润色（标题/节标题/表图题/表头底纹）==')
print('  ', sh([PY, os.path.join(HERE, 'beautify.py'), OUT]))
print('  ', sh([PY, os.path.join(HERE, 'beautify.py'), FINAL]))

z = zipfile.ZipFile(OUT)
s = z.read('word/styles.xml').decode('utf-8')
i = s.find('docDefaults')
print('   docDefaults:', s[i:i + 330])
print('   sectPr    :', re.search(r'<w:sectPr.*?</w:sectPr>',
                                  z.read('word/document.xml').decode('utf-8'), re.S).group(0)[:160])

print('== 6 页数估算 ==')
print(sh([PY, os.path.join(HERE, 'check_docx.py'), OUT]).splitlines()[-2:][0])
print(sh([PY, os.path.join(HERE, 'check_docx.py'), OUT]).splitlines()[-1])
