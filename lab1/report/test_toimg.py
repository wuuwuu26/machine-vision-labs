# -*- coding: utf-8 -*-
"""用极简文档验证 doc_to_image 服务是否可用。"""
import json
import subprocess
import sys

SKILL = r'C:\Program Files\WorkBuddy\resources\app.asar.unpacked\resources\plugins\workbuddy-builtin\skills\tencent-local-office-edit'
PY = r'C:\Users\user\.workbuddy\binaries\python\versions\3.13.12\python.exe'


def call(tool, **kw):
    cmd = [PY, 'edsdk.py', 'call', tool] + ['%s=%s' % (k, v) for k, v in kw.items()]
    r = subprocess.run(cmd, cwd=SKILL, capture_output=True, text=True, encoding='utf-8')
    o = (r.stdout or '').strip()
    return o


fid = sys.argv[1]
print('insert:', call('doc_insert_markdown', file_id=fid, idx=0,
                      markdown='# 测试标题\n\n这是一段用于验证转图服务的测试文字。')[:300])
print('save  :', call('save_file', file_id=fid,
                      file_path='C:/Users/user/Documents/workbuddy/report/preview/t.docx')[:300])
print('toimg :', call('doc_to_image',
                      file_path='C:/Users/user/Documents/workbuddy/report/preview/t.docx',
                      output_dir='C:/Users/user/Documents/workbuddy/report/preview')[:600])
