# -*- coding: utf-8 -*-
"""编译 XeLaTeX 版报告：xelatex 两次（交叉引用与总页数需要第二次）。

用法:  python build_tex.py
输入:  report.tex（图片由 \\includegraphics 直接引用 ../results/figs/*.png）
输出:  report.pdf（同时清理 aux/log/out）
"""
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TEX = os.path.join(HERE, 'report.tex')


def find_xelatex():
    for name in ('xelatex', 'xelatex.exe'):
        p = shutil.which(name)
        if p:
            return p
    for p in (r'C:\texlive\2026\bin\windows\xelatex.exe',
              r'C:\texlive\2025\bin\windows\xelatex.exe'):
        if os.path.exists(p):
            return p
    raise SystemExit('找不到 xelatex，请先安装 TeX Live / MiKTeX 并加入 PATH')


def main():
    xelatex = find_xelatex()
    for i in range(2):
        r = subprocess.run([xelatex, '-interaction=nonstopmode', '-halt-on-error', 'report.tex'],
                           cwd=HERE, capture_output=True, text=True,
                           encoding='utf-8', errors='ignore')
        if r.returncode != 0:
            log = os.path.join(HERE, 'report.log')
            if os.path.exists(log):
                lines = open(log, encoding='utf-8', errors='ignore').read().splitlines()
                for k, line in enumerate(lines):
                    if line.startswith('!'):
                        print('\n'.join(lines[k:k + 8]))
                        break
            raise SystemExit(f'xelatex 第 {i + 1} 次编译失败，详见 report.log')
        print(f'[OK] xelatex pass {i + 1}')

    log = open(os.path.join(HERE, 'report.log'), encoding='utf-8', errors='ignore').read()
    for line in log.splitlines():
        if line.startswith('Overfull \\hbox'):
            print('[WARN]', line)
    for line in log.splitlines():
        if 'Output written' in line:
            print(line)
    for ext in ('.aux', '.out', '.toc'):
        p = os.path.join(HERE, 'report' + ext)
        if os.path.exists(p):
            os.remove(p)


if __name__ == '__main__':
    main()
