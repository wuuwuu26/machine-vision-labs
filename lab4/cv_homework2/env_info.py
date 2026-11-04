"""把实验环境信息写入 results/env.json，供报告引用。"""
import os
import json
import platform
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, 'results')
os.makedirs(RES, exist_ok=True)

info = {'python': sys.version.split()[0], 'platform': platform.platform()}
try:
    import torch
    import torchvision
    info['torch'] = torch.__version__
    info['torchvision'] = torchvision.__version__
    info['cuda_available'] = torch.cuda.is_available()
    info['cuda_version'] = torch.version.cuda
    if torch.cuda.is_available():
        info['gpu'] = torch.cuda.get_device_name(0)
        info['gpu_mem_gb'] = round(torch.cuda.get_device_properties(0).total_memory / 1024 ** 3, 1)
except Exception as e:  # noqa
    info['error'] = str(e)
try:
    r = subprocess.run(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'],
                       capture_output=True, text=True, timeout=10)
    info['driver'] = r.stdout.strip()
except Exception:
    pass
json.dump(info, open(os.path.join(RES, 'env.json'), 'w'), indent=2, ensure_ascii=False)
print(json.dumps(info, ensure_ascii=False))
