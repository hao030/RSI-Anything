"""Read-only environment report; no models are loaded."""
import importlib.metadata
import json
import shutil
import socket
import subprocess
import sys
from .settings import ASSETS, BASE_WORLD, BASE_POLICY, PACKAGE, PORT, SETTINGS

EXPECTED = {'torch': '2.5.1', 'transformers': '5.17.0', 'peft': '0.21.0', 'accelerate': '1.10.1',
            'safetensors': '0.8.0', 'numpy': '2.2.6', 'scipy': '1.15.3', 'playwright': '1.55.0',
            'tokenizers': '0.23.2', 'huggingface-hub': '1.32.0', 'matplotlib': '3.10.7'}


def doctor(check_gpu=False):
    checks = {'python_3_10': sys.version_info[:2] == (3, 10),
              'official_task': (ASSETS / 'miniwob/click-checkboxes.html').is_file(),
              'world_model_present': (BASE_WORLD / 'config.json').is_file(),
              'policy_model_present': (BASE_POLICY / 'config.json').is_file()}
    versions = {}
    for name, expected in EXPECTED.items():
        try:
            actual = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            actual = None
        versions[name] = actual
        checks[name] = actual is not None and actual.split('+')[0] == expected
    with socket.socket() as s:
        try:
            s.bind(('127.0.0.1', PORT))
            checks['policy_port_free'] = True
        except OSError:
            checks['policy_port_free'] = False
    if versions['playwright']:
        from playwright.sync_api import sync_playwright
        from pathlib import Path
        with sync_playwright() as p:
            checks['chromium_installed'] = Path(p.chromium.executable_path).is_file()
    gpu = None
    if check_gpu:
        checks['nvidia_smi'] = shutil.which('nvidia-smi') is not None
        if checks['nvidia_smi']:
            from .orchestrate import gpu_free
            plan = SETTINGS['resource_plan']
            ids = set(plan['workers'] + [plan[k] for k in ['policy_gpu', 'rsi_world_gpu', 'rsi_policy_gpu']])
            for device in sorted(ids):
                try:
                    gpu_free(device)
                    checks['gpu_' + str(device) + '_free'] = True
                except (RuntimeError, subprocess.CalledProcessError, ValueError):
                    checks['gpu_' + str(device) + '_free'] = False
            gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=index,name,memory.total,driver_version', '--format=csv,noheader'], text=True).splitlines()
        if versions['torch']:
            import torch
            checks['cuda_available'] = torch.cuda.is_available()
            checks['cuda_build_12_1'] = torch.version.cuda == '12.1'
    return {'passed': all(checks.values()), 'checks': checks, 'versions': versions, 'gpus': gpu,
            'model_hashes_checked': False, 'next': 'verify-models and actual browser/world preflight are required'}
