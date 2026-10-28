import os
import sys
import tempfile
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / 'src'))
sys.path.insert(0, str(REPOSITORY / 'vendor/jepa-anything-core/src'))
_run = tempfile.TemporaryDirectory(prefix='rsi-anything-tests-')
os.environ['RSI_GUI_ROOT'] = _run.name
os.environ.pop('RSI_GUI_CONFIG', None)
