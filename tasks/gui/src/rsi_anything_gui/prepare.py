import json
from .common import ROOT
from .experiment import collect_initial
if __name__=='__main__':
    protocol=json.loads((ROOT/'protocol.json').read_text())
    collect_initial(protocol['splits']['terminal_calibration'],ROOT/'data/terminal_calibration.jsonl')
