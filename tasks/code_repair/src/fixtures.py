from cure_wm import Execution, TestResult

def execution_from_dict(payload):
    return Execution(tuple((TestResult(**r) for r in payload['results'])), payload['wall_seconds'])
