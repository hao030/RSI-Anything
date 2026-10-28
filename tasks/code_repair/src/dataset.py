import json
from cure_wm import TestCase

def load_rows(path):
    rows = json.loads(path.read_text())
    result = {str(row['task_id']): row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f'Duplicate task IDs in {path}')
    return result

def tests_from_row(row, visibility):
    if visibility == 'public':
        inputs, outputs = (row['example_input'], row['example_output'])
    elif visibility == 'hidden':
        inputs, outputs = (row['test_input'], row['test_output'])
    else:
        raise ValueError(visibility)
    if len(inputs) != len(outputs):
        raise ValueError(f"Mismatched test input/output for {row['task_id']}")
    return tuple((TestCase(str(i), str(o), f'{visibility}_{j}') for j, (i, o) in enumerate(zip(inputs, outputs))))
