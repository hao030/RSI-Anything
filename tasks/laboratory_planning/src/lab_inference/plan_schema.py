from __future__ import annotations
import ast
import json
import re
from dataclasses import dataclass
from typing import Any
FIELDS = {'id', 'context', 'goal', 'constraints', 'available_inputs', 'action_pool'}

@dataclass
class Argument:
    kind: str
    value: Any

@dataclass
class Step:
    output: str | None
    action: str
    inputs: dict[str, Argument]

def parse_argument(node: ast.AST) -> Argument:
    if isinstance(node, ast.Name):
        return Argument('name', node.id)
    if isinstance(node, ast.Constant):
        return Argument('literal', node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub) and isinstance(node.operand, ast.Constant):
        value = node.operand.value
        if isinstance(value, (int, float)):
            return Argument('literal', -value)
    return Argument('expression', ast.unparse(node))

def parse_steps(code: str) -> list[Step]:
    tree = ast.parse(code)
    steps: list[Step] = []
    for statement in tree.body:
        output: str | None = None
        call: ast.Call | None = None
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1 and isinstance(statement.targets[0], ast.Name):
            output = statement.targets[0].id
            if isinstance(statement.value, ast.Call):
                call = statement.value
        elif isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call):
            call = statement.value
        if call is None or not isinstance(call.func, ast.Name):
            continue
        if call.args:
            raise ValueError('Only keyword arguments are supported in the action sequence.')
        inputs = {keyword.arg: parse_argument(keyword.value) for keyword in call.keywords if keyword.arg}
        steps.append(Step(output=output, action=call.func.id, inputs=inputs))
    return steps

def extract_python_steps(text: str) -> str:
    fenced = re.findall('```(?:python)?\\s*(.*?)```', text, flags=re.IGNORECASE | re.DOTALL)
    for block in fenced:
        candidate = block.strip()
        if not candidate:
            continue
        try:
            if parse_steps(candidate):
                return candidate
        except SyntaxError:
            continue
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if re.match('^[A-Za-z_]\\w*\\s*=', stripped) or re.match('^[A-Za-z_]\\w*\\(', stripped):
            lines.append(stripped)
    candidate = '\n'.join(lines)
    if candidate:
        parse_steps(candidate)
    return candidate

def raw_input_names(available_inputs: str | None) -> set[str]:
    """Names of raw inputs from a Level 2 row's ``available_inputs`` JSON string.

    Mirrors ``evaluation.level2.prompts.raw_input_names`` but takes the raw
    string directly so this module stays free of the datasets dependency.
    """
    try:
        items = json.loads(available_inputs or '[]')
    except json.JSONDecodeError:
        return set()
    return {str(item.get('name')) for item in items if isinstance(item, dict) and item.get('name')}

def _latest_prior_producer(steps: list[Step], name: str, step_index: int) -> int | None:
    """Return the latest earlier step assigning a name.

    Action plans commonly reuse a handle name for an in-place update. Resolving
    against the last assignment in the whole program incorrectly labels that
    pattern as a forward reference.
    """
    for idx in range(step_index - 1, -1, -1):
        if steps[idx].output == name:
            return idx
    return None

def _argument_names(arg: Argument) -> set[str]:
    """Find variable reads in both simple and nested action arguments."""
    if arg.kind == 'literal':
        return set()
    if arg.kind == 'name':
        return {str(arg.value)}
    if arg.kind != 'expression':
        return set()
    tree = ast.parse(str(arg.value), mode='eval')
    return {node.id for node in ast.walk(tree)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)}

def is_valid_sequence(pred: list[Step], *, raw_inputs: set[str], action_pool_names: set[str] | None) -> tuple[bool, dict[str, Any]]:
    """I_Valid gate: actions in pool, no undefined / forward-referenced vars."""
    all_outputs = {step.output for step in pred if step.output}
    unknown_actions = sorted({step.action for step in pred if action_pool_names and step.action not in action_pool_names})
    unknown_vars: list[str] = []
    future_vars: list[str] = []
    for idx, step in enumerate(pred):
        for arg in step.inputs.values():
            for name in _argument_names(arg):
                if name in raw_inputs or _latest_prior_producer(pred, name, idx) is not None:
                    continue
                if name in all_outputs:
                    future_vars.append(name)
                else:
                    unknown_vars.append(name)
    valid = not (unknown_actions or unknown_vars or future_vars)
    return (valid, {'unknown_actions': unknown_actions, 'unknown_vars': sorted(set(unknown_vars)),
                   'future_generated_vars': sorted(set(future_vars))})

def code(text):
    fenced = re.search('```(?:python)?\\s*(.*?)```', text, re.IGNORECASE | re.DOTALL)
    if fenced:
        return fenced.group(1).strip()
    opened = re.search('```(?:python)?\\s*', text, re.IGNORECASE)
    return text[opened.end():].strip() if opened else text.strip()
