"""Normalize JEPA candidate plans and derive sampling seeds."""

import ast

import hashlib

import re

def code_from_completion(text: str) -> str:
    fenced = re.search('```(?:python)?\\s*(.*?)```', text, re.IGNORECASE | re.DOTALL)
    if fenced:
        return fenced.group(1).strip()
    open_fence = re.search('```(?:python)?\\s*', text, re.IGNORECASE)
    if open_fence:
        return text[open_fence.end():].strip()
    return text.strip()

def preprocess(text, action_pool):
    signatures = {n.name: [a.arg for a in n.args.posonlyargs + n.args.args] for n in ast.parse(action_pool).body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    lines = code_from_completion(text).splitlines()
    while lines:
        try:
            tree = ast.parse('\n'.join(lines))
            break
        except SyntaxError:
            lines.pop()
    else:
        return text
    changed = len(lines) != len(code_from_completion(text).splitlines())
    for statement in tree.body:
        call = statement.value if isinstance(statement, (ast.Assign, ast.Expr)) else None
        if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name) or (not call.args):
            continue
        params = signatures.get(call.func.id)
        existing = {kw.arg for kw in call.keywords}
        if params is None or len(call.args) > len(params) or any((n in existing for n in params[:len(call.args)])):
            continue
        call.keywords = [ast.keyword(arg=n, value=v) for n, v in zip(params, call.args)] + call.keywords
        call.args = []
        changed = True
    if len(tree.body) > 13:
        tree.body = tree.body[:13]
        changed = True
    ast.fix_missing_locations(tree)
    return ast.unparse(tree) if changed else text

def stable_seed(seed, task_id):
    return int.from_bytes(hashlib.sha256(f'{seed}|{task_id}'.encode()).digest()[:4], 'little')
