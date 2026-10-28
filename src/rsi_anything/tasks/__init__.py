"""Task adapters translate workflow recipes into task-native stages."""
from importlib import import_module


def adapter(name):
    if name not in {'code_repair', 'gui'}:
        raise ValueError('Unknown task: ' + name)
    return import_module('.' + name, __name__)
