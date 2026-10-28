"""LabHorizon Level 2 planning prompt."""


def build_messages(task):
    constraints = "\n".join(f"- {item}" for item in task["constraints"])
    prompt = (
        "You are solving a LabHorizon Level 2 long-horizon experimental planning task.\n"
        "Use only the provided action pool and raw inputs. Output only a Python fenced block with "
        "straight-line assignment statements calling action-pool functions. Use keyword arguments only "
        "in every function call; positional arguments are invalid.\n\n"
        f"Context:\n{task['context']}\n\n"
        f"Goal:\n{task['goal']}\n\n"
        f"Constraints:\n{constraints}\n\n"
        f"Available inputs:\n{task['available_inputs']}\n\n"
        f"Action pool:\n```python\n{task['action_pool']}\n```\n"
    )
    return [
        {"role": "system", "content": "You are a careful scientific experimental planning assistant."},
        {"role": "user", "content": prompt},
    ]
