import re
import time
from dataclasses import dataclass
from cure_wm import TestCase
CODE_FENCE = re.compile('```(?:python|py)?\\s*\\n(.*?)```', re.I | re.S)
TEST_INPUT = re.compile('\\*\\*Test Input:\\*\\*\\s*```[^\\n]*\\n(.*?)```', re.I | re.S)
TEST_OUTPUT = re.compile('\\*\\*Test Output:\\*\\*\\s*```[^\\n]*\\n(.*?)```', re.I | re.S)

def extract_code(text):
    matches = CODE_FENCE.findall(text)
    if matches:
        return matches[-1].strip() + '\n'
    tail = text.split('</think>')[-1].strip()
    lines = tail.splitlines()
    for i, line in enumerate(lines):
        if line.startswith(('import ', 'from ', 'def ', 'class ', 'if __name__', 'n =', 'n=', 'a =', 'a=', 's =', 's=')):
            return '\n'.join(lines[i:]).strip() + '\n'
    return ''

def extract_test(text, index):
    inputs, outputs = (TEST_INPUT.findall(text), TEST_OUTPUT.findall(text))
    if not inputs or not outputs:
        return None
    inp, out = (inputs[-1].strip(), outputs[-1].strip())
    if not inp or not out:
        return None
    return TestCase(inp + '\n', out, f'generated_{index}')

@dataclass(frozen=True)
class Generation:
    text: str
    input_tokens: int
    output_tokens: int
    seconds: float

class FrozenCoder:

    def __init__(self, model_path):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map='cuda:0', local_files_only=True).eval()

    def _generate(self, user_prompt, *, n=1, seed=0, max_new_tokens=2048):
        torch = self.torch
        prompt = self.tokenizer.apply_chat_template([{'role': 'system', 'content': 'You are a helpful assistant for solving Python coding problems.'}, {'role': 'user', 'content': user_prompt}], tokenize=False, add_generation_prompt=True)
        tokens = self.tokenizer(prompt, return_tensors='pt', truncation=False).to('cuda:0')
        input_length = tokens['input_ids'].shape[1]
        if input_length + max_new_tokens > 20000:
            raise ValueError(f'Prompt too long: {input_length} tokens')
        torch.manual_seed(seed)
        started = time.monotonic()
        with torch.inference_mode():
            outputs = self.model.generate(**tokens, do_sample=True, temperature=0.8, top_p=0.95, max_new_tokens=max_new_tokens, num_return_sequences=n, pad_token_id=self.tokenizer.eos_token_id)
        elapsed = time.monotonic() - started
        generations = []
        for seq in outputs:
            continuation = seq[input_length:].tolist()
            if self.tokenizer.eos_token_id in continuation:
                output_tokens = continuation.index(self.tokenizer.eos_token_id) + 1
            else:
                output_tokens = len(continuation)
            generations.append(Generation(self.tokenizer.decode(continuation[:output_tokens], skip_special_tokens=True), input_length, output_tokens, elapsed / n))
        return tuple(generations)

    def initial(self, problem, seed):
        prompt = f'Think first, then write a complete Python script for the problem below. Read from standard input and print the answer to standard output. Put the final runnable script in one ```python block.\n\nProblem:\n{problem}'
        raw = self._generate(prompt, seed=seed, max_new_tokens=4096)[0]
        return (extract_code(raw.text), raw)

    def refine(self, problem, current_code, feedback, visible_tests, *, k, seed):
        lines = []
        for test, result in zip(visible_tests, feedback.results):
            lines.append(f'Input: {test.input[:1000]!r}\nExpected: {test.expected[:500]!r}\nObserved: {result.actual[:500]!r}\nStatus: {result.status}\nError: {result.stderr[:500]}')
        prompt = f'Revise the following Python script to solve the problem. Read from standard input and print to standard output. Use the observed test feedback to fix the program. Return a complete runnable script in one ```python block.\n\nProblem:\n{problem}\n\nCurrent script:\n```python\n{current_code}\n```\n\nVisible test feedback:\n' + '\n\n'.join(lines)
        raws = self._generate(prompt, n=k, seed=seed, max_new_tokens=4096)
        return tuple(((extract_code(raw.text), raw) for raw in raws))

    def generate_tests(self, problem, public_tests, *, k, seed):
        example = public_tests[0] if public_tests else None
        example_text = f'Example input: {example.input!r}\nExample output: {example.expected!r}\n' if example else ''
        excluded_inputs = '\n'.join((repr(t.input[:500]) for t in public_tests[:5]))
        prompt = f'Generate one new, correct and discriminative test case for this coding problem. Its input must differ from every public example input listed below; do not copy or slightly reformat an example. Prefer a small edge case whose answer you can verify exactly. Reason about the answer, then use exactly these fields:\n**Test Input:**\n```\n<input>\n```\n**Test Output:**\n```\n<output>\n```\n\nProblem:\n{problem}\n{example_text}\nPublic inputs to avoid:\n{excluded_inputs}'
        raws = self._generate(prompt, n=k, seed=seed, max_new_tokens=1200)
        found = []
        seen = {t.input.strip() for t in public_tests}
        for j, raw in enumerate(raws):
            test = extract_test(raw.text, j)
            if test and test.input.strip() not in seen:
                found.append(test)
                seen.add(test.input.strip())
        return (tuple(found), raws)
