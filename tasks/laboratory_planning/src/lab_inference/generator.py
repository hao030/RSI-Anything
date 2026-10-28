"""Generate candidate laboratory plans for JEPA world-model selection."""
from .postprocess import preprocess, stable_seed
from .prompts import build_messages


class CandidateGenerator:
    def __init__(self, base_model, adapter, *, device="cuda:0", dtype="bf16"):
        import torch
        from peft import PeftModel
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.torch = torch
        self.device = device
        self.processor = AutoProcessor.from_pretrained(base_model, trust_remote_code=False)
        if not self.processor.chat_template and self.processor.tokenizer.chat_template:
            self.processor.chat_template = self.processor.tokenizer.chat_template
        self.tokenizer = self.processor.tokenizer
        precision = {"bf16": torch.bfloat16, "fp32": torch.float32}[dtype]
        base = AutoModelForImageTextToText.from_pretrained(
            base_model, dtype=precision, device_map={"": device}, trust_remote_code=False
        )
        self.model = PeftModel.from_pretrained(base, adapter, is_trainable=False).requires_grad_(False).eval()

    def candidates(self, task, seed):
        """Generate one greedy plan and three sampled plans."""
        prompt = self.processor.apply_chat_template(
            build_messages(task), tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        inputs = self.processor(text=[prompt], return_tensors="pt").to(self.device)
        length = inputs["input_ids"].shape[-1]
        with self.torch.inference_mode():
            greedy = self.model.generate(
                **inputs, do_sample=False, max_new_tokens=2048,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        task_seed = stable_seed(seed, task["id"])
        self.torch.manual_seed(task_seed)
        if self.torch.cuda.is_available():
            self.torch.cuda.manual_seed_all(task_seed)
        with self.torch.inference_mode():
            sampled = self.model.generate(
                **inputs, do_sample=True, num_return_sequences=3, temperature=0.7, top_p=0.95,
                max_new_tokens=2048, pad_token_id=self.tokenizer.eos_token_id,
            )
        candidates = []
        for index, tokens in enumerate([greedy[0], *sampled]):
            suffix = tokens[length:]
            raw = self.tokenizer.decode(suffix, skip_special_tokens=True)
            candidates.append({"candidate_id": f"a{index + 1}", "raw_completion": raw,
                               "completion": preprocess(raw, task["action_pool"]),
                               "output_tokens": len(suffix), "hit_token_limit": len(suffix) >= 2048,
                               "mode": "greedy" if index == 0 else "sampled_batch"})
        return candidates
