import copy, sys
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
CORE = Path(__file__).resolve().parents[1] / 'vendor' / 'jepa-anything-core' / 'src'
sys.path.insert(0, str(CORE))
from jepa_anything_core import OrthogonalFactorProjection, UnconstrainedMultiHeadPredictor, MLPPredictor, jepa_anything_objective, encoder_variance_loss, factor_activity_loss, audit_opf_geometry
D, K, R = (256, 4, 64)

class Overlength(ValueError):
    pass

class DomainEncoder(nn.Module):

    def __init__(self, base_path, max_length=16384):
        super().__init__()
        from transformers import AutoModel, AutoTokenizer
        from peft import LoraConfig, get_peft_model, TaskType
        self.tokenizer = AutoTokenizer.from_pretrained(base_path, local_files_only=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = 'right'
        self.max_length = max_length
        base = AutoModel.from_pretrained(base_path, dtype=torch.bfloat16, local_files_only=True, attn_implementation='sdpa')
        config = LoraConfig(task_type=TaskType.FEATURE_EXTRACTION, r=16, lora_alpha=32, lora_dropout=0.0, target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'])
        self.backbone = get_peft_model(base, config)
        self.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        self.backbone.enable_input_require_grads()
        self.backbone.config.use_cache = False
        self.projection = nn.Sequential(nn.Linear(base.config.hidden_size, D), nn.LayerNorm(D))

    def forward(self, texts):
        batch = self.tokenizer(texts, padding=True, truncation=False, return_tensors='pt')
        if batch['input_ids'].shape[-1] > self.max_length:
            raise Overlength('encoder input exceeds explicit limit')
        device = next(self.backbone.parameters()).device
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = self.backbone(**batch, return_dict=True, use_cache=False).last_hidden_state
        mask = batch['attention_mask'].unsqueeze(-1)
        pooled = (outputs.float() * mask).sum(1) / mask.sum(1).clamp_min(1)
        return self.projection(pooled)

class ExecutionWorld(nn.Module):

    def __init__(self, encoder, mode='opf', ema=0.996, include_target=True):
        super().__init__()
        self.mode = mode
        self.ema_decay = ema
        self.online = encoder
        self.target = copy.deepcopy(encoder).requires_grad_(False).eval() if mode != 'ce' and include_target else None
        if mode == 'opf':
            self.opf = OrthogonalFactorProjection(D, K, R, learnable=True, orthogonality_mode='soft_gram')
            self.predictor = UnconstrainedMultiHeadPredictor(D, D, K, R, 512)
        elif mode == 'multi':
            self.opf = None
            self.predictor = UnconstrainedMultiHeadPredictor(D, D, K, R, 640)
        elif mode in ('standard', 'ce'):
            self.opf = None
            self.predictor = MLPPredictor(D, D, 640)
        else:
            raise ValueError(mode)
        self.log_temperature = nn.Parameter(torch.zeros(()))
        self.readout = nn.Sequential(nn.LayerNorm(D), nn.Linear(D, 64), nn.GELU(), nn.Linear(64, 6))

    def train(self, mode=True):
        super().train(mode)
        if self.target is not None:
            self.target.eval()
        return self

    def predict_latent(self, context):
        p = self.predictor(context)
        if self.mode == 'opf':
            return (self.opf.compose(p), p)
        if self.mode == 'multi':
            return (p.flatten(-2), p)
        return (p, p.reshape(-1, K, R))

    def encode(self, texts, target=False, microbatch=1):
        encoder = self.target if target else self.online
        if encoder is None:
            raise RuntimeError('CE has no target')
        rows = []
        for i in range(0, len(texts), microbatch):
            rows.append(encoder(texts[i:i + microbatch]))
        return torch.cat(rows, dim=0)

    def training_loss(self, rows):
        contexts = self.encode([x['context'] for x in rows])
        latent, factors = self.predict_latent(contexts)
        output = self.readout(latent)
        labels = torch.tensor([x['status'] for x in rows], device=output.device)
        seconds = torch.tensor([x['seconds'] for x in rows], device=output.device)
        ce = F.cross_entropy(output[:, :5], labels)
        resource = F.smooth_l1_loss(F.softplus(output[:, 5]), seconds)
        ranking = output.sum() * 0
        pairs = 0
        for i, a in enumerate(rows):
            for j, b in enumerate(rows[:i]):
                if a['group_id'] == b['group_id'] and (a['status'] == 0) != (b['status'] == 0):
                    good, bad = (i, j) if a['status'] == 0 else (j, i)
                    scores = F.softmax(output[:, :5], dim=-1)[:, 0]
                    ranking = ranking + F.relu(0.1 - scores[good] + scores[bad])
                    pairs += 1
        ranking = ranking / max(pairs, 1)
        parts = {'status': ce, 'resource': resource, 'ranking': ranking}
        if self.target is not None:
            with torch.no_grad():
                target_states = self.encode([x['target'] for x in rows], target=True)
            if self.mode == 'opf':
                target_factors = self.opf(target_states.detach())
                objective = jepa_anything_objective(factors, target_factors, self.opf.analysis_basis(), contexts)
                parts.update(prediction=objective.prediction, orthogonality=objective.orthogonality, factor_activity=objective.factor_activity, encoder_variance=objective.encoder_variance)
                representation = objective.total
            else:
                target_factors = target_states.detach().reshape(-1, K, R)
                pred = F.mse_loss(latent, target_states.detach())
                variance = encoder_variance_loss(contexts)
                parts.update(prediction=pred, encoder_variance=variance)
                representation = pred + variance
        else:
            representation = output.sum() * 0
        loss = representation + ce + 0.1 * resource + 0.2 * ranking
        return (loss, parts, contexts, factors)

    @torch.no_grad()
    def update_ema(self):
        if self.target is None:
            return
        targets = dict(self.target.named_parameters())
        for name, p in self.online.named_parameters():
            if p.requires_grad:
                targets[name].mul_(self.ema_decay).add_(p, alpha=1 - self.ema_decay)

    @torch.no_grad()
    def score(self, contexts):
        self.eval()
        c = self.encode(contexts)
        z, _ = self.predict_latent(c)
        p = F.softmax(self.readout(z)[:, :5] / self.log_temperature.exp().clamp(0.2, 5.0), -1)
        return p

    @torch.no_grad()
    def details(self, contexts):
        self.eval()
        z = self.encode(contexts)
        latent, _ = self.predict_latent(z)
        out = self.readout(latent)
        return (F.softmax(out[:, :5] / self.log_temperature.exp().clamp(0.2, 5.0), -1), F.softplus(out[:, 5]))

    def optimizer(self, slow=True):
        lora = []
        head = []
        opf = []
        for name, p in self.named_parameters():
            if not p.requires_grad or name == 'log_temperature':
                continue
            if name.startswith('online.backbone'):
                lora.append(p)
            elif name.startswith('opf'):
                opf.append(p)
            else:
                head.append(p)
        return torch.optim.AdamW([{'params': lora, 'lr': 5e-06 if slow else 0.0}, {'params': head, 'lr': 0.0001}, {'params': opf, 'lr': 1e-05}], weight_decay=0.01)

    def geometry(self, context):
        if self.opf is None:
            return {}
        z = context.detach().float()
        f = self.opf(z)
        report = audit_opf_geometry(self.opf, z)
        return report.to_dict()

    def save(self, path, metadata):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.online.backbone.save_pretrained(path / 'online_lora')
        if self.target is not None:
            self.target.backbone.save_pretrained(path / 'ema_lora')
        torch.save({k: v.cpu() for k, v in self.state_dict().items() if not k.startswith(('online.backbone', 'target.backbone'))}, path / 'domain.pt')
        import json, hashlib
        metadata = dict(metadata, encoder_max_length=self.online.max_length, mode=self.mode, ema=self.ema_decay, dimensions=dict(d=D, K=K, r=R), core_commit='c6e6c88f3ef75a4ce7acd660d6fa5779d995512c')
        metadata['files'] = {str(p.relative_to(path)): hashlib.sha256(p.read_bytes()).hexdigest() for p in path.rglob('*') if p.is_file() and p.name != 'manifest.json'}
        (path / 'manifest.json').write_text(json.dumps(metadata, indent=2))

    @classmethod
    def load(cls, path, base_path, training=False, formal=True, device='cuda'):
        from artifacts import checkpoint_manifest
        from peft.utils.save_and_load import load_peft_weights, set_peft_model_state_dict
        metadata, identity = checkpoint_manifest(path, formal=formal)
        path = Path(path)
        encoder = DomainEncoder(base_path, max_length=metadata.get('encoder_max_length', 8192))
        model = cls(encoder, metadata['mode'], metadata.get('ema', 0.996), include_target=training)

        def adapter(module, folder):
            weights = load_peft_weights(str(path / folder), device='cpu')
            result = set_peft_model_state_dict(module, weights, adapter_name='default')
            if result.unexpected_keys:
                raise RuntimeError('Unexpected adapter checkpoint keys ' + str(result.unexpected_keys))
        adapter(model.online.backbone, 'online_lora')
        if model.target is not None:
            adapter(model.target.backbone, 'ema_lora')
        state = torch.load(path / 'domain.pt', map_location='cpu', weights_only=True)
        if model.target is None:
            state = {k: v for k, v in state.items() if not k.startswith('target.')}
        result = model.load_state_dict(state, strict=False)
        unexpected = [k for k in result.unexpected_keys if not k.startswith('target.')]
        missing = [k for k in result.missing_keys if not k.startswith(('online.backbone.', 'target.backbone.')) and k != 'log_temperature']
        if unexpected or missing:
            raise RuntimeError('Domain checkpoint mismatch ' + str((unexpected, missing)))
        model = model.to(device)
        model.identity = identity
        model.checkpoint_path = str(path)
        model.train(training)
        return model
