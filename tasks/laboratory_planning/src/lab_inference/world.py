"""Frozen OPF-JEPA forecasts for candidate laboratory plans."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from safetensors.torch import load_file
from .features import INPUT_DIM, relative_features
from .jepa_anything_core import OrthogonalFactorProjection
from .plan_schema import FIELDS

class OutcomeModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(INPUT_DIM, 128), nn.GELU(), nn.Linear(128, 32))
        self.predictor = nn.Sequential(nn.Linear(32, 64), nn.GELU(), nn.Linear(64, 32))
        self.head = nn.Linear(32, 8)
        self.opf = OrthogonalFactorProjection(32, 4, 8)

    def forward(self, x):
        z = self.encoder(x)
        p = self.predictor(z)
        return (torch.sigmoid(self.head(p)), z, p)

def predict_members(models, x):
    with torch.no_grad():
        return np.stack([m(torch.from_numpy(x).reshape(-1, INPUT_DIM))[0].numpy().reshape(*x.shape[:2], 8) for m in models])

class JEPAWorld:

    def __init__(self, checkpoint):
        checkpoint = Path(checkpoint)
        self.config = json.loads((checkpoint / 'config.json').read_text())
        if self.config['model_type'] != 'opf_jepa_laboratory_outcome_ensemble':
            raise ValueError('Expected a laboratory OPF-JEPA checkpoint')
        weights_path = checkpoint / 'world.safetensors'
        if hashlib.sha256(weights_path.read_bytes()).hexdigest() != self.config['weights_sha256']:
            raise ValueError('Checkpoint checksum mismatch')
        torch.set_num_threads(2)
        weights = load_file(str(weights_path), device='cpu')
        self.models = []
        expected_keys = set()
        for i in range(self.config['ensemble_members']):
            model = OutcomeModel()
            prefix = f'member_{i}.'
            state = {k[len(prefix):]: v for k, v in weights.items() if k.startswith(prefix)}
            model.load_state_dict(state, strict=True)
            expected_keys.update((prefix + k for k in model.state_dict()))
            self.models.append(model.requires_grad_(False).eval())
        if set(weights) != expected_keys:
            raise ValueError('Unexpected checkpoint tensors')
        self.mean = np.array(self.config['normalization']['mean'], dtype=np.float32)
        self.std = np.array(self.config['normalization']['std'], dtype=np.float32)
        if self.mean.shape != (INPUT_DIM,) or self.std.shape != (INPUT_DIM,) or np.any(self.std <= 0):
            raise ValueError('Invalid feature normalization')

    @torch.inference_mode()
    def forecast(self, task, candidates, include_factors=False):
        if set(task) != FIELDS:
            raise ValueError('Task fields must be id, context, goal, constraints, available_inputs and action_pool')
        for field in ['id', 'context', 'goal', 'available_inputs', 'action_pool']:
            if not isinstance(task[field], str):
                raise ValueError(f'Task field {field} must be a string')
        if not isinstance(task['constraints'], list) or not all((isinstance(x, str) for x in task['constraints'])):
            raise ValueError('Task constraints must be a list of strings')
        if len(candidates) != 4 or [c.get('candidate_id') for c in candidates] != ['a1', 'a2', 'a3', 'a4']:
            raise ValueError('Provide four candidates in a1, a2, a3, a4 order')
        if any((not isinstance(c.get('completion'), str) for c in candidates)):
            raise ValueError('Every candidate must contain a completion string')
        x, legal = relative_features(task, candidates)
        normalized = (x - self.mean) / self.std
        pred = predict_members(self.models, normalized[None])[:, 0]
        cal = self.config['calibration']
        avg = pred.mean(0)
        mean = np.clip(cal['slope'] * avg[:, 5] + cal['intercept'], 0, 1)
        sd = cal['uncertainty_scale'] * np.maximum(pred[:, :, 5].std(0), 0.03)
        weights = self.config['selection']
        score = mean + weights['delta_weight'] * (2 * avg[:, 6] - 1) - weights['risk_weight'] * avg[:, 7] - weights['uncertainty_weight'] * sd
        score[~legal] = -np.inf
        chosen = int(np.argmax(score)) if legal.any() else 0
        result = {'candidate_id': candidates[chosen]['candidate_id'], 'index': chosen, 'legal_candidates': int(legal.sum()), 'fallback_no_legal': not bool(legal.any()), 'predicted_Final': mean.tolist(), 'uncertainty_proxy': sd.tolist(), 'scores': [float(v) if np.isfinite(v) else None for v in score], 'predicted_gain': (2 * avg[:, 6] - 1).tolist(), 'predicted_risk': avg[:, 7].tolist(), 'legal': legal.tolist(), 'feedback': {name: avg[:, i].tolist() for i, name in enumerate(['action_sequence_similarity', 'parameter_accuracy', 'dependency', 'missing_fraction', 'redundant_fraction'])}}
        if include_factors:
            batch = torch.from_numpy(normalized)
            factors = torch.stack([model.opf.decompose(model(batch)[2]) for model in self.models])
            result['predicted_factors'] = factors.mean(0).tolist()
        return result
