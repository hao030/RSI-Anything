import copy
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from world import D, jepa_anything_objective
REVISION = 'world-update-v3-routed-suite-listwise-20261006-r2'
PUBLIC = 'per_public_test'
GLOBAL = 'whole_training_suite'

def validate_scope(row):
    scope = row.get('prediction_scope')
    if scope not in (PUBLIC, GLOBAL):
        raise ValueError('Every transition requires an explicit prediction scope')
    if scope == GLOBAL:
        if row.get('schema') != 'g6-whole-training-suite-transition-v1':
            raise ValueError('Whole-suite head requires the verified aggregate-state schema')
        partial = row['verified_full_suite_partial']
        allpass = row['verified_full_suite_allpass']
        utility = row['verified_training_utility']
        if not (math.isfinite(partial) and 0 <= partial <= 1):
            raise ValueError('Invalid observed pass fraction')
        if (row['status'] == 0) != bool(allpass) or bool(allpass) != (partial == 1):
            raise ValueError('Whole-suite AC, allpass and partial labels disagree')
        if not math.isclose(utility, float(allpass) + 0.25 * partial, abs_tol=1e-08):
            raise ValueError('Training utility does not match verified execution')
        provenance = row.get('provenance', {})
        required_false = ('context_contains_private_case_inputs', 'context_contains_private_expected_outputs', 'target_contains_candidate_code', 'gate_or_final_labels_used')
        if any((provenance.get(key) is not False for key in required_false)):
            raise ValueError('Invalid whole-suite supervision boundary')
    elif 'verified_full_suite_partial' in row:
        raise ValueError('Aggregate-state supervision was routed to a public-test head')
    return scope

def listwise_quality_loss(predicted_utility, rows, temperature=0.1):
    if temperature <= 0 or not math.isfinite(temperature):
        raise ValueError('Invalid ranking temperature')
    if len(predicted_utility) != len(rows):
        raise ValueError('Utility and transition counts differ')
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        if validate_scope(row) == GLOBAL:
            groups[row['problem_id'], row['group_id']].append(i)
    loss = predicted_utility.sum() * 0
    varied = tied = singleton = partial_only = 0
    for ids in groups.values():
        if len(ids) == 1:
            if rows[ids[0]]['candidate_id'] != 'keep_initial' or rows[ids[0]]['step_id'] != -1:
                raise ValueError('A candidate pool was fragmented into a singleton')
            singleton += 1
            continue
        if len(ids) != 4 or len({rows[i]['candidate_id'] for i in ids}) != 4:
            raise ValueError('Listwise supervision requires a complete four-candidate pool')
        if len({(rows[i]['trajectory_id'], rows[i]['step_id'], rows[i]['test_id']) for i in ids}) != 1:
            raise ValueError('Candidate group mixes different execution transitions')
        utilities = [rows[i]['verified_training_utility'] for i in ids]
        if max(utilities) == min(utilities):
            tied += 1
            continue
        observed = predicted_utility.new_tensor(utilities)
        target = F.softmax(observed / temperature, dim=0)
        loss = loss + temperature * F.kl_div(F.log_softmax(predicted_utility[ids] / temperature, dim=0), target, reduction='sum')
        varied += 1
        partial_only += not any((rows[i]['verified_full_suite_allpass'] for i in ids))
    return (loss / max(varied, 1), dict(global_groups=len(groups), utility_diverse_groups=varied, partial_only_diverse_groups=partial_only, tied_groups_skipped=tied, initial_singletons_skipped=singleton, utility_definition='P(full training suite passed) + 0.25 * predicted pass fraction', temperature=temperature))

class SemanticExecutionWorld(nn.Module):

    def __init__(self, public_world):
        super().__init__()
        if public_world.mode != 'opf':
            raise ValueError('Extension requires the native OPF world')
        self.public_world = public_world
        self.suite_readout = copy.deepcopy(public_world.readout)
        old = self.suite_readout[-1]
        final = nn.Linear(old.in_features, 7, device=old.weight.device, dtype=old.weight.dtype)
        with torch.no_grad():
            final.weight[:6].copy_(old.weight)
            final.bias[:6].copy_(old.bias)
            final.weight[6].zero_()
            final.bias[6].zero_()
        self.suite_readout[-1] = final
        self._original_trainable = {name for name, p in public_world.named_parameters() if p.requires_grad}
        self.stage = None
        self.set_update_stage('joint')

    def train(self, mode=True):
        super().train(mode)
        self.public_world.train(mode)
        return self

    def set_update_stage(self, stage):
        if stage not in ('heads', 'joint'):
            raise ValueError('Unknown update stage')
        for name, parameter in self.public_world.named_parameters():
            parameter.requires_grad_(name in self._original_trainable and (stage == 'joint' or name.startswith('readout.')) and (name != 'log_temperature'))
        if self.public_world.target is not None:
            self.public_world.target.requires_grad_(False).eval()
        self.suite_readout.requires_grad_(True)
        self.stage = stage

    def optimizer(self):
        groups = {'lora': [], 'opf': [], 'heads': []}
        for name, parameter in self.named_parameters():
            if not parameter.requires_grad:
                continue
            key = 'lora' if name.startswith('public_world.online.backbone.') else 'opf' if name.startswith('public_world.opf.') else 'heads'
            groups[key].append(parameter)
        rates = {'lora': 5e-06, 'opf': 1e-05, 'heads': 0.0001}
        return torch.optim.AdamW([dict(params=value, lr=rates[key]) for key, value in groups.items() if value], weight_decay=0.01)

    def training_loss(self, rows, reference_teacher=None, distillation_weight=0.0):
        if not rows or distillation_weight < 0:
            raise ValueError('Empty batch or invalid distillation weight')
        scopes = [validate_scope(row) for row in rows]
        public_ids = [i for i, scope in enumerate(scopes) if scope == PUBLIC]
        global_ids = [i for i, scope in enumerate(scopes) if scope == GLOBAL]
        world = self.public_world
        if world.target is None:
            raise ValueError('Training requires the native stop-gradient EMA target')
        with torch.set_grad_enabled(self.stage == 'joint'):
            contexts = world.encode([row['context'] for row in rows])
            latent, factors = world.predict_latent(contexts)
        public = world.readout(latent)
        suite = self.suite_readout(latent)
        output = torch.stack([suite[i, :6] if scope == GLOBAL else public[i] for i, scope in enumerate(scopes)])
        labels = torch.tensor([row['status'] for row in rows], device=output.device)
        seconds = output.new_tensor([row['seconds'] for row in rows])
        status = F.cross_entropy(output[:, :5], labels)
        resource = F.smooth_l1_loss(F.softplus(output[:, 5]), seconds)
        partial_loss = output.sum() * 0
        if global_ids:
            fractions = output.new_tensor([rows[i]['verified_full_suite_partial'] for i in global_ids])
            partial_loss = F.smooth_l1_loss(torch.sigmoid(suite[global_ids, 6]), fractions)
        predicted_utility = F.softmax(suite[:, :5], dim=-1)[:, 0] + 0.25 * torch.sigmoid(suite[:, 6])
        listwise, signal = listwise_quality_loss(predicted_utility, rows)
        public_ranking = output.sum() * 0
        public_pairs = 0
        public_scores = F.softmax(public[:, :5], dim=-1)[:, 0]
        for position, i in enumerate(public_ids):
            for j in public_ids[:position]:
                a, b = (rows[i], rows[j])
                if (a['problem_id'], a['group_id']) == (b['problem_id'], b['group_id']) and (a['status'] == 0) != (b['status'] == 0):
                    good, bad = (i, j) if a['status'] == 0 else (j, i)
                    public_ranking = public_ranking + F.relu(0.1 - public_scores[good] + public_scores[bad])
                    public_pairs += 1
        public_ranking = public_ranking / max(public_pairs, 1)
        with torch.no_grad():
            target = world.encode([row['target'] for row in rows], target=True)
        objective = jepa_anything_objective(factors, world.opf(target.detach()), world.opf.analysis_basis(), contexts)
        distillation = output.sum() * 0
        if distillation_weight:
            if reference_teacher is None or reference_teacher is world:
                raise ValueError('Distillation requires a separate frozen parent teacher')
            if any((p.requires_grad for p in reference_teacher.parameters())):
                raise ValueError('Reference teacher must be frozen')
            if public_ids:
                with torch.no_grad():
                    teacher = reference_teacher.score([rows[i]['context'] for i in public_ids])
                logits = public[public_ids, :5] / world.log_temperature.exp().clamp(0.2, 5.0)
                distillation = F.kl_div(F.log_softmax(logits, dim=-1), teacher.detach(), reduction='batchmean')
        parts = dict(status=status, resource=resource, partial=partial_loss, listwise=listwise, public_ranking=public_ranking, distillation=distillation, prediction=objective.prediction, orthogonality=objective.orthogonality, factor_activity=objective.factor_activity, encoder_variance=objective.encoder_variance)
        loss = objective.total + status + 0.1 * resource + 0.5 * partial_loss + 0.2 * listwise + 0.2 * public_ranking + distillation_weight * distillation
        return (loss, parts, contexts, factors, dict(signal, public_ranking_pairs=public_pairs, global_rows=len(global_ids), public_rows=len(public_ids), stage=self.stage, target_stop_gradient=True))

    @torch.no_grad()
    def update_ema(self):
        self.public_world.update_ema()

    @torch.no_grad()
    def score_suite(self, contexts):
        self.eval()
        encoded = self.public_world.encode(contexts)
        latent, _ = self.public_world.predict_latent(encoded)
        output = self.suite_readout(latent)
        probabilities = F.softmax(output[:, :5], dim=-1)
        partial = torch.sigmoid(output[:, 6])
        return dict(scope=GLOBAL, full_suite_pass_probability=probabilities[:, 0], predicted_pass_fraction=partial, utility=probabilities[:, 0] + 0.25 * partial, failure_status_probabilities=probabilities, predicted_mean_normalized_seconds=F.softplus(output[:, 5]), calibration_verified=False, semantic_correctness_guaranteed=False)

    @torch.no_grad()
    def score_public(self, contexts):
        return self.public_world.score(contexts)

    def save(self, path, metadata):
        from artifacts import atomic_json, checkpoint_manifest
        path = Path(path)
        if path.exists():
            raise FileExistsError('Do not overwrite an extension checkpoint')
        path.mkdir(parents=True)
        self.public_world.save(path / 'public_world', dict(metadata, formal_checkpoint=False, formal_G6_endpoint=False))
        checkpoint_manifest(path / 'public_world', formal=False)
        torch.save({key: value.detach().cpu() for key, value in self.suite_readout.state_dict().items()}, path / 'suite_readout.pt')
        manifest = dict(metadata, revision=REVISION, mode='semantic_opf', formal_checkpoint=False, formal_G6_endpoint=False, head_scopes=[PUBLIC, GLOBAL], stage=self.stage, arithmetic_precision='float32' if next(self.public_world.online.backbone.parameters()).dtype == torch.float32 else 'native_bfloat16_backbone_float32_domain', utility_definition='P(full training suite passed) + 0.25 * predicted pass fraction', calibration_verified=False, semantic_correctness_guaranteed=False, official_JEPA_Anything_core_changed=False)
        manifest['files'] = {str(p.relative_to(path)): hashlib.sha256(p.read_bytes()).hexdigest() for p in path.rglob('*') if p.is_file()}
        atomic_json(path / 'manifest.json', manifest)
        return checkpoint_manifest(path, formal=False)

    @classmethod
    def load(cls, path, base_path, training=False, device='cpu'):
        from artifacts import checkpoint_manifest
        from world import ExecutionWorld
        manifest, identity = checkpoint_manifest(path, formal=False)
        if manifest['revision'] != REVISION or manifest['mode'] != 'semantic_opf' or manifest['formal_checkpoint']:
            raise ValueError('Unexpected extension checkpoint type')
        world = ExecutionWorld.load(Path(path) / 'public_world', base_path, training=training, formal=False, device=device)
        if manifest.get('arithmetic_precision') == 'float32':
            world.float()
        elif manifest.get('arithmetic_precision') != 'native_bfloat16_backbone_float32_domain':
            raise ValueError('Unknown extension arithmetic precision')
        model = cls(world)
        state = torch.load(Path(path) / 'suite_readout.pt', map_location=device, weights_only=True)
        model.suite_readout.load_state_dict(state, strict=True)
        model.set_update_stage(manifest['stage'])
        model.to(device).train(training)
        model.identity = identity
        return model
