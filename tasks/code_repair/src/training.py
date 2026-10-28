import collections
import gc
import json
import random
from pathlib import Path

import torch

from artifacts import atomic_json, checkpoint_manifest
from development import evaluate
from diagnostics import delta, geometry
from grouping import grouped_batches
from incumbent import make_units, batches_for
from initial_batches import batch_indices
from io_utils import verify_training_hash, experiment, lines, sha, verify_assets
from losses import objective
from semantic_world import SemanticExecutionWorld
from world import DomainEncoder, ExecutionWorld


def batches(rows, stage, seed):
    if stage == 'initial':
        return [dict(indices=ids, padding_indices=[]) for ids in batch_indices(rows, seed)]
    if stage == 'incumbent':
        return batches_for(make_units(rows), seed)
    return [dict(indices=ids, **info) for ids, info in grouped_batches(rows, seed)]


def load(stage, parent, backbone):
    if stage == 'initial':
        return ExecutionWorld(DomainEncoder(backbone, max_length=16384), 'opf').cuda().train(), None
    if stage == 'semantic':
        base = ExecutionWorld.load(parent, backbone, training=True, device='cuda')
        model = SemanticExecutionWorld(base)
        teacher = ExecutionWorld.load(parent, backbone, training=False, device='cuda')
    else:
        model = SemanticExecutionWorld.load(parent, backbone, training=True, device='cuda')
        if stage == 'incumbent':
            teacher = SemanticExecutionWorld.load(parent, backbone, training=False, device='cuda')
        else:
            teacher = ExecutionWorld.load(Path(parent) / 'public_world', backbone, training=False, formal=False, device='cuda')
    teacher.requires_grad_(False).eval()
    return model, teacher


def loss_for(model, teacher, rows, stage, config):
    if stage == 'initial':
        loss, parts, _, _ = model.training_loss(rows)
        return loss, parts
    if stage == 'incumbent':
        loss, parts, _ = objective(model, teacher, rows, config['pair_weight'], config['anchor_weight'])
        return loss, parts
    loss, parts, _, _, _ = model.training_loss(rows, teacher, config['distillation_weight'])
    if stage == 'expanded':
        loss = loss + (config['listwise_weight'] - .2) * parts['listwise']
    return loss, parts


def fit(stage, assets, output, parent=None, max_steps=0, development=True):
    assets, output = Path(assets).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    config = experiment()['training'][stage]
    verify_assets(assets, prefix=[f'training/{stage}.jsonl', 'models/world/', 'development/'])
    rows = list(lines(assets / f'training/{stage}.jsonl'))
    assert len(rows) == config['rows']
    verify_training_hash(stage, assets / f'training/{stage}.jsonl')
    dev_ids = {row['task_id'] for row in lines(assets / 'development/public_cases.jsonl')}
    train_ids = {row['problem_id'] for row in rows}
    assert not train_ids & dev_ids and not train_ids & set(experiment()['evaluation']['tasks'])
    parent = Path(parent).resolve() if parent else assets / 'checkpoints' / str(config['parent'])
    if stage != 'initial':
        checkpoint_manifest(parent, formal=False)
    torch.set_num_threads(4)
    random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed_all(config['seed'])
    model, teacher = load(stage, parent, assets / 'models/world')
    entries = batches(rows, stage, config['seed'])
    if stage != 'initial':
        coverage = collections.Counter(i for b in entries for i in b['indices'] if i not in b['padding_indices'])
        assert coverage == collections.Counter(range(len(rows)))
    output.mkdir(parents=True)
    atomic_json(output / 'batches.json', entries)
    atomic_json(output / 'config.json', dict(stage=stage, settings=config, parent=str(parent),
        training_sha256=sha(assets / f'training/{stage}.jsonl'), smoke_steps=max_steps))
    before = {name: p.detach().float().cpu().clone() for name, p in model.named_parameters()
              if p.requires_grad or ('target.' in name and ('lora_' in name or '.projection.' in name))}
    if stage in ['semantic', 'expanded'] and not max_steps:
        atomic_json(output / 'baseline_geometry.json', geometry(model, rows))
    if stage == 'semantic' and development and not max_steps:
        evaluate(model, assets, output / 'baseline_development.json')
    phases = [('heads', 1), ('joint', 1)] if stage == 'semantic' else [('joint', 1)]
    steps = 0
    with (output / 'trace.jsonl').open('x') as stream:
        for phase, epochs in phases:
            if stage != 'initial':
                model.set_update_stage(phase)
            optimizer = model.optimizer()
            if stage == 'incumbent':
                for group in optimizer.param_groups:
                    group['lr'] *= config['lr_multiplier']
            seen = set()
            for entry in entries:
                model.train()
                optimizer.zero_grad(set_to_none=True)
                batch = [rows[i] for i in entry['indices']]
                loss, parts = loss_for(model, teacher, batch, stage, config)
                assert torch.isfinite(loss)
                loss.backward()
                base = model if stage == 'initial' else model.public_world
                assert all(p.grad is None for p in base.target.parameters())
                if teacher is not None:
                    assert all(p.grad is None for p in teacher.parameters())
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.grad is not None], 1., error_if_nonfinite=True)
                optimizer.step()
                model.update_ema()
                steps += 1
                seen.update(entry['indices'])
                record = dict(stage=stage, phase=phase, step=steps, indices=entry['indices'],
                    loss=float(loss.detach()), parts={k: float(v.detach()) for k, v in parts.items()})
                stream.write(json.dumps(record) + '\n')
                stream.flush()
                atomic_json(output / 'progress.json', dict(phase=phase, steps=steps, phase_rows_seen=len(seen)))
                if max_steps and steps >= max_steps:
                    break
            full = len(seen) == len(rows) and not max_steps
            assert max_steps or full
            if stage != 'initial' and development and not max_steps:
                evaluate(model, assets, output / f'{phase}_development.json')
            metadata = dict(stage=phase, seed=config['seed'], rows=len(rows), full_fit_completed=full and phase == 'joint',
                formal_checkpoint=stage == 'initial' and full, epochs_completed=int(full), training_steps=steps,
                native_JEPA_Anything_core_unchanged=True, incumbent_aware_update=stage == 'incumbent',
                final200_labels_used=False, training_sha256=sha(assets / f'training/{stage}.jsonl'))
            if stage != 'initial':
                metadata['parameter_change_norms'] = delta(model, before)
            model.save(output / f'{phase}_checkpoint', metadata)
            if max_steps and steps >= max_steps:
                break
    del optimizer, loss, parts, teacher
    gc.collect()
    torch.cuda.empty_cache()
    if stage != 'initial' and not max_steps:
        atomic_json(output / 'geometry.json', geometry(model, rows))
    checkpoint = output / f'{phase}_checkpoint'
    contexts = [row['context'] for row in rows[:4]]
    expected = (model.score(contexts) if stage == 'initial' else model.score_suite(contexts)['utility']).detach().cpu()
    del model, base
    gc.collect()
    torch.cuda.empty_cache()
    restored = (ExecutionWorld.load(checkpoint, assets / 'models/world', training=False, formal=False, device='cuda')
                if stage == 'initial' else SemanticExecutionWorld.load(checkpoint, assets / 'models/world', training=False, device='cuda'))
    observed = (restored.score(contexts) if stage == 'initial' else restored.score_suite(contexts)['utility']).detach().cpu()
    assert torch.allclose(expected, observed, rtol=2e-5, atol=2e-6)
    _, identity = checkpoint_manifest(checkpoint, formal=False)
    report = dict(stage=stage, checkpoint=str(checkpoint), identity=identity, rows=len(rows), steps=steps,
        full_fit_completed=not max_steps, checkpoint_reload_max_error=float((expected-observed).abs().max()))
    atomic_json(output / 'complete.json', report)
    return report


def run(args):
    if args.stage != 'all':
        print(json.dumps(fit(args.stage, args.assets, args.output, args.parent, args.max_steps, not args.no_development)), flush=True)
        return
    if args.parent or args.max_steps:
        raise ValueError('Use an individual stage for --parent or --max-steps')
    parent = None
    for stage in ['initial', 'semantic', 'expanded', 'incumbent']:
        result = fit(stage, args.assets, Path(args.output) / stage, parent, development=not args.no_development)
        parent = result['checkpoint']
        gc.collect()
        torch.cuda.empty_cache()
    print(json.dumps(result), flush=True)
