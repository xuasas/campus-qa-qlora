"""Validation-only rank-8 budget pilots; never starts formal rank experiments."""
import os
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
from pathlib import Path
import argparse, datetime, hashlib, json, math, subprocess, sys, time

ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / 'configs' / 'runtime'
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, choices=[1, 2])
parser.add_argument('--parent', type=Path)
args = parser.parse_args()
F = json.loads((CFG/'experiment_freeze.json').read_text(encoding='utf-8'))
DATA = Path(F['data_dir'])
if args.epochs is None:
    stamp = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).strftime('%Y%m%d_%H%M%S')
    parent = ROOT / 'runs' / ('budget_pilot_' + stamp)
    parent.mkdir(parents=True, exist_ok=False)
    selection = {
        'purpose': 'rank8 validation-only budget selection; not formal Q8',
        'selection_rule': 'Choose one epoch if its validation loss is at most 1.01 times the minimum of the two independently scheduled pilots; otherwise choose two epochs.',
        'rank': 8, 'alpha': 16, 'seed': 42,
        'heldout_test_not_used': True, 'formal_training_started': False,
    }
    (parent / 'selection_rule.json').write_text(json.dumps(selection, ensure_ascii=False, indent=2), encoding='utf-8')
    for epochs in [1, 2]:
        subprocess.run([sys.executable, '-X', 'utf8', str(Path(__file__).resolve()), '--epochs', str(epochs), '--parent', str(parent)], check=True)
    reports = [json.loads((parent / f'epochs_{n}' / 'pilot_report.json').read_text(encoding='utf-8')) for n in [1, 2]]
    assert all(p['status'] == 'pass' for p in reports)
    losses = [p['final_validation_loss'] for p in reports]
    chosen = 1 if losses[0] <= 1.01 * min(losses) else 2
    selection.update(status='pass', selected_epochs=chosen, final_validation_losses={'1': losses[0], '2': losses[1]},
                     optimizer_steps={str(p['epochs']): p['optimizer_steps'] for p in reports},
                     independent_learning_rate_schedules=True, pilot_reports=[str(parent / f'epochs_{n}' / 'pilot_report.json') for n in [1, 2]],
                     resource_limits='A single rank and seed; validation cross-entropy selects a practical budget, not model factual correctness or globally optimal hyperparameters.')
    (parent / 'budget_selection.json').write_text(json.dumps(selection, ensure_ascii=False, indent=2), encoding='utf-8')
    (ROOT / 'current_budget_selection.json').write_text(json.dumps({'report': str(parent / 'budget_selection.json')}, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(selection, ensure_ascii=False, indent=2), flush=True)
    sys.exit(0)

import torch, bitsandbytes as bnb, yaml
from transformers import TrainerCallback
from llamafactory.train.tuner import run_exp
assert args.parent is not None and torch.cuda.is_available()
run = args.parent / f'epochs_{args.epochs}'
run.mkdir(exist_ok=False)
cfg = yaml.safe_load((CFG / 'Q8.yaml').read_text(encoding='utf-8'))
cfg.update(dataset_dir=str(DATA), output_dir=str(run / 'adapter'), lora_rank=8, lora_alpha=16,
           num_train_epochs=float(args.epochs), logging_steps=5, eval_strategy='epoch',
           save_strategy='no', save_only_model=True, load_best_model_at_end=False,
           train_on_prompt=False, packing=False, overwrite_cache=False, resume_from_checkpoint=None)
cfg.pop('max_steps', None)
(run / 'pilot.yaml').write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding='utf-8')
report = {'purpose': 'Independent validation-only budget pilot, not formal Q8', 'epochs': args.epochs,
          'heldout_test_not_used': True, 'rank': 8, 'alpha': 16, 'data_dir': str(DATA),
          'config': cfg, 'gpu': torch.cuda.get_device_name(0), 'validation_history': [],
          'data_sha256': {name: hashlib.sha256((DATA / name).read_bytes()).hexdigest()
                          for name in ['campus_train.json', 'campus_val.json', 'split_manifest.json']}}

class Audit(TrainerCallback):
    def on_train_begin(self, args, state, control, model=None, **kw):
        self.trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        assert self.trainable and all('lora_' in n for n, p in self.trainable)
        self.before = {n: p.detach().cpu().clone() for n, p in self.trainable if 'lora_B' in n}
        layers = [m for m in model.modules() if isinstance(m, bnb.nn.Linear4bit)]
        assert layers and all(m.weight.quant_state.quant_type == 'nf4' and m.weight.quant_state.state2 is not None and m.compute_dtype == torch.bfloat16 for m in layers)
        report['trainable_parameters'] = sum(p.numel() for n, p in self.trainable)
        report['quantized_linear_layers'] = len(layers)
        torch.cuda.reset_peak_memory_stats()
        self.checked = False
    def on_pre_optimizer_step(self, args, state, control, **kw):
        if not self.checked:
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for n, p in self.trainable)
            assert any(torch.count_nonzero(p.grad).item() for n, p in self.trainable)
            self.checked = True
    def on_log(self, args, state, control, logs=None, **kw):
        if logs and 'loss' in logs:
            assert math.isfinite(logs['loss']) and math.isfinite(logs.get('grad_norm', 0))
    def on_evaluate(self, args, state, control, metrics=None, **kw):
        assert math.isfinite(metrics['eval_loss'])
        report['validation_history'].append({'epoch': state.epoch, 'step': state.global_step, 'eval_loss': metrics['eval_loss']})
    def on_train_end(self, args, state, control, **kw):
        assert self.checked and any(not torch.equal(self.before[n], p.detach().cpu()) for n, p in self.trainable if n in self.before)
        report.update(optimizer_steps=state.global_step, finite_gradients_and_parameter_updates=True,
                      peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved())

try:
    start = time.perf_counter()
    run_exp(cfg, callbacks=[Audit()])
    report['train_eval_save_seconds'] = time.perf_counter() - start
    ev = json.loads((run / 'adapter' / 'eval_results.json').read_text(encoding='utf-8'))
    report['final_validation_loss'] = ev['eval_loss']
    assert math.isfinite(ev['eval_loss']) and report['optimizer_steps'] == math.ceil(len(json.loads((DATA/'campus_train.json').read_text(encoding='utf-8')))/8) * args.epochs
    report['adapter_bytes'] = (run / 'adapter' / 'adapter_model.safetensors').stat().st_size
    report['status'] = 'pass'
except Exception as exc:
    report.update(status='fail', error=repr(exc))
    raise
finally:
    (run / 'pilot_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k not in ['config', 'data_sha256']}, ensure_ascii=False), flush=True)
