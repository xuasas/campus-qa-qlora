"""Run exactly one frozen rank; --check-only never starts training."""
import os
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['PYTHONUTF8'] = '1'
from pathlib import Path
import argparse, datetime, hashlib, importlib.metadata, json, math, subprocess, sys, time

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'configs' / 'runtime'
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--rank', type=int, choices=[4, 8, 16])
parser.add_argument('--check-only', action='store_true')
parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
parser.add_argument('--audit-dir', type=Path, help=argparse.SUPPRESS)
args = parser.parse_args()
freeze = json.loads((CONFIG / 'experiment_freeze.json').read_text(encoding='utf-8'))
if args.rank is None and not args.check_only: parser.error('--rank is required for training')

def check(rank):
    import yaml
    spec = freeze['formal_configs'][f'Q{rank}']
    path = Path(spec['path'])
    if hashlib.sha256(path.read_bytes()).hexdigest() != spec['sha256']:
        raise RuntimeError(f'Frozen config changed: {path}')
    for name, digest in freeze['data_files_sha256'].items():
        if hashlib.sha256((Path(freeze['data_dir']) / name).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f'Frozen data changed: {name}')
    for package, version in freeze['packages'].items():
        if importlib.metadata.version(package) != version:
            raise RuntimeError(f'Environment version changed: {package}')
    config = yaml.safe_load(path.read_text(encoding='utf-8'))
    if Path(config['output_dir']).exists():
        raise RuntimeError(f'Output already exists; do not overwrite or delete blindly: {config["output_dir"]}')
    from llamafactory.hparams import get_train_args
    model, data, training, tuning, generation = get_train_args(config)
    assert data.dataset == ['campus_train'] and data.eval_dataset == ['campus_val']
    assert tuning.lora_rank == rank and tuning.lora_alpha == rank * 2
    assert training.num_train_epochs == 2 and training.bf16 and not training.fp16
    import torch
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('CUDA/BF16 unavailable; activate the configured CUDA/BF16 environment')
    return config

if args.check_only:
    for rank in ([args.rank] if args.rank else [4, 8, 16]):
        check(rank)
    print('CHECK PASS: frozen configs, data, package versions and CUDA/BF16 verified. No training started.', flush=True)
    sys.exit(0)

config = check(args.rank)
if not args.worker:
    stamp = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).strftime('%Y%m%d_%H%M%S')
    audit = ROOT / 'reports' / 'formal_training' / f'Q{args.rank}_{stamp}'
    audit.mkdir(parents=True, exist_ok=False)
    print(f'Run Q{args.rank}; audit directory: {audit}', flush=True)
    command = [sys.executable, '-X', 'utf8', '-u', str(Path(__file__).resolve()), '--rank', str(args.rank), '--worker', '--audit-dir', str(audit)]
    start = time.perf_counter()
    child = None
    code = None
    try:
        with (audit / 'console.txt').open('w', encoding='utf-8') as log:
            child = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     text=True, encoding='utf-8', errors='replace', env=os.environ.copy())
            for line in child.stdout:
                print(line, end='', flush=True)
                log.write(line); log.flush()
            code = child.wait()
    except KeyboardInterrupt:
        if child is not None and child.poll() is None:
            child.terminate(); child.wait()
        code = 130
        print('Interrupted; keep logs and partial output for inspection.', flush=True)
    finally:
        (audit / 'launch_report.json').write_text(json.dumps({
            'group': f'Q{args.rank}', 'returncode': code,
            'process_wall_seconds': time.perf_counter() - start,
            'output_dir': config['output_dir'], 'console_path': str(audit / 'console.txt'),
        }, ensure_ascii=False, indent=2), encoding='utf-8')
    if code != 0:
        print(f'FAILED Q{args.rank}: inspect {audit}; do not start the next rank.', flush=True)
        sys.exit(code or 1)
    print(f'DONE Q{args.rank}: {config["output_dir"]}\nResource report: {audit / "training_report.json"}', flush=True)
    sys.exit(0)

assert args.audit_dir and args.audit_dir.resolve().is_relative_to((ROOT / 'reports' / 'formal_training').resolve())
import torch, bitsandbytes as bnb
from transformers import TrainerCallback
from llamafactory.train.tuner import run_exp
audit = args.audit_dir
report = {
    'group': f'Q{args.rank}', 'purpose': 'Formal fixed-budget QLoRA training',
    'gpu': torch.cuda.get_device_name(0), 'device_total_bytes': torch.cuda.get_device_properties(0).total_memory,
    'config': config, 'config_sha256': freeze['formal_configs'][f'Q{args.rank}']['sha256'],
    'data_sha256': freeze['data_files_sha256'], 'packages': freeze['packages'],
    'heldout_test_not_used': True, 'epoch_validation': [],
    'memory_scope': 'PyTorch allocation/reservation from on_train_begin through training, validation and save; not whole-device use or loading peak.',
}

class Resources(TrainerCallback):
    def on_train_begin(self, args, state, control, model=None, **kw):
        self.parameters = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        assert self.parameters and all('lora_' in n for n, p in self.parameters)
        self.before = {n: p.detach().cpu().clone() for n, p in self.parameters if 'lora_B' in n}
        layers = [m for m in model.modules() if isinstance(m, bnb.nn.Linear4bit)]
        assert layers and all(m.weight.quant_state.quant_type == 'nf4' and m.weight.quant_state.state2 is not None and m.compute_dtype == torch.bfloat16 for m in layers)
        assert model.is_gradient_checkpointing
        report.update(trainable_parameters=sum(p.numel() for n, p in self.parameters),
                      quantized_linear_layers=len(layers), target_modules=sorted(model.peft_config['default'].target_modules))
        torch.cuda.reset_peak_memory_stats()
        self.checked = False
    def on_pre_optimizer_step(self, args, state, control, **kw):
        if not self.checked:
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for n, p in self.parameters)
            assert any(torch.count_nonzero(p.grad).item() for n, p in self.parameters)
            self.checked = True
    def on_log(self, args, state, control, logs=None, **kw):
        if logs and 'loss' in logs:
            assert math.isfinite(logs['loss']) and math.isfinite(logs.get('grad_norm', 0))
    def on_evaluate(self, args, state, control, metrics=None, **kw):
        assert math.isfinite(metrics['eval_loss'])
        report['epoch_validation'].append({'step': state.global_step, 'epoch': state.epoch, 'eval_loss': metrics['eval_loss']})
    def on_train_end(self, args, state, control, **kw):
        assert state.global_step == freeze['expected_optimizer_steps_per_group']
        assert self.checked and any(not torch.equal(self.before[n], p.detach().cpu()) for n, p in self.parameters if n in self.before)
        report.update(optimizer_steps=state.global_step, finite_gradients_and_parameter_updates=True)

try:
    start = time.perf_counter()
    run_exp(config, callbacks=[Resources()])
    torch.cuda.synchronize()
    report['train_eval_save_seconds'] = time.perf_counter() - start
    report['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
    report['peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
    output = Path(config['output_dir'])
    result = json.loads((output / 'eval_results.json').read_text(encoding='utf-8'))
    assert math.isfinite(result['eval_loss'])
    report['final_validation_loss'] = result['eval_loss']
    report['adapter_bytes'] = (output / 'adapter_model.safetensors').stat().st_size
    adapter = json.loads((output / 'adapter_config.json').read_text(encoding='utf-8'))
    assert adapter['r'] == args.rank and adapter['lora_alpha'] == args.rank * 2
    report['status'] = 'pass'
except BaseException as exc:
    report.update(status='fail', error=repr(exc))
    raise
finally:
    (audit / 'training_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
