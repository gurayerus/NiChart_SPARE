#!/usr/bin/env python3
"""
NiChart_SPARE CLI

Two-stage workflow:
  train  — SVM training driven by a JSON config -> experiment directory with model + metadata
           (data prep is performed automatically and saved alongside the model)
  test   — model inference on raw CSV -> output directory with prepped.csv + predictions.csv
           (prep is applied automatically using parameters stored inside the model)
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime

from . import __version__


def _build_train_parser(sub):
    p = sub.add_parser('train', help='Train an SVM model from a JSON config file')
    p.add_argument('-c', '--config', required=True,
                   help='Path to a JSON training config file')
    p.add_argument('-n', '--notes', default='',
                   help='Free-text notes stored in the experiment meta.json')
    return p


def _build_test_parser(sub):
    p = sub.add_parser('test', help='Run inference or query model info')

    # Utility flags (short-circuit normal inference)
    p.add_argument('--list-tasks', action='store_true',
                   help='List all tasks in the registry and exit')
    p.add_argument('--model-info', action='store_true',
                   help='Print model info for --task/--version and exit')

    # I/O (required for inference, not needed for --list-tasks / --model-info)
    p.add_argument('-i', '--input', default=None,
                   help='Input CSV path — prepped or raw (required for inference)')
    p.add_argument('-o', '--output_dir', default=None,
                   help='Output directory; predictions.csv (and prepped.csv if prep runs) '
                        'are written inside (required for inference)')

    # Model selection — --task and --model are mutually exclusive
    model_group = p.add_mutually_exclusive_group()
    model_group.add_argument('-t', '--task', default=None,
                             help='Task name from the registry (e.g. AD, BA, BMI). '
                                  'Model is downloaded automatically on first use.')
    model_group.add_argument('-m', '--model', default=None,
                             help='Explicit path to a .joblib model file')

    p.add_argument('--version', default=None,
                   help='Model version to use with --task (default: task\'s default version)')

    return p


def main():
    parser = argparse.ArgumentParser(
        prog='NiChart_SPARE',
        description=f'NiChart_SPARE v{__version__} — SPARE scores from brain ROI volumes',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train a model from a JSON config (prep runs automatically)
  NiChart_SPARE train -c examples/SPARE_AD/input/train_config.json

  # Inference with a registered task (downloads model, preps input automatically)
  NiChart_SPARE test -t AD -i raw.csv -o ./results/

  # Inference with a specific version
  NiChart_SPARE test -t AD --version v3.0 -i raw.csv -o ./results/

  # Inference with a local model file
  NiChart_SPARE test -m model.joblib -i raw.csv -o ./results/

  # List available pre-trained tasks
  NiChart_SPARE test --list-tasks

  # Show info / download URL for a task
  NiChart_SPARE test -t AD --model-info
""",
    )
    parser.add_argument('--version', action='version', version=f'%(prog)s {__version__}')

    sub = parser.add_subparsers(dest='action', metavar='ACTION')
    sub.required = True
    _build_train_parser(sub)
    _build_test_parser(sub)

    args = parser.parse_args()

    try:
        if args.action == 'train':
            _run_train(args)
        elif args.action == 'test':
            _run_test(args)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


# ---------------------------------------------------------------------------
# Action handlers
# ---------------------------------------------------------------------------

def _run_train(args):
    from .train import train_model

    config_path = os.path.abspath(args.config)
    config_dir  = os.path.dirname(config_path)

    if not config_path.lower().endswith('.json'):
        raise ValueError("Training config must be a .json file.")

    with open(config_path) as fh:
        cfg = json.load(fh)

    desc      = cfg.get('description', {})
    inp       = cfg.get('input', {})
    out_cfg   = cfg.get('output', {})
    pre_proc  = cfg.get('pre-processing') or {}
    post_proc = cfg.get('post-processing') or {}
    model_cfg = cfg.get('model', {})

    # Validate required fields
    for section, key in [
        ('description', 'spare_type'), ('description', 'model_tag'),
        ('description', 'run_tag'), ('input', 'in_csv'), ('output', 'out_dir'),
        ('input', 'target_col'), ('model', 'svm_type'),
    ]:
        if not cfg.get(section, {}).get(key):
            raise ValueError(f"Config missing required field: '{section}.{key}'")

    spare_type    = str(desc['spare_type'])
    model_tag     = str(desc['model_tag'])
    model_version = str(desc.get('model_version', '1.0'))
    run_tag       = str(desc['run_tag'])

    # Input section
    in_dir        = str(inp.get('in_dir', '.'))
    in_csv        = str(inp['in_csv'])
    key_col       = str(inp.get('key_col', 'MRID'))
    target_col    = inp.get('target_col')
    data_cols     = inp.get('data_cols') or None   # feature column patterns; trailing '*' = prefix wildcard
    mappings      = inp.get('mappings') or None

    # Output section
    out_dir       = str(out_cfg['out_dir'])
    out_csv       = str(out_cfg.get('out_csv', 'predictions.csv'))
    out_cols      = out_cfg.get('out_cols') or None
    out_model_dir = str(out_cfg.get('out_model_dir', ''))

    input_file    = os.path.join(config_dir, in_dir, in_csv)
    output_folder = os.path.join(config_dir, out_dir)

    svm_type              = str(model_cfg['svm_type'])
    svm_kernel            = str(model_cfg.get('svm_kernel', 'linear'))
    hyperparameter_tuning = bool(model_cfg.get('hyperparameter_tuning', True))
    train_whole           = bool(model_cfg.get('train_whole', True))
    cv_fold               = int(model_cfg.get('cv_fold', 5))
    class_balancing       = bool(model_cfg.get('class_balancing', True))
    age_bias_correction   = int(model_cfg.get('age_bias_correction', 0))
    verbose               = int(model_cfg.get('verbose', 1))

    # run_tag determines the output directory — fail early if it already exists
    run_dir = os.path.join(output_folder, run_tag)
    if os.path.exists(run_dir):
        print(
            f"Warning: output directory already exists: {run_dir}\n"
            f"Change 'description.run_tag' in the config to use a different name.",
            file=sys.stderr,
        )
        sys.exit(1)
    os.makedirs(run_dir)

    # Open timestamped log (writes alongside stdout for every key event)
    log_path = os.path.join(run_dir, 'train.log')

    def _log(msg: str, log_fh) -> None:
        ts   = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        line = f"[{ts}]  {msg}"
        print(line)
        log_fh.write(line + '\n')
        log_fh.flush()

    with open(log_path, 'w') as log_fh:
        _log(f"Training started", log_fh)
        _log(f"Config     : {config_path}", log_fh)
        _log(f"Run tag    : {run_tag}", log_fh)
        _log(f"SPARE type : {spare_type}  |  model: {model_tag} v{model_version}", log_fh)
        _log(f"Output dir : {run_dir}", log_fh)

        shutil.copy2(config_path, os.path.join(run_dir, 'config.json'))

        # Capture git commit (best-effort)
        try:
            git_commit = subprocess.check_output(
                ['git', 'rev-parse', 'HEAD'],
                text=True, stderr=subprocess.DEVNULL,
                cwd=config_dir,
            ).strip()
        except Exception:
            git_commit = None

        meta = {
            'run_tag':       run_tag,
            'timestamp':     datetime.now().isoformat(),
            'model_tag':     model_tag,
            'model_version': model_version,
            'git_commit':    git_commit,
            'notes':         args.notes or '',
        }
        with open(os.path.join(run_dir, 'meta.json'), 'w') as fh:
            json.dump(meta, fh, indent=2)

        # Prep step — always runs; saves prepped.csv alongside the model
        from .prep_data import prep_data
        _log(f"Prep       : {in_csv} -> prepped.csv", log_fh)
        raw_file   = input_file
        input_file = os.path.join(run_dir, 'prepped.csv')
        _, cvm_mean_age = prep_data(
            input_file=raw_file,
            key_col=key_col,
            target_col=target_col,
            data_cols=data_cols,
            mappings=mappings,
            preprocessing=pre_proc or None,
            output_file=input_file,
        )
        _log(f"Prep done  : {input_file}", log_fh)

        # Prep config saved into the model for automatic replay at inference time
        prep_config = {
            'key_col':       key_col,
            'target_col':    target_col,
            'data_cols':     data_cols,
            'mappings':      mappings,
            'preprocessing': pre_proc or None,
            'cvm_mean_age':  cvm_mean_age,   # None when no residualization
        }

        # Output config: filename/column filter + post-processing steps for inference
        output_config = {
            'out_csv':         out_csv,
            'out_cols':        out_cols,
            'post_processing': post_proc or None,
        }

        # Model lives in out_model_dir subdir within run_dir (if specified)
        model_dir = os.path.join(run_dir, out_model_dir) if out_model_dir else run_dir
        if out_model_dir:
            os.makedirs(model_dir)
        model_path = os.path.join(model_dir, f"{run_tag}.joblib")
        _log(
            f"SVM        : kernel={svm_kernel}  cv_fold={cv_fold}"
            f"  hp_tuning={hyperparameter_tuning}  class_bal={class_balancing}",
            log_fh,
        )
        _log(f"Training started", log_fh)
        t0 = datetime.now()

        cv_results = train_model(
            input_file=input_file,
            model_path=model_path,
            spare_type=spare_type,
            svm_type=svm_type,
            kernel=svm_kernel,
            tune_hyperparameters=hyperparameter_tuning,
            cv_fold=cv_fold,
            class_balancing=class_balancing,
            cross_validate=cv_fold != 0,
            train_whole_set=train_whole,
            bias_correction=age_bias_correction,
            verbose=verbose,
            model_tag=model_tag,
            model_version=model_version,
            prep_config=prep_config,
            output_config=output_config,
        )

        elapsed = str(datetime.now() - t0).split('.')[0]
        _log(f"Training complete (elapsed: {elapsed})", log_fh)
        _log(f"Model saved: {model_path}", log_fh)

        if cv_results:
            import numpy as np
            import pandas as pd

            class _NumpyEncoder(json.JSONEncoder):
                def default(self, obj):
                    if isinstance(obj, np.integer):
                        return int(obj)
                    if isinstance(obj, np.floating):
                        return float(obj)
                    if isinstance(obj, np.ndarray):
                        return obj.tolist()
                    if isinstance(obj, pd.DataFrame):
                        return obj.to_dict(orient='list')
                    if isinstance(obj, pd.Series):
                        return obj.tolist()
                    return super().default(obj)

            with open(os.path.join(run_dir, 'metrics.json'), 'w') as fh:
                json.dump(cv_results, fh, indent=2, cls=_NumpyEncoder)
            _log(f"Metrics    : metrics.json", log_fh)

        _log(f"Done. Output: {run_dir}", log_fh)


def _run_test(args):
    from .task_registry import list_tasks, print_model_info, resolve_model_path
    from .inference import infer_model

    # Utility-only flags — no inference needed
    if args.list_tasks:
        list_tasks()
        return

    if args.model_info:
        if not args.task:
            raise ValueError("--model-info requires --task")
        print_model_info(args.task, args.version)
        return

    # For actual inference, validate required args
    if not args.task and not args.model:
        raise ValueError(
            "One of --task or --model is required for inference.\n"
            "Run 'NiChart_SPARE test --list-tasks' to see available tasks."
        )
    if not args.input:
        raise ValueError("--input is required for inference.")
    if not args.output_dir:
        raise ValueError("--output_dir is required for inference.")

    model_path = resolve_model_path(
        task=args.task,
        version=args.version,
        model=args.model,
    )

    os.makedirs(args.output_dir, exist_ok=True)

    infer_model(
        input_file=args.input,
        model_path=model_path,
        output_dir=args.output_dir,
    )


if __name__ == '__main__':
    main()
