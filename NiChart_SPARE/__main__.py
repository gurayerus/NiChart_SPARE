#!/usr/bin/env python3
"""
NiChart_SPARE CLI

Three-stage workflow:
  prep   — task-specific preprocessing  → standardized CSV [MRID, target, features]
  train  — SVM training driven by a JSON config → experiment directory with model + metadata
  test   — model inference on prepped CSV → output directory with predictions.csv
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime

from . import __version__


def _add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument('-kv', '--key_variable', default='MRID',
                   help='Column that uniquely identifies each sample (default: MRID)')
    p.add_argument('-v', '--verbose', type=int, default=1,
                   help='Verbosity level (0–3)')


def _build_prep_parser(sub):
    p = sub.add_parser('prep', help='Prepare raw data for training or inference')
    p.add_argument('-i', '--input',   required=True,  help='Input CSV path')
    p.add_argument('-o', '--output',  required=True,  help='Output prepared CSV path')
    p.add_argument('-t', '--type',    required=True,
                   help='SPARE type: CL, RG, AD, BA, CVM, HT, T2B, SM, BMI')
    p.add_argument('-tc', '--target_column', default=None,
                   help='Target column name (omit for inference inputs)')
    p.add_argument('-ic', '--ignore_columns', default='',
                   help='Comma-separated columns to drop (e.g. Study,SITE,Sex)')
    p.add_argument('-icv', '--icv_correction', default='False',
                   help='Divide MUSE ROI volumes by ICV (True/False)')
    p.add_argument('-icvc', '--icv_column', default='DL_MUSE_Volume_702',
                   help='ICV column name')
    p.add_argument('--age_col', default='Age',   help='Age column (CVM only)')
    p.add_argument('--sex_col', default='Sex',   help='Sex column (CVM only)')
    _add_common_args(p)
    return p


def _build_train_parser(sub):
    p = sub.add_parser('train', help='Train an SVM model from a YAML config file')
    p.add_argument('-c', '--config', required=True,
                   help='Path to a YAML training config file')
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

    # Optional inline prep (if --prep-type is given, input is treated as raw)
    p.add_argument('-pt', '--prep-type', default=None,
                   help='Run prep on --input before inference. '
                        'Same values as NiChart_SPARE prep -t (CL, RG, AD, BA, CVM, …)')
    p.add_argument('-ic', '--ignore-columns', default='',
                   help='Comma-separated columns to drop during prep (used with --prep-type)')
    p.add_argument('-icv', '--icv-correction', default='False',
                   help='Divide MUSE ROI volumes by ICV before prep (True/False)')
    p.add_argument('-icvc', '--icv-column', default='DL_MUSE_Volume_702',
                   help='ICV column name (used with --icv-correction)')
    p.add_argument('--age-col', default='Age',
                   help='Age column name for CVM residualization (used with --prep-type CVM)')
    p.add_argument('--sex-col', default='Sex',
                   help='Sex column name for CVM residualization (used with --prep-type CVM)')

    p.add_argument('--append_spare_tag', default='',
                   help='Rename SPARE_<type> → SPARE_<tag> in output column')
    _add_common_args(p)
    return p


def main():
    parser = argparse.ArgumentParser(
        prog='NiChart_SPARE',
        description=f'NiChart_SPARE v{__version__} — SPARE scores from brain ROI volumes',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List available pre-trained tasks
  NiChart_SPARE test --list-tasks

  # Quick inference using a registered task (downloads model automatically)
  NiChart_SPARE test -t AD -i prepped.csv -o ./results/

  # Specific version
  NiChart_SPARE test -t AD --version v3.0-raw -i prepped.csv -o ./results/

  # Explicit local model file
  NiChart_SPARE test -m model.joblib -i prepped.csv -o ./results/

  # Show download URL and manual-use instructions for a task
  NiChart_SPARE test -t AD --model-info

  # Prepare training data (CVM residualization applied automatically)
  NiChart_SPARE prep -t CVM -i raw.csv -o prepped.csv -tc Disease -ic Study,SITE

  # Train from a YAML config file
  NiChart_SPARE train -c configs/train_cvm.yaml

  # Prepare test data (same type, no target column required)
  NiChart_SPARE prep -t CVM -i raw_test.csv -o prepped_test.csv
""",
    )
    parser.add_argument('--version', action='version', version=f'%(prog)s {__version__}')

    sub = parser.add_subparsers(dest='action', metavar='ACTION')
    sub.required = True
    _build_prep_parser(sub)
    _build_train_parser(sub)
    _build_test_parser(sub)

    args = parser.parse_args()

    try:
        if args.action == 'prep':
            _run_prep(args)
        elif args.action == 'train':
            _run_train(args)
        elif args.action == 'test':
            _run_test(args)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


# ---------------------------------------------------------------------------
# Action handlers
# ---------------------------------------------------------------------------

def _run_prep(args):
    from .prep_data import prep_data

    ignore = [c.strip() for c in args.ignore_columns.split(',') if c.strip()]
    icv_correction = args.icv_correction.lower() == 'true'

    prep_data(
        input_file=args.input,
        spare_type=args.type,
        key_variable=args.key_variable,
        target_column=args.target_column,
        ignore_columns=ignore or None,
        output_file=args.output,
        icv_correction=icv_correction,
        icv_column=args.icv_column,
        age_col=args.age_col,
        sex_col=args.sex_col,
    )


def _run_train(args):
    from .train import train_model

    config_path = os.path.abspath(args.config)
    config_dir  = os.path.dirname(config_path)

    if not config_path.lower().endswith('.json'):
        raise ValueError("Training config must be a .json file.")

    with open(config_path) as fh:
        cfg = json.load(fh)

    desc      = cfg.get('description', {})
    data      = cfg.get('data', {})
    variables = cfg.get('variables', {})
    model_cfg = cfg.get('model', {})

    # Validate required fields
    for section, key in [('description', 'spare_type'), ('description', 'model_tag'),
                          ('description', 'run_tag'),
                          ('data', 'input'), ('data', 'output')]:
        if not cfg.get(section, {}).get(key):
            raise ValueError(f"Config missing required field: '{section}.{key}'")

    spare_type    = str(desc['spare_type'])
    model_tag     = str(desc['model_tag'])
    model_version = str(desc.get('model_version', '1.0'))
    run_tag       = str(desc['run_tag'])
    input_file    = os.path.join(config_dir, data['input'])
    output_folder = os.path.join(config_dir, data['output'])

    key_variable   = str(variables.get('key_variable', 'MRID'))
    target_column  = variables.get('target_column')
    ignore_columns = str(variables.get('ignore_columns', ''))
    icv_correction = bool(variables.get('icv_correction', False))
    icv_column     = str(variables.get('icv_column', 'DL_MUSE_Volume_702'))
    age_col        = str(variables.get('age_col', 'Age'))
    sex_col        = str(variables.get('sex_col', 'Sex'))

    svm_kernel            = str(model_cfg.get('svm_kernel', 'linear'))
    hyperparameter_tuning = bool(model_cfg.get('hyperparameter_tuning', True))
    train_whole           = bool(model_cfg.get('train_whole', True))
    cv_fold               = int(model_cfg.get('cv_fold', 5))
    class_balancing       = bool(model_cfg.get('class_balancing', True))
    bias_correction       = int(model_cfg.get('bias_correction', 0))
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

        # Inline prep — runs when target_column is set in the variables section
        if target_column:
            from .prep_data import prep_data
            _log(f"Prep       : {data['input']} → prepped.csv", log_fh)
            raw_file   = input_file
            input_file = os.path.join(run_dir, 'prepped.csv')
            ignore = [c.strip() for c in ignore_columns.split(',') if c.strip()]
            prep_data(
                input_file=raw_file,
                spare_type=spare_type,
                key_variable=key_variable,
                target_column=target_column,
                ignore_columns=ignore or None,
                output_file=input_file,
                icv_correction=icv_correction,
                icv_column=icv_column,
                age_col=age_col,
                sex_col=sex_col,
            )
            _log(f"Prep done  : {input_file}", log_fh)

        model_path = os.path.join(run_dir, f"{run_tag}.joblib")
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
            kernel=svm_kernel,
            tune_hyperparameters=hyperparameter_tuning,
            cv_fold=cv_fold,
            class_balancing=class_balancing,
            cross_validate=cv_fold != 0,
            train_whole_set=train_whole,
            bias_correction=bias_correction,
            verbose=verbose,
            model_tag=model_tag,
            model_version=model_version,
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

    # Inline prep — if --prep-type is given, treat --input as raw data
    input_file = args.input
    if args.prep_type:
        from .prep_data import prep_data
        prepped_path = os.path.join(args.output_dir, 'prepped.csv')
        ignore = [c.strip() for c in args.ignore_columns.split(',') if c.strip()]
        prep_data(
            input_file=args.input,
            spare_type=args.prep_type,
            key_variable=args.key_variable,
            target_column=None,
            ignore_columns=ignore or None,
            output_file=prepped_path,
            icv_correction=args.icv_correction.lower() == 'true',
            icv_column=args.icv_column,
            age_col=args.age_col,
            sex_col=args.sex_col,
        )
        input_file = prepped_path
        print(f"Prepped data saved to: {prepped_path}")

    infer_model(
        input_file=input_file,
        model_path=model_path,
        output_dir=args.output_dir,
        key_variable=args.key_variable,
        append_spare_tag=args.append_spare_tag,
    )


if __name__ == '__main__':
    main()
