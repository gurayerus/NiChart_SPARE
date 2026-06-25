#!/usr/bin/env python3
"""
NiChart_SPARE CLI

Three-stage workflow:
  prep   — task-specific preprocessing  → standardized CSV [MRID, target, features]
  train  — SVM training on prepped CSV  → model .joblib
  test   — model inference on prepped CSV → predictions CSV
"""
import argparse
import os
import sys

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
    p = sub.add_parser('train', help='Train an SVM model on prepared data')
    p.add_argument('-i',  '--input',        required=True, help='Prepared CSV path')
    p.add_argument('-mo', '--model_output', required=True, help='Output model path (.joblib)')
    p.add_argument('-t',  '--type',         required=True,
                   help='SPARE type: CL, RG, AD, BA, CVM, HT, T2B, SM, BMI')
    p.add_argument('-sk', '--svm_kernel',   default='linear',
                   help='SVM kernel: linear_fast, linear, rbf, poly, sigmoid')
    p.add_argument('-ht', '--hyperparameter_tuning', default='True',
                   help='Run GridSearchCV (True/False)')
    p.add_argument('-tw', '--train_whole',  default='True',
                   help='Train final model on the full dataset (True/False)')
    p.add_argument('-cf', '--cv_fold',      type=int, default=5,
                   help='Cross-validation folds (0 to skip CV, default: 5)')
    p.add_argument('-cb', '--class_balancing', default='True',
                   help='Enable class_weight="balanced" (True/False, classification only)')
    p.add_argument('-bc', '--bias_correction', default='0',
                   help='Bias correction: 0=none, 1=Beheshti et al., 2=Cole et al. (regression only)')
    _add_common_args(p)
    return p


def _build_test_parser(sub):
    p = sub.add_parser('test', help='Run inference with a trained model on prepared data')
    p.add_argument('-i', '--input',  required=True, help='Prepared CSV path')
    p.add_argument('-m', '--model',  required=True, help='Model file path (.joblib)')
    p.add_argument('-o', '--output', required=True, help='Output predictions CSV path')
    p.add_argument('--append_spare_tag', default='',
                   help='Rename SPARE_<type> → SPARE_<tag> in output')
    _add_common_args(p)
    return p


def main():
    parser = argparse.ArgumentParser(
        prog='NiChart_SPARE',
        description=f'NiChart_SPARE v{__version__} — SPARE scores from brain ROI volumes',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 1. Prepare training data (CVM residualization applied automatically)
  NiChart_SPARE prep -t CVM -i raw.csv -o prepped_train.csv -tc Disease -ic Study,SITE

  # 2. Train a linear SVM classifier
  NiChart_SPARE train -t CVM -i prepped_train.csv -mo model.joblib -sk linear -ht True -cf 5

  # 3. Prepare test data (same prep type, no target column required)
  NiChart_SPARE prep -t CVM -i raw_test.csv -o prepped_test.csv

  # 4. Run inference
  NiChart_SPARE test -i prepped_test.csv -m model.joblib -o predictions.csv
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

    tune   = args.hyperparameter_tuning.lower() == 'true'
    whole  = args.train_whole.lower() == 'true'
    bal    = args.class_balancing.lower() == 'true'
    cv     = args.cv_fold != 0
    bc     = int(args.bias_correction)

    train_model(
        input_file=args.input,
        model_path=args.model_output,
        spare_type=args.type,
        kernel=args.svm_kernel,
        tune_hyperparameters=tune,
        cv_fold=args.cv_fold,
        class_balancing=bal,
        cross_validate=cv,
        train_whole_set=whole,
        bias_correction=bc,
        verbose=args.verbose,
    )


def _run_test(args):
    from .inference import infer_model

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    infer_model(
        input_file=args.input,
        model_path=args.model,
        output_file=args.output,
        key_variable=args.key_variable,
        append_spare_tag=args.append_spare_tag,
    )


if __name__ == '__main__':
    main()
