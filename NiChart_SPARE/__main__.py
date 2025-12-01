#!/usr/bin/env python3
"""
NiChart_SPARE - Main entry point for SPARE scores calculation

This script provides command-line interface for training and inference of SPARE models.
Supported SPARE types: 
    - BA (Brain Age)
    - AD (Alzheimer's)
    - CVMs: HT (Hypertension), HL (Hyperlipidemia), T2B (Diabetes), SM (Smoking), OB (Obesity)
"""
import os
import argparse
import sys
from .svm import (
	train_svm_model, 
	infer_svm_model
)

# Entry point & CLI Args
def main():
    """Main entry point for NiChart_SPARE"""
    parser = argparse.ArgumentParser(
        description="NiChart_SPARE - SPARE scores calculation from Brain ROI Volumes",
        epilog="""
            Examples:
            # Train AD model with hyperparameter tuning
            NiChart_SPARE -a trainer -t AD -i data.csv -mo model.pkl -v True
            
            # Train final model after tuning
            NiChart_SPARE -a trainer -t AD -i data.csv -mo model.pkl -v True -f True
            
            # Make predictions
            NiChart_SPARE -a inference -t AD -i test.csv -m model.pkl -o predictions.csv
        """
    )
    
    # Required arguments
    parser.add_argument('-a', '--action', required=True, choices=['trainer', 'inference', 'analysis'],
                       help='Action to perform: trainer, inference, or analysis')
    parser.add_argument('-t', '--type', required=True, 
                       help="SPARE type: CL (Classfication), RG (Regression), BA (Brain Age), AD (Alzheimer\'s),\n CVM (Cardiovascular and metabolic (Govindarajan et al.))")
    parser.add_argument('-i', '--input', required=True,
                       help='Input CSV file path')
    # Model specific arguments
    parser.add_argument('-mt', '--model_type', type=str, default='SVM',
                        help='Type of ML model. Currently supported: SVM')
    ## SVM specific
    parser.add_argument('-sk', '--svm_kernel', type=str, default='linear',
                       help='SVM kernel type (linear, poly, rbf, sigmoid)')
    # parser.add_argument('-bc', '--bias_correction', type=str, default='False',
    #                    help='Perform bias correction for linearSVM (linear_fast) models.')
    parser.add_argument('-bc', '--bias_correction', required=False, type=str, default='1',
                       help='Perform bias correction for regression task. 0 for disabling. 1 for Beheshti et al. 2 for Cole et al.')
    # ICV correction
    parser.add_argument('-icv','--icv_correction', type=str, default='False',
                        help='Perform ICV correction (all ROI features divided by ICV). (True/False)')
    parser.add_argument('-icvc','--icv_column', type=str, default='DL_MUSE_Volume_702',
                        help='Name of the ICV column in the input csv.')
    ## MLP specific
    ### TBA
    # Train/Test specific arguments
    parser.add_argument('-ht', '--hyperparameter_tuning', type=str, default='True',
                       help='Perform hyperparameter tuning job. Takes a while. (True/False)')
    parser.add_argument('-tw', '--train_whole', type=str, default='True',
                       help='Train final model on entire dataset (True/False)')
    parser.add_argument('-cf', '--cv_fold', type=int, default=5,
                       help='Number of folds for CV (Default: 5)')
    parser.add_argument('-mo', '--model_output', 
                       help='Output model file path (for training)')
    # Inference specific arguments
    parser.add_argument('-m', '--model', 
                       help='Input model file path (for inference)')
    parser.add_argument('-o', '--output', 
                       help='Output CSV file path (for inference)')
    # Analysis specific arguments
    parser.add_argument('-di', '--disease', type=str, default='AD',
                         help='Name of column indicating unique disease')
    # data preprocessing arguments
    parser.add_argument('-kv', '--key_variable',
                       help='Name of column indicating unique data points in the input CSV', default="MRID")
    parser.add_argument('-tc', '--target_column', default='target',
                       help='Name of target column in CSV')
    parser.add_argument('-ic', '--ignore_column', default='',
                       help='Comma-separated list of column names to drop from input CSV')
    parser.add_argument('-cb', '--class_balancing', type=str, default='True',
                        help='Enable SVM Class Balancing for Training')
    # Misc arguments
    parser.add_argument('-v', '--verbose', type=int, default=0,
                       help='Control the amount of output messages (0, 1, 2, 3)')
    parser.add_argument('--append-spare-tag', type=str, default='',
                       help='Post-process SPARE output CSV by applying the provided tag, e.g. SPARE_score becomes SPARE_{tag}. Mostly useful for pipelining. (Inference only).')
    args = parser.parse_args()
    
    # Convert string arguments to boolean
    tune_hyperparameters = args.hyperparameter_tuning.lower() == 'true'
    train_whole_set = args.train_whole.lower() == 'true'
    class_balancing = args.class_balancing.lower() == 'true'
    icv_correction = args.icv_correction.lower() == 'true'

    bias_correction = int(args.bias_correction)
    cross_validate = int(args.cv_fold) != 0
    
    # Parse columns to drop
    if ',' in args.ignore_column:
        ignore_columns = args.ignore_column.split(',')
    elif args.ignore_column==None:
        ignore_columns = None
    else:
        ignore_columns = [args.ignore_column]
    
    try:
        if args.action == 'trainer':
            if not args.model_output:
                raise ValueError("Model output path (-mo) is required for training")
            print(f"Training {args.model_type} model")

            if args.model_type == 'SVM':
                train_svm_model(
                    input_file=args.input,
                    model_path=args.model_output,
                    spare_type=args.type,
                    target_column=args.target_column,
                    kernel=args.svm_kernel,
                    tune_hyperparameters=tune_hyperparameters,
                    cv_fold=args.cv_fold,
                    class_balancing=class_balancing,
                    cross_validate=cross_validate,
                    train_whole_set=train_whole_set,
                    bias_correction=bias_correction,
                    icv_correction = icv_correction,
                    drop_columns=ignore_columns + [args.key_variable],
                    verbose=args.verbose
                )
            elif args.model_type == 'MLP':
                print("MLP is coming soon!")
            else:
                print(f"{args.model_type} is an unsupported model type.")
            
        elif args.action == 'inference':
            # Check Arguments
            if not args.model:
                raise ValueError("Model path (-m) is required for inference")
            if not args.output:
                raise ValueError("Output path (-o) is required for inference")
            # Create output directory if it doesn't exist
            if not os.path.exists(os.path.dirname(args.output)):
                print(f"Output directory does not exist. Creating f{os.path.dirname(args.output)}...")
                os.mkdir(os.path.dirname(args.output))
            # Run inference
            if args.model_type == 'SVM':
                infer_svm_model(
                    input_file=args.input,
                    model_path=args.model,
                    spare_type=args.type,
                    output_file=args.output,
                    key_variable=args.key_variable,
                    append_spare_tag=args.append_spare_tag,
                )
            elif args.model_type == 'MLP':
                print("MLP is coming soon!")
            else:
                print(f"{args.model_type} is an unsupported model type.")
            
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
