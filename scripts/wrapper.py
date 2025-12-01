# This wrapper takes standard NiChart_DLMUSE/merged demographics CSV files and handles preprocessing steps necessary for SPARE, then invokes SPARE.

import argparse
import pandas as pd
import sys
import os
import tempfile
import os
from pathlib import Path

parser = argparse.ArgumentParser(description="Preprocess CSV and run spare_scores")
parser.add_argument('-i', '--input', required=True, help='Input CSV file')
parser.add_argument('-demog','--demographics_csv', required=False, help='Optional demographics CSV to merge with input csv')
args, unknown_args = parser.parse_known_args()

if not args.input.lower().endswith('.csv'):
    print("Error: input file must be a CSV file.")
    sys.exit(1)
if not os.path.exists(args.input):
    print(f"Error: Input file '{args.input}' does not exist.")
    sys.exit(1)
if args.demographics_csv is not None:
    if not args.demographics_csv.lower().endswith('.csv'):
        print("Error: demographics file must be a CSV file.")
        sys.exit(1)
    if not os.path.exists(args.demographics_csv):
        print(f"Error: Input file '{args.demographics_csv}' does not exist.")
        sys.exit(1)

df_original = pd.read_csv(args.input)

if args.demographics_csv is not None:
    print("Detected optional demographics file. Merging files to ensure all data is available for SPARE.")
    df_demog = pd.read_csv(args.demographics_csv)
    len_original = len(df_original)
    len_demog = len(df_demog)
    print(f"Length of original data: { len_original }")
    df_original = pd.merge(df_original, df_demog, on='MRID', how='inner')
    len_merged = len(df_original)
    print(f"Length of demographic data: { len_demog }")
    print(f"Length of merged data: { len_merged }")
    if len_merged < len_original:
        print("WARNING: Merged CSV has fewer entries than the original data. Check demographics and input CSVs for completeness.")

if 'Sex' not in df_original.columns:
    print("Error: Required 'Sex' column not found in the CSV file.")
    sys.exit(1)

df_original['Sex_M'] = df_original['Sex'].apply(lambda x: 1 if x=='M' else 0)

if 'DL_MUSE_Volume_702' in df_original.columns:
    df_original['702'] = df_original['DL_MUSE_Volume_702']
if 'H_DL_MUSE_Volume_702' in df_original.columns:
    df_original['702'] = df_original['H_DL_MUSE_Volume_702']

with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_file:
    temp_csv_path = tmp_file.name
    df_original.to_csv(temp_csv_path, index=False)

command_parts = ['NiChart_SPARE', '-kv', 'MRID', '-i', temp_csv_path] + unknown_args
command = ' '.join(f"{part}" if ' ' in part else part for part in command_parts)

exit_code = os.system(command)

if os.WEXITSTATUS(exit_code) > 0:
    sys.exit(exit_code)



