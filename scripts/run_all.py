# This wrapper takes standard NiChart_DLMUSE/merged demographics CSV files and handles preprocessing steps necessary for SPARE, then invokes SPARE.

import argparse
import pandas as pd
import sys
import os
import tempfile
import re
import joblib
from pathlib import Path

parser = argparse.ArgumentParser(description="Run many SPARE predictions in one invocation (CVM or non-CVM). Preprocess CSV and run spare_scores")
parser.add_argument('-i', '--input', required=True, help='Input CSV file')
parser.add_argument('-demog','--demographics_csv', required=False, help='Optional demographics CSV to merge with input csv')
parser.add_argument('-o', '--output', required=True, help='Output CSV file')
parser.add_argument('--category', default='misc', help='Category to run. Values: "cvm", "misc"')
parser.add_argument('--harmonize', default=False, action='store_true', help='Pass to run harmonized SPARE models only (default: only run non-harmonized).')
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

def list_filtered_files(directory, forbidden_substrings):
    """
    Return all filenames in `directory` that do NOT contain
    any of the substrings in `forbidden_substrings`.
    """
    files = []
    for name in os.listdir(directory):
        full = os.path.join(directory, name)
        if os.path.isfile(full):
            if not any(sub in name for sub in forbidden_substrings):
                files.append(full)
    return files

def merge_dfs_on_mrid(df_list):
    """
    Merge a list of dataframes on the 'MRID' column.
    Returns a single merged dataframe.
    """
    if not df_list:
        raise ValueError("No dataframes provided.")

    # Start with the first dataframe
    merged = df_list[0]

    # Iteratively merge all remaining
    for df in df_list[1:]:
        merged = merged.merge(df, on="MRID", how="outer")

    return merged

def merge_csvs_on_mrid(csv_paths):
    """
    Load all CSVs in `csv_paths` and merge them on the 'MRID' column.
    Returns a single merged dataframe.
    """
    if not csv_paths:
        raise ValueError("No CSV paths provided.")

    # Load the first one as the base
    merged = pd.read_csv(csv_paths[0])

    # Iteratively merge all remaining
    for path in csv_paths[1:]:
        df = pd.read_csv(path)
        merged = merged.merge(df, on="MRID", how="outer")

    return merged

def extract_spare_tag(path):
    """
    Given a filename like SPARE-BA-RAW-xyz.joblib,
    return the tag after SPARE-, e.g. 'BA'.
    """
    name = os.path.basename(path)
    m = re.search(r"SPARE-([A-Za-z0-9]+)-", name)
    return m.group(1) if m else None


def rename_spare_columns(df, tag):
    """
    Rename SPARE/GT columns using the provided tag (e.g., 'MDD').
    Returns a new dataframe.
    """

    mapping = {}

    # SPARE_CL  -> SPARE_<tag>
    if "SPARE_CL" in df.columns:
        mapping["SPARE_CL"] = f"SPARE_{tag}"

    # SPARE_RG  -> SPARE_<tag>
    if "SPARE_RG" in df.columns:
        mapping["SPARE_RG"] = f"SPARE_{tag}"

    # SPARE_CL_decision_function -> SPARE_<tag>_decision_function
    if "SPARE_CL_decision_function" in df.columns:
        mapping["SPARE_CL_decision_function"] = f"SPARE_{tag}_decision_function"

    # GT_RG -> <tag>_GT_RG
    if "GT_RG" in df.columns:
        mapping["GT_RG"] = f"{tag}_GT_RG"

    return df.rename(columns=mapping)

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

non_cvm_models = list_filtered_files('/spare_score/Models/final_models', ['HYPERTENSION', 'DIABETES', 'OBESITY', 'SMOKING', 'HYPERLIPIDEMIA'])
cvm_models = list_filtered_files('/spare_score/Models/withDLWMLS', [])

model_location_dict = {'cvm': cvm_models, 'misc': non_cvm_models}

all_tmp_csvs = []
all_successful_tmp_csvs = []
successful_tags = []
exit_codes = []
encountered_spare_tags = []
for model in model_location_dict[args.category]:
    print(f"Model {model}")
    if 'harmonized' in model.lower():
        if not args.harmonize:
            print("Skipping...")
            continue
    else:
        if args.harmonize:
            print("Skipping...")
            continue
    model_joblib = joblib.load(model)
    inference_mode = model_joblib['meta_data']['spare_type']
    spare_tag = extract_spare_tag(model)
    if spare_tag in encountered_spare_tags:
        print("Skipping due to duplicate spare tag...")
        continue
    else:
        encountered_spare_tags.append(spare_tag)

    print(f"Spare tag {spare_tag}, inference mode {inference_mode}")
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_out:
        temp_out_path = tmp_out.name
        all_tmp_csvs.append(temp_out_path)
        
        command_parts = ['NiChart_SPARE', '-kv', 'MRID', '-i', temp_csv_path, '-o', temp_out_path, '-t', inference_mode, '-m', model] + unknown_args
        command = ' '.join(f"{part}" if ' ' in part else part for part in command_parts)
        
        exit_code = os.system(command)
        exit_codes.append(exit_code)
        if os.WEXITSTATUS(exit_code) > 0:
            print(f"Nonzero exit code encountered while executing SPARE model {model}")
        else:
            successful_tags.append(spare_tag)
            all_successful_tmp_csvs.append(temp_out_path)

dfs = []
for index, csv in  enumerate(all_successful_tmp_csvs):
    df = pd.read_csv(csv)
    df = rename_spare_columns(df, successful_tags[index])
    dfs.append(df)

df_merged = merge_dfs_on_mrid(dfs)
df_merged.to_csv(args.output)

if any([os.WEXITSTATUS(exit_code) > 0 for exit_code in exit_codes]):
    print("Some SPARE scores did not succeed, but those that did have been written to the specified output.")
    sys.exit(exit_code)



