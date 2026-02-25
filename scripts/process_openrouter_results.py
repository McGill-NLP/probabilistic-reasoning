import os 
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import json 
import re 
from typing import Dict, Any

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw-output-filepath', type=str) # Path to raw output file
    parser.add_argument('--processed-output-dir', type=str, default='./results/')
    parser.add_argument('--original-dataset-filepath', type=str, default='./datasets/probcopa_items.jsonl') # Path to original dataset file
    parser.add_argument('--results-tag', type=str)
    args = parser.parse_args()
    process_openrouter_results(args)

def extract_answer_from_structured_text(answer_text: str) -> str:
    answer = answer_text.strip()
    try:
        answer = re.findall(r"\<answer>([-+]?[0-9]*\.?[0-9]+)<\/answer>", answer)[0]
        return answer
    except:
        return 'error'

def get_answer_from_raw_response(raw_response: Dict[str, Any]) -> str:
    choices = raw_response['choices']
    if choices is None:
        return 'error'
    return extract_answer_from_structured_text(choices[0]['message']['content'])


def try_converting_to_float(x):
    try:
        return float(x)
    except:
        return x

def process_openrouter_results(args):
    raw_output_filepath = args.raw_output_filepath
    processed_output_dir = args.processed_output_dir
    results_tag = args.results_tag
    if not os.path.exists(processed_output_dir):
        os.makedirs(processed_output_dir)
    processed_output_filepath = os.path.join(processed_output_dir, f"{results_tag}.jsonl")
    df = pd.read_json(raw_output_filepath, lines=True)
    df['answer'] = df['raw_response'].apply(lambda x: get_answer_from_raw_response(x))
    df['finish_reason'] = df['raw_response'].apply(lambda x: x['choices'][0]['finish_reason'] if x['choices'] is not None else 'error')
    df['answer'] = df['answer'].apply(try_converting_to_float) # Convert to int if possible
    df['reasoning_token_count'] = df['raw_response'].apply(lambda x: x['usage']['completion_tokens_details']['reasoning_tokens'] if x['usage'] is not None else 'error')
    if 'persona_id' not in df.columns:
        df_to_merge = df[['UID', 'answer', 'finish_reason']].copy()
        df_to_merge['run'] = df_to_merge.groupby('UID').cumcount()
    else:
        df_to_merge = df[['UID', 'persona_id', 'answer', 'finish_reason']].copy()
    # Now, link to original dataset:
    original_df = pd.read_json(args.original_dataset_filepath, lines=True)
    processed_df = df_to_merge.merge(original_df, on='UID', how='left')
    processed_df['summary'] = None # Grok does not support provide reasoning summaries
    processed_df.to_json(processed_output_filepath, orient='records', lines=True)
    print("Done!")

if __name__ == "__main__":
    main()

