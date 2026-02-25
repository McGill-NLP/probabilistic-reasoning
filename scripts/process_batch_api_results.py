import os 
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import json 
import re 
from typing import Dict, Any
from transformers import AutoTokenizer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw-output-filepath', type=str) # Path to raw output file
    parser.add_argument('--processed-output-dir', type=str, default='./results/')
    parser.add_argument('--original-dataset-filepath', type=str, default='./datasets/probcopa_items.jsonl') # Path to original dataset file
    parser.add_argument('--results-tag', type=str)
    parser.add_argument('--provider', type=str)
    parser.add_argument('--model', type=str, help="Only required for Together AI models")
    args = parser.parse_args()
    process_batch_api_results(args)

def extract_answer_and_summary_from_unstructured_text(raw_text: str) -> Dict[str, Any]:
    # Prefer explicit <answer> tags if present
    answer_match = re.search(r'<answer>\s*([^<]+?)\s*</answer>', raw_text, flags=re.IGNORECASE | re.DOTALL)
    if answer_match:
        answer = answer_match.group(1).strip()
        try:
            answer = re.findall(r"[-+]?[0-9]*\.?[0-9]+", answer)[0]
        except Exception:
            pass
        try:
            summary = re.search(r'^(.*?)<answer>\s*[^<]+?\s*</answer>', raw_text, re.DOTALL | re.IGNORECASE).group(1).strip()
        except Exception:
            summary = ''
        return {'answer': answer, 'summary': summary, 'raw_text': raw_text}
    else:
        summary = raw_text.strip()
        return {'answer': 'error', 'summary': summary, 'raw_text': raw_text}

def extract_answer_from_structured_text(answer_text: str) -> str:
    answer = answer_text.strip()
    try:
        answer = re.findall(r"\<answer>([-+]?[0-9]*\.?[0-9]+)<\/answer>", answer)[0]
        return answer
    except:
        return 'error'

def try_converting_to_float(x):
    try:
        return float(x)
    except:
        return x

def process_batch_api_results(args):
    raw_output_filepath = args.raw_output_filepath
    processed_output_dir = args.processed_output_dir
    results_tag = args.results_tag
    if not os.path.exists(processed_output_dir):
        os.makedirs(processed_output_dir)
    processed_output_filepath = os.path.join(processed_output_dir, f"{results_tag}.jsonl")
    provider = args.provider
    model = args.model
    if model is not None:
        tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        def get_reasoning_token_count_with_tokenizer(summary):
            try:
                encoded = tokenizer.encode(summary)
                return len(encoded)
            except:
                return 'error'
    #
    else:
        tokenizer = None
    if provider == 'openai':
        df = pd.read_json(raw_output_filepath, lines=True)
        df['UID'] = df['custom_id'].str.split('-').str[1].apply(lambda x: int(x))
        df['run'] = df['custom_id'].str.split('-').str[2]+"_"+df['custom_id'].str.split('-').str[3]
        df['response_outputs'] = df['response'].apply(lambda x: x['body']['output'])
        df['finish_reason'] = df['response'].apply(lambda x: x['body']['incomplete_details'])
        df['reasoning_token_count'] = df['response'].apply(lambda x: x['body']['usage']['output_tokens_details']['reasoning_tokens'] if x['body']['usage'] is not None else 'error')
        def get_summary_from_response_outputs(response_outputs):
            summaries = []
            try:
                for output in response_outputs:
                    if output['type'] == 'reasoning':
                        for summary in output['summary']:
                            summaries.append(summary['text'])
                return "\n".join(summaries)
            except:
                return 'error'
        df['summary'] = df['response_outputs'].apply(get_summary_from_response_outputs)
        def get_answer_from_response_outputs(response_outputs):
            try:
                for output in response_outputs:
                    if output['type'] == 'message':
                        raw_answer = output['content'][0]['text']
                        answer = extract_answer_from_structured_text(raw_answer)
                        return answer
            except:
                return 'error'
        df['answer'] = df['response_outputs'].apply(get_answer_from_response_outputs)
        df['answer'] = df['answer'].apply(try_converting_to_float) # Convert to int if possible
        # Drop other columns and now stitch together with original dataframe to get clean results:
        df = df[['UID', 'run', 'answer', 'summary', 'finish_reason', 'reasoning_token_count']]
    elif provider == 'anthropic':
        df = pd.read_json(raw_output_filepath, lines=True)
        df['UID'] = df['custom_id'].str.split('-').str[1].apply(lambda x: int(x))
        df['run'] = df['custom_id'].str.split('-').str[2]+"_"+df['custom_id'].str.split('-').str[3]
        df['response_outputs'] = df['result'].apply(lambda x: x['message']['content'])
        df['finish_reason'] = df['result'].apply(lambda x: x['message']['stop_reason'])
        df['reasoning_token_count'] = df['result'].apply(lambda x: x['message']['usage']['output_tokens'] if x['message']['usage'] is not None else 'error')
        # This is not exactly correct, but close enough for now -- Claude doesn't return the reasoning token count, but total output tokens will be a close enough approximation
        def get_summary_from_response_outputs(response_outputs):
            try:
                for output in response_outputs:
                    if output['type'] == 'thinking':
                        return output['thinking']
            except:
                return 'error'
        df['summary'] = df['response_outputs'].apply(get_summary_from_response_outputs)
        def get_answer_from_response_outputs(response_outputs):
            try:
                for output in response_outputs:
                    if output['type'] == 'text':
                        raw_answer = output['text']
                        answer = extract_answer_from_structured_text(raw_answer)
                        return answer
            except:
                return 'error'
        df['answer'] = df['response_outputs'].apply(get_answer_from_response_outputs)
        df['answer'] = df['answer'].apply(try_converting_to_float) # Convert to int if possible
        df = df[['UID', 'run', 'answer', 'summary', 'finish_reason', 'reasoning_token_count']]
    elif provider == 'google':
        df = pd.read_json(raw_output_filepath, lines=True)
        df['UID'] = df['key'].str.split('-').str[1].apply(lambda x: int(x))
        df['run'] = df['key'].str.split('-').str[2]+"_"+df['key'].str.split('-').str[3]
        df['response_outputs'] = df['response'].apply(lambda x: x['candidates'][0]['content']['parts'])
        df['finish_reason'] = df['response'].apply(lambda x: x['candidates'][0]['finishReason'])
        df['reasoning_token_count'] = df['response'].apply(lambda x: x['usageMetadata']['thoughtsTokenCount'] if x['usageMetadata'] is not None else 'error')
        def get_summary_from_response_outputs(response_outputs):
            try:
                thoughts = []
                for output in response_outputs:
                    if "thought" in output.keys():
                        thoughts.append(output['text'])
                return "\n".join(thoughts)
            except:
                return 'error'
        df['summary'] = df['response_outputs'].apply(get_summary_from_response_outputs)
        def get_answer_from_response_outputs(response_outputs):
            try:
                for output in response_outputs:
                    if "thought" not in output.keys():
                        raw_answer = output['text']
                        answer = extract_answer_from_structured_text(raw_answer)
                        return answer
            except:
                return 'error'
        df['answer'] = df['response_outputs'].apply(get_answer_from_response_outputs)
        df['answer'] = df['answer'].apply(try_converting_to_float) # Convert to int if possible
        df = df[['UID', 'run', 'answer', 'summary', 'finish_reason', 'reasoning_token_count']]
    elif provider == 'together':
        df = pd.read_json(raw_output_filepath, lines=True)
        df['UID'] = df['custom_id'].str.split('-').str[1].apply(lambda x: int(x))
        df['run'] = df['custom_id'].str.split('-').str[2]+"_"+df['custom_id'].str.split('-').str[3]
        df['response_outputs'] = df['response'].apply(lambda x: x['body']['choices'][0])
        df['finish_reason'] = df['response'].apply(lambda x: x['body']['choices'][0]['finish_reason'])
        def get_summary_and_answer_from_response_outputs_no_reasoning_field(response_outputs):
            try:
                full_response = response_outputs['message']['content']
                parsed = extract_answer_and_summary_from_unstructured_text(full_response)
                answer = parsed['answer']
                summary = parsed['summary']
                return answer, summary
            except:
                return 'error', 'error'
        def get_summary_and_answer_from_response_outputs_reasoning_field(response_outputs):
            try:
                summary = response_outputs['message']['reasoning']
                answer = extract_answer_from_structured_text(response_outputs['message']['content'])
                return answer, summary
            except Exception:
                return 'error', 'error'
        if ('qwen' in raw_output_filepath.casefold()) or ('deepseek' in raw_output_filepath.casefold()): # For Qwen and DeepSeek, reasoning chains are in the main message content, not in the reasoning field
            answers_and_summaries = df['response_outputs'].apply(get_summary_and_answer_from_response_outputs_no_reasoning_field)
            df['answer'] = answers_and_summaries.apply(lambda x: x[0])
            df['answer'] = df['answer'].apply(try_converting_to_float) # Convert to int if possible
            df['summary'] = answers_and_summaries.apply(lambda x: x[1])
            if tokenizer is not None:
                df['reasoning_token_count'] = df['summary'].apply(get_reasoning_token_count_with_tokenizer)
            else:
                print("No tokenizer provided for model, so reasoning token count cannot be calculated")
                df['reasoning_token_count'] = 'error'
            df = df[['UID', 'run', 'answer', 'summary', 'finish_reason', 'reasoning_token_count']]
        else: # For other models, reasoning chains are in the reasoning field
            answers_and_summaries = df['response_outputs'].apply(get_summary_and_answer_from_response_outputs_reasoning_field)
            df['answer'] = answers_and_summaries.apply(lambda x: x[0])
            df['answer'] = df['answer'].apply(try_converting_to_float) # Convert to int if possible
            df['summary'] = answers_and_summaries.apply(lambda x: x[1])
            if tokenizer is not None:
                df['reasoning_token_count'] = df['summary'].apply(get_reasoning_token_count_with_tokenizer)
            else:
                print("No tokenizer provided for model, so reasoning token count cannot be calculated")
                df['reasoning_token_count'] = 'error'
            df = df[['UID', 'run', 'answer', 'summary', 'finish_reason', 'reasoning_token_count']]
    else:
        raise ValueError(f"Provider {provider} not supported")
    # Now, link to original dataset:
    original_df = pd.read_json(args.original_dataset_filepath, lines=True)
    df = df.merge(original_df, on='UID', how='left')
    df.to_json(processed_output_filepath, orient='records', lines=True)
    print("Done!")

if __name__ == "__main__":
    main()

