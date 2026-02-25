import os 
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import json 

# Provider SDKs are optional; import lazily where possible
try:
    from openai import OpenAI as OpenAIClient
except Exception:
    OpenAIClient = None  # type: ignore

try:
    from anthropic import Anthropic as AnthropicClient
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming as AnthropicMessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request as AnthropicRequest
except Exception:
    AnthropicClient = None  # type: ignore

try:
    from google.genai import Client as GoogleClient
    from google.genai import types
except Exception:
    GoogleClient = None  # type: ignore

try:
    from together import Together as TogetherClient
except Exception:
    TogetherClient = None  # type: ignore

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-job-info-filepaths', type=str, nargs='+') # List of paths to batch job info files
    parser.add_argument('--raw-output-dir', type=str, default='./results/raw_outputs/')
    args = parser.parse_args()
    fetch_batch_api_results(args)

def fetch_batch_api_results(args):
    batch_job_info_filepaths = args.batch_job_info_filepaths
    raw_output_dir = args.raw_output_dir
    if not os.path.exists(raw_output_dir):
        os.makedirs(raw_output_dir)
    for batch_job_info_filepath in batch_job_info_filepaths:
        with open(batch_job_info_filepath, 'r') as f:
            batch_job_info = json.load(f)
        results_tag = batch_job_info['results_tag']
        batch_id = batch_job_info['batch_id']
        batch_status = batch_job_info['batch_status']
        batch_created_at = batch_job_info['batch_created_at']
        if 'batch_completed_at' not in batch_job_info:
            batch_completed_at = "NA: There is no field for when the batch was completed (probably because this is a Together AI batch)."
        else:
            batch_completed_at = batch_job_info['batch_completed_at']
        provider = batch_job_info['provider']
        print("------- CURRENT INFO -------")
        print(f"Results tag: {results_tag}")
        print(f"Batch ID: {batch_id}")
        print(f"Batch status: {batch_status}")
        print(f"Batch created at: {batch_created_at}")
        print(f"Batch completed at: {batch_completed_at}")
        print("--------------------------------")
        print("Fetching updates...")
        if provider == 'openai':
            client = OpenAIClient()
            batch_job = client.batches.retrieve(batch_id)
            new_batch_status = batch_job.status
            new_batch_completed_at = batch_job.completed_at
            output_file_id = batch_job.output_file_id
            if new_batch_status != batch_status:
                print(f"Batch status: {new_batch_status}")
                print(f"Batch completed at: {new_batch_completed_at}")
                batch_status = new_batch_status
                batch_completed_at = new_batch_completed_at
                batch_job_info['batch_status'] = new_batch_status
                batch_job_info['batch_completed_at'] = str(new_batch_completed_at)
                with open(batch_job_info_filepath, 'w') as f:
                    json.dump(batch_job_info, f)
            output_file_path = os.path.join(raw_output_dir, f'{results_tag}_raw.jsonl')
            if (batch_status == 'completed') and (not os.path.exists(output_file_path)):
                print(f"Fetching output file {output_file_id}...")
                output_file = client.files.content(output_file_id)
                output_data = output_file.text
                print(f"Writing output data to {output_file_path}...")
                with open(output_file_path, 'w') as f:
                    f.write(output_data)
            else:
                if batch_status != 'completed':
                    print("Job is not completed. Skipping download.")
                elif os.path.exists(output_file_path):
                    print(f"Output file {output_file_path} already exists. Skipping download.")
        elif provider == 'anthropic':
            client = AnthropicClient()
            batch_job = client.messages.batches.retrieve(batch_id)
            new_batch_status = batch_job.processing_status
            new_batch_completed_at = batch_job.ended_at
            if new_batch_status != batch_status:
                print(f"Batch status: {new_batch_status}")
                print(f"Batch completed at: {new_batch_completed_at}")
                batch_status = new_batch_status
                batch_completed_at = new_batch_completed_at
                batch_job_info['batch_status'] = new_batch_status
                batch_job_info['batch_completed_at'] = str(new_batch_completed_at)
                with open(batch_job_info_filepath, 'w') as f:
                    json.dump(batch_job_info, f)
            output_file_path = os.path.join(raw_output_dir, f'{results_tag}_raw.jsonl')
            if (not os.path.exists(output_file_path)) and (batch_job.processing_status == 'ended'):
                print(f"Processing output file...")
                with open(output_file_path, 'w') as f:
                    for result in client.messages.batches.results(batch_id):
                        f.write(json.dumps(result.to_dict()) + '\n')
            else:
                if batch_job.processing_status != 'ended':
                    print("Job is not completed. Skipping download.")
                elif os.path.exists(output_file_path):
                    print(f"Output file {output_file_path} already exists. Skipping download.")
        elif provider == 'google':
            client = GoogleClient()
            batch_job = client.batches.get(name=batch_id)
            new_batch_status = batch_job.state.name
            new_batch_completed_at = batch_job.end_time
            if new_batch_status != batch_status:
                print(f"Batch status: {new_batch_status}")
                print(f"Batch completed at: {new_batch_completed_at}")
                batch_status = new_batch_status
                batch_completed_at = new_batch_completed_at
                batch_job_info['batch_status'] = new_batch_status
                batch_job_info['batch_completed_at'] = str(new_batch_completed_at)
                with open(batch_job_info_filepath, 'w') as f:
                    json.dump(batch_job_info, f)
            output_file_path = os.path.join(raw_output_dir, f'{results_tag}_raw.jsonl')
            if (not os.path.exists(output_file_path)) and (batch_job.state.name == 'JOB_STATE_SUCCEEDED'):
                print(f"Fetching output file...")
                results_file_name = batch_job.dest.file_name
                file_content = client.files.download(file=results_file_name)
                file_content = file_content.decode('utf-8')
                print(f"Writing output data to {output_file_path}...")
                with open(output_file_path, 'w') as f:
                    f.write(file_content)
            else:
                if batch_job.state.name != 'JOB_STATE_SUCCEEDED':
                    print("Job is not completed. Skipping download.")
                elif os.path.exists(output_file_path):
                    print(f"Output file {output_file_path} already exists. Skipping download.")
        elif provider == 'together':
            client = TogetherClient()
            batch_job = client.batches.get_batch(batch_id)
            new_batch_status = batch_job.status
            new_batch_completed_at = "NA: From Together API docs, there is no field for when the batch was completed."
            output_file_id = batch_job.output_file_id
            if new_batch_status != batch_status:
                print(f"Batch status: {new_batch_status}")
                print(f"Batch completed at: {new_batch_completed_at}")
                batch_status = new_batch_status
                batch_completed_at = new_batch_completed_at
                batch_job_info['batch_status'] = new_batch_status
                with open(batch_job_info_filepath, 'w') as f:
                    json.dump(batch_job_info, f)
            output_file_path = os.path.join(raw_output_dir, f'{results_tag}_raw.jsonl')
            if (batch_status == 'COMPLETED') and (not os.path.exists(output_file_path)):
                print(f"Fetching and writing output file {output_file_id}...")
                client.files.retrieve_content(id=output_file_id, output=output_file_path)
            else:
                if batch_status != 'COMPLETED':
                    print("Job is not completed. Skipping download.")
                elif os.path.exists(output_file_path):
                    print(f"Output file {output_file_path} already exists. Skipping download.")

if __name__ == '__main__':
    main()
