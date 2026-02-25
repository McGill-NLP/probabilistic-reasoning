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
    # Data and prompts
    parser.add_argument('--dataset-path', type=str, required=True)
    parser.add_argument('--batch-api-file-dir', type=str, default='./batch_api_files/')
    parser.add_argument('--base-prompt', type=str, default=(
        "Consider the following situation and possible effect.\n\n"
        "Situation: {1}\nPossible Effect: {2}\n\n"
        "Given the situation, how likely is this effect?"
        "Respond with a numerical value between 0 and 100, where 0 indicates that this is DEFINITELY NOT the effect, "
        "and 100 indicates that this is DEFINITELY the effect."
    ))
    parser.add_argument('--system-prompt', type=str, default=(
        "You provide responses to questions about the likelihood of an effect given some situation. "
        "After any internal reasoning, reply with a single number between 0 and 100, enclosed in <answer> tags."
        "You can use the following descriptions of numerical ranges to help guide your response:"
        "\n0: Absolutely no chance"
        "\n1-5: Almost no chance"
        "\n6-15: Highly unlikely"
        "\n16-34: Unlikely"
        "\n35-49: Somewhat unlikely"
        "\n50: Totally even chance"
        "\n51-65: Somewhat likely"
        "\n66-84: Likely"
        "\n85-94: Highly likely"
        "\n95-99: Almost certain"
        "\n100: Absolutely certain"
    ))
    parser.add_argument('--path-to-persona-prompts', type=str, default=None)

    # Provider/model selection
    parser.add_argument('--provider', type=str, default='openai', choices=['openai', 'anthropic', 'google', 'together'])
    parser.add_argument('--model', type=str, default='gpt-5')

    # Reasoning controls (provider-specific semantics)
    # OpenAI Responses API supports reasoning effort/summary depth.
    parser.add_argument('--reasoning-effort', type=str, default='medium', help='OpenAI and DeepSeek: none|low|medium|high (set to "None" to disable)')
    parser.add_argument('--reasoning-summary-depth', type=str, default='detailed', help='OpenAI-only: brief|detailed (set to "None" to disable)')
    # Anthropic thinking toggle/budget
    parser.add_argument('--enable-thinking', type=bool, default=True,
                        help='Anthropic and Gemini: enable internal thinking. Auto enables if reasoning-effort is set.')
    parser.add_argument('--thinking-budget', type=int, default=1024, help='Anthropic and Gemini: thinking budget tokens when enabled')
    # Google thinking model selection (Gemini). Some models expose thinking via separate model IDs.
    # Sampling
    parser.add_argument('--temperature', type=float, default=None)
    parser.add_argument('--n-responses', type=int, default=30)
    parser.add_argument('--max-output-tokens', type=int, default=2048)
    # IO
    parser.add_argument('--output-dir', type=str, default='./results/')
    parser.add_argument('--results-tag', type=str, required=True)
    parser.add_argument('--notes', type=str, default=None)
    parser.add_argument('--random-seed', type=int, default=3535)
    #
    args = parser.parse_args()
    run_batch_api(args)

def format_prompt(base_prompt: str, variable: str, statement1: str, statement2: str) -> str:
    return base_prompt.format(variable, statement1, statement2)

def write_batch_dict(provider: str, system_prompt: str, formatted_prompt: str, custom_id: str, model: str, temperature: float, max_output_tokens: int, reasoning_effort: str, reasoning_summary_depth: str, enable_thinking: bool, thinking_budget: int):
    if provider == 'openai':
        kwargs = {
                'model': model,
                'instructions': system_prompt,
                'input': formatted_prompt
            }
        reasoning = {}
        if reasoning_effort and reasoning_effort != 'None':
            reasoning['effort'] = reasoning_effort
        if reasoning_summary_depth and reasoning_summary_depth != 'None':
            reasoning['summary'] = reasoning_summary_depth
        if reasoning:
            kwargs['reasoning'] = reasoning
        if temperature is not None:
            kwargs['temperature'] = temperature
        if max_output_tokens is not None:
            kwargs['max_output_tokens'] = max_output_tokens
        #
        batch_request_dict = {
            'custom_id': custom_id,
            'method': 'POST',
            'url': '/v1/responses',
            'body': kwargs
        }
        #
    if provider == 'anthropic':
        kwargs = {
            'model': model,
            'messages': [{'role': 'user', 'content': formatted_prompt}],
            'system': system_prompt,
        }
        if enable_thinking:
            if thinking_budget is not None:
                thinking_budget_tokens = 1024
            kwargs['thinking'] = {'type': 'enabled', 'budget_tokens': thinking_budget_tokens}
        if temperature is not None:
            kwargs['temperature'] = temperature
        if max_output_tokens is not None:
            kwargs['max_tokens'] = max_output_tokens
        #
        batch_request_dict = AnthropicRequest(
            custom_id=custom_id,
            params=AnthropicMessageCreateParamsNonStreaming(**kwargs)
        )
        #
    elif provider == 'google':
        '''
        Note/rant: insane, but as of 04.12.2025, while Gemini Docs SAY you should use thinking_level instead of thinking_budget, the Batch API only supports thinking_budget, and does not recognize thinking_level as a valid argument.
        So, for now, I'm using only thinking_budget, with the same default as before (1024 tokens), and will do ablations to see if higher/lower thinking budgets make any difference.
        In other words, I'm running Gemini 3 Pro the exact same as Gemini 2.5 Pro.
        '''
        generation_config = {}
        if enable_thinking:
            include_thoughts_bool = True
            if thinking_budget is not None:
                thinking_budget_tokens = thinking_budget
            else:
                thinking_budget_tokens = -1
        else:
            include_thoughts_bool = False
            thinking_budget_tokens = 0
        generation_config['thinking_config'] = {
            'include_thoughts': include_thoughts_bool,
            'thinking_budget': thinking_budget_tokens
        }
        if temperature is not None:
            generation_config['temperature'] = temperature
        if max_output_tokens is not None:
            generation_config['max_output_tokens'] = max_output_tokens
        #
        # Note that with the Google API, the model is only specified when running the request, not when creating the batch file
        batch_request_dict = {
            'key': custom_id,
            'request': {
                'system_instruction': {'role': 'system', 'parts': [{'text': system_prompt}]},
                'contents': [{'parts': [{'text': formatted_prompt}]}],
                'generation_config': generation_config
            }
        }
        #
    elif provider == 'together':
        formatted_prompt_plus_system = system_prompt + "\n\n" + formatted_prompt
        kwargs = {
            'model': model,
            'messages': [{'role': 'user', 'content': formatted_prompt_plus_system}],
            }
        if reasoning_effort is not None:
            kwargs['reasoning_effort'] = reasoning_effort
        if temperature is not None:
            kwargs['temperature'] = temperature
        if max_output_tokens is not None:
            kwargs['max_tokens'] = max_output_tokens
        batch_request_dict = {
            'custom_id': custom_id,
            'body': kwargs
        }
        #
    return batch_request_dict



def run_batch_api(args):
    dataset_path = args.dataset_path
    batch_api_file_dir = args.batch_api_file_dir
    base_prompt = args.base_prompt
    system_prompt = args.system_prompt
    path_to_persona_prompts = args.path_to_persona_prompts
    provider = args.provider
    model = args.model
    reasoning_effort = args.reasoning_effort
    reasoning_summary_depth = args.reasoning_summary_depth
    enable_thinking = args.enable_thinking
    thinking_budget = args.thinking_budget
    temperature = args.temperature
    n_responses = args.n_responses
    max_output_tokens = args.max_output_tokens
    results_tag = args.results_tag
    notes = args.notes
    random_seed = args.random_seed

    if path_to_persona_prompts is not None:
        persona_prompts = pd.read_json(path_to_persona_prompts, lines=True)
    else:
        persona_prompts = None

    df = pd.read_json(dataset_path, lines=True)
    print(f"Loaded {len(df)} rows from {dataset_path}")
    df['formatted_prompt'] = df.apply(lambda x: format_prompt(base_prompt, x['asks-for'], x['premise'], x['hypothesis']), axis=1)
    print(f"Formatted {len(df)} prompts")
    batch_request_dicts = []
    for i, row in df.iterrows():
        if persona_prompts is not None:
            for j, persona_row in persona_prompts.iterrows():
                custom_id = f"UID-{row['UID']}-Persona-{persona_row['persona_id']}"
                persona_prompt = persona_row['description']
                system_prompt_plus_persona = persona_prompt + " " + system_prompt
                batch_request_dict = write_batch_dict(provider, system_prompt_plus_persona, row['formatted_prompt'], custom_id, model, temperature, max_output_tokens, reasoning_effort, reasoning_summary_depth, enable_thinking, thinking_budget)
                batch_request_dicts.append(batch_request_dict)
        else:
            for j in range(n_responses):
                custom_id = f"UID-{row['UID']}-Response-{j}"
                batch_request_dict = write_batch_dict(provider, system_prompt, row['formatted_prompt'], custom_id, model, temperature, max_output_tokens, reasoning_effort, reasoning_summary_depth, enable_thinking, thinking_budget)
                batch_request_dicts.append(batch_request_dict)
    print(f"Created {len(batch_request_dicts)} batch request dicts")
    if provider != 'anthropic':
        file_path = f'{batch_api_file_dir}/{results_tag}.jsonl'
        batch_requests_df = pd.DataFrame(batch_request_dicts) # Only to make it easier to write to file
        batch_requests_df.to_json(file_path, orient='records', lines=True)
        print(f"Successfully wrote {len(batch_request_dicts)} batch request dicts to {file_path}")
    # Sending batch requests
    # OpenAI
    if provider == 'openai':
        client = OpenAIClient()
        batch_input_file  = client.files.create(
            file=open(file_path, 'rb'),
            purpose='batch'
        )
        batch_input_file_id = batch_input_file.id
        batch_job = client.batches.create(
            input_file_id=batch_input_file_id,
            endpoint='/v1/responses',
            completion_window="24h",
            metadata={
                "notes": notes,
                "results_tag": results_tag
            }
        )
        batch_job_info_dict = {"results_tag": results_tag, "provider": provider, "batch_id": batch_job.id, "batch_status": batch_job.status, "batch_created_at": str(batch_job.created_at), "batch_completed_at": str(batch_job.completed_at)}
        print(f"Successfully created batch job {batch_job_info_dict['batch_id']}")
        batch_job_info_path = f'{batch_api_file_dir}/batch_job_info_{results_tag}.json'
        with open(batch_job_info_path, 'w') as f:
            json.dump(batch_job_info_dict, f)
        print(f"Successfully wrote batch job info to {batch_job_info_path}")
    
    # Google
    if provider == 'google':
        client = GoogleClient()
        uploaded_file = client.files.upload(
            file=open(file_path, 'rb'),
            config=types.UploadFileConfig(
                display_name=results_tag,
                mime_type='jsonl'
                )
        )
        uploaded_file_name = uploaded_file.name
        print(f"Successfully uploaded file {uploaded_file_name}")
        file_batch_job = client.batches.create(
            model=model,
            src=uploaded_file_name,
            config={
                "display_name": results_tag
            }
        )
        batch_job_info_dict = {"results_tag": results_tag, "provider": provider, "batch_id": file_batch_job.name, "batch_status": file_batch_job.state, "batch_created_at": str(file_batch_job.create_time), "batch_completed_at": str(file_batch_job.end_time)}
        print(f"Successfully created batch job {batch_job_info_dict['batch_id']}")
        batch_job_info_path = f'{batch_api_file_dir}/batch_job_info_{results_tag}.json'
        with open(batch_job_info_path, 'w') as f:
            json.dump(batch_job_info_dict, f)
        print(f"Successfully wrote batch job info to {batch_job_info_path}")
    
    # Together
    if provider == 'together':
        client = TogetherClient()
        file_response = client.files.upload(
            file=file_path,
            purpose='batch-api'
        )
        print(f"Successfully uploaded file {file_response.id}")
        file_id = file_response.id
        file_batch_job = client.batches.create_batch(
            file_id,
            endpoint='/v1/chat/completions'
        )
        batch_job_info_dict = {"results_tag": results_tag, "provider": provider, "batch_id": file_batch_job.id, "batch_status": file_batch_job.status, "batch_created_at": str(file_batch_job.created_at)} # Together doesn't have a completed_at field
        print(f"Successfully created batch job {batch_job_info_dict['batch_id']}")
        batch_job_info_path = f'{batch_api_file_dir}/batch_job_info_{results_tag}.json'
        with open(batch_job_info_path, 'w') as f:
            json.dump(batch_job_info_dict, f)
        print(f"Successfully wrote batch job info to {batch_job_info_path}")
    
    # Anthropic
    if provider == 'anthropic':
        # From what I can tell, the Anthropic Batch API doesn't involve file uploads.
        # Instead, you call the API using requests, and the API takes a list of prompts and returns a list of responses.
        client = AnthropicClient()
        message_batch = client.messages.batches.create(
            requests=batch_request_dicts
            )
        print(f"Successfully created message batch {message_batch.id}")
        batch_job_info_dict = {"results_tag": results_tag, "provider": provider, "batch_id": message_batch.id, "batch_status": message_batch.processing_status, "batch_created_at": str(message_batch.created_at), "batch_completed_at": str(message_batch.ended_at)}
        print(f"Successfully created batch job {batch_job_info_dict['batch_id']}")
        batch_job_info_path = f'{batch_api_file_dir}/batch_job_info_{results_tag}.json'
        with open(batch_job_info_path, 'w') as f:
            json.dump(batch_job_info_dict, f)
        print(f"Successfully wrote batch job info to {batch_job_info_path}")

if __name__ == '__main__':
    main()
