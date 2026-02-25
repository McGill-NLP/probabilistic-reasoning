import os
import argparse
import asyncio
import numpy as np
import pandas as pd
from tqdm.asyncio import tqdm as atqdm
import json 
from typing import Optional, List, Tuple, Dict, Any, Union
from openai import AsyncOpenAI as AsyncOpenAIClient
from openai import APIError, RateLimitError, APITimeoutError, APIConnectionError

def main():
    parser = argparse.ArgumentParser()
    # Data and prompts
    parser.add_argument('--dataset-path', type=str, required=True)
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
    parser.add_argument('--model', type=str, default='x-ai/grok-4.1-fast')

    # Reasoning controls (provider-specific semantics)
    parser.add_argument('--reasoning', type=bool, default=True, help='Whether to enable reasoning')
    # Sampling
    parser.add_argument('--temperature', type=float, default=None)
    parser.add_argument('--n-responses', type=int, default=30)
    parser.add_argument('--max-output-tokens', type=int, default=2048)
    # Concurrency control
    parser.add_argument('--max-concurrent', type=int, default=10, help='Maximum number of concurrent API calls')
    # IO
    parser.add_argument('--output-dir', type=str, default='./results/raw_outputs/')
    parser.add_argument('--results-tag', type=str, required=True)
    parser.add_argument('--notes', type=str, default=None)
    parser.add_argument('--random-seed', type=int, default=3535)
    args = parser.parse_args()
    print(args)
    asyncio.run(generate_model_responses_async(args))


def format_prompt(base_prompt: str, variable: str, statement1: str, statement2: str) -> str:
    return base_prompt.format(variable, statement1, statement2)


async def call_openrouter_async(client: AsyncOpenAIClient, formatted_prompt: str, system_prompt: str, *, 
                                 model: str, temperature: Optional[float], max_output_tokens: Optional[int], 
                                 reasoning: bool, max_retries: int = 5, 
                                 initial_wait: float = 1.0) -> Union[Any, Dict[str, str]]:
    """
    Call OpenRouter API asynchronously with exponential backoff retry logic.
    
    Args:
        client: AsyncOpenAI client instance
        formatted_prompt: The user prompt
        system_prompt: The system prompt
        model: Model identifier
        temperature: Sampling temperature
        max_output_tokens: Maximum output tokens
        reasoning: Whether to enable reasoning
        max_retries: Maximum number of retry attempts (default: 5)
        initial_wait: Initial wait time in seconds for exponential backoff (default: 1.0)
    
    Returns:
        API response object on success, or dict with error info on failure
    """
    messages = [
        {
            'role': 'system',
            'content': system_prompt
        },
        {
            'role': 'user',
            'content': formatted_prompt
        }
    ]
    kwargs = {"reasoning": {"enabled": reasoning}}
    if max_output_tokens is not None:
        kwargs['max_completion_tokens'] = max_output_tokens
    if temperature is not None:
        kwargs['temperature'] = temperature
    
    for attempt in range(max_retries):
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                extra_body=kwargs
            )
            return response
        
        except RateLimitError as e:
            wait_time = initial_wait * (2 ** attempt)  # Exponential backoff
            print(f"\nRate limit hit. Waiting {wait_time:.1f}s before retry {attempt + 1}/{max_retries}...")
            if attempt < max_retries - 1:
                await asyncio.sleep(wait_time)
            else:
                error_msg = f"Rate limit error after {max_retries} attempts: {str(e)}"
                print(f"\n{error_msg}")
                return {"error": error_msg, "error_type": "RateLimitError"}
        
        except APITimeoutError as e:
            wait_time = initial_wait * (2 ** attempt)
            print(f"\nAPI timeout. Waiting {wait_time:.1f}s before retry {attempt + 1}/{max_retries}...")
            if attempt < max_retries - 1:
                await asyncio.sleep(wait_time)
            else:
                error_msg = f"API timeout after {max_retries} attempts: {str(e)}"
                print(f"\n{error_msg}")
                return {"error": error_msg, "error_type": "APITimeoutError"}
        
        except APIConnectionError as e:
            wait_time = initial_wait * (2 ** attempt)
            print(f"\nConnection error. Waiting {wait_time:.1f}s before retry {attempt + 1}/{max_retries}...")
            if attempt < max_retries - 1:
                await asyncio.sleep(wait_time)
            else:
                error_msg = f"Connection error after {max_retries} attempts: {str(e)}"
                print(f"\n{error_msg}")
                return {"error": error_msg, "error_type": "APIConnectionError"}
        
        except APIError as e:
            # Generic API error - could be 500, 503, etc.
            wait_time = initial_wait * (2 ** attempt)
            print(f"\nAPI error: {str(e)}. Waiting {wait_time:.1f}s before retry {attempt + 1}/{max_retries}...")
            if attempt < max_retries - 1:
                await asyncio.sleep(wait_time)
            else:
                error_msg = f"API error after {max_retries} attempts: {str(e)}"
                print(f"\n{error_msg}")
                return {"error": error_msg, "error_type": "APIError"}
        
        except Exception as e:
            # Catch-all for unexpected errors
            wait_time = initial_wait * (2 ** attempt)
            print(f"\nUnexpected error: {str(e)}. Waiting {wait_time:.1f}s before retry {attempt + 1}/{max_retries}...")
            if attempt < max_retries - 1:
                await asyncio.sleep(wait_time)
            else:
                error_msg = f"Unexpected error after {max_retries} attempts: {str(e)}"
                print(f"\n{error_msg}")
                return {"error": error_msg, "error_type": "UnexpectedError"}
    
    # Fallback (should never reach here)
    return {"error": "Maximum retries exceeded", "error_type": "MaxRetriesExceeded"}


async def process_datapoint_async(
    client: AsyncOpenAIClient,
    datapoint: pd.Series,
    system_prompt: str,
    model: str,
    max_output_tokens: Optional[int],
    reasoning: bool,
    temperature: Optional[float],
    n_responses: int,
    persona_prompts: Optional[pd.DataFrame],
    semaphore: asyncio.Semaphore
) -> Tuple[List[Dict[str, Any]], int, int]:
    """
    Process a single datapoint and generate all required responses.
    
    Returns:
        Tuple of (list of response dicts, success_count, error_count)
    """
    UID = datapoint['UID']
    premise = datapoint['premise']
    hypothesis = datapoint['hypothesis']
    asks_for = datapoint['asks-for']
    formatted_prompt = datapoint['formatted_prompt']
    
    async def call_with_semaphore(sys_prompt: str, persona_id: Optional[str] = None):
        async with semaphore:
            response = await call_openrouter_async(
                client, 
                formatted_prompt, 
                sys_prompt,
                model=model,
                max_output_tokens=max_output_tokens,
                reasoning=reasoning,
                temperature=temperature
            )
            
            # Check if response is an error dict
            is_error = isinstance(response, dict) and 'error' in response
            
            response_dict: Dict[str, Any] = {
                'UID': UID,
                'premise': premise,
                'hypothesis': hypothesis,
                'asks-for': asks_for,
                'raw_response': response,
            }
            if persona_id is not None:
                response_dict['persona_id'] = persona_id
            
            return response_dict, is_error
    
    # Gather all tasks for this datapoint
    tasks = []
    if persona_prompts is not None:
        for _, row in persona_prompts.iterrows():
            persona_id = row['persona_id']
            persona_prompt = row['description']
            system_prompt_with_persona = persona_prompt + " " + system_prompt
            tasks.append(call_with_semaphore(system_prompt_with_persona, persona_id))
    else:
        tasks = [call_with_semaphore(system_prompt) for _ in range(n_responses)]
    
    # Execute all tasks for this datapoint
    results = await asyncio.gather(*tasks)
    
    # Unpack results
    responses = []
    success_count = 0
    error_count = 0
    for response_dict, is_error in results:
        responses.append(response_dict)
        if is_error:
            error_count += 1
        else:
            success_count += 1
    
    return responses, success_count, error_count


async def generate_model_responses_async(args: argparse.Namespace) -> None:
    dataset_path = args.dataset_path
    base_prompt = args.base_prompt
    system_prompt = args.system_prompt
    model = args.model
    # Arguments:
    reasoning = args.reasoning
    temperature = args.temperature
    n_responses = args.n_responses
    max_output_tokens = args.max_output_tokens
    max_concurrent = args.max_concurrent
    output_dir = args.output_dir
    results_tag = args.results_tag
    random_seed = args.random_seed
    path_to_persona_prompts = args.path_to_persona_prompts

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = os.path.join(output_dir, f'{results_tag}_raw.jsonl')

    print(f"Loading dataset from {dataset_path}")
    df = pd.read_json(dataset_path, lines=True)
    print("Done!")
    df['formatted_prompt'] = df.apply(
        lambda x: format_prompt(base_prompt, x['asks-for'], x['premise'], x['hypothesis']), 
        axis=1
    )
    
    if path_to_persona_prompts is not None:
        print(f"Loading persona prompts from {path_to_persona_prompts}")
        persona_prompts = pd.read_json(path_to_persona_prompts, lines=True)
        print("Done!")
        print(f"Generating model responses for model={model} using persona prompts from {path_to_persona_prompts}")
    else:
        persona_prompts = None
        print(f"Generating model responses for model={model} using n={n_responses} responses (sampling)")
    
    # Initialize OpenRouter client (async version)
    openrouter_client = AsyncOpenAIClient(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv('OPENROUTER_API_KEY'),
    )
    
    # Create semaphore for concurrency control
    semaphore = asyncio.Semaphore(max_concurrent)
    
    # Track errors
    error_count = 0
    success_count = 0
    master_list: List[Dict[str, Any]] = []

    # Process all datapoints with async progress bar
    tasks = [
        process_datapoint_async(
            openrouter_client,
            df.iloc[i],
            system_prompt,
            model,
            max_output_tokens,
            reasoning,
            temperature,
            n_responses,
            persona_prompts,
            semaphore
        )
        for i in range(len(df))
    ]
    
    # Use tqdm for async progress tracking
    results = await atqdm.gather(*tasks, desc="Processing datapoints")
    
    # Aggregate results
    for responses, succ_count, err_count in results:
        master_list.extend(responses)
        success_count += succ_count
        error_count += err_count
    
    # Save final results
    output_df = pd.DataFrame(master_list)
    print(f"\nSaving {len(output_df)} responses to {output_path}")
    print(f"Success: {success_count}, Errors: {error_count}")
    output_df.to_json(output_path, orient='records', lines=True)
    print("Done!")
    
    # Final summary
    print(f"\n{'='*60}")
    print(f"FINAL SUMMARY:")
    print(f"Total successful API calls: {success_count}")
    print(f"Total failed API calls: {error_count}")
    if (success_count + error_count) > 0:
        print(f"Success rate: {success_count / (success_count + error_count) * 100:.2f}%")
    print(f"{'='*60}")
    
    # Close the client
    await openrouter_client.close()


if __name__ == "__main__":
    main()

