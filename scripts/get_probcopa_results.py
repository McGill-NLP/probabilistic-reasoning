import os
import re
import argparse
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from tqdm import tqdm

# Provider SDKs are optional; import lazily where possible
try:
    from openai import OpenAI as OpenAIClient
except Exception:
    OpenAIClient = None  # type: ignore

try:
    from anthropic import Anthropic as AnthropicClient
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
    parser.add_argument('--base-prompt', type=str, default=(
        "Consider the following situation and possible effect.\n\n"
        "Situation: {1}\nPossible Effect: {2}\n\n"
        "Given the situation, how likely is this effect?"
        "Respond with a numerical value between 0 and 100, where 0 indicates that this is DEFINITELY NOT the effect, "
        "and 100 indicates that this is DEFINITELY the effect."
    ))
    parser.add_argument('--system-prompt', type=str, default=(
        "You provide responses to questions about the likelihood of causes and effects. "
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
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--n-responses', type=int, default=30)
    parser.add_argument('--max-output-tokens', type=int, default=2048)
    # IO
    parser.add_argument('--output-dir', type=str, default='./results/')
    parser.add_argument('--results-tag', type=str, required=True)
    parser.add_argument('--random-seed', type=int, default=3535)
    #
    args = parser.parse_args()
    print(args)
    generate_model_responses(args)


def format_prompt(base_prompt: str, variable: str, statement1: str, statement2: str) -> str:
    return base_prompt.format(variable, statement1, statement2)


def extract_answer_and_summary_from_text(raw_text: str) -> Dict[str, Any]:
    # Prefer explicit <answer> tags if present
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


# ---------------------------- Provider calls ---------------------------- #

def call_openai(client, formatted_prompt: str, system_prompt: str, *, model: str, temperature: float,
                reasoning_effort: Optional[str], reasoning_summary_depth: Optional[str],
                max_output_tokens: Optional[int]) -> Tuple[str, Optional[List[str]]]:
    # OpenAI Responses API (docs: https://platform.openai.com/docs/api-reference/responses)
    kwargs: Dict[str, Any] = {
        'model': model,
        'instructions': system_prompt,
        'input': formatted_prompt,
        'temperature': temperature,
    }
    if max_output_tokens is not None:
        kwargs['max_output_tokens'] = max_output_tokens
    reasoning: Dict[str, Any] = {}
    if reasoning_effort and reasoning_effort != 'None':
        reasoning['effort'] = reasoning_effort
    if reasoning_summary_depth and reasoning_summary_depth != 'None':
        reasoning['summary'] = reasoning_summary_depth
    if reasoning:
        kwargs['reasoning'] = reasoning

    response = client.responses.create(**kwargs)
    # For gpt-5 style structured responses with summaries
    try:
        summaries = [summary.text for summary in response.output[0].summary]
    except Exception:
        summaries = None
    try:
        raw_answer_text = response.output[1].content[0].text
        answer_text = extract_answer_and_summary_from_text(raw_answer_text)['answer']
    except Exception:
        # Fallback to flatten text
        try:
            flat_text = ''.join(getattr(chunk, 'text', '') for chunk in getattr(response.output[0], 'content', []))
        except Exception:
            flat_text = str(response)
        parsed = extract_answer_and_summary_from_text(flat_text)
        return parsed['answer'], [parsed['summary']] if parsed['summary'] else None
    return answer_text, summaries


def call_anthropic(client, formatted_prompt: str, system_prompt: str, *, model: str, temperature: float,
                   enable_thinking: bool, thinking_budget_tokens: int,
                   max_output_tokens: Optional[int]) -> Tuple[str, Optional[List[str]]]:
    # Anthropic Messages API with optional thinking
    # Docs: https://docs.anthropic.com/en/api/messages
    kwargs: Dict[str, Any] = {
        'model': model,
        'max_tokens': max_output_tokens or 1024,
        'messages': [{
            'role': 'user',
            'content': formatted_prompt,
        }],
        'system': system_prompt,
        'temperature': temperature,
    }
    if enable_thinking:
        if thinking_budget_tokens is None:
            thinking_budget_tokens = 1024
        # Docs: https://docs.anthropic.com/en/docs/build-with-claude/tokens-thinking#enable-thinking
        kwargs['thinking'] = {'type': 'enabled', 'budget_tokens': thinking_budget_tokens}

    message = client.messages.create(**kwargs)
    # Collect thinking and text parts
    thinking_texts: List[str] = []
    visible_texts: List[str] = []
    try:
        for block in message.content:
            block_type = getattr(block, 'type', None) or (isinstance(block, dict) and block.get('type'))
            if block_type == 'thinking':
                text_val = getattr(block, 'thinking', None) or (isinstance(block, dict) and block.get('thinking'))
                if text_val:
                    thinking_texts.append(text_val)
            elif block_type == 'text':
                text_val = getattr(block, 'text', None) or (isinstance(block, dict) and block.get('text'))
                if text_val:
                    visible_texts.append(text_val)
    except Exception:
        # Fallback: best-effort stringify
        visible_texts.append(str(message))

    full_visible_text = '\n'.join(visible_texts).strip()
    parsed = extract_answer_and_summary_from_text(full_visible_text)
    # Prefer actual thinking as summaries if available
    summaries: Optional[List[str]] = None
    if thinking_texts:
        summaries = ['\n'.join(thinking_texts)]
    elif parsed.get('summary'):
        summaries = [parsed['summary']]
    return parsed['answer'], summaries

def call_google(client, formatted_prompt: str, system_prompt: str, *, model: str, temperature: float,
                enable_thinking: bool, thinking_budget_tokens: Optional[int],
                max_output_tokens: Optional[int]) -> Tuple[str, Optional[List[str]]]:
    
    # Google Gemini via google-generativeai
    # Docs: https://ai.google.dev/api/python/google/generativeai
    if enable_thinking:
        include_thoughts_bool = True
        if thinking_budget_tokens:
            thinking_budget_tokens = thinking_budget_tokens
        else:
            thinking_budget_tokens = -1 # -1 means dynamic budget
    else:
        include_thoughts_bool = False
        thinking_budget_tokens = 0 # 0 means no budget
    #
    gemini_config = types.GenerateContentConfig(
        system_instruction=system_prompt,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        thinking_config=types.ThinkingConfig(
            include_thoughts=include_thoughts_bool,
            thinking_budget=thinking_budget_tokens,
        )
    )
    response = client.models.generate_content(
        model=model,
        contents=formatted_prompt,
        config=gemini_config,
    )
    # Response has .text and .candidates; no explicit thinking unless using special models
    raw_answer = response.text
    if enable_thinking:
        summaries = [part.text for part in response.parts if part.thought]
        answer = extract_answer_and_summary_from_text(raw_answer)['answer']
    else:
        parsed = extract_answer_and_summary_from_text(raw_answer)
        answer = parsed['answer']
        summaries = [parsed['summary']] if parsed.get('summary') else None
    return answer, summaries


def call_together_deepseek(client, formatted_prompt: str, system_prompt: str, *, model: str, temperature: float,
                           max_output_tokens: Optional[int], reasoning_effort: Optional[str]) -> Tuple[str, Optional[List[str]]]:
    # Use Chat Completions API (OpenAI-compatible)
    formatted_prompt = system_prompt + "\n\n" + formatted_prompt # Together AI Docs suggest avoiding system prompts for DeepSeek-R1, and putting them in the prompt directly
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {'role': 'user', 'content': formatted_prompt},
        ],
        temperature=temperature,
        max_tokens=max_output_tokens or 1024,
        reasoning_effort=reasoning_effort
    )
    try:
        raw_text = resp.choices[0].message.content
    except Exception:
        raw_text = str(resp)
    parsed = extract_answer_and_summary_from_text(raw_text)
    summaries = [parsed['summary']] if parsed.get('summary') else None
    return parsed['answer'], summaries


def generate_model_responses(args: argparse.Namespace) -> None:
    dataset_path = args.dataset_path
    base_prompt = args.base_prompt
    system_prompt = args.system_prompt
    model = args.model
    provider = args.provider

    # Create client based on provider
    if provider == 'openai':
        if OpenAIClient is None:
            raise RuntimeError('openai package is not available. Install openai>=1.37.0')
        else:
            openai_api_key = os.environ.get('OPENAI_API_KEY')
            if not openai_api_key:
                raise RuntimeError('Set OPENAI_API_KEY for OpenAI access.')
            client = OpenAIClient()  
    elif provider == 'anthropic':
        if AnthropicClient is None:
            raise RuntimeError('anthropic package is not available. Install anthropic>=0.34.0')
        else:
            anthropic_api_key = os.environ.get('ANTHROPIC_API_KEY')
            if not anthropic_api_key:
                raise RuntimeError('Set ANTHROPIC_API_KEY for Anthropic access.')
            client = AnthropicClient()
    elif provider == 'google':
        if GoogleClient is None:
            raise RuntimeError('google package is not available. Install google>=0.7.0')
        else:
            google_api_key = os.environ.get('GOOGLE_API_KEY')
            if not google_api_key:
                raise RuntimeError('Set GOOGLE_API_KEY for Google access.')
            client = GoogleClient()
    elif provider == 'together':
        if TogetherClient is None:
            raise RuntimeError('together package is not available. Install together>=0.7.0')
        else:
            together_api_key = os.environ.get('TOGETHER_API_KEY')
            if not together_api_key:
                raise RuntimeError('Set TOGETHER_API_KEY for Together access.')
            client = TogetherClient()
    # Normalize reasoning flags per provider
    reasoning_effort = args.reasoning_effort if args.reasoning_effort != 'None' else None
    reasoning_summary_depth = args.reasoning_summary_depth if args.reasoning_summary_depth != 'None' else None
    temperature = args.temperature if args.temperature != -1 else None # -1 means no temperature specified
    n_responses = args.n_responses
    max_output_tokens = args.max_output_tokens if args.max_output_tokens != -1 else None # -1 means no max output tokens specified
    enable_thinking = args.enable_thinking # Only for Anthropic and Gemini
    thinking_budget = args.thinking_budget # Only for Anthropic and Gemini
    output_dir = args.output_dir
    results_tag = args.results_tag
    random_seed = args.random_seed
    path_to_persona_prompts = args.path_to_persona_prompts

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = os.path.join(output_dir, f'{results_tag}.jsonl')

    print(f"Loading dataset from {dataset_path}")
    df = pd.read_json(dataset_path, lines=True)
    print("Done!")

    if path_to_persona_prompts:
        print(f"Loading persona prompts from {path_to_persona_prompts}")
        persona_prompts = pd.read_json(path_to_persona_prompts, lines=True)
        print("Done!")
        print(f"Generating model responses for provider={provider}, model={model} using persona prompts from {path_to_persona_prompts}")
    else:
        persona_prompts = None
        print(f"Generating model responses for provider={provider}, model={model} using n={n_responses} responses (sampling)")

    master_list: List[Dict[str, Any]] = []
    
    for i in tqdm(range(len(df))):
        datapoint = df.iloc[i]
        UID = datapoint['UID']
        premise = datapoint['premise']
        hypothesis = datapoint['hypothesis']
        asks_for = datapoint['asks-for']
        formatted_prompt = format_prompt(base_prompt, asks_for, premise, hypothesis)

        def run_once(sp: str) -> Dict[str, Any]:
            if provider == 'openai':
                ans, sums = call_openai(
                    client=client,
                    formatted_prompt=formatted_prompt,
                    system_prompt=sp,
                    model=model,
                    temperature=temperature,
                    reasoning_effort=reasoning_effort,
                    reasoning_summary_depth=reasoning_summary_depth,
                    max_output_tokens=max_output_tokens,
                )
            elif provider == 'anthropic':
                ans, sums = call_anthropic(
                    client=client,
                    formatted_prompt=formatted_prompt,
                    system_prompt=sp,
                    model=model,
                    temperature=temperature,
                    enable_thinking=enable_thinking,
                    thinking_budget_tokens=thinking_budget,
                    max_output_tokens=max_output_tokens,
                )
            elif provider == 'google':
                ans, sums = call_google(
                    client=client,
                    formatted_prompt=formatted_prompt,
                    system_prompt=sp,
                    model=model,
                    temperature=temperature,
                    enable_thinking=enable_thinking,
                    thinking_budget_tokens=thinking_budget,
                    max_output_tokens=max_output_tokens,
                )
            elif provider == 'together':
                # Defaults: model name for DeepSeek-R1 on Together is often 'deepseek-ai/DeepSeek-R1'
                ans, sums = call_together_deepseek(
                    client=client,
                    formatted_prompt=formatted_prompt,
                    system_prompt=sp,
                    model=model,
                    temperature=temperature,
                    max_output_tokens=max_output_tokens,
                    reasoning_effort=reasoning_effort,
                )
            else:
                raise ValueError(f"Unsupported provider: {provider}")
            response_dict: Dict[str, Any] = {
                'UID': UID,
                'premise': premise,
                'hypothesis': hypothesis,
                'asks-for': asks_for,
                'answer': ans,
                'summary': sums if sums is not None else [],
            }
            return response_dict

        if persona_prompts is not None:
            for _, row in persona_prompts.iterrows():
                persona_id = row['persona_id']
                persona_prompt = row['description']
                system_prompt_with_persona = persona_prompt + " " + system_prompt
                response_dict = run_once(system_prompt_with_persona)
                response_dict['persona_id'] = persona_id
                master_list.append(response_dict)
        else:
            for _ in range(n_responses):
                master_list.append(run_once(system_prompt))

    output_df = pd.DataFrame(master_list)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = os.path.join(output_dir, f'{results_tag}.jsonl')
    print(f"Saving results to {output_path}")
    output_df.to_json(output_path, orient='records', lines=True)
    print("Done!")


if __name__ == "__main__":
    main()


