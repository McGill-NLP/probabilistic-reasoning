# Scripts Documentation

This folder contains Python scripts for running experiments and generating visualizations. Please reach out if you have any issues running the scripts!

## Main Inference Scripts

These are the scripts used to generate the results reported in the paper:

- `probcopa_inference_batch_api.py`: **Primary inference script** for running experiments using batch APIs (OpenAI, Anthropic, Google, Together AI). This script creates batch job files and submits them to provider batch APIs for cost-efficient, large-scale inference. 

- `probcopa_inference_openrouter_async.py`: **Async inference script** for running experiments via OpenRouter API. Uses asynchronous requests with concurrency control for models not available through batch APIs. We only used this for Grok-4.1, but in theory this could be used for other models accessed via OpenRouter (though some of the processing scripts may have to be changed accordingly).

- `get_probcopa_results.py`: General-purpose inference script supporting multiple LLM providers with synchronous API calls. Useful for interactive experimentation and smaller-scale runs. This was only used for testing though, and was not used for the results we report in the paper.

## Fetching and Processing Scripts

- `fetch_batch_api_results.py`: Fetches completed batch job results from provider APIs. Takes batch job info files (containing batch IDs), downloads raw results from the provider.

- `process_batch_api_results.py`: Processes raw batch API results into standardized format. Takes raw output files and formats them consistently.

- `process_openrouter_results.py`: Processes and formats raw results from OpenRouter asynchronous inference runs.

## Utility Scripts

- `create_probcopa_samples.py`: Creates random samples of the ProbCOPA dataset for validation studies and smaller-scale experiments. Used to create the validation sample that we run temperature and reasoning effort experiments on.

## Canary String Scripts

- `add_canary_strings.py`: Adds canary strings to JSONL data files for data contamination detection. Inserts synthetic canary entries at random positions using a seed for reproducibility. Used to protect human annotation files distributed in the repository.

- `remove_canary_strings.py`: Removes canary strings from JSONL data files. Must be run before the analysis notebooks, since the canary-protected versions (`_CANARY` suffix) are what's tracked in git, while the cleaned versions are gitignored.

## Statistical Test Scripts

- `silverman_test.R`: Uses Silverman's test of multimodality to see whether the ProbCOPA human annotations show unimodal or multimodal distributions. Results are saved in `results/ProbCOPA_silverman_test_significances.csv`.

## Visualization

- `analyze_and_plot_results_models.ipynb`: Statistical tests and plots for model results and model-human comparisons. Includes model response distributions, Wasserstein distance comparisons, entropy analysis, temperature ablations, reasoning effort/thinking budget ablations, and persona prompting experiments.

- `analyze_and_plot_results_human.ipynb`: Statistical tests and plots for human results. Includes human response distributions, entropy analysis, and comparison with the Pavlick & Kwiatkowski (2019) NLI dataset.

## Usage Examples

### Generating Model Responses (Batch API - Recommended for Paper Results)

The `probcopa_inference_batch_api.py` script is what was used to generate most results in the paper. It's more cost-efficient for large-scale experiments:

**Example with GPT-5:**
```bash
# Step 1: Create and submit batch job
# This step creates batch requests that are submitted to the respective providers. For Google, OpenAI and Together AI, this involves the intermediate step of writing a batch request file that is saved in ./batch_api_files/ (with the Anthropic batch API, the batch requests are directly sent without this intermediate request file being saved). More importantly, the step creates a batch job info file in ./batch_api_files/, that is used in Step 3 to retrieve the processed batch request.

python scripts/probcopa_inference_batch_api.py \
    --dataset-path ./datasets/probcopa_items.jsonl \
    --provider openai \
    --model gpt-5 \
    --reasoning-effort medium \
    --n-responses 30 \
    --results-tag probcopa_gpt-5 \
    --batch-api-file-dir ./batch_api_files/



# Step 2: Wait for batch to complete (check provider dashboard, or run `scripts/fetch_batch_api_results.py` for updates)


# Step 3: Fetch raw results from provider
python scripts/fetch_batch_api_results.py \
    --batch-job-info-filepaths ./batch_api_files/probcopa_gpt-5_batch_job_info.json \
    --raw-output-dir ./results/raw_outputs/

# Step 4: Process raw results into standardized format
python scripts/process_batch_api_results.py \
    --raw-output-filepath ./results/raw_outputs/probcopa_gpt-5_raw.jsonl \
    --processed-output-dir ./results/ \
    --results-tag probcopa_gpt-5 \
    --provider openai
```

To run models from different providers, change the `model` and `provider` arguments, along with any other differences in provider-specific argument names (e.g., the Anthropic API takes the input `thinking_budget`, while the OpenAI API uses `reasoning_effort`).


### Generating Model Responses (OpenRouter Async - For Grok and Other Models)

The `probcopa_inference_openrouter_async.py` script was used for Grok-4.1, which was available through OpenRouter (**NO batching**). In theory it could be used for other models too, though this may require some small changes to the processing scripts.

```bash
python scripts/probcopa_inference_openrouter_async.py \
    --dataset-path ./datasets/probcopa_items.jsonl \
    --model x-ai/grok-4.1-fast \
    --n-responses 30 \
    --temperature 1.0 \
    --max-concurrent-requests 10 \
    --results-tag probcopa_grok-4.1-fast \
    --output-dir ./results/
```

Then process:
```bash
python scripts/process_openrouter_results.py \
    --input-path ./results/probcopa_grok-4.1-fast_raw.jsonl \
    --output-path ./results/probcopa_grok-4.1-fast.jsonl
```

### Generating Model Responses (Synchronous API)

The `get_probcopa_results.py` script is useful for interactive experiments or smaller runs, though note that the experimental results reported in this paper were not generated using this script. Most of the arguments are the same as in the `probcopa_inference_batch_api.py` script, except that results are now processed and written directly to file  (no separate processing script, or batch request file).

**Example with GPT-5:**
```bash
python scripts/get_probcopa_results.py \
    --dataset-path ./datasets/probcopa_items.jsonl \
    --provider openai \
    --model gpt-5 \
    --reasoning-effort medium \
    --n-responses 30 \
    --temperature 1.0 \
    --results-tag probcopa_gpt-5 \
    --output-dir ./results/
```

### Environment Variables Required

Set the appropriate API keys as environment variables:
- `OPENAI_API_KEY` for OpenAI (batch and sync)
- `ANTHROPIC_API_KEY` for Anthropic (batch and sync)
- `GOOGLE_API_KEY` for Google (batch and sync)
- `TOGETHER_API_KEY` for Together AI (batch and sync)
- `OPENROUTER_API_KEY` for OpenRouter (async inference)

### Which Script to Use?

**For reproducing paper results (recommended):**
1. Use the batch API workflow for OpenAI, Anthropic, Google, Together AI models:
   - `probcopa_inference_batch_api.py` → `fetch_batch_api_results.py` → `process_batch_api_results.py`
   - More cost-efficient (50% discount on most providers)
   - Better for large-scale experiments (210 items × 30 responses = 6,300 requests)
   - Requires 3-step workflow (submit → fetch → process)

2. Use `probcopa_inference_openrouter_async.py` for Grok and other OpenRouter models
   - Async requests with concurrency control
   - Good rate limit handling with retries
   - Saves results directly (may need cleanup with `process_openrouter_results.py`)

**For interactive experimentation:**
- Use `get_probcopa_results.py` for quick tests with small samples
- Synchronous API calls (slower but simpler)
- Good for testing prompts, debugging, or single-item exploration

### Creating Dataset Samples

Finally, `scripts/create_probcopa_samples.py` creates the 30-item random sample from the full ProbCOPA dataset that we used for validation studies, as well as the temperature and reasoning effort experiments.

```bash
python scripts/create_probcopa_samples.py
```


### Removing Canary Strings (Required Before Analysis)

Before running the analysis notebooks, remove canary strings from the human annotation files:

```bash
python scripts/remove_canary_strings.py \
    --input-path datasets/probcopa_CANARY.jsonl \
    --output-path datasets/probcopa.jsonl

python scripts/remove_canary_strings.py \
    --input-path results/probcopa_human_results_annotated_CANARY.jsonl \
    --output-path results/probcopa_human_results_annotated.jsonl

python scripts/remove_canary_strings.py \
    --input-path results/probcopa_random_sample_validation_round_human_results_CANARY.jsonl \
    --output-path results/probcopa_random_sample_validation_round_human_results_cleaned.jsonl

python scripts/remove_canary_strings.py \
    --input-path results/probcopa_random_sample_prompt_validation_round_human_results_CANARY.jsonl \
    --output-path results/probcopa_random_sample_prompt_validation_round_human_results_cleaned.jsonl
```

### Generating Plots

Open and run the two analysis notebooks to generate all figures from the paper:

- `scripts/analyze_and_plot_results_models.ipynb` — model analysis and model-human comparison plots
- `scripts/analyze_and_plot_results_human.ipynb` — human response analysis plots

All plots are saved as PDFs in the `plots/` directory.
