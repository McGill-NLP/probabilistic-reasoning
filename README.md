# Humans and LLMs Diverge on Probabilistic Inferences

*Gaurav Kamath*, *Sreenath Madathil*, *Sebastian Schuster*, *Marie-Catherine de Marneffe*, *Siva Reddy*

This repository contains code and data for the paper *Humans and LLMs Diverge on Probabilistic Inferences*.

You can access the paper on ArXiV here: PLACEHOLDER: ADD LINK.

You can also explore the data and results with our interactive data visualizer here: https://grvkamath.github.io/probcopa-demo/index.html.

## Contents

Below are the structure and contents of this repository.

- `datasets/`: The ProbCOPA dataset and related data files used in this study.
- `results/`: Model and human annotation results from all experiments.
- `plots/`: All plots generated as part of this study (PDFs).
- `scripts/`: Python scripts for running experiments, analysis notebooks for generating plots, and canary string utilities.
- `assets/`: Additional configuration files (model argument limits, structured personas).
- `requirements.txt`: Python libraries required to run the code in this repository.
- `ProbabilisticInferences.pdf`: The paper manuscript.

## The ProbCOPA Dataset

The ProbCOPA dataset consists of 210 handcrafted probabilistic inferences in English, each annotated for inference likelihood by 25-30 human participants. The dataset is available in `datasets/probcopa_items.jsonl`.

Each item in the dataset contains:
- `UID`: Unique identifier for the item
- `premise`: The premise statement
- `hypothesis`: The hypothesis statement (possible effect)
- `asks-for`: The relationship being queried (always "effect" in this dataset)

The full dataset with human annotations is in `datasets/probcopa_CANARY.jsonl` (see [Canary Strings](#canary-strings) below).

## Canary Strings

Several data files containing human annotations are distributed with **canary strings** — synthetic entries inserted at random positions to detect potential data contamination if future LLMs are trained on this data. Files with canary strings have a `_CANARY` suffix (e.g., `probcopa_CANARY.jsonl`).

Before running the analysis notebooks, you must remove the canary strings:

```bash
python scripts/remove_canary_strings.py \
    --input-path datasets/probcopa_CANARY.jsonl \
    --output-path datasets/probcopa.jsonl
```

The cleaned output files (without canary strings) are listed in `.gitignore` and are not tracked in the repository. To re-add canary strings to a cleaned file:

```bash
python scripts/add_canary_strings.py \
    --input-path datasets/probcopa.jsonl \
    --output-path datasets/probcopa_CANARY.jsonl
```

## Running Experiments

### Generating Model Responses

The paper results were generated using batch APIs for cost efficiency and scale. The workflow involves three steps:

```bash
# Step 1: Create and submit batch job
python scripts/probcopa_inference_batch_api.py \
    --dataset-path ./datasets/probcopa_items.jsonl \
    --provider openai \
    --model gpt-5 \
    --n-responses 30 \
    --results-tag probcopa_gpt-5 \
    --batch-api-file-dir ./batch_api_files/

# Step 2: Wait for completion, then fetch raw results
python scripts/fetch_batch_api_results.py \
    --batch-job-info-filepaths ./batch_api_files/probcopa_gpt-5_batch_job_info.json \
    --raw-output-dir ./results/raw_outputs/

# Step 3: Process raw results into standardized format
python scripts/process_batch_api_results.py \
    --raw-output-filepath ./results/raw_outputs/probcopa_gpt-5_raw.jsonl \
    --processed-output-dir ./results/ \
    --results-tag probcopa_gpt-5 \
    --provider openai
```

For models available through OpenRouter (e.g., Grok), use the async inference script:

```bash
python scripts/probcopa_inference_openrouter_async.py \
    --dataset-path ./datasets/probcopa_items.jsonl \
    --model x-ai/grok-4.1-fast \
    --n-responses 30 \
    --results-tag probcopa_grok-4.1-fast \
    --output-dir ./results/
```

See `scripts/README.md` for detailed documentation of all scripts.

### Generating Plots

All plots from the paper can be reproduced using the two analysis notebooks in the `scripts/` folder:

- `analyze_and_plot_results_models.ipynb`: Statistical tests and plots for model results and model-human comparisons (response distributions, Wasserstein distances, entropy analysis, temperature/reasoning effort ablations, persona prompting experiments).
- `analyze_and_plot_results_human.ipynb`: Statistical tests and plots for human results (response distributions, entropy analysis, comparison with Pavlick & Kwiatkowski 2019).

**Note:** Before running the notebooks, you must first remove canary strings from the data files (see [Canary Strings](#canary-strings) above).

## Key Results

Our study finds that while LLMs generally align with human judgments for highly likely or highly unlikely inferences, they consistently struggle with:
1. Inferences where humans show more uncertainty (middle-range likelihood scores)
2. Producing human-like distributions of judgments across sampled responses
3. Matching the variation in human responses

We also conduct follow-up experiments examining:
- **Temperature effects**: How sampling temperature affects response distributions
- **Reasoning effort / 'thinking budget'**: How reasoning budget/effort affects alignment with human judgments
- **Persona prompting**: How demographic and psychological persona prompts affect model responses

See the paper for complete results and analysis.

## Citation

If you use this dataset or code in your work, please cite:

```bibtex
# PLACEHOLDER: FILL THIS IN
```

## License

This work is licensed under the MIT License. See [LICENSE](./LICENSE) for details.
