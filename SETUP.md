# Setup Guide

This guide will help you set up the environment to run experiments and generate plots.

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/probabilistic-reasoning-clean.git
cd probabilistic-reasoning-clean
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Up API Keys

Create a `.env` file in the project root with your API keys:

```bash
# OpenAI
export OPENAI_API_KEY="your-openai-api-key"

# Anthropic
export ANTHROPIC_API_KEY="your-anthropic-api-key"

# Google
export GOOGLE_API_KEY="your-google-api-key"

# Together AI
export TOGETHER_API_KEY="your-together-api-key"

# OpenRouter (for Grok and other models)
export OPENROUTER_API_KEY="your-openrouter-api-key"
```

Then source the file:

```bash
source .env
```

## Running Experiments

### Generate Model Responses

See `scripts/README.md` for detailed examples. Basic usage:

```bash
python scripts/get_probcopa_results.py \
    --dataset-path ./datasets/probcopa_items.jsonl \
    --provider openai \
    --model gpt-5 \
    --n-responses 30 \
    --results-tag probcopa_gpt-5 \
    --output-dir ./results/
```

### Generate Plots

Before running the analysis notebooks, remove canary strings from the data files:

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

Then open and run the analysis notebooks:

```bash
jupyter notebook scripts/analyze_and_plot_results_models.ipynb
jupyter notebook scripts/analyze_and_plot_results_human.ipynb
```

## Repository Structure

```
probabilistic-reasoning-clean/
├── README.md                   # Main documentation
├── SETUP.md                    # This file
├── LICENSE                     # MIT License
├── requirements.txt            # Python dependencies
├── .gitignore                  # Git ignore rules
├── ProbabilisticInferences.pdf  # Paper manuscript
├── datasets/                   # Data files
│   ├── probcopa_items.jsonl                              # ProbCOPA items (210 items)
│   ├── probcopa_CANARY.jsonl                             # Full dataset with human annotations (canary-protected)
│   ├── probcopa_30_samples.jsonl                         # Random sample for validation
│   └── pavlick_kwiatkowski_2019_sentencepair_data.jsonl  # NLI comparison dataset
├── results/                    # Experiment results
│   ├── probcopa_human_results_annotated_CANARY.jsonl     # Human annotations (canary-protected)
│   ├── probcopa_{model}.jsonl                            # Model results (8 models)
│   ├── temperature_experiments/                          # Temperature ablations
│   ├── reasoning_effort_experiments/                     # Reasoning effort & thinking budget ablations
│   └── persona_prompt_experiments/                       # Persona prompting experiments
├── plots/                      # Generated figures (31 PDFs)
├── scripts/                    # Code
│   ├── README.md                                  # Script documentation
│   ├── analyze_and_plot_results_models.ipynb      # Model analysis & plots
│   ├── analyze_and_plot_results_human.ipynb       # Human analysis & plots
│   ├── probcopa_inference_batch_api.py            # Batch API inference (primary)
│   ├── probcopa_inference_openrouter_async.py     # OpenRouter async inference
│   ├── get_probcopa_results.py                    # Synchronous inference (testing)
│   ├── fetch_batch_api_results.py                 # Fetch batch results
│   ├── process_batch_api_results.py               # Process batch results
│   ├── process_openrouter_results.py              # Process OpenRouter results
│   ├── create_probcopa_samples.py                 # Dataset sampling
│   ├── add_canary_strings.py                      # Add canary strings to data files
│   ├── remove_canary_strings.py                   # Remove canary strings from data files
│   └── silverman_test.R                           # Silverman's test of multimodality
└── assets/                     # Configuration files
    ├── model_argument_limits.json
    ├── structured_personas_demographic.jsonl
    └── structured_personas_psychological.jsonl
```

## Troubleshooting

### API Rate Limits

If you encounter rate limits, you can:
- Reduce `--n-responses` to generate fewer samples per item
- Add delays between API calls (modify the script)
- Use batch processing for supported providers

### Memory Issues

If you run out of memory when generating plots:
- Process models one at a time instead of loading all at once
- Reduce the number of items processed simultaneously
- Close other applications

### Missing Data Files

Some result files may not be included in the repository due to size constraints. You can:
- Generate them yourself using the inference scripts
- Contact the authors for access to the full dataset

### Canary Strings

If the analysis notebooks fail because data files are missing, you likely need to run `remove_canary_strings.py` first (see [Generate Plots](#generate-plots) above). The cleaned files are not tracked by git.

## Citation

If you use this code or data, please cite:

```bibtex

```

## Support

For questions or issues, please open an issue on GitHub.
