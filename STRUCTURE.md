# Repository Structure

```
probabilistic-reasoning-clean/
│
├── README.md                    # Main documentation
├── SETUP.md                     # Installation and setup guide
├── SUMMARY.md                   # Summary of changes from original repo
├── TODO.md                      # Checklist of remaining tasks
├── STRUCTURE.md                 # This file
├── COMPLETION_REPORT.md         # Completion report from initial cleanup
├── LICENSE                      # MIT License
├── requirements.txt             # Python dependencies
├── .gitignore                   # Git ignore rules
├── ProbabilisticInferences.pdf   # Paper manuscript
│
├── datasets/                    # Dataset files
│   ├── probcopa_items.jsonl                              # ProbCOPA items (210 items, no annotations)
│   ├── probcopa_CANARY.jsonl                             # Full dataset with human annotations (canary-protected)
│   ├── probcopa_30_samples.jsonl                         # Random 30-item sample for validation
│   └── pavlick_kwiatkowski_2019_sentencepair_data.jsonl  # NLI comparison dataset (Pavlick & Kwiatkowski, 2019)
│
├── results/                     # Experiment results
│   ├── probcopa_human_results_annotated_CANARY.jsonl     # Human annotations (canary-protected)
│   ├── probcopa_random_sample_validation_round_human_results_CANARY.jsonl
│   ├── probcopa_random_sample_prompt_validation_round_human_results_CANARY.jsonl
│   ├── reasoning_chain_100_samples_annotated.jsonl       # Annotated reasoning chains
│   ├── ProbCOPA_silverman_test_significances.csv          # Silverman test results
│   │
│   ├── probcopa_gpt-5.jsonl                # Main model results (8 models)
│   ├── probcopa_claude-sonnet-4.5.jsonl
│   ├── probcopa_DeepSeek-R1.jsonl
│   ├── probcopa_gemini-3-pro-preview.jsonl
│   ├── probcopa_Kimi-K2-Thinking.jsonl
│   ├── probcopa_Qwen3-235B-Thinking.jsonl
│   ├── probcopa_GLM-4.6.jsonl
│   ├── probcopa_grok-4.1-fast.jsonl
│   │
│   ├── temperature_experiments/             # Temperature ablations (30 files)
│   │   └── {model}_temperature_{0.4,0.8,1.2,1.6,2.0}.jsonl
│   │       # 6 models: DeepSeek-R1, GLM-4.6, Kimi-K2-Thinking,
│   │       #           Qwen3-235B-Thinking, gemini-3-pro-preview, grok-4.1-fast
│   │
│   ├── reasoning_effort_experiments/        # Reasoning effort & thinking budget ablations (24 files)
│   │   ├── {model}_reasoning_effort_{low,medium,high}.jsonl
│   │   │   # 6 models: gpt-5, claude-opus-4.6, DeepSeek-R1,
│   │   │   #           GLM-4.6, Kimi-K2-Thinking, Qwen3-235B-Thinking
│   │   ├── claude-sonnet-4.5_thinking_budget_{512,2048,4096}.jsonl
│   │   └── gemini-3-pro-preview_thinking_budget_{512,2048,4096}.jsonl
│   │
│   └── persona_prompt_experiments/          # Persona prompting experiments (16 files)
│       ├── {model}_structured_personas_demographic.jsonl
│       └── {model}_structured_personas_psychological.jsonl
│           # 8 models: all main models
│
├── plots/                       # Generated figures (31 PDFs)
│   ├── overall_human_response_distribution.pdf
│   ├── sample_human_response_distribution.pdf
│   ├── pavlick_data_histograms.pdf
│   ├── model_response_distributions*.pdf
│   ├── model_vs_human_wasserstein_distance_*.pdf
│   ├── model_vs_human_median_responses_*.pdf
│   ├── model_vs_human_entropy_*.pdf
│   ├── model_vs_human_entropy_vs_reasoning_chain_length_*.pdf
│   ├── reasoning_chain_length_vs_*.pdf
│   ├── unified_temperature_ablation.pdf
│   ├── unified_reasoning_effort_thinking_budget_ablation.pdf
│   ├── unified_persona_prompting_ablation.pdf
│   ├── median_response_vs_entropy.pdf
│   ├── time_vs_entropy.pdf
│   ├── human_joint_four_comparisons.pdf
│   ├── gemini_joint_four_comparisons.pdf
│   └── ensemble_wasserstein_distance_boxplot.pdf
│
├── scripts/                     # Code
│   ├── README.md                                  # Script documentation
│   │
│   ├── analyze_and_plot_results_models.ipynb      # Model analysis & plotting notebook
│   ├── analyze_and_plot_results_human.ipynb       # Human analysis & plotting notebook
│   │
│   ├── probcopa_inference_batch_api.py            # Batch API inference (primary)
│   ├── probcopa_inference_openrouter_async.py     # OpenRouter async inference
│   ├── get_probcopa_results.py                    # Synchronous inference (for testing)
│   ├── fetch_batch_api_results.py                 # Fetch completed batch results
│   ├── process_batch_api_results.py               # Process and standardize batch results
│   ├── process_openrouter_results.py              # Process OpenRouter results
│   ├── create_probcopa_samples.py                 # Dataset sampling utility
│   │
│   ├── add_canary_strings.py                      # Add canary strings to data files
│   ├── remove_canary_strings.py                   # Remove canary strings from data files
│   │
│   └── silverman_test.R                           # Silverman's test of multimodality
│
└── assets/                      # Configuration files
    ├── model_argument_limits.json                 # Model parameter limits
    ├── structured_personas_demographic.jsonl       # Demographic persona prompts
    └── structured_personas_psychological.jsonl     # Psychological persona prompts
```

## File Counts by Type

| Type | Count | Description |
|------|-------|-------------|
| Documentation (root) | 6 | README, SETUP, SUMMARY, TODO, STRUCTURE, COMPLETION_REPORT |
| Documentation (scripts) | 1 | scripts/README.md |
| Python Scripts | 9 | Inference, processing, canary string utilities, sampling |
| Jupyter Notebooks | 2 | Model analysis, human analysis |
| R Scripts | 1 | Silverman's test |
| Datasets | 4 | ProbCOPA items + annotations + sample + comparison |
| Results | 86 | Human + 8 models + 30 temp + 24 reasoning/thinking + 16 persona |
| Plots | 31 | Publication-ready figures (PDFs) |
| Assets | 3 | Model limits, persona prompts |
| Config | 3 | LICENSE, requirements.txt, .gitignore |
| Paper | 1 | ProbabilisticInferences.pdf |

## Canary String Files

Files with a `_CANARY` suffix contain canary strings for data contamination detection. Before analysis, use `scripts/remove_canary_strings.py` to produce cleaned versions. The cleaned files are listed in `.gitignore` and not tracked by git:

| Canary-protected file | Cleaned output |
|----------------------|----------------|
| `datasets/probcopa_CANARY.jsonl` | `datasets/probcopa.jsonl` |
| `results/probcopa_human_results_annotated_CANARY.jsonl` | `results/probcopa_human_results_annotated.jsonl` |
| `results/probcopa_random_sample_validation_round_human_results_CANARY.jsonl` | `results/probcopa_random_sample_validation_round_human_results_cleaned.jsonl` |
| `results/probcopa_random_sample_prompt_validation_round_human_results_CANARY.jsonl` | `results/probcopa_random_sample_prompt_validation_round_human_results_cleaned.jsonl` |

## Key Files Description

### Scripts

**Inference Scripts (used to generate paper results):**
- **probcopa_inference_batch_api.py**: Primary script for batch API inference (OpenAI, Anthropic, Google, Together AI). More cost-efficient for large-scale experiments.
- **probcopa_inference_openrouter_async.py**: Async inference for models available through OpenRouter (e.g., Grok). Handles concurrency and rate limiting.

**Fetching and Processing Scripts:**
- **fetch_batch_api_results.py**: Fetches completed batch results from provider APIs.
- **process_batch_api_results.py**: Processes raw batch API results into standardized format.
- **process_openrouter_results.py**: Processes OpenRouter async inference results.

**Utility Scripts:**
- **get_probcopa_results.py**: General-purpose synchronous inference script for interactive experimentation.
- **create_probcopa_samples.py**: Creates random samples from the dataset for validation studies.
- **add_canary_strings.py**: Inserts canary strings into JSONL files for contamination detection.
- **remove_canary_strings.py**: Removes canary strings from JSONL files before analysis.

**Statistical Tests & Plots:**
- **silverman_test.R**: Silverman's test of multimodality on ProbCOPA human annotations.
- **analyze_and_plot_results_models.ipynb**: Model response analysis, model-human comparisons, and ablation plots.
- **analyze_and_plot_results_human.ipynb**: Human response analysis and comparison with Pavlick & Kwiatkowski (2019).

### Datasets

- **probcopa_items.jsonl**: ProbCOPA dataset with 210 handcrafted probabilistic inferences (items only, no annotations).
- **probcopa_CANARY.jsonl**: Full ProbCOPA dataset with human annotations (canary-protected).
- **probcopa_30_samples.jsonl**: Random sample of 30 items for validation studies.
- **pavlick_kwiatkowski_2019_sentencepair_data.jsonl**: Comparison NLI dataset from Pavlick & Kwiatkowski (2019).

---

**Last Updated:** 2026-02-23
