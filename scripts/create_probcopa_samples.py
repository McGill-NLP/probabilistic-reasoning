import pandas as pd
import numpy as np
import os

# For random sample:
n_samples_random = 30
random_seed = 3535
dataset_dir = "./datasets/"
dataset_path = os.path.join(dataset_dir, "probcopa_items.jsonl")
sample_name = f"probcopa_{n_samples_random}_samples"
sample_path = os.path.join(dataset_dir, f"{sample_name}.jsonl")

dataset = pd.read_json(dataset_path, lines=True)
dataset = dataset.sample(n=n_samples_random, random_state=random_seed).sort_values(by='UID')
dataset.to_json(sample_path, orient='records', lines=True)

print(f"Created {n_samples_random} random samples from {dataset_path}")
print(f"Saved to {sample_path}")
print(f"Random seed: {random_seed}")
print(f"Number of samples: {len(dataset)}")
