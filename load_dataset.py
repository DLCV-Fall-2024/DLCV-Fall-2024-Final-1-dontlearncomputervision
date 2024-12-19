from datasets import load_dataset

dataset = load_dataset("ntudlcv/dlcv_2024_final1")
dataset.save_to_disk("data/dlcv_2024_final1")