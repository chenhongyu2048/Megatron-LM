# STANDERD DATASET DOWNLOAD CODE
# from datasets import load_dataset

ds_name = "derek-thomas/ScienceQA"
ds_dir = "~/run/dataset/scienceqa"
ds = load_dataset(ds_name, token="YOUR_HF_TOKEN")
ds.save_to_disk(ds_dir)

# DOWNLOAD MMMU DATASET
import os
from datasets import load_dataset, concatenate_datasets, DatasetDict

ds_name = "MMMU/MMMU"
save_dir = os.path.expanduser("~/run/dataset/MMMU_all")

configs = [
    'Accounting', 'Agriculture', 'Architecture_and_Engineering', 'Art',
    'Art_Theory', 'Basic_Medical_Science', 'Biology', 'Chemistry',
    'Clinical_Medicine', 'Computer_Science', 'Design',
    'Diagnostics_and_Laboratory_Medicine', 'Economics', 'Electronics',
    'Energy_and_Power', 'Finance', 'Geography', 'History', 'Literature',
    'Manage', 'Marketing', 'Materials', 'Math', 'Mechanical_Engineering',
    'Music', 'Pharmacy', 'Physics', 'Psychology', 'Public_Health', 'Sociology'
]

# Collect all subsets by split
split_buckets = {}  # {"dev": [...], "validation": [...], "test": [...]}

for cfg in configs:
    print(f"Loading {cfg} ...")
    ds = load_dataset(ds_name, cfg)  # add token=... if needed
    for split_name, split_ds in ds.items():
        # add a subject column to distinguish sources
        split_ds = split_ds.add_column("subject", [cfg] * len(split_ds))
        split_buckets.setdefault(split_name, []).append(split_ds)

# Merge each split
merged = DatasetDict({
    split: concatenate_datasets(dsets)
    for split, dsets in split_buckets.items()
})

print(merged)
merged.save_to_disk(save_dir)