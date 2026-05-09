import os
from datasets import load_dataset, concatenate_datasets, DatasetDict
from tqdm import tqdm

# MMMU's 30 subjects, each with dev/validation/test splits (test has no answers)
MMMU_SUBJECTS = [
    # Art & Design
    "Art", "Art_Theory", "Design", "Music",
    # Business
    "Accounting", "Economics", "Finance", "Manage", "Marketing",
    # Science
    "Biology", "Chemistry", "Geography", "Math", "Physics",
    # Health & Medicine
    "Basic_Medical_Science", "Clinical_Medicine",
    "Diagnostics_and_Laboratory_Medicine", "Pharmacy", "Public_Health",
    # Humanities & Social Science
    "History", "Literature", "Sociology", "Psychology",
    # Tech & Engineering
    "Agriculture", "Architecture_and_Engineering", "Computer_Science",
    "Electronics", "Energy_and_Power", "Materials", "Mechanical_Engineering",
]


def download_mmmu(
    save_dir: str = "~/run/dataset/mmmu",
    splits_to_keep=("dev", "validation", "test"),
    add_subject_field: bool = True,
):
    """Download all 30 MMMU subjects and merge by split.

    Resulting directory structure:
        mmmu/
            dev/            # all subjects dev merged
            validation/     # all subjects validation merged
            test/           # all subjects test merged (test has no answers)
    """
    save_dir = os.path.expanduser(save_dir)
    os.makedirs(save_dir, exist_ok=True)

    # Collect Dataset objects for each subject within each split
    split_buckets = {s: [] for s in splits_to_keep}

    print(f"Downloading MMMU: {len(MMMU_SUBJECTS)} subjects "
          f"× {len(splits_to_keep)} splits ...\n")

    for subject in tqdm(MMMU_SUBJECTS, desc="Subjects"):
        try:
            ds = load_dataset("MMMU/MMMU", subject, token="YOUR_HF_TOKEN")
        except Exception as e:
            print(f"  [WARN] failed to load subject '{subject}': {e}")
            continue

        for split in splits_to_keep:
            if split not in ds:
                continue
            split_ds = ds[split]

            # add a 'subject' field for per-subject analysis during training/eval
            if add_subject_field:
                split_ds = split_ds.add_column(
                    "subject", [subject] * len(split_ds)
                )

            split_buckets[split].append(split_ds)

    # Merge all subjects within each split
    print("\nConcatenating per split ...")
    merged = {}
    for split, parts in split_buckets.items():
        if not parts:
            print(f"  [skip] split '{split}' has no data")
            continue
        merged[split] = concatenate_datasets(parts)
        print(f"  {split:12s}  {len(merged[split]):>6d} samples")

    final = DatasetDict(merged)

    print(f"\nSaving to: {save_dir}")
    final.save_to_disk(save_dir)
    print("Done.")

    # Print sample schema for debugging
    any_split = next(iter(final))
    print(f"\nSample columns ({any_split} split):")
    for col in final[any_split].column_names:
        print(f"  - {col}")

    return final

# DATASET_TO_DOWNLOAD = "scienceqa"  # "mmmu"
DATASET_TO_DOWNLOAD = "mmmu" 

if DATASET_TO_DOWNLOAD == "scienceqa":
    ds_name = "derek-thomas/ScienceQA"
    ds_dir = "~/run/dataset/scienceqa"
    ds = load_dataset(ds_name, token="YOUR_HF_TOKEN")
    ds.save_to_disk(ds_dir)
elif DATASET_TO_DOWNLOAD == "mmmu":
    download_mmmu(
        save_dir="~/run/dataset/mmmu",
        splits_to_keep=("dev", "validation", "test"),
        add_subject_field=True,
    )
