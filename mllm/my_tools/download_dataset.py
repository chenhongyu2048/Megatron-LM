from datasets import load_dataset

ds_name = "laion/relaion-pop"
ds_dir = "~/run/dataset/laion_relaion-pop"
ds = load_dataset(ds_name, token="YOUR_HF_TOKEN")
ds.save_to_disk(ds_dir)

ds_name = "derek-thomas/ScienceQA"
ds_dir = "~/run/dataset/scienceqa"
ds = load_dataset(ds_name, token="YOUR_HF_TOKEN")
ds.save_to_disk(ds_dir)