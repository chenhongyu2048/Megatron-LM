import os
import io
import json
from datasets import load_from_disk
import webdataset as wds

SAMPLES_PER_SHARD = 1000  # 每个 tar 包的样本数，可按需调整

def format_scienceqa(hf_sample, idx):
    question = hf_sample["question"]
    choices = "\n".join([f"{i}: {c}" for i, c in enumerate(hf_sample["choices"])])
    full_prompt = f"Question: {question}\nChoices:\n{choices}\nAnswer:"

    answer_idx = hf_sample["answer"]
    correct_answer = hf_sample["choices"][answer_idx]

    image = hf_sample.get("image", None)

    sample = {
        "__key__": f"scienceqa_{idx:07d}",
        # 将 context 和 answers 分别存为不同扩展名的文件
        "context.txt": full_prompt,
        "answers.json": json.dumps([correct_answer]),
    }

    if image is not None:
        # 将 PIL Image 转为 JPEG bytes
        buf = io.BytesIO()
        image.save(buf, format="JPEG")
        sample["image.jpg"] = buf.getvalue() # type: ignore
    else:
        return None # skip samples without images

    return sample


def main():
    hf_dir = os.path.expanduser("~/run/dataset/scienceqa")
    energon_out_dir = os.path.expanduser("~/run/dataset/energon_scienceqa")
    os.makedirs(energon_out_dir, exist_ok=True)

    # 1. 加载 HuggingFace 数据
    print("Loading HF dataset...")
    ds = load_from_disk(hf_dir)
    train_ds = ds["train"] if "train" in ds else ds

    # 2. 用 webdataset TarWriter 写入 tar 分片
    total = len(train_ds)
    print(f"Converting {total} samples to WebDataset tar shards...")

    shard_idx = 0
    sink = None

    for idx, sample in enumerate(train_ds):
        # 每 SAMPLES_PER_SHARD 条开一个新 tar
        if idx % SAMPLES_PER_SHARD == 0:
            if sink is not None:
                sink.close()
            shard_path = os.path.join(energon_out_dir, f"shard-{shard_idx:06d}.tar")
            sink = wds.TarWriter(shard_path)
            shard_idx += 1

        wds_sample = format_scienceqa(sample, idx)
        if wds_sample is not None:
            sink.write(wds_sample)

        if (idx + 1) % 500 == 0:
            print(f"  [{idx + 1}/{total}]")

    if sink is not None:
        sink.close()

    print(f"\nDone! {shard_idx} shard(s) written to: {energon_out_dir}")
    print("Next step: run the following command to generate Energon metadata:\n")
    print(f"  energon prepare {energon_out_dir}")


# python convert_to_energon.py
# energon prepare /data/home/scyb683/run/dataset/energon_scienceqa
# python check_energon_dataset.py --data_path /data/home/scyb683/run/dataset/energon_scienceqa

if __name__ == "__main__":
    main()