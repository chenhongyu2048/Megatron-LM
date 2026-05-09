# convert_to_energon.py
import os
import io
import re
import ast
import json
import random
import argparse
from typing import Optional, List

from PIL import Image
from datasets import load_from_disk, concatenate_datasets
import webdataset as wds


SAMPLES_PER_SHARD = 1000

# Fixed random seed for reproducible fake answers when comparing framework performance
_FAKE_ANSWER_RNG = random.Random(20251231)


# ---------- Common utilities ----------

def pil_to_jpeg_bytes(image: Image.Image, quality: int = 95) -> bytes:
    if image.mode != "RGB":
        image = image.convert("RGB")
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def combine_images_vertical(
    images: List[Image.Image],
    target_width: int = 1024,
    gap: int = 8,
    bg_color=(255, 255, 255),
) -> Image.Image:
    """Vertically concatenate multiple PIL images at equal width into one tall image."""
    if len(images) == 1:
        return images[0].convert("RGB") if images[0].mode != "RGB" else images[0]

    resized = []
    for img in images:
        if img.mode != "RGB":
            img = img.convert("RGB")
        scale = target_width / img.width
        new_h = max(1, int(img.height * scale))
        resized.append(img.resize((target_width, new_h), Image.LANCZOS))

    total_h = sum(img.height for img in resized) + gap * (len(resized) - 1)
    canvas = Image.new("RGB", (target_width, total_h), bg_color)
    y = 0
    for img in resized:
        canvas.paste(img, (0, y))
        y += img.height + gap
    return canvas


# ---------- ScienceQA ----------

def format_scienceqa(hf_sample, idx: int) -> Optional[dict]:
    question = hf_sample["question"]
    choices = hf_sample["choices"]
    answer_idx = hf_sample["answer"]
    image = hf_sample.get("image", None)

    if image is None:
        return None

    choices_str = "\n".join([f"{i}: {c}" for i, c in enumerate(choices)])
    # --- Key: add explicit <image> placeholder to align with MMMU ---
    prompt = (
        f"Question: <image> {question}\n"
        f"Choices:\n{choices_str}\n"
        f"Answer:"
    )
    correct_answer = choices[answer_idx]

    return {
        "__key__": f"scienceqa_{idx:07d}",
        "context.txt": prompt,
        "answers.json": json.dumps([correct_answer]),
        "image.jpg": pil_to_jpeg_bytes(image),
    }

# ---------- MMMU ----------

_MMMU_LETTERS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
_IMG_TAG_RE = re.compile(r"<image\s*\d+>")
# Placeholder values in MMMU test split answer field
_IMAGE_PLACEHOLDER = "<image>"
_MMMU_PLACEHOLDER_ANSWERS = {"?", "", None}


def _parse_mmmu_options(options_raw) -> List[str]:
    if options_raw is None:
        return []
    if isinstance(options_raw, list):
        return list(options_raw)
    if isinstance(options_raw, str):
        try:
            val = ast.literal_eval(options_raw)
            if isinstance(val, list):
                return list(val)
        except Exception:
            pass
    return []


def format_mmmu(hf_sample, idx: int) -> Optional[dict]:
    """MMMU: 1-7 images + multiple-choice or open-ended.

    Note: MMMU test split answer fields are all placeholders "?".
    This script is only for training framework performance testing (not accuracy), so:
      - multiple-choice:  randomly pick A/B/C/D as fake answer
      - open:            use a fixed placeholder string
    This allows loss to be computed normally, but values are not meaningful.

    Image placeholder strategy (multi-image already vertically stacked into 1 image):
      - <image N> in question:     compressed to exactly 1 <image>
      - <image N> in options:      replaced with "[img]" (plain text marker)
      - <image> in answer:         also replaced with "[img]" (must not contain <image>)
    This ensures exactly 1 <image> placeholder in context, aligned with pixel_values.shape[0].
    """
    question = hf_sample["question"]
    options = _parse_mmmu_options(hf_sample.get("options"))
    answer = hf_sample.get("answer", "")
    question_type = hf_sample.get("question_type", "multiple-choice")

    images: List[Image.Image] = []
    for i in range(1, 8):
        img = hf_sample.get(f"image_{i}", None)
        if img is not None:
            images.append(img)

    if len(images) == 0:
        return None

    # --- Normalize <image N> placeholders ---
    def _normalize_img_tags(text: str) -> str:
        text = _IMG_TAG_RE.sub(_IMAGE_PLACEHOLDER, text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    question_clean = _normalize_img_tags(question)
    options_clean = [_normalize_img_tags(str(o)) for o in options]

    # Rule 1: replace all <image> in options with "[img]" (no placeholder)
    options_clean = [
        (re.sub(r"<image>", "[img]", o).strip() or "[img]")
        for o in options_clean
    ]

    # Rule 2: compress <image> in question to exactly 1 at the beginning
    n_img_in_q = question_clean.count(_IMAGE_PLACEHOLDER)
    if n_img_in_q == 0:
        question_clean = f"{_IMAGE_PLACEHOLDER} {question_clean}"
    elif n_img_in_q > 1:
        question_clean = question_clean.replace(_IMAGE_PLACEHOLDER, "").strip()
        question_clean = re.sub(r"\s+", " ", question_clean)
        question_clean = f"{_IMAGE_PLACEHOLDER} {question_clean}"
    # n_img_in_q == 1: leave as-is

    # --- Build prompt and answer ---
    if question_type == "multiple-choice" and options_clean:
        opts_str = "\n".join(
            [f"{_MMMU_LETTERS[i]}: {o}" for i, o in enumerate(options_clean)]
        )
        prompt = f"Question: {question_clean}\nOptions:\n{opts_str}\nAnswer:"

        if answer in _MMMU_PLACEHOLDER_ANSWERS:
            # test split "?" -> randomly pick a valid letter (limited to actual option count)
            n_opts = min(len(options_clean), 4)  # pool at most A/B/C/D
            fake_letter = _FAKE_ANSWER_RNG.choice(_MMMU_LETTERS[:n_opts])
            correct_answer = options_clean[_MMMU_LETTERS.index(fake_letter)]
        elif isinstance(answer, str) and answer in _MMMU_LETTERS:
            a_idx = _MMMU_LETTERS.index(answer)
            correct_answer = (
                options_clean[a_idx] if 0 <= a_idx < len(options_clean) else answer
            )
        else:
            correct_answer = str(answer)
    else:
        prompt = f"Question: {question_clean}\nAnswer:"
        if answer in _MMMU_PLACEHOLDER_ANSWERS:
            correct_answer = "placeholder"
        else:
            correct_answer = str(answer)

    # Rule 3: answer must not contain <image>
    correct_answer = re.sub(r"<image>", "[img]", str(correct_answer)).strip()
    if not correct_answer:
        correct_answer = "[img]"

    combined = combine_images_vertical(images, target_width=576)

    return {
        "__key__": f"mmmu_{idx:07d}",
        "context.txt": prompt,
        "answers.json": json.dumps([correct_answer]),
        "image.jpg": pil_to_jpeg_bytes(combined),
    }

# ---------- Dataset registry ----------

DATASET_REGISTRY = {
    "scienceqa": {
        "default_hf_dir": "~/run/dataset/scienceqa",
        "default_out_dir": "~/run/dataset/energon_scienceqa",
        "default_split": "train",
        "formatter": format_scienceqa,
    },
    "mmmu": {
        "default_hf_dir": "~/run/dataset/mmmu",
        "default_out_dir": "~/run/dataset/energon_mmmu",
        "default_split": "all",   # changed default: merge dev+validation+test
        "formatter": format_mmmu,
    },
}


# ---------- Main flow ----------

def _resolve_split(ds, split: str):
    """Return a Dataset object based on split name.

    - split="all" and ds is DatasetDict:  concat all splits
    - split is in ds:                     return that split
    - otherwise:                          return ds as a single Dataset
    """
    is_dict = hasattr(ds, "keys") and hasattr(ds, "values")

    if split == "all":
        if is_dict:
            parts = list(ds.values())
            names = list(ds.keys())
            print(f"  merging splits {names} -> single pool "
                  f"({sum(len(p) for p in parts)} samples total)")
            return concatenate_datasets(parts)
        else:
            print("  split='all' but dataset is not a DatasetDict; using as-is")
            return ds

    if is_dict and split in ds:
        return ds[split]

    print(f"  split '{split}' not found; using whole dataset")
    return ds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=list(DATASET_REGISTRY.keys()),
        required=True,
    )
    parser.add_argument("--hf_dir", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--split", type=str, default=None,
                        help="train / validation / test / all (merge all splits)")
    parser.add_argument("--samples_per_shard", type=int,
                        default=SAMPLES_PER_SHARD)
    args = parser.parse_args()

    cfg = DATASET_REGISTRY[args.dataset]
    hf_dir = os.path.expanduser(args.hf_dir or cfg["default_hf_dir"])
    out_dir = os.path.expanduser(args.out_dir or cfg["default_out_dir"])
    split = args.split or cfg["default_split"]
    formatter = cfg["formatter"]

    os.makedirs(out_dir, exist_ok=True)

    print(f"[{args.dataset}] Loading HF dataset from: {hf_dir}")
    ds = load_from_disk(hf_dir)

    data_split = _resolve_split(ds, split)

    total = len(data_split)
    print(f"[{args.dataset}] Converting {total} samples (split={split}) ...")

    shard_idx = 0
    sink = None
    written = 0
    skipped = 0

    for idx, sample in enumerate(data_split):
        if idx % args.samples_per_shard == 0:
            if sink is not None:
                sink.close()
            shard_path = os.path.join(out_dir, f"shard-{shard_idx:06d}.tar")
            sink = wds.TarWriter(shard_path)
            shard_idx += 1

        wds_sample = formatter(sample, idx)
        if wds_sample is None:
            skipped += 1
        else:
            sink.write(wds_sample)
            written += 1

        if (idx + 1) % 500 == 0:
            print(f"  [{idx + 1}/{total}]  written={written}  skipped={skipped}")

    if sink is not None:
        sink.close()

    print(f"\n[{args.dataset}] Done! {shard_idx} shard(s) written to: {out_dir}")
    print(f"  written = {written}")
    print(f"  skipped = {skipped}")
    print("\nNext step — generate Energon metadata:")
    print(f"  energon prepare {out_dir}")


# Usage examples:
#   # ScienceQA: train split
#   python convert_to_energon.py --dataset scienceqa
#
#   # MMMU: default merge dev+validation+test (for performance benchmarking)
#   python convert_to_energon.py --dataset mmmu
#
#   energon prepare ~/run/dataset/energon_mmmu

if __name__ == "__main__":
    main()