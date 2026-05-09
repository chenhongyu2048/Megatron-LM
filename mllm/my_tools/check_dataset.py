import argparse
import math
import os

import matplotlib.pyplot as plt
import numpy as np
from datasets import load_from_disk
from PIL import Image
from tqdm import tqdm

PATCH_SIZE = 14


def count_pixels(images):
    """Total pixel count across all non-None images in a sample."""
    total = 0
    for img in images:
        if img is not None and isinstance(img, Image.Image):
            total += img.width * img.height
    return total


def count_image_tokens(images):
    """Total vision tokens (patches) across all non-None images."""
    total = 0
    for img in images:
        if img is not None and isinstance(img, Image.Image):
            total += math.ceil(img.width / PATCH_SIZE) * math.ceil(img.height / PATCH_SIZE)
    return total


def count_text_tokens(text):
    """Simple whitespace-based token count."""
    return len(text.split())


def process_mmmu(ds_dir):
    """Load MMMU dataset and compute per-sample statistics for each split."""
    ds = load_from_disk(os.path.expanduser(ds_dir))
    results = {}

    for split_name in ds.keys():
        split_ds = ds[split_name]
        pixel_counts = []
        image_token_counts = []
        text_token_counts = []

        for sample in tqdm(split_ds, desc=f"MMMU/{split_name}"):
            images = [sample.get(f"image_{i}") for i in range(1, 8)]
            images = [img for img in images if img is not None]

            pixel_counts.append(count_pixels(images))
            image_token_counts.append(count_image_tokens(images))

            question = sample.get("question", "")
            options = str(sample.get("options", ""))
            explanation = str(sample.get("explanation", ""))
            full_text = f"{question} {options} {explanation}"
            text_token_counts.append(count_text_tokens(full_text))

        results[split_name] = {
            "num_samples": len(split_ds),
            "pixel_counts": pixel_counts,
            "image_token_counts": image_token_counts,
            "text_token_counts": text_token_counts,
        }

    return results


def process_scienceqa(ds_dir):
    """Load ScienceQA dataset and compute per-sample statistics for each split."""
    ds = load_from_disk(os.path.expanduser(ds_dir))
    results = {}

    for split_name in ds.keys():
        split_ds = ds[split_name]
        pixel_counts = []
        image_token_counts = []
        text_token_counts = []

        for sample in tqdm(split_ds, desc=f"SQA/{split_name}"):
            img = sample.get("image")
            images = [img] if img is not None else []

            pixel_counts.append(count_pixels(images))
            image_token_counts.append(count_image_tokens(images))

            question = sample.get("question", "")
            choices = " ".join(sample.get("choices", []))
            hint = sample.get("hint", "")
            lecture = sample.get("lecture", "")
            solution = sample.get("solution", "")
            full_text = f"{question} {choices} {hint} {lecture} {solution}"
            text_token_counts.append(count_text_tokens(full_text))

        results[split_name] = {
            "num_samples": len(split_ds),
            "pixel_counts": pixel_counts,
            "image_token_counts": image_token_counts,
            "text_token_counts": text_token_counts,
        }

    return results


def plot_distributions(mmmu_results, sqa_results, output_path):
    """Plot histograms comparing MMMU and ScienceQA distributions.

    Layout: 2 rows (MMMU, SQA) x 3 cols (pixels, image tokens, text tokens)
    per split.
    """
    plt.style.use("seaborn-v0_8-whitegrid")
    split_names = sorted(set(list(mmmu_results.keys()) + list(sqa_results.keys())))
    num_splits = len(split_names)

    fig, axes = plt.subplots(
        num_splits * 2, 3, figsize=(18, 5 * num_splits)
    )

    if num_splits == 1:
        axes = axes.reshape(2, 3)

    # Gather global maxima for unified x-axis
    all_pixel = []
    all_img_tok = []
    all_text_tok = []
    for r in [mmmu_results, sqa_results]:
        for split_info in r.values():
            all_pixel.extend(split_info["pixel_counts"])
            all_img_tok.extend(split_info["image_token_counts"])
            all_text_tok.extend(split_info["text_token_counts"])

    max_pixel = max(all_pixel) if all_pixel else 1
    max_img_tok = max(all_img_tok) if all_img_tok else 1
    max_text_tok = max(all_text_tok) if all_text_tok else 1

    def plot_log_hist(ax, data, color, xlabel, global_max):
        filtered = [x for x in data if x > 0]
        if not filtered:
            ax.set_title("(No Data > 0)", fontsize=14)
            return
        min_val = min(filtered)
        bins = np.logspace(np.log10(min_val), np.log10(global_max), 200)
        ax.hist(filtered, bins=bins, alpha=0.7, color=color, density=True, log=True)
        ax.set_xscale("log")
        ax.set_xlabel(xlabel, fontsize=14)
        ax.set_ylabel("Log Density", fontsize=14)
        ax.tick_params(axis="both", which="major", labelsize=12)
        ax.set_xlim(left=min_val, right=global_max)

    for row_idx, split_name in enumerate(split_names):
        # MMMU row
        m_row = row_idx * 2
        if split_name in mmmu_results:
            r = mmmu_results[split_name]
            axes[m_row, 0].set_title(
                f"MMMU/{split_name}: Pixels (n={r['num_samples']})", fontsize=14
            )
            plot_log_hist(axes[m_row, 0], r["pixel_counts"], "blue",
                          "Pixels (Log)", max_pixel)
            axes[m_row, 1].set_title(
                f"MMMU/{split_name}: Image Tokens (patch={PATCH_SIZE})", fontsize=14
            )
            plot_log_hist(axes[m_row, 1], r["image_token_counts"], "blue",
                          "Image Tokens (Log)", max_img_tok)
            axes[m_row, 2].set_title(
                f"MMMU/{split_name}: Text Tokens", fontsize=14
            )
            plot_log_hist(axes[m_row, 2], r["text_token_counts"], "blue",
                          "Text Tokens (Log)", max_text_tok)
        else:
            for col in range(3):
                axes[m_row, col].set_visible(False)

        # ScienceQA row
        s_row = row_idx * 2 + 1
        if split_name in sqa_results:
            r = sqa_results[split_name]
            axes[s_row, 0].set_title(
                f"SQA/{split_name}: Pixels (n={r['num_samples']})", fontsize=14
            )
            plot_log_hist(axes[s_row, 0], r["pixel_counts"], "orange",
                          "Pixels (Log)", max_pixel)
            axes[s_row, 1].set_title(
                f"SQA/{split_name}: Image Tokens (patch={PATCH_SIZE})", fontsize=14
            )
            plot_log_hist(axes[s_row, 1], r["image_token_counts"], "orange",
                          "Image Tokens (Log)", max_img_tok)
            axes[s_row, 2].set_title(
                f"SQA/{split_name}: Text Tokens", fontsize=14
            )
            plot_log_hist(axes[s_row, 2], r["text_token_counts"], "orange",
                          "Text Tokens (Log)", max_text_tok)
        else:
            for col in range(3):
                axes[s_row, col].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Chart saved as {output_path}")


def print_summary(mmmu_results, sqa_results):
    """Print per-split sample counts and basic statistics."""
    print("\n" + "=" * 70)
    print("MMMU")
    print("=" * 70)
    for split_name in sorted(mmmu_results.keys()):
        r = mmmu_results[split_name]
        pixels = r["pixel_counts"]
        img_toks = r["image_token_counts"]
        text_toks = r["text_token_counts"]
        print(f"  {split_name}: {r['num_samples']} samples")
        print(f"    pixels:      mean={np.mean(pixels):.0f}  median={np.median(pixels):.0f}  "
              f"min={np.min(pixels)}  max={np.max(pixels)}")
        print(f"    img_tokens:  mean={np.mean(img_toks):.1f}  median={np.median(img_toks):.1f}  "
              f"min={np.min(img_toks)}  max={np.max(img_toks)}")
        print(f"    text_tokens: mean={np.mean(text_toks):.1f}  median={np.median(text_toks):.1f}  "
              f"min={np.min(text_toks)}  max={np.max(text_toks)}")

    print()
    print("=" * 70)
    print("ScienceQA")
    print("=" * 70)
    for split_name in sorted(sqa_results.keys()):
        r = sqa_results[split_name]
        pixels = r["pixel_counts"]
        img_toks = r["image_token_counts"]
        text_toks = r["text_token_counts"]
        print(f"  {split_name}: {r['num_samples']} samples")
        print(f"    pixels:      mean={np.mean(pixels):.0f}  median={np.median(pixels):.0f}  "
              f"min={np.min(pixels)}  max={np.max(pixels)}")
        print(f"    img_tokens:  mean={np.mean(img_toks):.1f}  median={np.median(img_toks):.1f}  "
              f"min={np.min(img_toks)}  max={np.max(img_toks)}")
        print(f"    text_tokens: mean={np.mean(text_toks):.1f}  median={np.median(text_toks):.1f}  "
              f"min={np.min(text_toks)}  max={np.max(text_toks)}")


def main():
    parser = argparse.ArgumentParser(
        description="Compute pixel count and text token distributions for MMMU and ScienceQA."
    )
    parser.add_argument(
        "--mmmu_dir", type=str, default="~/run/dataset/mmmu",
        help="Path to MMMU HF dataset directory."
    )
    parser.add_argument(
        "--sqa_dir", type=str, default="~/run/dataset/scienceqa",
        help="Path to ScienceQA HF dataset directory."
    )
    parser.add_argument(
        "--output", type=str, default="token_distribution_mmmu_sqa.png",
        help="Output image path for the histogram chart."
    )
    parser.add_argument(
        "--patch_size", type=int, default=14,
        help="Patch size for vision token calculation (default: 16)."
    )
    args = parser.parse_args()

    global PATCH_SIZE
    PATCH_SIZE = args.patch_size

    print(f"Loading MMMU from: {args.mmmu_dir}")
    mmmu_results = process_mmmu(args.mmmu_dir)

    print(f"\nLoading ScienceQA from: {args.sqa_dir}")
    sqa_results = process_scienceqa(args.sqa_dir)

    print_summary(mmmu_results, sqa_results)
    plot_distributions(mmmu_results, sqa_results, args.output)
    plt.show()


if __name__ == "__main__":
    main()
