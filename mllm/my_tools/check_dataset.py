from datasets import load_from_disk
from PIL import Image
import math
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import os

# Define dataset paths
laion_dir = os.path.expanduser("~/run/dataset/laion_relaion-pop")
scienceqa_dir = os.path.expanduser("~/run/dataset/scienceqa")
patch_size = 16

def process_dataset(ds_dir, is_laion):
    """
    Process dataset and return statistics list for each sample.
    """
    try:
        dataset_all = load_from_disk(ds_dir)
    except FileNotFoundError:
        print(f"Warning: dataset directory '{ds_dir}' does not exist, skipping.")
        return [], [], []

    dataset = dataset_all['train']
    print(f"Successfully loaded dataset ({'LAION' if is_laion else 'ScienceQA'}), total samples: {len(dataset)}")

    text_tokens = []
    image_tokens = []
    ratios = []

    for sample in tqdm(dataset, desc=f"Processing {'LAION' if is_laion else 'ScienceQA'}"):
        if is_laion:
            width = sample.get('width', 0)
            height = sample.get('height', 0)
            image_token = math.ceil(width / patch_size) * math.ceil(height / patch_size)
            
            cogvlm_text = sample.get('cogvlm_caption', '')
            text_count = len(cogvlm_text.split())
        else:
            image_token = 0
            img = sample.get('image')

            if img is not None and isinstance(img, Image.Image):
                width, height = img.size
                image_token = math.ceil(width / patch_size) * math.ceil(height / patch_size)

            question = sample.get('question', '')
            choices = " ".join(sample.get('choices', []))
            hint = sample.get('hint', '')
            lecture = sample.get('lecture', '')
            solution = sample.get('solution', '')

            full_text = f"{question} {choices} {hint} {lecture} {solution}"
            text_count = len(full_text.split())
        
        text_tokens.append(text_count)
        image_tokens.append(image_token)
        
        # calculate ratio, avoid division by zero
        if text_count > 0:
            ratios.append(image_token / text_count)

    return text_tokens, image_tokens, ratios

# 1. Collect data from both datasets
laion_texts, laion_images, laion_ratios = process_dataset(laion_dir, is_laion=True)
sqa_texts, sqa_images, sqa_ratios = process_dataset(scienceqa_dir, is_laion=False)

if not laion_texts and not sqa_texts:
    raise ValueError("No datasets loaded successfully, cannot plot.")

# 2. Plot charts (2 rows x 3 columns)
plt.style.use('seaborn-v0_8-whitegrid')
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Get global max values to unify X axis range
max_text = max(max(laion_texts, default=0), max(sqa_texts, default=0))
max_image = max(max(laion_images, default=0), max(sqa_images, default=0))
max_ratio = max(max(laion_ratios, default=0), max(sqa_ratios, default=0))

def plot_log_hist(ax, data, color, title, xlabel, global_max):
    """
    Helper: plot log-log histogram on given subplot.
    """
    # Filter out zeros since log(0) is undefined
    data_filtered = [x for x in data if x > 0]
    
    if not data_filtered:
        # also increase title font size for no-data case
        ax.set_title(f"{title} (No Data > 0)", fontsize=16)
        return

    min_val = min(data_filtered)
    # Use log-evenly-spaced bins for uniform bar width in log scale
    bins = np.logspace(np.log10(min_val), np.log10(global_max), 200)
    
    ax.hist(data_filtered, bins=bins, alpha=0.7, color=color, density=True, log=True)
    ax.set_xscale('log')
    
    # Use fontsize to increase title and axis label sizes
    ax.set_title(title, fontsize=20)          # title font size 16
    ax.set_xlabel(xlabel, fontsize=18)        # X axis label font size 14
    ax.set_ylabel('Log Density', fontsize=18) # Y axis label font size 14
    
    # Uncomment below to increase tick label font size (e.g. 10^1, 10^2)
    ax.tick_params(axis='both', which='major', labelsize=16)
    
    # Force X axis max to global max
    ax.set_xlim(left=min_val, right=global_max)

# --- Row 1: LAION dataset ---
plot_log_hist(axes[0, 0], laion_texts, 'blue', 'LAION: Text Token Count', 'Text Tokens (Log)', max_text)
plot_log_hist(axes[0, 1], laion_images, 'blue', 'LAION: Image Token Count', 'Image Tokens (Log)', max_image)
plot_log_hist(axes[0, 2], laion_ratios, 'blue', 'LAION: Image/Text Ratio', 'Ratio (Log)', max_ratio)

# --- Row 2: ScienceQA dataset ---
plot_log_hist(axes[1, 0], sqa_texts, 'orange', 'ScienceQA: Text Token Count', 'Text Tokens (Log)', max_text)
plot_log_hist(axes[1, 1], sqa_images, 'orange', 'ScienceQA: Image Token Count', 'Image Tokens (Log)', max_image)
plot_log_hist(axes[1, 2], sqa_ratios, 'orange', 'ScienceQA: Image/Text Ratio', 'Ratio (Log)', max_ratio)

plt.tight_layout()
plt.savefig("token_distribution_2x3_logX.png", dpi=300)
print("Chart saved as token_distribution_2x3_logX.png")
plt.show()