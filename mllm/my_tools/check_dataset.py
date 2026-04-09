from datasets import load_from_disk
from PIL import Image
import math
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import os

# 定义数据集路径
laion_dir = os.path.expanduser("~/run/dataset/laion_relaion-pop")
scienceqa_dir = os.path.expanduser("~/run/dataset/scienceqa")
patch_size = 16

def process_dataset(ds_dir, is_laion):
    """
    处理数据集并返回每个样本的统计信息列表
    """
    try:
        dataset_all = load_from_disk(ds_dir)
    except FileNotFoundError:
        print(f"警告: 数据集目录 '{ds_dir}' 不存在，跳过该数据集。")
        return [], [], []

    dataset = dataset_all['train']
    print(f"成功加载数据集 ({'LAION' if is_laion else 'ScienceQA'})，总样本数: {len(dataset)}")

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
        
        # 计算比例，避免除以0的情况
        if text_count > 0:
            ratios.append(image_token / text_count)

    return text_tokens, image_tokens, ratios

# 1. 收集两个数据集的数据
laion_texts, laion_images, laion_ratios = process_dataset(laion_dir, is_laion=True)
sqa_texts, sqa_images, sqa_ratios = process_dataset(scienceqa_dir, is_laion=False)

if not laion_texts and not sqa_texts:
    raise ValueError("没有成功加载任何数据集，无法绘图。")

# 2. 绘制图表 (2行3列)
plt.style.use('seaborn-v0_8-whitegrid')
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# 获取各项数据的全局最大值，用于统一 X 轴的最大范围
max_text = max(max(laion_texts, default=0), max(sqa_texts, default=0))
max_image = max(max(laion_images, default=0), max(sqa_images, default=0))
max_ratio = max(max(laion_ratios, default=0), max(sqa_ratios, default=0))

def plot_log_hist(ax, data, color, title, xlabel, global_max):
    """
    辅助函数：在指定的子图上绘制双对数(X和Y均为对数)直方图
    """
    # 过滤掉 0 值，因为 log(0) 无意义
    data_filtered = [x for x in data if x > 0]
    
    if not data_filtered:
        # 同样增大无数据时的标题字号
        ax.set_title(f"{title} (No Data > 0)", fontsize=16)
        return

    min_val = min(data_filtered)
    # 为了在对数坐标下柱子宽度一致，生成对数等距的 bins
    bins = np.logspace(np.log10(min_val), np.log10(global_max), 200)
    
    ax.hist(data_filtered, bins=bins, alpha=0.7, color=color, density=True, log=True)
    ax.set_xscale('log')
    
    # 修改这里：通过 fontsize 参数增大标题和 xy 轴标签的字号
    ax.set_title(title, fontsize=20)          # 标题字号设为 16
    ax.set_xlabel(xlabel, fontsize=18)        # X轴标签字号设为 14
    ax.set_ylabel('Log Density', fontsize=18) # Y轴标签字号设为 14
    
    # 补充：如果还需要增大坐标轴上的刻度数字(比如 10^1, 10^2)的字号，可以取消下面这行的注释
    ax.tick_params(axis='both', which='major', labelsize=16)
    
    # 强制设置 X 轴最大值为全局最大值
    ax.set_xlim(left=min_val, right=global_max)

# --- 第一行: LAION 数据集 ---
plot_log_hist(axes[0, 0], laion_texts, 'blue', 'LAION: Text Token Count', 'Text Tokens (Log)', max_text)
plot_log_hist(axes[0, 1], laion_images, 'blue', 'LAION: Image Token Count', 'Image Tokens (Log)', max_image)
plot_log_hist(axes[0, 2], laion_ratios, 'blue', 'LAION: Image/Text Ratio', 'Ratio (Log)', max_ratio)

# --- 第二行: ScienceQA 数据集 ---
plot_log_hist(axes[1, 0], sqa_texts, 'orange', 'ScienceQA: Text Token Count', 'Text Tokens (Log)', max_text)
plot_log_hist(axes[1, 1], sqa_images, 'orange', 'ScienceQA: Image Token Count', 'Image Tokens (Log)', max_image)
plot_log_hist(axes[1, 2], sqa_ratios, 'orange', 'ScienceQA: Image/Text Ratio', 'Ratio (Log)', max_ratio)

plt.tight_layout()
plt.savefig("token_distribution_2x3_logX.png", dpi=300)
print("图表已保存为 token_distribution_2x3_logX.png")
plt.show()