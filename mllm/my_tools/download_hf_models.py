import os
import argparse
from transformers import CLIPVisionModel, AutoModelForCausalLM, AutoTokenizer, CLIPVisionConfig

def download_models(download_dir):
    """
    Downloads the CLIP Vision Model and Vicuna-7B (LLM) from Hugging Face
    and saves them to the specified local directory.
    """
    # Create target directories
    clip_dir = os.path.join(download_dir, "clip-vit-large-patch14-336")
    llm_dir = os.path.join(download_dir, "vicuna-7b-v1.5")
    
    os.makedirs(clip_dir, exist_ok=True)
    os.makedirs(llm_dir, exist_ok=True)

    print("="*50)
    print(f"Downloading CLIP model to {clip_dir}...")
    print("="*50)
    # Download and save CLIP
    clip_model_name = "openai/clip-vit-large-patch14-336"
    clip_model = CLIPVisionModel.from_pretrained(clip_model_name)
    clip_model.save_pretrained(clip_dir)
    # Also save the config
    clip_config = CLIPVisionConfig.from_pretrained(clip_model_name)
    clip_config.save_pretrained(clip_dir)
    print("CLIP model downloaded successfully!\n")

    print("="*50)
    print(f"Downloading LLM (Vicuna-7B) to {llm_dir}...")
    print("="*50)
    # Download and save LLM
    # Note: LLaVA-1.5 uses vicuna-7b-v1.5. You need HF access to LLaMA weights if required.
    llm_model_name = "lmsys/vicuna-7b-v1.5"
    
    try:
        # Download tokenizer
        tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
        tokenizer.save_pretrained(llm_dir)
        
        # Download model weights (this will take a while, it's ~13GB)
        # Using float16 to save RAM during download/save if possible
        llm_model = AutoModelForCausalLM.from_pretrained(
            llm_model_name, 
            torch_dtype="auto", 
            low_cpu_mem_usage=True
        )
        
        # Fix GenerationConfig validation error in transformers when saving
        if hasattr(llm_model, "generation_config"):
            llm_model.generation_config.do_sample = True
            
        llm_model.save_pretrained(llm_dir)
        print("LLM downloaded successfully!\n")
    except Exception as e:
        print(f"Error downloading LLM: {e}")
        print("Make sure you have enough disk space and proper Hugging Face permissions (if using explicit LLaMA models).")

    print("="*50)
    print("Download Complete!")
    print(f"CLIP Path: {clip_dir}")
    print(f"LLM Path : {llm_dir}")
    print("="*50)
    
    print("\nTo use the LLM checkpoint in your Megatron training script, run:")
    print(f"./run_vlm_train.sh /path/to/dataset {llm_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download LLaVA dependencies (CLIP and Vicuna).")
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default=os.path.expanduser("~/run/.local"), 
        help="The local directory to save the downloaded models."
    )
    args = parser.parse_args()
    
    download_models(args.output_dir)
