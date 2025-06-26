#!/usr/bin/env python
import os
import json
import sys

def check_model_files(model_path=None):
    """Check model files without loading anything into memory"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    
    if model_path is None:
        model_path = os.path.join(project_root, "models", "medical_diagnosis_model", "best_model")
    
    print(f"Checking model files in: {model_path}")
    
    # Check if path exists
    if not os.path.exists(model_path):
        print(f"ERROR: Path {model_path} does not exist!")
        return False
    
    # List all files
    print("\nFiles in model directory:")
    for file in os.listdir(model_path):
        file_path = os.path.join(model_path, file)
        file_size = os.path.getsize(file_path) / (1024 * 1024)  # Convert to MB
        print(f"  - {file} ({file_size:.2f} MB)")
    
    # Check specific files
    adapter_config_path = os.path.join(model_path, "adapter_config.json")
    
    if os.path.exists(adapter_config_path):
        print("\nAdapter config exists. Checking content...")
        try:
            with open(adapter_config_path, 'r') as f:
                config = json.load(f)
                
            print("  Target modules:", config.get("target_modules", []))
            print("  Base model name:", config.get("base_model_name_or_path", "Not specified"))
            print("  Task type:", config.get("task_type", "Not specified"))
            print("  Inference mode:", config.get("inference_mode", "Not specified"))
            print("  r:", config.get("r", "Not specified"))
            print("  lora_alpha:", config.get("lora_alpha", "Not specified"))
            print("  lora_dropout:", config.get("lora_dropout", "Not specified"))
            
            # Check common issues
            if not config.get("target_modules"):
                print("WARNING: No target modules specified in adapter config")
            
            # Check for the target modules that might not exist in the model
            suspect_modules = ["c_attn", "c_proj"]
            found_suspicious = False
            for module in suspect_modules:
                if module in str(config.get("target_modules", [])):
                    found_suspicious = True
                    print(f"WARNING: Module '{module}' found in target_modules")
            
            if found_suspicious:
                print("\nPotential issue: The target modules may not match the base model architecture.")
                print("This could cause errors when loading the model or during generation.")
        except Exception as e:
            print(f"ERROR: Could not parse adapter config: {e}")
    else:
        print("ERROR: adapter_config.json not found!")
    
    # Check tokenizer files
    tokenizer_files = ["tokenizer_config.json", "vocab.json", "merges.txt"]
    missing_tokenizer_files = [f for f in tokenizer_files if not os.path.exists(os.path.join(model_path, f))]
    
    if missing_tokenizer_files:
        print(f"\nWARNING: Missing tokenizer files: {missing_tokenizer_files}")
        print("The model may need to fall back to the base model's tokenizer.")
    else:
        print("\nAll required tokenizer files are present.")
    
    return True

def main():
    """Main function to check model files"""
    model_path = None
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    
    check_model_files(model_path)
    
    print("\nAlso checking alternative models...")
    
    # Check the lora_adapter directory too
    lora_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                           "models", "lora_adapter", "best_model")
    if os.path.exists(lora_path):
        print("\n" + "="*60)
        print(f"Checking lora_adapter model: {lora_path}")
        print("="*60)
        check_model_files(lora_path)

if __name__ == "__main__":
    main()
