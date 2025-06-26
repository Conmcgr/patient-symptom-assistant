import os
import torch
import argparse
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs", "diagnostics.log"))
    ]
)
logger = logging.getLogger(__name__)

def diagnose_model(model_path=None, base_model="gpt2", skip_lora=False):
    """Run diagnostics on the model and adapter"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    
    if model_path is None:
        model_path = os.path.join(project_root, "models", "medical_diagnosis_model", "best_model")
    
    logger.info(f"Running diagnostics on {model_path}")
    logger.info(f"Base model: {base_model}")
    
    # Check model path existence
    if not os.path.exists(model_path):
        logger.error(f"Model path {model_path} does not exist!")
        return False

    # Check adapter files
    adapter_files = ["adapter_config.json", "adapter_model.safetensors"]
    missing_files = [f for f in adapter_files if not os.path.exists(os.path.join(model_path, f))]
    if missing_files:
        logger.error(f"Missing adapter files: {missing_files}")
        logger.info("Available files in adapter directory:")
        for file in os.listdir(model_path):
            logger.info(f"  - {file}")
        return False
    else:
        logger.info("All required adapter files found")
    
    # Try to load tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(base_model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        logger.info("Tokenizer loaded successfully")
    except Exception as e:
        logger.error(f"Error loading tokenizer: {e}")
        return False
    
    # Try to load base model
    try:
        base_model_obj = AutoModelForCausalLM.from_pretrained(
            base_model, 
            low_cpu_mem_usage=True,
            torch_dtype=torch.float32
        )
        logger.info(f"Base model loaded successfully with {sum(p.numel() for p in base_model_obj.parameters())} parameters")
        
        # Run a simple generation with base model
        test_input = "The patient has fever and headache"
        inputs = tokenizer(test_input, return_tensors="pt")
        with torch.no_grad():
            outputs = base_model_obj.generate(
                inputs["input_ids"],
                max_new_tokens=20,
                num_beams=1,
                do_sample=False
            )
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info(f"Base model test generation: {generated}")
        
    except Exception as e:
        logger.error(f"Error loading base model: {e}")
        return False
    
    # Skip LoRA testing if requested
    if skip_lora:
        logger.info("Skipping LoRA adapter testing as requested")
        return True
    
    # Try to load LoRA adapter
    try:
        logger.info("Attempting to load LoRA adapter")
        peft_model = PeftModel.from_pretrained(
            base_model_obj,
            model_path,
            is_trainable=False,
            torch_dtype=torch.float32
        )
        logger.info("LoRA adapter loaded successfully")
        
        # Test generation with LoRA model
        try:
            logger.info("Testing generation with LoRA model")
            test_input = "The patient has fever and headache"
            inputs = tokenizer(test_input, return_tensors="pt")
            with torch.no_grad():
                outputs = peft_model.generate(
                    inputs["input_ids"],
                    max_new_tokens=20,
                    num_beams=1,
                    do_sample=False
                )
            generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
            logger.info(f"LoRA model test generation: {generated}")
            return True
        except Exception as e:
            logger.error(f"Error during LoRA model generation: {e}")
            return False
        
    except Exception as e:
        logger.error(f"Error loading LoRA adapter: {e}")
        return False

def test_interface_without_lora(symptoms="I have a headache and fever"):
    """Test the interface without using LoRA to isolate issues"""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import time
        
        logger.info("Testing interface without LoRA")
        base_model = "gpt2"
        
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(base_model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(base_model, low_cpu_mem_usage=True)
        model.eval()
        
        # Prepare input
        input_text = f"Based on the following symptoms, provide a possible medical diagnosis:\n\nSymptoms: {symptoms}\n\nDiagnosis:"
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=128)
        
        # Generate
        start_time = time.time()
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=50,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )
        
        # Decode
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract diagnosis
        diagnosis = "No specific diagnosis provided"
        if "Diagnosis:" in generated_text:
            diagnosis = generated_text.split("Diagnosis:")[-1].strip()
        
        elapsed_time = time.time() - start_time
        logger.info(f"Base model test completed in {elapsed_time:.2f} seconds")
        logger.info(f"Input: {symptoms}")
        logger.info(f"Output: {diagnosis}")
        
        print("\n" + "="*60)
        print("BASE MODEL TEST RESULTS")
        print("="*60)
        print(f"Symptoms: {symptoms}")
        print(f"Diagnosis: {diagnosis}")
        print(f"Generation time: {elapsed_time:.2f} seconds")
        print("="*60)
        
        return True
    except Exception as e:
        logger.error(f"Error in test interface: {e}")
        print(f"Error testing interface: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Model diagnostics for medical diagnosis assistant")
    parser.add_argument("--model_path", type=str, help="Path to the LoRA adapter model")
    parser.add_argument("--base_model", type=str, default="gpt2", help="Base model name")
    parser.add_argument("--skip_lora", action="store_true", help="Skip LoRA adapter testing")
    parser.add_argument("--test_interface", action="store_true", help="Test interface without LoRA")
    parser.add_argument("--symptoms", type=str, default="I have a headache and fever", help="Symptoms to test with")
    
    args = parser.parse_args()
    
    logs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
    os.makedirs(logs_dir, exist_ok=True)
    
    print("Running model diagnostics...")
    
    if args.test_interface:
        test_interface_without_lora(args.symptoms)
    else:
        success = diagnose_model(args.model_path, args.base_model, args.skip_lora)
        
        if success:
            print("\nDiagnostics completed successfully!")
            print("Check the logs for detailed information.")
        else:
            print("\nDiagnostics found issues with the model.")
            print("Please check the logs for detailed error information.")
            print("\nRecommendations:")
            print("1. Try running with --skip_lora to test only the base model")
            print("2. Try running with --test_interface to test basic generation")
            print("3. Check that the adapter files are correctly formatted")
            print("4. Re-run training with fix for target modules")

if __name__ == "__main__":
    main()
