import torch
import os
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import time
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs", "interface.log"))
    ]
)
logger = logging.getLogger(__name__)

class MedicalDiagnosisAssistant:
    def __init__(self, model_path=None, base_model=None):
        """Initialize the medical diagnosis assistant with the specified model"""
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(current_dir)
            
            if model_path is None:
                model_path = os.path.join(project_root, "models", "medical_diagnosis_model", "best_model")
            
            if base_model is None:
                base_model = "gpt2"
            
            logger.info(f"Loading model from {model_path}")
            logger.info(f"Using base model: {base_model}")
            
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Using device: {self.device}")
            
            # Check if the model path exists first
            if not os.path.exists(model_path):
                logger.warning(f"Model path {model_path} not found. Using base model only.")
                
                # Load tokenizer and model directly
                self.tokenizer = AutoTokenizer.from_pretrained(base_model)
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                
                model_kwargs = {}
                if self.device == "cuda":
                    model_kwargs["torch_dtype"] = torch.float16
                
                self.model = AutoModelForCausalLM.from_pretrained(
                    base_model,
                    **model_kwargs
                )
            else:
                # Load tokenizer from the adapter path if possible
                try:
                    self.tokenizer = AutoTokenizer.from_pretrained(model_path)
                    logger.info("Loaded tokenizer from adapter path")
                except Exception as e:
                    logger.warning(f"Could not load tokenizer from adapter path: {e}")
                    self.tokenizer = AutoTokenizer.from_pretrained(base_model)
                    logger.info("Loaded tokenizer from base model")
                
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                
                # Load base model with appropriate configuration
                model_kwargs = {}
                if self.device == "cuda":
                    model_kwargs["torch_dtype"] = torch.float16
                
                # Load the base model with lower precision to reduce memory usage
                self.base_model = AutoModelForCausalLM.from_pretrained(
                    base_model,
                    low_cpu_mem_usage=True,  # Reduce memory usage
                    **model_kwargs
                )
                
            # Load the LoRA adapter with additional safety
            try:
                # Configure base model for better compatibility with LoRA
                if hasattr(self.base_model, "config") and hasattr(self.base_model.config, "pad_token_id"):
                    if self.base_model.config.pad_token_id is None:
                        self.base_model.config.pad_token_id = self.tokenizer.pad_token_id
                
                # Try loading the model with safe parameters
                self.model = PeftModel.from_pretrained(
                    self.base_model,
                    model_path,
                    is_trainable=False,
                    device_map="auto" if torch.cuda.is_available() else None,
                    torch_dtype=torch.float32  # Use full precision for stability
                )
                
                logger.info("LoRA adapter loaded successfully")
            except Exception as e:
                logger.error(f"Error loading LoRA adapter: {e}")
                logger.warning("Falling back to base model")
                self.model = self.base_model
            
            # Move model to device and set to evaluation mode
            self.model.to(self.device)
            self.model.eval()
            
            # Clear CUDA cache if using GPU
            if self.device == "cuda":
                torch.cuda.empty_cache()
                
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error initializing model: {str(e)}")
            raise

    def predict(self, symptoms, max_length=100, temperature=0.7, top_p=0.9, top_k=50):
        """Generate a medical diagnosis based on the provided symptoms"""
        try:
            start_time = time.time()
            
            # Sanitize input
            symptoms = symptoms.strip()
            if not symptoms:
                return "No symptoms provided. Please describe your symptoms."
            
            # Prepare the prompt with clear formatting
            input_text = f"Based on the following symptoms, provide a possible medical diagnosis:\n\nSymptoms: {symptoms}\n\nDiagnosis:"
            logger.info(f"Processing input: {input_text[:100]}...")
            
            # Tokenize with safety checks
            try:
                inputs = self.tokenizer(input_text, return_tensors="pt", truncation=True, 
                                       max_length=512)  # Add truncation with reasonable max length
                inputs = inputs.to(self.device)
            except Exception as e:
                logger.error(f"Tokenization error: {e}")
                return "Error processing your symptoms. Please try again with a clearer description."
            
            # Generate with much more robust error handling
            try:
                # First, check if we can generate without raising errors
                logger.info("Starting text generation")
                
                # Use minimal generation parameters to avoid issues
                generation_config = {
                    "input_ids": inputs["input_ids"],
                    "attention_mask": inputs["attention_mask"],
                    "max_new_tokens": 50,  # Start with very limited tokens
                    "temperature": 1.0,  # Default temperature
                    "do_sample": False,  # Deterministic generation is more stable
                    "pad_token_id": self.tokenizer.pad_token_id
                }
                
                # Try with minimal parameters first
                with torch.no_grad():
                    outputs = self.model.generate(**generation_config)
                
                logger.info("Basic generation succeeded, attempting with full parameters")
                
                # If basic generation succeeds, try with requested parameters
                if temperature < 1.0 or top_p < 1.0 or top_k < 50:
                    with torch.no_grad():
                        outputs = self.model.generate(
                            input_ids=inputs["input_ids"],
                            attention_mask=inputs["attention_mask"],
                            max_new_tokens=min(max_length, 100),  # Further limit tokens
                            temperature=temperature,
                            top_p=top_p,
                            top_k=top_k,
                            do_sample=True,  # Only use sampling if parameters require it
                            pad_token_id=self.tokenizer.pad_token_id,
                            num_beams=1
                        )
                
            except RuntimeError as e:
                error_msg = str(e)
                if "CUDA out of memory" in error_msg or "DefaultCPUAllocator: can't allocate memory" in error_msg:
                    logger.error(f"Memory error during generation: {e}")
                    # Clear cache if using GPU
                    if self.device == "cuda":
                        torch.cuda.empty_cache()
                    return "Memory error occurred. Please try a shorter symptom description."
                elif "bus error" in error_msg.lower() or "segmentation fault" in error_msg.lower():
                    logger.error(f"Critical memory access error: {e}")
                    return "A critical error occurred. The system may need to be restarted."
                else:
                    logger.error(f"Generation error: {e}")
                    return "Error generating diagnosis. Technical details: " + error_msg[:100]
            except Exception as e:
                logger.error(f"Unexpected error during generation: {e}")
                return f"Error generating diagnosis: {str(e)[:100]}"
            
            # Decode with error handling
            try:
                generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Extract diagnosis
                if "Diagnosis:" in generated_text:
                    diagnosis = generated_text.split("Diagnosis:")[-1].strip()
                    # Verify we have actual content
                    if diagnosis and len(diagnosis) > 0:
                        elapsed_time = time.time() - start_time
                        logger.info(f"Prediction generated in {elapsed_time:.2f} seconds")
                        return diagnosis
                    else:
                        return "Could not determine a specific diagnosis from the symptoms provided."
                else:
                    logger.warning("Generated text doesn't contain 'Diagnosis:' marker")
                    return generated_text.strip()  # Return whatever was generated
            except Exception as e:
                logger.error(f"Error decoding or extracting diagnosis: {e}")
                return "Error processing the diagnosis. Please try again."
            
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            return "Error in prediction. Please try again."

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Medical Diagnosis Assistant")
    parser.add_argument("--model_path", type=str, help="Path to the fine-tuned model")
    parser.add_argument("--base_model", type=str, help="Base model to use")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--max_length", type=int, default=100, help="Maximum length of generated text")
    return parser.parse_args()

def main():
    """Main function to run the medical diagnosis assistant"""
    try:
        args = parse_args()
        
        logs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
        os.makedirs(logs_dir, exist_ok=True)
        
        print("Loading medical diagnosis assistant...")
        assistant = MedicalDiagnosisAssistant(
            model_path=args.model_path,
            base_model=args.base_model
        )
        
        print("\n" + "="*60)
        print("MEDICAL DIAGNOSIS ASSISTANT")
        print("="*60)
        print("\nMEDICAL DISCLAIMER:")
        print("This is an AI assistant for educational purposes only.")
        print("Always consult with qualified healthcare professionals for medical advice.")
        print("This system is not a substitute for professional medical diagnosis.")
        print("="*60)
        
        while True:
            try:
                symptoms = input("\nDescribe your symptoms (or type 'quit' to exit): ")
                if symptoms.lower() in ['quit', 'exit', 'q']:
                    print("\nThank you for using the Medical Diagnosis Assistant.")
                    break
                
                if not symptoms.strip():
                    print("Please enter your symptoms.")
                    continue
                
                print("\nAnalyzing symptoms...")
                diagnosis = assistant.predict(
                    symptoms,
                    max_length=args.max_length,
                    temperature=args.temperature
                )
                
                print("\n" + "-"*60)
                print(f"POSSIBLE DIAGNOSIS: {diagnosis}")
                print("-"*60)
                print("\nREMINDER: This is an AI prediction. Please consult a healthcare professional.")
                
            except KeyboardInterrupt:
                print("\n\nExiting the program...")
                break
            except Exception as e:
                logger.error(f"Error in main loop: {str(e)}")
                print(f"\nAn error occurred: {str(e)}")
                print("Please try again or restart the program.")
    
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        print(f"Fatal error: {str(e)}")
        print("Please restart the program.")

if __name__ == "__main__":
    main()
