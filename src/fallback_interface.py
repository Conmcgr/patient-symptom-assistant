import torch
import os
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs", "fallback.log"))
    ]
)
logger = logging.getLogger(__name__)

class FallbackMedicalAssistant:
    def __init__(self, base_model="gpt2"):
        """Initialize the fallback medical assistant with base model only"""
        try:
            logger.info(f"Loading base model: {base_model}")
            
            # Force CPU to avoid CUDA memory issues
            self.device = "cpu"
            logger.info(f"Using device: {self.device}")
            
            # Load tokenizer with safety checks
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(base_model)
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                logger.info("Tokenizer loaded successfully")
            except Exception as e:
                logger.error(f"Error loading tokenizer: {e}")
                raise
            
            # Load base model with minimal settings
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    base_model,
                    low_cpu_mem_usage=True,
                    torch_dtype=torch.float32,  # Use full precision for stability
                )
                self.model.eval()  # Set to evaluation mode
                logger.info("Model loaded successfully")
            except Exception as e:
                logger.error(f"Error loading model: {e}")
                raise
                
        except Exception as e:
            logger.error(f"Initialization error: {e}")
            raise

    def predict(self, symptoms, max_length=50):
        """Generate a medical diagnosis based on the provided symptoms"""
        try:
            start_time = time.time()
            
            # Process input
            symptoms = symptoms.strip()
            input_text = f"Based on the following symptoms, provide a possible medical diagnosis:\n\nSymptoms: {symptoms}\n\nDiagnosis:"
            
            logger.info(f"Processing input: {input_text[:100]}...")
            
            # Tokenize
            try:
                inputs = self.tokenizer(
                    input_text, 
                    return_tensors="pt", 
                    truncation=True, 
                    max_length=128
                )
            except Exception as e:
                logger.error(f"Tokenization error: {e}")
                return "Error processing input"
            
            # Generate text
            try:
                with torch.no_grad():
                    outputs = self.model.generate(
                        inputs["input_ids"],
                        max_new_tokens=max_length,
                        do_sample=False,  # Deterministic for stability
                        pad_token_id=self.tokenizer.pad_token_id
                    )
                
                # Decode
                generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Process result
                if "Diagnosis:" in generated_text:
                    diagnosis = generated_text.split("Diagnosis:")[-1].strip()
                else:
                    diagnosis = generated_text.replace(input_text, "").strip()
                
                elapsed_time = time.time() - start_time
                logger.info(f"Generated diagnosis in {elapsed_time:.2f} seconds")
                
                return diagnosis
                
            except Exception as e:
                logger.error(f"Generation error: {e}")
                return f"Error generating diagnosis: {str(e)}"
                
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return "System error during prediction"

def main():
    """Main function to run the fallback assistant"""
    parser = argparse.ArgumentParser(description="Fallback Medical Assistant")
    parser.add_argument("--base_model", type=str, default="gpt2", help="Base model to use")
    args = parser.parse_args()
    
    logs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
    os.makedirs(logs_dir, exist_ok=True)
    
    try:
        print("Loading fallback medical assistant...")
        assistant = FallbackMedicalAssistant(base_model=args.base_model)
        
        print("\n" + "="*60)
        print("FALLBACK MEDICAL ASSISTANT")
        print("="*60)
        print("\nMEDICAL DISCLAIMER:")
        print("This is an AI assistant for educational purposes only.")
        print("Always consult with qualified healthcare professionals for medical advice.")
        print("This system is not a substitute for professional medical diagnosis.")
        print("\nNOTE: This is running in FALLBACK MODE with the base model only.")
        print("Fine-tuned medical knowledge may be limited.")
        print("="*60)
        
        while True:
            try:
                symptoms = input("\nDescribe your symptoms (or type 'quit' to exit): ")
                if symptoms.lower() in ['quit', 'exit', 'q']:
                    print("\nThank you for using the Medical Assistant.")
                    break
                
                if not symptoms.strip():
                    print("Please enter your symptoms.")
                    continue
                
                print("\nAnalyzing symptoms...")
                diagnosis = assistant.predict(symptoms)
                
                print("\n" + "-"*60)
                print(f"POSSIBLE DIAGNOSIS: {diagnosis}")
                print("-"*60)
                print("\nREMINDER: This is an AI prediction. Please consult a healthcare professional.")
                
            except KeyboardInterrupt:
                print("\n\nExiting the program...")
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                print(f"\nAn error occurred: {e}")
                print("Please try again or restart the program.")
    
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        print(f"Fatal error: {e}")
        print("Please restart the program.")

if __name__ == "__main__":
    main()
