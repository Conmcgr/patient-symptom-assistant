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
                base_model = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
            
            logger.info(f"Loading model from {model_path}")
            logger.info(f"Using base model: {base_model}")
            
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Using device: {self.device}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(base_model)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            model_kwargs = {}
            if self.device == "cuda":
                model_kwargs["torch_dtype"] = torch.float16
            
            self.base_model = AutoModelForCausalLM.from_pretrained(
                base_model,
                **model_kwargs
            )
            
            if os.path.exists(model_path):
                self.model = PeftModel.from_pretrained(
                    self.base_model,
                    model_path,
                    is_trainable=False
                )
                logger.info("LoRA adapter loaded successfully")
            else:
                logger.warning(f"Model path {model_path} not found. Using base model only.")
                self.model = self.base_model
            
            self.model.to(self.device)
            self.model.eval()
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error initializing model: {str(e)}")
            raise

    def predict(self, symptoms, max_length=100, temperature=0.7, top_p=0.9, top_k=50):
        """Generate a medical diagnosis based on the provided symptoms"""
        try:
            start_time = time.time()
            
            input_text = f"Based on the following symptoms, provide a possible medical diagnosis:\n\nSymptoms: {symptoms.strip()}\n\nDiagnosis:"
            
            inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_length,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    attention_mask=inputs.get("attention_mask", None)
                )
            
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            diagnosis = generated_text.split("Diagnosis:")[-1].strip()
            
            elapsed_time = time.time() - start_time
            logger.info(f"Prediction generated in {elapsed_time:.2f} seconds")
            
            return diagnosis
            
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
