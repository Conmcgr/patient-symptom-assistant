import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from peft import PeftModel
import os

class SymptomDiagnosisModel:
    def __init__(self):
        # Get absolute path to model
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        model_path = os.path.join(project_root, "models", "lora_adapter", "best_model")
        
        # Force CPU and float32
        self.device = "cpu"
        torch.set_default_dtype(torch.float32)
        
        # Load tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load base model
        self.base_model = GPT2LMHeadModel.from_pretrained("gpt2")
        self.base_model.config.pad_token_id = self.base_model.config.eos_token_id
        
        # Load adapter
        self.model = PeftModel.from_pretrained(
            self.base_model,
            model_path,
            is_trainable=False
        )
        self.model.eval()

    def predict(self, symptoms):
        """Predict diagnosis based on symptoms"""
        try:
            # Format input
            input_text = f"Symptoms: {symptoms.strip()}\nDiagnosis: "
            input_ids = self.tokenizer.encode(input_text, return_tensors="pt")
            
            # Generate tokens one at a time
            max_new_tokens = 16
            generated_text = input_text
            
            for _ in range(max_new_tokens):
                    # Get model predictions
                    with torch.no_grad():
                        outputs = self.model(input_ids)
                        logits = outputs.logits[:, -1, :]
                        
                        # Apply softmax to get probabilities
                        probs = torch.nn.functional.softmax(logits, dim=-1)
                        
                        # Sample from top-k tokens
                        top_k = 5
                        top_k_probs, top_k_indices = torch.topk(probs, top_k)
                        # Ensure dimensions match
                        chosen_idx = top_k_indices[0][torch.multinomial(top_k_probs[0], 1).item()]
                    
                    # Convert to token and add to text
                    new_token = self.tokenizer.decode([chosen_idx])  # Convert to list
                    generated_text += new_token
                    
                    # Update input_ids for next iteration
                    chosen_idx_tensor = torch.tensor([chosen_idx], dtype=torch.long).unsqueeze(0)  # Make it [1,1]
                    input_ids = torch.cat([input_ids, chosen_idx], dim=1)
                    
                    # Stop if we generate a newline or period
                    if new_token in ["\n", "."]:
                        break
            
            # Extract diagnosis
            try:
                diagnosis = generated_text.split("Diagnosis: ")[-1].strip()
                if not diagnosis:
                    return "No diagnosis generated"
                return diagnosis
            except:
                return generated_text
                
        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            return "Error in prediction"

def main():
    print("Loading model...")
    model = SymptomDiagnosisModel()
    
    print("\nMEDICAL DISCLAIMER:")
    print("This is an AI assistant for educational purposes only.")
    print("Always consult with qualified healthcare professionals for medical advice.")
    
    while True:
        symptoms = input("\nDescribe your symptoms (or type 'quit' to exit): ")
        if symptoms.lower() == 'quit':
            break
            
        print("\nAnalyzing symptoms...")
        diagnosis = model.predict(symptoms)
        
        if diagnosis:
            print(f"\nPossible condition: {diagnosis}")
            print("\nREMINDER: This is an AI prediction. Please consult a healthcare professional.")
        else:
            print("\nUnable to make a prediction. Please try rephrasing or consult a healthcare professional.")

if __name__ == "__main__":
    main()
