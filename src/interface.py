import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from peft import PeftModel
import os

class SymptomDiagnosisModel:
    def __init__(self, model_path="../models/lora_adapter/best_model"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.base_model = GPT2LMHeadModel.from_pretrained("gpt2").to(self.device)
        self.model = PeftModel.from_pretrained(self.base_model, model_path).to(self.device)
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.model.eval()

    def predict(self, symptoms, max_length=32):
        """
        Predict diagnosis based on symptoms
        """
        try:
            inputs = self.tokenizer(symptoms, return_tensors="pt", truncation=True, max_length=128)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=inputs["input_ids"],
                    max_length=max_length,
                    num_return_sequences=1,
                    no_repeat_ngram_size=2,
                    temperature=0.7
                )
            
            prediction = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return prediction.strip()
            
        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            return None

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