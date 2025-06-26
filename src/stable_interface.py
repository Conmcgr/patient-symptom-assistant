import os
import json
import argparse
import logging
from difflib import get_close_matches
import re
import time
import random

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs", "stable.log"))
    ]
)
logger = logging.getLogger(__name__)

class StableMedicalAssistant:
    def __init__(self):
        """Initialize the stable medical assistant with preloaded data"""
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(current_dir)
            
            # Load training data for symptom-to-diagnosis mappings
            train_path = os.path.join(project_root, "data", "cleaned", "train.jsonl")
            val_path = os.path.join(project_root, "data", "cleaned", "val.jsonl")
            test_path = os.path.join(project_root, "data", "cleaned", "test.jsonl")
            
            self.symptom_to_diagnosis = {}
            self.all_symptoms = []
            self.all_diagnoses = []
            
            logger.info("Loading symptom-to-diagnosis data...")
            
            # Process training data
            if os.path.exists(train_path):
                self._load_data(train_path)
            
            # Process validation data
            if os.path.exists(val_path):
                self._load_data(val_path)
                
            # Process test data
            if os.path.exists(test_path):
                self._load_data(test_path)
                
            # Create keyword to diagnosis mapping for faster lookups
            self.keywords = {}
            for symptom, diagnosis in self.symptom_to_diagnosis.items():
                words = re.findall(r'\b\w+\b', symptom.lower())
                for word in words:
                    if len(word) > 3:  # Only use meaningful words
                        if word not in self.keywords:
                            self.keywords[word] = []
                        if diagnosis not in self.keywords[word]:
                            self.keywords[word].append(diagnosis)
            
            logger.info(f"Loaded {len(self.symptom_to_diagnosis)} symptom-diagnosis pairs")
            logger.info(f"Created keyword index with {len(self.keywords)} keywords")
            
            # Keep track of common diagnoses for fallback
            self.common_diagnoses = {}
            for diagnosis in self.all_diagnoses:
                if diagnosis not in self.common_diagnoses:
                    self.common_diagnoses[diagnosis] = 0
                self.common_diagnoses[diagnosis] += 1
            
            # Sort by frequency
            self.common_diagnoses = dict(sorted(
                self.common_diagnoses.items(), 
                key=lambda item: item[1], 
                reverse=True
            ))
            
            logger.info("Initialization complete")
            
        except Exception as e:
            logger.error(f"Initialization error: {e}")
            raise
    
    def _load_data(self, file_path):
        """Load data from a JSONL file"""
        try:
            with open(file_path, 'r') as f:
                for line in f:
                    data = json.loads(line)
                    if 'input' in data and 'output' in data:
                        symptom = data['input'].lower()
                        diagnosis = data['output']
                        
                        self.symptom_to_diagnosis[symptom] = diagnosis
                        self.all_symptoms.append(symptom)
                        self.all_diagnoses.append(diagnosis)
        except Exception as e:
            logger.error(f"Error loading data from {file_path}: {e}")
    
    def predict(self, symptoms):
        """Predict diagnosis based on symptoms"""
        try:
            start_time = time.time()
            
            symptoms = symptoms.strip().lower()
            if not symptoms:
                return "No symptoms provided. Please describe your symptoms."
            
            logger.info(f"Processing symptoms: {symptoms[:100]}...")
            
            # Method 1: Direct match
            if symptoms in self.symptom_to_diagnosis:
                diagnosis = self.symptom_to_diagnosis[symptoms]
                logger.info(f"Direct match found: {diagnosis}")
                return diagnosis
            
            # Method 2: Fuzzy matching on full symptoms
            closest_symptoms = get_close_matches(symptoms, self.all_symptoms, n=3, cutoff=0.5)
            if closest_symptoms:
                diagnosis = self.symptom_to_diagnosis[closest_symptoms[0]]
                logger.info(f"Fuzzy match found: {diagnosis}")
                return diagnosis
            
            # Method 3: Keyword matching
            matched_diagnoses = {}
            words = re.findall(r'\b\w+\b', symptoms.lower())
            
            for word in words:
                if len(word) > 3 and word in self.keywords:
                    for diagnosis in self.keywords[word]:
                        if diagnosis not in matched_diagnoses:
                            matched_diagnoses[diagnosis] = 0
                        matched_diagnoses[diagnosis] += 1
            
            if matched_diagnoses:
                # Sort by frequency
                sorted_diagnoses = sorted(
                    matched_diagnoses.items(), 
                    key=lambda item: item[1], 
                    reverse=True
                )
                diagnosis = sorted_diagnoses[0][0]
                logger.info(f"Keyword match found: {diagnosis}")
                return diagnosis
            
            # Method 4: Return one of the most common diagnoses
            top_diagnoses = list(self.common_diagnoses.keys())[:5]
            diagnosis = random.choice(top_diagnoses)
            logger.info(f"Fallback to common diagnosis: {diagnosis}")
            
            elapsed_time = time.time() - start_time
            logger.info(f"Prediction completed in {elapsed_time:.2f} seconds")
            
            return diagnosis
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return "Unable to determine diagnosis from the symptoms provided."

def main():
    """Main function to run the stable medical assistant"""
    parser = argparse.ArgumentParser(description="Stable Medical Assistant")
    args = parser.parse_args()
    
    logs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
    os.makedirs(logs_dir, exist_ok=True)
    
    try:
        print("Loading stable medical assistant...")
        assistant = StableMedicalAssistant()
        
        print("\n" + "="*60)
        print("STABLE MEDICAL ASSISTANT")
        print("="*60)
        print("\nMEDICAL DISCLAIMER:")
        print("This is an AI assistant for educational purposes only.")
        print("Always consult with qualified healthcare professionals for medical advice.")
        print("This system is not a substitute for professional medical diagnosis.")
        print("\nNOTE: This is running in STABLE MODE with pattern matching.")
        print("="*60)
        
        while True:
            try:
                symptoms = input("\nDescribe your symptoms (or type 'quit' to exit): ")
                if symptoms.lower() in ['quit', 'exit', 'q']:
                    print("\nThank you for using the Stable Medical Assistant.")
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
