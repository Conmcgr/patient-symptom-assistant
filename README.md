# Patient Symptom Assistant

A medical diagnosis assistant powered by fine-tuned language models that provides possible diagnoses based on patient-reported symptoms.

## 📋 Overview

This project uses a medical-domain language model fine-tuned on symptom-to-diagnosis data to provide possible medical diagnoses based on natural language descriptions of symptoms. The system is designed for educational purposes and to assist healthcare professionals, not to replace professional medical advice.

## 🔍 Key Features

- **Medical-Domain Language Model**: Uses a biomedical language model fine-tuned specifically for symptom-to-diagnosis tasks
- **Natural Language Input**: Accepts symptoms described in natural language
- **Efficient Fine-Tuning**: Implements Parameter-Efficient Fine-Tuning (PEFT) with LoRA for resource-efficient training
- **Comprehensive Evaluation**: Includes metrics like BLEU and ROUGE for model performance assessment
- **User-Friendly Interface**: Simple command-line interface for interacting with the model

## 🛠️ Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/patient-symptom-assistant.git
   cd patient-symptom-assistant
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Usage

### Data Preprocessing

Preprocess the raw symptom-to-diagnosis data:

```bash
python src/preprocess.py
```

This will:

- Clean and format the raw data
- Split it into training, validation, and test sets
- Save the processed data in JSONL format

### Model Fine-Tuning

Fine-tune the model on the preprocessed data:

```bash
python src/finetune.py
```

Advanced options:

```bash
python src/finetune.py --base_model microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext --epochs 10 --batch_size 8 --learning_rate 2e-5 --use_8bit
```

### Using the Assistant

Run the medical diagnosis assistant:

```bash
python src/interface.py
```

Advanced options:

```bash
python src/interface.py --temperature 0.7 --max_length 100
```

## 📊 Model Architecture

- **Base Model**: Microsoft BiomedNLP-PubMedBERT (specialized for biomedical text)
- **Fine-Tuning Method**: Parameter-Efficient Fine-Tuning (PEFT) with Low-Rank Adaptation (LoRA)
- **Training Approach**: Instruction-tuning with symptom-to-diagnosis pairs

## 📈 Performance

The model is evaluated using:

- BLEU score for n-gram precision
- ROUGE scores for recall-oriented evaluation
- Manual evaluation of diagnosis quality

## 📁 Project Structure

```
patient-symptom-assistant/
├── data/
│   ├── raw/             # Raw symptom-to-diagnosis data
│   └── cleaned/         # Preprocessed data in JSONL format
├── models/              # Saved model checkpoints
│   └── medical_diagnosis_model/
│       ├── best_model/  # Best model checkpoint
│       └── final_model/ # Final model checkpoint
├── logs/                # Training and inference logs
├── src/
│   ├── preprocess.py    # Data preprocessing script
│   ├── finetune.py      # Model fine-tuning script
│   └── interface.py     # User interface for the assistant
├── requirements.txt     # Project dependencies
└── README.md            # Project documentation
```

## ⚠️ Medical Disclaimer

This system is intended for educational purposes only and is not a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- The Symptom2Disease dataset for providing the training data
- Hugging Face for their transformers library and model hosting
- The PEFT library for efficient fine-tuning techniques
