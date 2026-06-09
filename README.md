
# BERT Sentiment Analysis with Hugging Face & PyTorch

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/🤗-Transformers-yellow.svg)](https://huggingface.co/docs/transformers)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Fine-tune **BERT-base-uncased** for binary sentiment classification using the Hugging Face Transformers library and PyTorch.

## 🎯 Overview

This project demonstrates how to:
- Load and fine-tune a pre-trained BERT model for sequence classification
- Tokenize text data using BERT's WordPiece tokenizer
- Train a sentiment classifier (Positive/Negative) with PyTorch
- Perform inference on new text samples

## 🧠 Model Architecture

```
Input Text → BERT Tokenizer → BERT-base-uncased (110M params) → Classification Head → Sentiment
                                    ↓
                            [CLS] token representation
                                    ↓
                          Dense Layer (768 → 2 classes)
                                    ↓
                              Softmax (Positive/Negative)
```

## 📊 Dataset

**Sample data used for demonstration:**
| Text | Sentiment |
|------|-----------|
| "I loved this movie, it was fantastic!" | Positive (1) |
| "The product broke after just two days of use." | Negative (0) |
| "This restaurant has amazing food and great service." | Positive (1) |
| "Terrible experience, would not recommend to anyone." | Negative (0) |
| "The book was okay, but not as good as I expected." | Negative (0) |

> Replace with your own dataset (e.g., IMDb, Twitter sentiment, product reviews)

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/bert-sentiment-analysis.git
cd bert-sentiment-analysis

# Install dependencies
pip install torch transformers scikit-learn pandas
```

### Run Training

```python
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
import torch

# Sample data
texts = ["I loved this movie!", "Terrible experience."]
labels = [1, 0]

# Tokenization
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
encodings = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")

# Model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
optimizer = AdamW(model.parameters(), lr=5e-5)

# Training loop (simplified)
model.train()
# ... (see full notebook for complete training code)
```

### Inference Example

```python
def predict_sentiment(text):
    encoding = tokenizer(text, padding=True, truncation=True, max_length=128, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**encoding)
    probs = torch.softmax(outputs.logits, dim=1)
    return "Positive" if torch.argmax(probs).item() == 1 else "Negative"

print(predict_sentiment("This was amazing!"))  # Positive
```

## 📈 Training Results

```
Epoch 1, Loss: 0.7197
Epoch 2, Loss: 0.6598
Epoch 3, Loss: 0.5701
Validation Accuracy: 1.00

Test: "This was an absolutely wonderful experience!"
Predicted sentiment: Positive ✅
```

## 🔧 Key Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Model | `bert-base-uncased` | 12-layer, 768-hidden, 12-heads (110M params) |
| Learning Rate | `5e-5` | Standard fine-tuning LR |
| Epochs | `3` | Sufficient for small datasets |
| Max Sequence Length | `128` | Truncates longer texts |
| Batch Size | `8` | Adjust based on GPU memory |
| Optimizer | `AdamW` | Weight decay variant of Adam |

## 📁 Repository Structure

```
bert-sentiment-analysis/
│
├── BERT.ipynb                # Complete Jupyter notebook
├── README.md                 # This file
├── requirements.txt          # Dependencies
└── LICENSE                   # MIT License
```

## 🛠️ Dependencies

```txt
torch>=2.0.0
transformers>=4.30.0
scikit-learn>=1.2.0
pandas>=1.5.0
```

## 💡 Key Concepts Explained

### BERT (Bidirectional Encoder Representations from Transformers)
- Pre-trained on massive text corpus (BooksCorpus + Wikipedia)
- Bidirectional context understanding (reads left-to-right AND right-to-left)
- Uses **Masked Language Modeling (MLM)** and **Next Sentence Prediction (NSP)** pre-training objectives

### Fine-tuning Process
1. **Load pre-trained BERT** with a classification head
2. **Tokenize** input text into subword tokens
3. **Add special tokens** (`[CLS]` for classification, `[SEP]` for separation)
4. **Train only the classification head** (or full model) on your task
5. **Use `[CLS]` token output** as sentence representation

### Why BERT for Sentiment?
- ✅ State-of-the-art performance on NLP tasks
- ✅ Contextual embeddings (unlike static Word2Vec/GloVe)
- ✅ Handles polysemy (words with multiple meanings)
- ✅ Pre-trained knowledge reduces need for large datasets

## 🚀 Real-World Use Cases

- **Product review analysis** (Amazon, Yelp, etc.)
- **Social media monitoring** (Twitter sentiment for brands)
- **Customer support ticket prioritization**
- **Movie/Book review classification**
- **Market sentiment analysis from news**

## 🔄 Improving the Model

| Improvement | How to Implement |
|-------------|------------------|
| Larger dataset | Load IMDb, SST-2, or Twitter datasets |
| Data augmentation | Back-translation, synonym replacement |
| Hyperparameter tuning | Grid search over LR, batch size, epochs |
| Cross-validation | K-Fold for robust evaluation |
| Class weights | Handle imbalanced datasets |
| Early stopping | Prevent overfitting |
| Learning rate scheduling | Warmup + linear decay |

## 📚 Resources

- [BERT Paper (Devlin et al., 2019)](https://arxiv.org/abs/1810.04805)
- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers)
- [Original BERT GitHub](https://github.com/google-research/bert)
- [The Illustrated BERT](http://jalammar.github.io/illustrated-bert/)

## 🤝 Contributing

Contributions welcome! Ideas:
- Add evaluation metrics (F1, Precision, Recall)
- Implement learning rate scheduler
- Add support for multi-class classification
- Create Streamlit/Gradio web app
- Add training curves visualization

## 📝 License

MIT License – see [LICENSE](LICENSE) file.

## 🙏 Acknowledgements

- Google Research for BERT
- Hugging Face for Transformers library
- PyTorch team

---

⭐ **Star this repo** if you found it helpful for learning BERT fine-tuning!
```

This README provides complete documentation suitable for GitHub, including setup instructions, code examples, explanations of BERT concepts, and improvement suggestions. Replace `yourusername` with your actual GitHub username before publishing.
