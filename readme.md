# Mental Health Multi-Label Classification

This project trains and deploys a multi-label RoBERTa-based classifier that detects mental health conditions in Reddit posts. Each post may be associated with multiple conditions such as depression, anxiety, PTSD, ADHD, and more.

---

## 🔍 Background

Mental health forums like Reddit contain rich, unstructured user-generated content where individuals often discuss symptoms or emotional states. This classifier aims to identify relevant mental health tags for each post, which can aid moderation, mental health research, or automated screening tools.

We fine-tune `roberta-base` from Hugging Face using a dataset of labeled Reddit posts. The model is trained to output multiple labels per post using binary cross-entropy loss.

---

## 📁 Project Structure

```
MentalHealthClassifier/
├── README.md
├── requirements.txt
├── data/
│   └── cleaned_paper.csv
├── models/
│   └── best_roberta_multilabel.pt
├── notebooks/
│   ├── train_model.ipynb
│   ├── evaluate_model.ipynb
│   └── demo_model.ipynb
└── src/
    ├── __init__.py
    ├── data_preprocessing.py
    ├── model.py
    └── utils.py
```

---

## 🚀 Quick Start

### 1. Clone and Install

```bash
git clone https://github.com/chillsahil/Mental-Health-RoBERTa-Model.git
cd Mental-Health-RoBERTa-Model
pip install -r requirements.txt
```

### 2. Download Resources

- Place `cleaned_paper.csv` and `best_roberta_multilabel.pt` in the appropriate directories from:
  [Google Drive Link](https://drive.google.com/drive/folders/1G5UkFdD6eAYwcWwtK2ebKjjFY6STjuoB)

---

## 🧪 Notebooks

### `demo_model.ipynb` ⭐

- Run instant predictions on example or user-inputted text
- No need to retrain the model — ideal for demos!

### `train_model.ipynb`

- Trains RoBERTa on the Reddit mental health dataset
- Saves best checkpoint

### `evaluate_model.ipynb`

- Evaluates the model on the entire dataset
- Reports overall and per-label accuracy

---

## 📦 Requirements

```text
pandas
numpy
scikit-learn
torch
transformers
emoji
nltk
swifter
```

Install with:

```bash
pip install -r requirements.txt
```

---

## 🧠 Supported Labels

- depression
- anxiety
- OCD
- PTSD
- autism
- eatingdisorders
- adhd
- bipolar
- schizophrenia

---

## 📬 Contact

**Author**: Sahil Prusti  
**Email**: prustisahil@gmail.com  
**LinkedIn**: [saprusti](https://www.linkedin.com/in/saprusti/)
