# README

This repository contains two Jupyter notebooks for training and evaluating a multi-label RoBERTa-based classifier on Reddit mental health data.

MESSAGE TO GRADER:

It would make more sense to run evaluate_metrics.ipynb because it is my whole code minus the model training, which took about 2.5 hours on my machine. 

I've supplied both notebooks however and you can read about each one down below. 
## Notebooks

### 1. `training.ipynb`

**Purpose:**
- Load and preprocess the dataset (`cleaned_paper.csv`)
- Train a RoBERTa-based multi-label classifier on Reddit posts
- Save the best checkpoint (`best_roberta_multilabel.pt`)
- Perform a final evaluation on the full dataset (subset-accuracy, per-label metrics)

**Usage:**
1. Ensure you have Python 3.8+ installed along with the required dependencies listed below.
2. Place `cleaned_paper.csv` in the notebook directory.
3. Open `training.ipynb` in JupyterLab or Jupyter Notebook.
4. Run all cells sequentially:
   - Installs and imports
   - Data loading and aggressive text cleaning
   - Multilabel encoding and tokenization
   - Dataset and DataLoader setup
   - Model architecture, training loop, and checkpoint saving
   - Final evaluation and metrics printing
5. After completion, you should see:
   - A saved checkpoint file: `best_roberta_multilabel.pt`
   - Printed metrics (subset-accuracy, classification report, per-label accuracy)


### 2. `evaluate_metrics.ipynb`

**Purpose:**
- Load a pre-trained model checkpoint (`best_roberta_multilabel.pt`)
- Preprocess the dataset (`cleaned_paper.csv`) with the same pipeline
- Evaluate the model on the full dataset or specified split
- Print overall subset-accuracy, per-label metrics, and classification report

**Usage:**
1. Ensure you have Python 3.8+ and the same dependencies installed.
2. Place both `cleaned_paper.csv` and `best_roberta_multilabel.pt` in the notebook directory.
3. Open `evaluate_metrics.ipynb`.
4. Run all cells sequentially:
   - Imports and environment setup
   - Data loading and text cleaning
   - Multilabel encoding and tokenization
   - Model instantiation and checkpoint loading
   - Inference loop collecting predictions
   - Printing subset-accuracy and per-label metrics
5. Review printed metrics to understand model performance.

## Requirements

Install the required Python packages:

```bash
pip install -r requirements.txt
```

A sample `requirements.txt` might include:

```
pandas
numpy
emoji
nltk
transformers
torch
scikit-learn
```

## Directory Structure

```
├── cleaned_paper.csv            # Preprocessed dataset CSV
├── training.ipynb               # Notebook to train the model
├── evaluate_metrics.ipynb       # Notebook to evaluate the trained model
├── best_roberta_multilabel.pt   # Saved checkpoint (after training)
└── requirements.txt             # Python dependencies
```

## Notes

- Ensure that `cleaned_paper.csv` and the checkpoint file are in the same directory as the notebooks.
- Use a GPU-enabled environment (CUDA) for faster training.
- Adjust hyperparameters in `training.ipynb` as needed (epochs, batch size, learning rate).

