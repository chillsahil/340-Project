import torch
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

class RedditMentalHealthDataset(Dataset):
    def __init__(self, df, mlb, tokenizer, max_length=128):
        self.texts = df['clean_text'].tolist()
        self.labels = mlb.transform(df['labels'])
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        tokenized = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        return {
            'input_ids': tokenized['input_ids'].squeeze(0),
            'attention_mask': tokenized['attention_mask'].squeeze(0),
            'labels': torch.tensor(self.labels[idx], dtype=torch.float32)
        }

def evaluate_model(model, dataloader, device, mlb):
    model.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].cpu().numpy()

            outputs = model(input_ids, attention_mask)
            probs = torch.sigmoid(outputs).cpu().numpy()
            preds = (probs > 0.5).astype(int)

            all_preds.append(preds)
            all_true.append(labels)

    all_preds = np.vstack(all_preds)
    all_true = np.vstack(all_true)

    print(f"Overall subset-accuracy: {accuracy_score(all_true, all_preds):.4f}\n")
    print(classification_report(all_true, all_preds, target_names=mlb.classes_, digits=4))

    acc_per_label = {
        label: accuracy_score(all_true[:, i], all_preds[:, i])
        for i, label in enumerate(mlb.classes_)
    }
    print("\nPer-label accuracy:")
    for label, acc in acc_per_label.items():
        print(f"{label}: {acc:.4f}")
