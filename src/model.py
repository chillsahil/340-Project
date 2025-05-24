import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

class RoBERTaMultiLabelClassifier(nn.Module):
    def __init__(self, num_labels, dropout_rate=0.2):
        super().__init__()
        config = AutoConfig.from_pretrained("roberta-base")
        self.roberta = AutoModel.from_pretrained("roberta-base", config=config)
        for name, param in self.roberta.named_parameters():
            if "encoder.layer" in name and int(name.split(".")[2]) < 6:
                param.requires_grad = False
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        return self.classifier(self.dropout(cls_output))
