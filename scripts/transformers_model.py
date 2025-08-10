from torch import nn

class Transformer(nn.Module):
    def __init__(self, transformer, num_classes=4):
        super(Transformer, self).__init__()
        self.transformer = transformer
        self.fc = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, input_ids, attention_mask, numerical_features=None):
        transformer_output = self.transformer(input_ids=input_ids, attention_mask=attention_mask).pooler_output
        logits = self.fc(transformer_output)
        return logits