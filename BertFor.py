import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# 1. Load Data
df = pd.read_csv('final_events_with_categories.csv')

# 2. Convert tabular features to text
def features_to_text(row):
    text_parts = []
    for i in range(1, 6):  # Assuming you have event1. to event5. features
        event_features = [f"{col.split('.')[1]}:{row[col]}" 
                         for col in df.columns 
                         if col.startswith(f'event{i}.')]
        text_parts.append(f"Event {i}: " + ", ".join(event_features))
    return ". ".join(text_parts)

df['text'] = df.apply(features_to_text, axis=1)

# 3. Encode Labels
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['Category'])

# 4. Split Data
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# 5. Define Dataset Class (same as before)
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

# 6. Initialize Tokenizer & Model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=len(label_encoder.classes_)
)

# 7. Create DataLoaders
train_dataset = TextDataset(train_df['text'].values, train_df['label'].values, tokenizer)
test_dataset = TextDataset(test_df['text'].values, test_df['label'].values, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16)

# 7. Training Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
optimizer = AdamW(model.parameters(), lr=2e-5)  # Now using correct import

# 8. Training Loop
for epoch in range(3):  # 3 epochs
    model.train()
    for batch in tqdm(train_loader, desc=f'Epoch {epoch + 1}'):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

# 9. Evaluation
model.eval()
predictions, true_labels = [], []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1)
        predictions.extend(preds.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())

# 10. Print Classification Report
from sklearn.metrics import classification_report
print(classification_report(true_labels, predictions, target_names=label_encoder.classes_))