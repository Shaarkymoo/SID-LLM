from transformers import BertTokenizer, BertForSequenceClassification
import torch
from torch.utils.data import Dataset, DataLoader
import csv

# Define your dataset class
class CustomDataset(Dataset):
    def __init__(self, prompts, labels, tokenizer, max_len):
        self.prompts = prompts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        prompt = str(self.prompts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            prompt,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'prompt': prompt,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

# Load pre-trained model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2) # 2 classes: malicious or non-malicious

# Define your training parameters
MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 2e-5

# Define device
device = torch.device("cuda")
print(device)

# Move model to device
model.to(device)

#preparing dataset
filename = r'D:\college\Sem5\dsn2099\beavertails-main\train300.csv'
fields = []
train_prompts = []
train_labels = []
bene_count = 0
mali_count = 0
with open(filename,'r',encoding='utf8') as box:
    csvreader = csv.reader(box)
    fields = next(csvreader)

    for row in csvreader:
        if len(row)==0:
            continue
        if row[0] == 'benevolent':
            bene_count+=1
            if bene_count%2==0 and bene_count%3!=0 and bene_count%5!=0:
                train_prompts.append(row[1])
                train_labels.append(1)
        else:
            mali_count+=1
            if mali_count%2==0:
                train_prompts.append(row[1])
                train_labels.append(0)

print(len(train_prompts),len(train_labels))
print(mali_count,bene_count)
print(train_labels.count(0),train_labels.count(1))

# loading dataset
train_dataset = CustomDataset(train_prompts, train_labels, tokenizer, max_len=MAX_LEN)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

print(len(train_loader))

# Define optimizer and loss function
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
criterion = torch.nn.CrossEntropyLoss()

epoch_count = 0
batch_count = 0
# Training loop
for epoch in range(EPOCHS):
    model.train()
    batch_count=0
    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        batch_count+=1
        print("batch",batch_count,"of epoch",epoch_count," done")
    epoch_count+=1

# Evaluation (you need to implement validation dataset and evaluation loop)
# Testing (you need to implement test dataset and testing loop)
# Deployment (save the trained model and deploy it in your desired environment)

# Define the path where you want to save the model
MODEL_PATH = "maliciousOrNot_prompt_classifier.pt"

# Save the trained model
torch.save(model.state_dict(), MODEL_PATH)

# Now your model is ready for inference
# You can use it to classify new prompts as malicious or non-malicious

