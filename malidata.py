from transformers import BertTokenizer, BertForSequenceClassification
import torch
from torch.utils.data import Dataset, DataLoader
import csv

maliciousOrNot_file = "maliciousOrNot_prompt_classifier.pt"
maliciousType_file = "maliciousType_prompt_classifier.pt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


# Load the model for inference
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=7)
model.load_state_dict(torch.load(maliciousType_file))
model.to(device)  # Move the model to the device (GPU or CPU)
labels = ['chemical_biological', 'illegal', 'misinformation_disinformation', 'harmful', 'harassment_bullying', 'cybercrime_intrusion', 'copyright']

# Example of how to use the model for inference
def predict_prompt(prompt):
    model.eval()  # Set the model to evaluation mode

    # Tokenize the prompt
    inputs = tokenizer(prompt, return_tensors="pt", max_length=200, truncation=True, padding='max_length')
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    # Perform inference
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()

    # Map predicted class index to label
    predicted_label = labels[predicted_class]
    return predicted_label

dataset = "D:\college\Sem5\dsn2099\malicious datasets\sst_test_cases.csv"
promptlist = []
with open(dataset,'r',encoding='utf8') as box:
    csvreader = csv.reader(box)
    fields = next(csvreader)
    for row in csvreader:
        promptlist.append(row[4])

rows = []
for prompt in promptlist:
    predicted_label = predict_prompt(prompt)
    row = [predicted_label,prompt]
    rows.append(row)
    print(row)

writefile = "D:\college\Sem5\dsn2099\malicious datasets\malicious3.csv"
with open(writefile,'w',encoding='utf8') as box:
    csvwriter = csv.writer(box)
    csvwriter.writerows(rows)


