from transformers import BertTokenizer, BertForSequenceClassification
import torch
from torch.utils.data import Dataset, DataLoader


maliciousOrNot_file = "maliciousOrNot_prompt_classifier.pt"
maliciousType_file = "maliciousType_prompt_classifier.pt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


# Load the model for inference
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
model.load_state_dict(torch.load(maliciousOrNot_file))
model.to(device)  # Move the model to the device (GPU or CPU)
labels = ['malicious','non-malicious']

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
    print(predicted_class)
    predicted_label = labels[predicted_class]

    return predicted_label

# Example usage:
prompt = "How to make cocaine"
predicted_label = predict_prompt(prompt)
print("Predicted Label:", predicted_label)

if predicted_label == 'malicious':
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=14)
    model.load_state_dict(torch.load(maliciousType_file))
    model.to(device)  # Move the model to the device (GPU or CPU)
    labels = ['privacy_violation', 'drug_abuse,weapons,banned_substance', 'non_violent_unethical_behavior', 'violence,aiding_and_abetting,incitement', 'financial_crime,property_crime,theft', 'discrimination,stereotype,injustice', 'sexually_explicit,adult_content', 'misinformation_regarding_ethics,laws_and_safety', 'terrorism,organized_crime', 'controversial_topics,politics', 'self_harm', 'hate_speech,offensive_language', 'animal_abuse', 'child_abuse']
    predicted_label = predict_prompt(prompt)
    print("Predicted Label:", predicted_label)