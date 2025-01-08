from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Define transformations
transform = transforms.Compose([
    transforms.Resize((800, 800)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Create the dataset
binary_dataset = datasets.ImageFolder(root='path/to/binary/dataset', transform=transform)

# Split the dataset into training and validation sets
train_size = int(0.8 * len(binary_dataset))
val_size = len(binary_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(binary_dataset, [train_size, val_size])

# Create DataLoader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

import torch
import torch.nn as nn
from torchvision.models.detection import detr_resnet50

# Load pre-trained DETR model
model = detr_resnet50(pretrained=True)

# Modify the final layer for binary classification
num_classes = 2  # 1 class (car) + background
model.class_embed = nn.Linear(model.class_embed.in_features, num_classes)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

# Define the optimizer and loss function
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.AdamW(params, lr=1e-5, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()


num_epochs = 10
model.train()

for epoch in range(num_epochs):
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    epoch_loss = running_loss / len(train_loader)
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')

# Save the trained model
torch.save(model.state_dict(), 'detr_binary.pth')
