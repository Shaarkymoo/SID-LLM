import torch
import torch.nn as nn
from torchvision.models.detection import detr_resnet50
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.Resize((800, 800)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Create the dataset
multi_class_dataset = datasets.ImageFolder(root='path/to/multi_class/dataset', transform=transform)

# Split the dataset into training and validation sets
train_size = int(0.8 * len(multi_class_dataset))
val_size = len(multi_class_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(multi_class_dataset, [train_size, val_size])

# Create DataLoader
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

model = detr_resnet50(pretrained=True)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

# Load the previously trained binary model
model.load_state_dict(torch.load('detr_binary.pth'))

# Modify the final layer for multi-class classification
num_classes = 3  # car, bike, not_car_bike
model.class_embed = nn.Linear(model.class_embed.in_features, num_classes)
model.to(device)

# Redefine the optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-4)

num_epochs = 10
model.train()

for epoch in range(num_epochs):
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    epoch_loss = running_loss / len(train_loader)
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')

# Save the fine-tuned model
torch.save(model.state_dict(), 'detr_multi_class.pth')
