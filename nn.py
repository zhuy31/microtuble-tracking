import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
import json

# Define the dataset class
class MicrotubuleDataset(Dataset):
    def __init__(self, image_dir, label_file, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_files = sorted(os.listdir(image_dir))
        with open(label_file, 'r') as f:
            self.labels = json.load(f)
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(self.labels[self.image_files[idx]], dtype=torch.float32)
        return image, label

# Define the neural network
class MicrotubuleNet(nn.Module):
    def __init__(self):
        super(MicrotubuleNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=16, stride=1, padding=7)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=16, stride=1, padding=7)
        
        self.rnn1 = nn.LSTM(16 * 16 * 16, 128, batch_first=True)
        self.rnn2 = nn.LSTM(128, 128, batch_first=True)
        
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 2)  # Assuming 2D coordinates
        
    def forward(self, x):
        batch_size, seq_length, c, h, w = x.size()
        x = x.view(batch_size * seq_length, c, h, w)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        
        x = x.view(batch_size, seq_length, -1)
        
        x, _ = self.rnn1(x)
        x, _ = self.rnn2(x)
        
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x

# Define the training function
def train(model, dataloader, criterion, optimizer, num_epochs=25):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in dataloader:
            inputs, labels = inputs.cuda(), labels.cuda()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(dataloader)}')

# Define the evaluation function
def evaluate(model, dataloader, criterion):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.cuda(), labels.cuda()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
    print(f'Loss: {total_loss/len(dataloader)}')

# Placeholder directories and label file
train_dir = 'path/to/train/images'
train_label_file = 'path/to/train/labels.json'
test_dir = 'path/to/test/images'
test_label_file = 'path/to/test/labels.json'

# Define the data loaders
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

train_dataset = MicrotubuleDataset(train_dir, train_label_file, transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

test_dataset = MicrotubuleDataset(test_dir, test_label_file, transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Initialize the model, loss function, and optimizer
model = MicrotubuleNet().cuda()
criterion = nn.MSELoss()  # Using MSELoss for coordinate regression
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
train(model, train_loader, criterion, optimizer, num_epochs=25)

# Evaluate the model
evaluate(model, test_loader, criterion)

