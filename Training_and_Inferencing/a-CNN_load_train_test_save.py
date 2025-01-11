import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

# Hyperparameters
batch_size = 32
learning_rate = 1e-4
num_epochs = 5

# Paths to datasets
train_csv = "./Training_and_Inferencing/train_dataset.csv"
test_csv = "./Training_and_Inferencing/test_dataset.csv"
model_save_path = "./Training_and_Inferencing/multiclass_model.pt"

# Set device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Dataset Class
class AudioDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.features = self.data.iloc[:, :-1].values  # All columns except the last
        self.labels = self.data.iloc[:, -1].astype('category').cat.codes.values  # Encode class labels as integers

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = torch.tensor(self.features[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y

# CNN Model
class MultiClassCNN(nn.Module):
    def __init__(self, input_feature_size, num_classes):
        super(MultiClassCNN, self).__init__()
        # First convolutional layer
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=1)  # Increased neurons
        # Second convolutional layer
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=1)  # Increased neurons
        # Third convolutional layer
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=1)  # New layer
        # Max pooling
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Dynamically calculate flattened size for the fully connected layer
        self.fc1_input_size = self._calculate_flattened_size(input_feature_size)
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.fc1_input_size, 128)  # Increased size of the fully connected layer
        self.fc2 = nn.Linear(128, num_classes)
        
        # Dropout layer
        self.dropout = nn.Dropout(0.5)
        # self.softmax = nn.Softmax(dim=1)

    def _calculate_flattened_size(self, input_feature_size):
        dummy_input = torch.zeros(1, 1, input_feature_size)  # (batch_size, channels, length)
        x = self.pool(torch.relu(self.conv1(dummy_input)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))  # Include the new layer
        flattened_size = x.numel()
        print(f"Calculated flattened size: {flattened_size}")
        return flattened_size

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension: (batch_size, 1, features)
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))  # Pass through the new layer
        x = x.view(x.size(0), -1)  # Flatten for fully connected layers
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)  # Apply Dropout
        x = self.fc2(x)
        # return self.softmax(x)
        return x


# Training Loop
def train_model(model, dataloader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    for features, labels in dataloader:
        features, labels = features.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(dataloader)

# Testing Loop
def test_model(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for features, labels in dataloader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            predictions = torch.argmax(outputs, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    accuracy = correct / total
    return accuracy

# Main Script
if __name__ == "__main__":
    # Load datasets
    train_dataset = AudioDataset(train_csv)
    test_dataset = AudioDataset(test_csv)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Retrieve the actual input feature size and number of classes
    input_feature_size = train_dataset[0][0].shape[0]
    num_classes = len(pd.unique(pd.read_csv(train_csv).iloc[:, -1]))
    print(f"Input feature size: {input_feature_size}, Number of classes: {num_classes}")

    # Initialize model, criterion, and optimizer
    model = MultiClassCNN(input_feature_size, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training and Testing
    for epoch in range(num_epochs):
        train_loss = train_model(model, train_loader, criterion, optimizer)
        test_accuracy = test_model(model, test_loader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {train_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

    # Save the trained model
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")
