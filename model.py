import torch
import torch.nn as nn
import torch.nn.functional as F

# class Net(nn.Module):
#     """A simple fully connected network suitable for time-series EV battery data."""

#     def __init__(self, num_classes) -> None:
#         super(Net, self).__init__()
#         # Adjust input size to match the 8 features in each sample
#         self.fc1 = nn.Linear(8, 64)
#         self.fc2 = nn.Linear(64, 32)
#         self.fc3 = nn.Linear(32, 16)
#         self.fc4 = nn.Linear(16, 1)  # Output layer for binary classification

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = F.relu(self.fc3(x))
#         x = torch.sigmoid(self.fc4(x))  # Sigmoid for binary classification
#         return x.squeeze()  # Squeeze output to fit BCEWithLogitsLoss

import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    """A simple fully connected network suitable for time-series EV battery data with dropout for regularization."""

    def __init__(self, num_classes) -> None:
        super(Net, self).__init__()
        # Adjust input size to match the 8 features in each sample
        self.fc1 = nn.Linear(8, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 1)  # Output layer for binary classification
        
        # Dropout layers with a rate of 0.3
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = self.dropout(x)  # Apply dropout after first layer
        
        x = F.relu(self.fc2(x))
        x = self.dropout(x)  # Apply dropout after second layer
        
        x = F.relu(self.fc3(x))
        x = self.dropout(x)  # Apply dropout after third layer
        
        x = torch.sigmoid(self.fc4(x))  # Sigmoid for binary classification
        return x.squeeze()  # Squeeze output to fit BCEWithLogitsLoss

def train(net, trainloader, optimizer, epochs, device: str):
    """Train the network on the training set."""
    criterion = torch.nn.BCELoss()  # Binary Cross Entropy Loss
    net.train()
    net.to(device)
    for _ in range(epochs):
        for batch in trainloader:
            features, labels = batch['features'].to(device), batch['label'].to(device)
            optimizer.zero_grad()
            outputs = net(features)
            loss = criterion(outputs, labels.float())  # Ensure labels are float for BCE
            loss.backward()
            optimizer.step()

def test(net, testloader, device: str):
    """Validate the network on the entire test set and report loss and accuracy."""
    criterion = torch.nn.BCELoss()
    correct, loss = 0, 0.0
    net.eval()
    net.to(device)
    with torch.no_grad():
        for batch in testloader:
            features, labels = batch['features'].to(device), batch['label'].to(device)
            # print("DEBUG: BEFORE> features.shape, labels.shape", features.shape, labels.shape)
            
            outputs = net(features)

            # print("DEBUG: AFTER> outputs.shape, labels.shape", outputs.shape, labels.shape)
            loss += criterion(outputs, labels.float()).item()
            predicted = (outputs >= 0.5).float()  # Threshold to get binary predictions
            correct += (predicted == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    return loss, accuracy



# import torch 
# import torch.nn as nn
# import torch.nn.functional as F

# class Net(nn.Module):
#     """An improved fully connected network suitable for time-series EV battery data."""

#     def __init__(self, num_classes) -> None:
#         super(Net, self).__init__()
#         self.fc1 = nn.Linear(8, 128)  # Increased size
#         self.bn1 = nn.BatchNorm1d(128)  # Batch normalization
#         self.fc2 = nn.Linear(128, 64)
#         self.bn2 = nn.BatchNorm1d(64)  # Batch normalization
#         self.fc3 = nn.Linear(64, 32)
#         self.bn3 = nn.BatchNorm1d(32)  # Batch normalization
#         self.fc4 = nn.Linear(32, 16)
#         self.bn4 = nn.BatchNorm1d(16)  # Batch normalization
#         self.dropout = nn.Dropout(0.5)  # Dropout layer
#         self.fc5 = nn.Linear(16, 1)  # Output layer for binary classification

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = F.relu(self.bn1(self.fc1(x)))
#         x = self.dropout(x)  # Apply dropout
#         x = F.relu(self.bn2(self.fc2(x)))
#         x = self.dropout(x)  # Apply dropout
#         x = F.relu(self.bn3(self.fc3(x)))
#         x = self.dropout(x)  # Apply dropout
#         x = F.relu(self.bn4(self.fc4(x)))
#         x = torch.sigmoid(self.fc5(x))  # Sigmoid for binary classification
#         return x.squeeze()  # Squeeze output to fit BCEWithLogitsLoss

# def train(net, trainloader, optimizer, epochs, device: str):
#     """Train the network on the training set."""
#     criterion = torch.nn.BCELoss()  # Binary Cross Entropy Loss
#     net.train()
#     net.to(device)
#     for _ in range(epochs):
#         for batch in trainloader:
#             features, labels = batch['features'].to(device), batch['label'].to(device)
#             optimizer.zero_grad()
#             outputs = net(features)
#             loss = criterion(outputs, labels.float())  # Ensure labels are float for BCE
#             loss.backward()
#             optimizer.step()

# def test(net, testloader, device: str):
#     """Validate the network on the entire test set and report loss and accuracy."""
#     criterion = torch.nn.BCELoss()
#     correct, loss = 0, 0.0
#     net.eval()
#     net.to(device)
#     with torch.no_grad():
#         for batch in testloader:
#             features, labels = batch['features'].to(device), batch['label'].to(device)
#             outputs = net(features)
#             loss += criterion(outputs, labels.float()).item()
#             predicted = (outputs >= 0.5).float()  # Threshold to get binary predictions
#             correct += (predicted == labels).sum().item()
#     accuracy = correct / len(testloader.dataset)
#     return loss, accuracy
