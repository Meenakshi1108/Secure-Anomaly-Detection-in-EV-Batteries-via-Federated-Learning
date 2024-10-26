import torch
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import os
import random


class EVBatteryDataset(Dataset):
    """Custom Dataset for loading EV battery time-series data and meta-information."""

    def __init__(self, csv_file):
        # Read the CSV file
        self.data = pd.read_csv(csv_file)

        # Extract time-series features and labels
        self.features = self.data[[
            "volt", "current", "soc", "max_single_volt", "min_single_volt", 
            "max_temp", "min_temp", "timestamp"]].values
        
        self.labels = (self.data["label"].values==10)  # Label for each charging snippet

        # Optionally, you can also extract meta-information if needed
        # self.meta_info = self.data[["charge_segment", "mileage"]].values

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = {
            'features': torch.tensor(self.features[idx], dtype=torch.float32),
            'label': torch.tensor(self.labels[idx], dtype=torch.float32)
        }
        return sample


def prepare_dataset(data_dir: str, num_clients: int, batch_size: int, val_ratio: float = 0.1, test_ratio: float = 0.1):
    """Load EV dataset by iterating over CSV files and generate partitions for training, validation, and testing."""

    # Get all CSV files from the normal and abnormal directories
    normal_files = [os.path.join(data_dir, 'normal', file) for file in os.listdir(os.path.join(data_dir, 'normal')) if file.endswith('.csv')]
    abnormal_files = [os.path.join(data_dir, 'abnormal', file) for file in os.listdir(os.path.join(data_dir, 'abnormal')) if file.endswith('.csv')]

    # print(normal_files)
    # print(abnormal_files)

    # Randomize the selection within both normal and abnormal files
    random.shuffle(normal_files)
    random.shuffle(abnormal_files)

    # Split the clients: half from normal, half from abnormal
    num_normal_clients = num_clients // 2
    num_abnormal_clients = num_clients - num_normal_clients

    selected_normal_files = normal_files[:num_normal_clients]
    selected_abnormal_files = abnormal_files[:num_abnormal_clients]

    # Combine the selected normal and abnormal files
    selected_files = selected_normal_files + selected_abnormal_files

    trainloaders, valloaders = [], []
    test_subsets = []  # List to store each test subset
    
    for csv_file in selected_files:
        # Open each CSV file and load the dataset for the vehicle
        dataset = EVBatteryDataset(csv_file)
                
        num_total = len(dataset)
        
        # print("DEBUG: csv_file, num_total: ",csv_file,num_total)

        # Calculate sizes for training, validation, and testing
        num_test = int(test_ratio * num_total) #10
        num_val = int(val_ratio * (num_total - num_test)) #9
        num_train = num_total - num_test - num_val

        # Split dataset into training, validation, and testing sets
        train_indices, temp_indices = train_test_split(range(num_total), test_size=num_val + num_test, random_state=2023)
        val_indices, test_indices = train_test_split(temp_indices, test_size=num_test, random_state=2023)
        
        train_subset = torch.utils.data.Subset(dataset, train_indices)
        val_subset = torch.utils.data.Subset(dataset, val_indices)
        test_subset = torch.utils.data.Subset(dataset, test_indices)
        # print("DEBUG: len(train_subset): ",len(train_subset))
        # print("DEBUG: len(val_subset): ",len(val_subset))
        # print("DEBUG: len(test_subset): ",len(test_subset))
        # print()

        test_subsets.append(test_subset)

        # Create DataLoaders for training and validation
        trainloaders.append(DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=2))
        valloaders.append(DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=2))

        # print("DEBUG: len(train_subset): ",len(train_subset))
        # print("DEBUG: len(val_subset): ",len(val_subset))

        # The CSV file will automatically be closed once it's loaded into memory by pandas.
        # This ensures that the file is not kept open unnecessarily.
    
    # Create a single DataLoader for the combined test set
    test_dataset = torch.utils.data.ConcatDataset(test_subsets)
    # print("DEBUG: len(test_dataset): ",len(test_dataset))
    testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return trainloaders, valloaders, testloader
