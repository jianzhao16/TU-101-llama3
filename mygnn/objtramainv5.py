import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torchvision import transforms
import torch_geometric
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from torch_geometric.data import DataLoader, Data
from PIL import Image
import pandas as pd
import numpy as np
import os

import os

class ImageGraphDataset(Dataset):
    def __init__(self, input_image_files, ground_truth_files, tracking_info, transform=None):
        self.input_image_files = input_image_files
        self.ground_truth_files = ground_truth_files
        self.tracking_info = tracking_info
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.input_image_files)

    def __getitem__(self, idx):
        input_img_path = self.input_image_files[idx]
        ground_truth_path = self.ground_truth_files[idx]

        input_image = Image.open(input_img_path).convert("RGB")
        ground_truth = Image.open(ground_truth_path).convert("RGB")

        if self.transform:
            input_image = self.transform(input_image)
            ground_truth = self.transform(ground_truth)

        frame_info = self.tracking_info[self.tracking_info['frame_number'] == idx + 1]
        x = torch.tensor(frame_info[['object_id', 'object_position']].values, dtype=torch.float)
        edge_index = torch.tensor([[i, j] for i in range(len(frame_info)) for j in range(i+1, len(frame_info))], dtype=torch.long).t().contiguous()
        graph = Data(x=x, edge_index=edge_index)
        label = torch.tensor(frame_info['label'].values, dtype=torch.float)

        return input_image, ground_truth, graph, label

class SiameseGNN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SiameseGNN, self).__init__()
        self.convnet = nn.Sequential(
            nn.Conv2d(3, 64, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(64, 128, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Flatten()
        )
        self.gnn = GCNConv(in_channels, out_channels)
        self.fc = nn.Sequential(
            nn.Linear(128 * 53 * 53 + out_channels, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 2)
        )

    def forward_once(self, x, graph):
        cnn_features = self.convnet(x)
        gnn_features = self.gnn(graph.x, graph.edge_index)
        combined_features = torch.cat([cnn_features, gnn_features], dim=1)
        return self.fc(combined_features)

    def forward(self, x1, graph1, x2, graph2):
        output1 = self.forward_once(x1, graph1)
        output2 = self.forward_once(x2, graph2)
        return output1, output2

def contrastive_loss(output1, output2, label, margin=2.0):
    euclidean_distance = F.pairwise_distance(output1, output2)
    loss = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                      label * torch.pow(torch.clamp(margin - euclidean_distance, min=0.0), 2))
    return loss

def train(model, dataloader, optimizer, criterion, epochs=5):
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for input_image, ground_truth, graph, label in dataloader:
            optimizer.zero_grad()
            output1, output2 = model(input_image, graph, ground_truth, graph)
            loss = criterion(output1, output2, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch + 1}, Loss: {total_loss / len(dataloader)}')


def find_tif_files_in_directory(directory_path,file_ext):
    """
    Finds files ending with '.tif' in a directory.

    Args:
    directory_path (str): The path to the directory.

    Returns:
    List[str]: List of file names ending with '.tif'.
    """
    try:
        # List all files in the directory
        files = os.listdir(directory_path)

        # return relative path
        #tif_files = [file_name for file_name in files if file_name.endswith(file_ext)]

        # Return full path of the files
        #tif_files = [os.path.join(directory_path, file_name) for file_name in files if file_name.endswith('.tif')]
        tif_files = [os.path.abspath(os.path.join(directory_path, file_name)) for file_name in files if
                     file_name.endswith('.tif')]

        return tif_files

    except FileNotFoundError:
        print(f"Directory '{directory_path}' not found.")
        return []
    except PermissionError:
        print(f"No permission to access files in directory '{directory_path}'.")
        return []
    except Exception as e:
        print(f"An error occurred: {e}")
        return []


# Example usage:
file_ext='.tif'
input_image_dir='./01'
input_image_files=find_tif_files_in_directory(input_image_dir,file_ext)


# Example usage:
ground_truth_dir= './01_GT/TRA'
ground_truth_files=find_tif_files_in_directory(ground_truth_dir,file_ext)


man_tra_file='./01_GT/TRA/man_track.txt'
# Parsing the tracking information
tracking_info = pd.read_csv(man_tra_file, delim_whitespace=True, header=None, names=['frame_number', 'object_id', 'object_position', 'label'])

# Files

# Creating the dataset and dataloader
dataset = ImageGraphDataset(input_image_files, ground_truth_files, tracking_info)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# Model setup
model = SiameseGNN(in_channels=2, out_channels=64)
optimizer = optim.Adam(model.parameters(), lr=0.0005)
criterion = contrastive_loss

# Train
train(model, dataloader, optimizer, criterion)

