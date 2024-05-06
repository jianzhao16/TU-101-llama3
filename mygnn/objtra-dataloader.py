import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch_geometric
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from PIL import Image

class ImageGraphDataset(Dataset):
    def __init__(self, image_dir, graph_dir, transform=None):
        self.image_dir = image_dir
        self.graph_dir = graph_dir
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.image_files = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.tif')])
        self.graph_files = sorted([os.path.join(graph_dir, f) for f in os.listdir(graph_dir) if f.endswith('.pt')])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        graph_path = self.graph_files[idx]

        image = Image.open(img_path)
        graph = torch.load(graph_path)

        if self.transform:
            image = self.transform(image)

        return image, graph, torch.tensor([0])  # Placeholder label

# Instantiate the dataset
img_dir = "C:/Users/tus35240/Documents/GitHub/llama3/mygnn/ctcdata/Fluo-C2DL-Huh7-Training/Fluo-C2DL-Huh7/01"
graph_dir = "C:/Users/tus35240/Documents/GitHub/llama3/mygnn/ctcdata/Fluo-C2DL-Huh7-Training/Fluo-C2DL-Huh7/01graph"

dataset = ImageGraphDataset(img_dir, graph_dir)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
