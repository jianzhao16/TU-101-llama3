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
        for x1, graph1, x2, graph2, label in dataloader:
            optimizer.zero_grad()
            output1, output2 = model(x1, graph1, x2, graph2)
            loss = criterion(output1, output2, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch + 1}, Loss: {total_loss / len(dataloader)}')


model = SiameseGNN(in_channels=128, out_channels=64)
optimizer = optim.Adam(model.parameters(), lr=0.0005)
criterion = contrastive_loss

train(model, dataloader, optimizer, criterion)
