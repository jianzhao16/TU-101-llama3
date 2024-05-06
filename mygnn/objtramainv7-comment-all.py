import torch  # Importing PyTorch
import torch.nn as nn  # Importing neural network modules
import torch.optim as optim  # Importing optimization algorithms
import torch.nn.functional as F  # Importing functional interface
from torch.utils.data import Dataset, DataLoader, random_split  # Importing PyTorch dataset utilities
from torch_geometric.data import Data  # Importing data structure for graph data
from torch_geometric.nn import GCNConv  # Importing graph convolutional network layer
from torch_geometric.loader import DataLoader as GeoDataLoader  # Importing PyTorch Geometric dataloader
from torchvision import transforms  # Importing image transformations
from PIL import Image  # Importing image processing from PIL
import pandas as pd  # Importing pandas for data handling
import numpy as np  # Importing numpy for numerical operations
import os  # Importing os for file operations
from sklearn.metrics import accuracy_score  # Importing accuracy metric


class ImageGraphDataset(Dataset):
    """
    Custom dataset that combines image and graph data.
    Each sample consists of an input image, a ground truth image, a graph, and a label.
    """
    def __init__(self, input_image_files, ground_truth_files, tracking_info, transform=None):
        self.input_image_files = input_image_files  # List of input image file paths
        self.ground_truth_files = ground_truth_files  # List of ground truth image file paths
        self.tracking_info = tracking_info  # Tracking information
        self.transform = transform or transforms.Compose([  # Image transformations
            transforms.Resize((224, 224)),  # Resize image to 224x224
            transforms.ToTensor(),  # Convert image to tensor
            transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),  # Repeat grayscale to RGB
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize image
        ])

    def __len__(self):
        return len(self.input_image_files)  # Return the number of samples

    def __getitem__(self, idx):
        """
        Returns a single data sample consisting of an input image, a ground truth image, a graph, and a label.
        """
        input_img_path = self.input_image_files[idx]  # Get input image path
        ground_truth_path = self.ground_truth_files[idx]  # Get ground truth image path
        input_image = Image.open(input_img_path).convert("RGB")  # Open input image
        ground_truth = Image.open(ground_truth_path).convert("RGB")  # Open ground truth image

        # Apply transformations
        if self.transform:
            input_image = self.transform(input_image)  # Transform input image
            ground_truth = self.transform(ground_truth)  # Transform ground truth image

        frame_info = self.tracking_info[self.tracking_info['frame_number'] == idx + 1]  # Get frame info
        x = torch.tensor(frame_info[['object_id', 'object_position']].values, dtype=torch.float)  # Convert to tensor
        edge_index = torch.tensor([[i, j] for i in range(len(frame_info)) for j in range(i + 1, len(frame_info))], dtype=torch.long).t().contiguous()  # Create edge index

        if edge_index.numel() == 0:  # if no edges, create a self-loop
            edge_index = torch.tensor([[0], [0]], dtype=torch.long)

        graph = Data(x=x, edge_index=edge_index)  # Create graph data object
        label = torch.tensor(frame_info['label'].values, dtype=torch.float)  # Create label tensor

        return input_image, ground_truth, graph, label  # Return the sample


class SiameseGNN(nn.Module):
    """
    Siamese network with a convolutional neural network for image data
    and a graph convolutional network for graph data.
    """
    def __init__(self, in_channels, out_channels):
        super(SiameseGNN, self).__init__()
        self.convnet = nn.Sequential(  # Sequential container for CNN layers
            nn.Conv2d(3, 64, 5),  # 2D convolution layer
            nn.ReLU(),  # ReLU activation
            nn.MaxPool2d(2, stride=2),  # 2D max pooling layer
            nn.Conv2d(64, 128, 5),  # Another 2D convolution layer
            nn.ReLU(),  # ReLU activation
            nn.MaxPool2d(2, stride=2),  # Another max pooling layer
            nn.Flatten()  # Flatten the tensor
        )
        self.gnn = GCNConv(in_channels, out_channels)  # Graph convolutional network layer
        self.fc = nn.Sequential(  # Sequential container for fully connected layers
            nn.Linear(128 * 53 * 53 + out_channels, 256),  # Fully connected layer
            nn.ReLU(),  # ReLU activation
            nn.Linear(256, 256),  # Another fully connected layer
            nn.ReLU(),  # ReLU activation
            nn.Linear(256, 2)  # Output layer
        )

    def forward_once(self, x, graph):
        """
        Forward pass for one branch of the Siamese network.
        """
        cnn_features = self.convnet(x)  # Extract features from the image using the CNN
        gnn_features = self.gnn(graph.x, graph.edge_index)  # Extract features from the graph using the GNN
        combined_features = torch.cat([cnn_features, gnn_features], dim=1)  # Concatenate features
        return self.fc(combined_features)  # Pass through fully connected layers

    def forward(self, x1, graph1, x2, graph2):
        """
        Forward pass for the Siamese network.
        """
        output1 = self.forward_once(x1, graph1)  # Forward pass for the first input
        output2 = self.forward_once(x2, graph2)  # Forward pass for the second input
        return output1, output2  # Return the outputs


def contrastive_loss(output1, output2, label, margin=2.0):
    """
    Computes the contrastive loss for a pair of outputs.
    """
    euclidean_distance = F.pairwise_distance(output1, output2)  # Calculate pairwise distance
    loss = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                      label * torch.pow(torch.clamp(margin - euclidean_distance, min=0.0), 2))  # Compute contrastive loss
    return loss


def train(model, dataloader, optimizer, criterion, epochs, threshold):
    """
    Trains the Siamese network using contrastive loss.
    """
    for epoch in range(epochs):
        model.train()  # Set the model to training mode
        total_loss = 0  # Initialize total loss
        for input_image, ground_truth, graph, label in dataloader:  # Iterate through the dataloader
            optimizer.zero_grad()  # Zero the gradients
            output1, output2 = model(input_image, graph, ground_truth, graph)  # Forward pass
            loss = criterion(output1, output2, label)  # Calculate loss
            loss.backward()  # Backpropagation
            optimizer.step()  # Update parameters
            total_loss += loss.item()  # Accumulate loss

        average_loss = total_loss / len(dataloader)  # Calculate average loss
        print(f'Epoch {epoch + 1}, Loss: {average_loss}')  # Print epoch and loss

        # Exit condition if loss goes below threshold
        if average_loss < threshold:
            print(f"Stopping early at epoch {epoch + 1} due to loss below threshold of {threshold}")
            break


def find_tif_files_in_directory(directory_path, file_ext='.tif'):
    """
    Finds files ending with specified file extension in a directory.

    Args:
    directory_path (str): The path to the directory.
    file_ext (str): The file extension to search for (default is '.tif').

    Returns:
    List[str]: List of file names ending with specified file extension.
    """
    try:
        files = os.listdir(directory_path)  # List all files in the directory
        tif_files = [os.path.abspath(os.path.join(directory_path, file_name)) for file_name in files if file_name.endswith(file_ext)]  # Find TIF files
        tif_files.sort()  # Sort the list of file names in ascending order
        return tif_files

    except FileNotFoundError:
        print(f"Directory '{directory_path}' not found.")  # Directory not found
        return []
    except PermissionError:
        print(f"No permission to access files in directory '{directory_path}'.")  # Permission error
        return []
    except Exception as e:
        print(f"An error occurred: {e}")  # General error
        return []


def split_dataset(dataset, split_ratio=0.8):
    """
    Splits the dataset into training and testing sets.
    """
    train_size = int(split_ratio * len(dataset))  # Calculate training set size
    test_size = len(dataset) - train_size  # Calculate testing set size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])  # Split dataset
    return train_dataset, test_dataset  # Return train and test datasets


def test(model, dataloader, criterion):
    """
    Evaluates the Siamese network on a test dataset.
    """
    model.eval()  # Set the model to evaluation mode
    total_loss = 0  # Initialize total loss
    all_outputs = []  # List to store outputs
    all_labels = []  # List to store labels
    with torch.no_grad():  # Disable gradient calculation
        for input_image, ground_truth, graph, label in dataloader:  # Iterate through the dataloader
            output1, output2 = model(input_image, graph, ground_truth, graph)  # Forward pass
            loss = criterion(output1, output2, label)  # Calculate loss
            total_loss += loss.item()  # Accumulate loss
            all_outputs.extend((output1 > 0.5).int().numpy())  # Store output
            all_labels.extend(label.numpy())  # Store label

    accuracy = accuracy_score(all_labels, all_outputs)  # Calculate accuracy
    average_loss = total_loss / len(dataloader)  # Calculate average loss
    return average_loss, accuracy  # Return loss and accuracy


# Example usage:
input_image_dir = './01'  # Input image directory
input_image_files = find_tif_files_in_directory(input_image_dir)  # Find input image files

ground_truth_dir = './01_GT/TRA'  # Ground truth directory
ground_truth_files = find_tif_files_in_directory(ground_truth_dir)  # Find ground truth files

man_tra_file = './01_GT/TRA/man_track.txt'  # Tracking file path
tracking_info = pd.read_csv(man_tra_file, delim_whitespace=True, header=None, names=['frame_number', 'object_id', 'object_position', 'label'])  # Load tracking info

# Creating the dataset and dataloader
dataset = ImageGraphDataset(input_image_files, ground_truth_files, tracking_info)  # Create dataset
dataloader = GeoDataLoader(dataset, batch_size=2, shuffle=True)  # Create dataloader

# Split the dataset
train_dataset, test_dataset = split_dataset(dataset, split_ratio=0.8)  # Split dataset
train_dataloader = GeoDataLoader(train_dataset, batch_size=2, shuffle=True)  # Create training dataloader
test_dataloader = GeoDataLoader(test_dataset, batch_size=2, shuffle=False)  # Create testing dataloader

# Model setup
model = SiameseGNN(in_channels=2, out_channels=64)  # Initialize model

# Optimizer and loss function
lr_rate = 0.0005  # Learning rate
optimizer = optim.Adam(model.parameters(), lr=lr_rate)  # Create Adam optimizer
criterion = contrastive_loss  # Set loss function

# Train
epochs = 1000  # Number of epochs
threshold = 1e-3  # Loss threshold
train(model, train_dataloader, optimizer, criterion, epochs, threshold)  # Train the model

# Test
test_loss, test_accuracy = test(model, test_dataloader, criterion)  # Test the model
print(f'Test Loss: {test_loss}, Test Accuracy: {test_accuracy}')  # Print test results
