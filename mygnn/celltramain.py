import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
import os
from torch import nn, optim
import torch.nn.functional as F



from google.colab import drive
drive.mount('/content/drive')


class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.images = sorted([os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith('.tif')])
        self.labels = {idx: idx % 10 for idx in range(len(self.images))}  # Simulated labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        image = read_image(img_name)
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

transform = Compose([
    Resize((224, 224)),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])



class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.convnet = nn.Sequential(
            nn.Conv2d(3, 64, 5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(64, 128, 5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2)
        )

        self.fc = nn.Sequential(
            nn.Linear(128 * 53 * 53, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 2)
        )

    def forward_once(self, x):
        output = self.convnet(x)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2

def contrastive_loss(output1, output2, label, margin=2.0):
    euclidean_distance = F.pairwise_distance(output1, output2)
    loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                  (label) * torch.pow(torch.clamp(margin - euclidean_distance, min=0.0), 2))
    return loss_contrastive

model = SiameseNetwork().cuda()
optimizer = optim.Adam(model.parameters(), lr=0.0005)

def train(model, dataloader, optimizer, epochs=5):
    for epoch in range(epochs):
        for i, data in enumerate(dataloader, 0):
            img0, label0 = data[0].cuda(), data[1].cuda()
            img1, label1 = data[0].cuda(), data[1].cuda()  # Example: same images for now, adjust accordingly
            optimizer.zero_grad()
            output1, output2 = model(img0, img1)
            loss = contrastive_loss(output1, output2, label0 == label1)
            loss.backward()
            optimizer.step()
            if i % 10 == 0:
                print(f'Epoch number {epoch+1}, Training loss: {loss.item()}')


dataset = ImageDataset(root_dir='/content/drive/MyDrive/Colabdata/Fluo-C2DL-Huh7-Training/Fluo-C2DL-Huh7/01', transform=transform)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
train(model, dataloader, optimizer)
