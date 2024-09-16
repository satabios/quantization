import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
from torch import optim
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
import os
from torch.utils.data import DataLoader
from torchvision.datasets import *
from torchvision.transforms import *
import math
import torch
import os

import ipdb

# Make torch deterministic
_ = torch.manual_seed(0)


image_size = 512
transforms = {
#     "train": Compose([
#         RandomCrop(image_size, padding=4),
#         RandomHorizontalFlip(),
#         ToTensor(),
#     ]),
#     "test": ToTensor(),
    'train': Compose([
                ToTensor(),
                Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize between -1 and 1
            ]),
    'test': Compose([
                ToTensor(),
                Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize between -1 and 1
            ])
}
dataset = {}
for split in ["train", "test"]:
    dataset[split] = CIFAR10(
        root="../../data/dataset/cifar10",
        train=(split == "train"),
        download=True,
        transform=transforms[split],
    )
dataloader = {}
for split in ['train', 'test']:
    dataloader[split] = DataLoader(
        dataset[split],
        batch_size=512,
        shuffle=(split == 'train'),
        num_workers=0,
        pin_memory=True,
        drop_last = True
    )


class QuantizedNet(nn.Module):
    def __init__(self, hidden_size_1=100, hidden_size_2=100):
        super(QuantizedNet,self).__init__()
        self.quant = torch.quantization.QuantStub()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 8 * 8, 512)  # After two maxpool layers, size is reduced to 8x8
        self.fc2 = nn.Linear(512, 10)  # CIFAR-10 has 10 classes
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)  # Flatten
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.dequant(x)
        return x

def evaluate_model(model, test_loader, device ='cpu'):
    model.eval()
    model.to(device)
    correct = 0
    total = 0
    with torch.no_grad():
        for data in tqdm(test_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return accuracy


device = 'cpu'
net_quantized = QuantizedNet().to(device)
net_quantized.load_state_dict(torch.load('../data/weights/best_model.pth', map_location=torch.device('cpu'), weights_only=True))

quantized_model = net_quantized

net_quantized.qconfig = torch.ao.quantization.default_qconfig
quantized_model = torch.ao.quantization.prepare(quantized_model) # Insert observers


evaluate_model(quantized_model,dataloader['test'], device ='cpu')


torch.backends.quantized.engine = 'qnnpack'
fquantized_model = torch.ao.quantization.convert(quantized_model)
print(fquantized_model)


