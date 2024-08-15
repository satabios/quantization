from Fusion import Fuse
from dataset import dataloader
import torch
from models import SimpleCNN, VGG
from Chunker import Chunker
from tqdm import tqdm
import torch.nn.functional as F
from sconce import sconce
import torch.nn as nn
import torch.optim as optim

# resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18')
# mbnet = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2')


def evaluate_model(model, test_loader):
    # Set the model to evaluation mode
    model.eval()

    # Initialize counters for correct predictions and total samples
    correct = 0
    total = 0

    # Disable gradient computation since we are in inference mode
    with torch.no_grad():
        for data in tqdm(test_loader, desc="Evaluating", leave=False):
            inputs, labels = data
            outputs = model(inputs)

            # Get the predicted class (the one with the highest probability)
            _, predicted = torch.max(outputs.data, 1)

            # Update the total and correct counters
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # Calculate accuracy
    accuracy = 100 * correct / total

    return accuracy


vgg = VGG()
vgg.load_state_dict(torch.load("vgg.cifar.pretrained.pth"))
# smp = SimpleCNN()
# test_data = torch.rand(1, 3, 128, 128)


print(f"Original Model Accuracy : {evaluate_model(vgg, dataloader['test'])}")
fuser = Fuse(vgg.eval(), dataloader['test'])
fused_model = fuser.fused_model.train()
quantized_model = Chunker(fused_model, dataloader['test']).model
print(quantized_model)
print(f"Quantized Model Accuracy : {evaluate_model(quantized_model, dataloader['test'])}")
# Define all parameters

# from sconce import sconce
#
# sconces = sconce()
# sconces.model= quantized_model # Model Definition
# sconces.criterion = nn.CrossEntropyLoss() # Loss
# sconces.optimizer= optim.Adam(sconces.model.parameters(), lr=1e-4)
# sconces.scheduler = optim.lr_scheduler.CosineAnnealingLR(sconces.optimizer, T_max=200)
# sconces.dataloader = dataloader
# sconces.epochs = 5 #Number of time we iterate over the data
# print(sconces.evaluate())