from Fusion import Fuse
from dataset import dataloader
import torch
from models import SimpleCNN, VGG
from Chunker import Chunker
from tqdm import tqdm

from sconce import sconce
import torch.nn as nn
import torch.optim as optim

# resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18')
# mbnet = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2')
vgg = VGG()
vgg.load_state_dict(torch.load("vgg.cifar.pretrained.pth"))
# smp = SimpleCNN()
test_data = torch.rand(1, 3, 128, 128)


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

for model in tqdm([vgg]):#tqdm([resnet, mbnet, vgg, smp]):
    print(f"Original Model Accuracy : {evaluate_model(model, dataloader['test'])}")
    fuser = Fuse(model.eval(), dataloader['test'])
    fused_model = fuser.fused_model.train()
    quantized_model = Chunker(fused_model, dataloader['test']).model
    print(f"Quantized Model Accuracy : {evaluate_model(quantized_model, dataloader['test'])}")