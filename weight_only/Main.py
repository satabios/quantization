from Fusion import Fuse
import torch
from models import VGG
from Chunker import Chunker
from tqdm import tqdm
import torch.nn as nn
def evaluate_model(model, test_loader, device ='cuda'):
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


from dataset import Dataset
import torchvision
# model = torchvision.models.mobilenet_v3_large(weights='MobileNet_V3_Large_Weights.DEFAULT')
# torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
dataloader = Dataset('cifar10')
#
model = VGG()
model.load_state_dict(torch.load("/home/sathya/Desktop/Projects/quantization/weight_only/vgg.cifar.pretrained.pth"))
# print(f"Original Model Accuracy : {evaluate_model(model, dataloader,device='cuda')}")

quantized_model = Chunker(model, dataloader).model
print(f"Quantized Model Accuracy : {evaluate_model(quantized_model, dataloader)}")


