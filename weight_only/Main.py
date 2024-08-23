from Fusion import Fuse
from dataset import dataloader
import torch
from models import VGG
from Chunker import Chunker
from tqdm import tqdm

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

vgg = VGG()
vgg.load_state_dict(torch.load("vgg.cifar.pretrained.pth"))
fuser = Fuse(vgg.eval(), dataloader['test'])
fused_model = fuser.fused_model.train()
print(f"Fused Model Accuracy : {evaluate_model(fused_model, dataloader['test'])}")
quantized_model = Chunker(fused_model, dataloader['test']).model
print(f"Quantized Model Accuracy : {evaluate_model(quantized_model, dataloader['test'])}")
