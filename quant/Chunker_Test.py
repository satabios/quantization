from Fusion import Fuse
from dataset import dataloader
import torch
from models import SimpleCNN, VGG
from Chunker import Chunker
from tqdm import tqdm


resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
mbnet = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2')#, pretrained=True)
vgg = VGG()
smp = SimpleCNN()

for model in tqdm([resnet, mbnet, vgg, smp]):
    test = torch.rand(1, 3, 128, 128)

    fuser = Fuse(model.eval(), dataloader['test'])
    fused_model = fuser.fused_model.train()

    quantized_model = Chunker(fused_model, dataloader['test']).model