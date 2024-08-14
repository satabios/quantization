
from models import SimpleCNN, VGG
import torch

from model_analyzer import ModelAnalyzer
from chunker import Chunker
from fusion import fuse
from dataset import dataloader


model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)

    # torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
# checkpoint = torch.load('vgg.cifar.pretrained.pth',weights_only=False)
# model.load_state_dict(checkpoint)


ma = ModelAnalyzer(model.train(),dataloader['test'])
mapped_layers = ma.mapped_layers
print(mapped_layers['calibiration_data'].keys())
print(len(mapped_layers['calibiration_data'].keys()))
print(mapped_layers.keys())

from torchsummary import summary
summary(model.to('cuda'), input_size=(3, 224, 224))
