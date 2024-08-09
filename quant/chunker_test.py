
from models import SimpleCNN, VGG
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import *
from torchvision.transforms import *

from model_analyzer import ModelAnalyzer
from chunker import Chunker
from fusion import fuse
from dataset import dataloader


# model = VGG()
# checkpoint = torch.load('vgg.cifar.pretrained.pth',weights_only=False)
# model.load_state_dict(checkpoint)
model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18').eval()
# model = SimpleCNN()
ma = ModelAnalyzer(model.train(),dataloader['test'])
mapped_layers = ma.mapped_layers




#Replace Layers

for layer_name, layer_type in zip(mapped_layers['name_list'], mapped_layers['type_list']):
    if(layer_type=='Conv2d' or layer_type=='Linear'):

         setattr(eval('.'.join(layer_name.split('.')[:-1])),layer_name.split('.')[-1],Chunker.from_float(module=eval(layer_name), act_quant='tensor'))



# test = torch.randint(0,255,(512,3,32,32))#.type(torch.int8)

test = torch.rand(1, 3, 32, 32)
print(f"Quantized Model: {model}")
out = model(test)
print(f"Output: {out}")

#
# image_size = 32
# transforms = {
#     "train": Compose([
#         RandomCrop(image_size, padding=4),
#         RandomHorizontalFlip(),
#         ToTensor(),
#     ]),
#     "test": ToTensor(),
# }
# dataset = {}
# for split in ["train", "test"]:
#   dataset[split] = CIFAR10(
#     root="data/cifar10",
#     train=(split == "train"),
#     download=True,
#     transform=transforms[split],
#   )
# dataloader = {}
# for split in ['train', 'test']:
#   dataloader[split] = DataLoader(
#     dataset[split],
#     batch_size=512,
#     shuffle=(split == 'train'),
#     num_workers=0,
#     pin_memory=True,
#   )
