import copy
from sconce import sconce
import torch.optim as optim
import torch.nn as nn
from Fusion import Fuse
from dataset import dataloader
import torch
from models import SimpleCNN, VGG
from Chunker import Chunker
from tqdm import tqdm

# original_model = VGG()
# original_model.load_state_dict(torch.load("vgg.cifar.pretrained.pth"))
# original_model.eval()
#
# ##### Custom Quant ##########
# custom_copy = copy.deepcopy(original_model)
#
# custom_quantized_model = Chunker(custom_copy, dataloader['test']).model.to('cpu')
#
# ##### torch quant ####
#
# from torch.quantization import get_default_qconfig, quantize
#
#
# torch_copy = copy.deepcopy(original_model)
# # Specify the quantization configuration to use - typical for x86 is QConfig with FBGEMM
# torch_copy.qconfig = get_default_qconfig('fbgemm')
#
# # Prepare the model for static quantization
# torch.quantization.prepare(torch_copy, inplace=True)
#
# for images, _ in dataloader['test']:
#     torch_copy(images)  # Run the model to calibrate it with the test data
#
# torch_quantized_model = torch.quantization.convert(torch_copy, inplace=True)
#
#
# sconces = sconce()
#
#
# for model in [torch_quantized_model]:
#
#     sconces.model = model.to('cpu')  # Model Definition
#     sconces.criterion = nn.CrossEntropyLoss()  # Loss
#     sconces.optimizer = optim.Adam(sconces.model.parameters(), lr=1e-4)
#     sconces.scheduler = optim.lr_scheduler.CosineAnnealingLR(sconces.optimizer, T_max=200)
#     sconces.dataloader = dataloader
#     sconces.epochs = 5  # Number of time we iterate over the data
#     # sconces.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # print( sconces.evaluate(device='cpu'))

resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18')
mbnet = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2')
vgg = VGG()
# vgg.load_state_dict(torch.load("vgg.cifar.pretrained.pth"))
smp = SimpleCNN()

for model in [mbnet, resnet, vgg, smp]:#tqdm([resnet, mbnet, vgg, smp]):
    test = torch.rand(1, 3, 128, 128)



    # sconces = sconce()
    # sconces.model = model  # Model Definition
    # sconces.criterion = nn.CrossEntropyLoss()  # Loss
    # sconces.optimizer = optim.Adam(sconces.model.parameters(), lr=1e-4)
    # sconces.scheduler = optim.lr_scheduler.CosineAnnealingLR(sconces.optimizer, T_max=200)
    # sconces.dataloader = dataloader
    # sconces.epochs = 5  # Number of time we iterate over the data
    # sconces.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(sconces.evaluate(device='cpu'))

    fuser = Fuse(model.eval(), dataloader['test'])
    fused_model = fuser.fused_model.train()

    quantized_model = Chunker(fused_model, dataloader['test']).model
    print(quantized_model)
    # sconces.model = quantized_model
    #
    # print(sconces.evaluate(device='cpu'))
