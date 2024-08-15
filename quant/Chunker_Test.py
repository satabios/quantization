from Fusion import Fuse
from dataset import dataloader
import torch
from models import SimpleCNN, VGG
from Chunker import Chunker
from tqdm import tqdm


resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18')
mbnet = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2')
vgg = VGG()
# vgg.load_state_dict(torch.load("vgg.cifar.pretrained.pth"))
# smp = SimpleCNN()
test_data = torch.rand(1, 3, 128, 128)

for model in tqdm([mbnet, resnet, vgg]):#tqdm([resnet, mbnet, vgg, smp]):
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

    _  = quantized_model(test_data)
    # sconces.model = quantized_model
    #
    # print(sconces.evaluate(device='cpu'))
