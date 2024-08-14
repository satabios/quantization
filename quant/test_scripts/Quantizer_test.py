from models import SimpleCNN, VGG
import torch
from model_analyzer import ModelAnalyzer
from Quantizer import Quantizer
from fusion import fuse
from dataset import dataloader


# model = SimpleCNN()
# checkpoint = torch.load('vgg.cifar.pretrained.pth',weights_only=False)
# model.load_state_dict(checkpoint)
model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18').eval()
# model = SimpleCNN()
# model = VGG()
ma = ModelAnalyzer(model.train(),dataloader['test'])
mapped_layers = ma.mapped_layers

#Replace Layers

#Run through configs for each layer and compute the error!
for layer_name, layer_data in mapped_layers['calibiration_data'].items():

    # Weights, Activations
        # Per: Group, Channel, Tensor, etc..
        # Dtype: int8, fp8, etc..
        # Symmentric: True, False
    # Compute Error for each qlayer and get the least mse error of qlayer_op and layer_data['output']
    qlayer = Quantizer.from_float(module=eval(layer_name), act_quant='tensor',activations=layer_data['activations'])
    setattr(eval('.'.join(layer_name.split('.')[:-1])), layer_name.split('.')[-1], qlayer)


test = torch.rand(1, 3, 128, 128)
print(f"Quantized Model: {model}")
out = model(test)
# print(f"Output: {out}")
