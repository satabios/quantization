import torch
from models import VGG
import torch.nn as nn
from model_analyzer import ModelAnalyzer
from smooth_quant_cnn import W8A8


model = VGG()

ma = ModelAnalyzer(model)
mapped_layers = ma.mapped_layers


print(model)



#Replace Layers

for layer_name, layer_type in zip(mapped_layers['name_list'], mapped_layers['type_list']):
    if(layer_type=='Conv2d' or layer_type=='Linear'):
         setattr(eval('.'.join(layer_name.split('.')[:-1])),layer_name.split('.')[-1],W8A8.from_float(eval(layer_name)))

print(model)

test = torch.rand(512,3,32,32)#.dtype(torch.int8)
out = model(test)
print(out)
