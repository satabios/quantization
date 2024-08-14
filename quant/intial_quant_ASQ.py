import numpy as np
import torch
from ModelAnalyzer import ModelAnalyzer
import torch.nn as nn
from activation_awareness import _search_module_scale, auto_clip_layer, get_weight_scale, get_act_scale, smq_scale
import torch.optim as optim
from models import VGG
from dataset import dataloader
import gc
from sconce import sconce
from Chunker import Chunker


model = VGG()
model.load_state_dict(torch.load("vgg.cifar.pretrained.pth"))

sconces = sconce()
sconces.model= model # Model Definition
sconces.criterion = nn.CrossEntropyLoss() # Loss
sconces.optimizer= optim.Adam(sconces.model.parameters(), lr=1e-4)
sconces.scheduler = optim.lr_scheduler.CosineAnnealingLR(sconces.optimizer, T_max=200)
sconces.dataloader = dataloader
sconces.epochs = 5 #Number of time we iterate over the data
sconces.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Prior Scaling:", sconces.evaluate())

quantized_model = Chunker(sconces.model.to('cpu'), dataloader['test']).model
sconces.model= quantized_model.to('cpu')
print("Pre  AWQ-Quant:", sconces.evaluate(device='cpu'))

mapped_layers = ModelAnalyzer(sconces.model, sconces.dataloader['test']).mapped_layers

layer_list = [(idx,l_n) for idx, l_n in enumerate(mapped_layers['name_list'])
              if isinstance(eval(l_n), (nn.Conv2d,nn.Linear))]
# Activation Aware Scaling
final_scales = []
with (torch.no_grad()):
  for layer_name_idx in range(1,len(layer_list)):
    prev_layer_data , curr_layer_data = mapped_layers['calibiration_data'][layer_list[layer_name_idx-1][1]], \
                                        mapped_layers['calibiration_data'][layer_list[layer_name_idx][1]]

    #  Find Activation Scales
    x, w, y, module = curr_layer_data['activations'], curr_layer_data['weights'], \
      curr_layer_data['outputs'], curr_layer_data['layer']

    scales = _search_module_scale(x, w, module, y)
    final_scales.append(scales)

    prev_module = prev_layer_data['layer']

    #  Apply A-Scales
    if(prev_module.weight.data.dim()==4):
      prev_module.weight.data.div_(scales.view(-1, 1, 1, 1))
    elif(prev_module.weight.data.dim()==2):
      prev_module.weight.data.div_(scales.view(1, -1))

    #Update Dict Data
    curr_layer_data['weights'] = module.weight.data
    prev_layer_data['weights'] = prev_module.weight.data


sconces.model= model
print("Post  AWQ:", sconces.evaluate())

quantized_model = Chunker(sconces.model.to('cpu'), dataloader['test']).model
sconces.model= quantized_model.to('cpu')
print("Post  AWQ-Quant:", sconces.evaluate(device='cpu'))
#Run on calib data and record hooks
#Apply Smoothing



with torch.no_grad():
    for layer_name_idx in range(1, len(layer_list)):
        prev_layer_data, curr_layer_data = mapped_layers['calibiration_data'][layer_list[layer_name_idx - 1]], \
        mapped_layers['calibiration_data'][layer_list[layer_name_idx]]

        curr_layer, layer_inputs = curr_layer_data['layer'],curr_layer_data['activations']
        prev_layer = prev_layer_data['layer']

        act_scales = get_act_scale(layer_inputs)
        weight_scales = get_act_scale(curr_layer.weight.data)
        alpha = 0.5
        scales = (
            (act_scales.pow(alpha) / weight_scales.pow(1 - alpha))
            .clamp(min=1e-5)

        )
        # Multiply Current Weights with Scale along Input Dimension
        if(curr_layer.weight.data.dim()==4):
            curr_layer.weight.data.mul_(scales.view(1, -1, 1, 1))
        else:
            curr_layer.weight.data.mul_(scales.view(1, -1))

        # Multiply Current Weights with Scale along Input Dimension
        if(prev_layer.weight.data.dim()==4):
            prev_layer.weight.data.mul_(scales.view(-1, 1, 1, 1))
        else:
            prev_layer.weight.data.mul_(scales.view(-1, 1))

        # break
# Find Clipping Range
# best_max_val = auto_clip_layer( prev_w, prev_inp, n_bit=8, module=prev_module)
    #Applying Clipping

sconces.model= model
print("Post  Scaling:", sconces.evaluate())

#Replace Layers and Real-Quantize



# final_scales.append(torch.zeros(1))
# for idx in range(len(final_scales)):
#   print(mapped_layers['catcher']['name_list'][idx],":",mapped_layers['catcher']['type_list'][idx][1],":",mapped_layers['catcher']['w'][idx].size(),":", final_scales[idx].shape)


gc.collect()
torch.cuda.empty_cache()