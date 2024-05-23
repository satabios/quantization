import torch
from model_analyzer import ModelAnalyzer
import torch.nn as nn
from activation_awareness import _search_module_scale, auto_clip_layer
import torch.optim as optim
from models import VGG
from dataset import dataloader
from sconce import sconce


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

# print("Prior Scaling:", sconces.evaluate())

ma = ModelAnalyzer(model)
mapped_layers = ma.mapped_layers
layer_list = [idx for idx, l_n in enumerate(mapped_layers['catcher']['name_list']) if isinstance(eval(l_n), (nn.Conv2d,nn.Linear))]
final_scales = []
with torch.no_grad():
  for idxr in range(1,len(layer_list)):
    idx_prev, idx_curr = layer_list[idxr-1], layer_list[idxr]

    #  Find Activation Scales
    x, w, y, module = mapped_layers['catcher']['x'][idx_curr], mapped_layers['catcher']['w'][idx_curr], \
      mapped_layers['catcher']['y'][idx_curr], eval(mapped_layers['catcher']['name_list'][idx_curr])
    scales = _search_module_scale(x, w, module, y)
    final_scales.append(scales)

    prev_module = eval(mapped_layers['catcher']['name_list'][idx_prev])

    #  Apply A-Scales
    if(prev_module.weight.data.dim()==4):
      prev_module.weight.data.div_(scales.view(-1, 1, 1, 1))
    elif(prev_module.weight.data.dim()==2):
      prev_module.weight.data.div_(scales.view(1, -1))
    mapped_layers['catcher']['w'][idx_prev] = prev_module.weight.data

#Apply Smoothing

# Find Clipping Range
    #Applying Clipping
    # best_max_val = auto_clip_layer( prev_w, prev_inp, n_bit=8, module=prev_module)


#Replace Layers and Real-Quantize

sconces.model= model
print("Post Scaling:", sconces.evaluate())

# final_scales.append(torch.zeros(1))
# for idx in range(len(final_scales)):
#   print(mapped_layers['catcher']['name_list'][idx],":",mapped_layers['catcher']['type_list'][idx][1],":",mapped_layers['catcher']['w'][idx].size(),":", final_scales[idx].shape)