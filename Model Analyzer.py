#!/usr/bin/env python
# coding: utf-8

# In[1]:


import timm
import torch
import torch.nn as nn
from collections import OrderedDict
import re
import numpy as np
from torchsummary import summary
from collections import defaultdict, OrderedDict
import ipdb
import torch.nn.functional as F
from collections import defaultdict, OrderedDict

import numpy as np
import torch
from torch import nn
from torch.optim import *
from torch.optim.lr_scheduler import *
from torch.utils.data import DataLoader
from torchvision.datasets import *
from torchvision.transforms import *
import torch.optim as optim


from collections import defaultdict, OrderedDict

import copy
import numpy as np
import torch
from torch import nn
from torch.optim import *
from torch.optim.lr_scheduler import *
from torch.utils.data import DataLoader
from torchvision.datasets import *
from torchvision.transforms import *
import torch.optim as optim

from collections import defaultdict, OrderedDict

import copy
import numpy as np
import torch
from torch import nn
from torch.optim import *
from torch.optim.lr_scheduler import *
from torch.utils.data import DataLoader
from torchvision.datasets import *
from torchvision.transforms import *
import torch.optim as optim


get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('matplotlib', 'notebook')




# In[ ]:


image_size = 32
transforms = {
    "train": Compose([
        RandomCrop(image_size, padding=4),
        RandomHorizontalFlip(),
        ToTensor(),
    ]),
    "test": ToTensor(),
}
dataset = {}
for split in ["train", "test"]:
  dataset[split] = CIFAR10(
    root="data/cifar10",
    train=(split == "train"),
    download=True,
    transform=transforms[split],
  )
dataloader = {}
for split in ['train', 'test']:
  dataloader[split] = DataLoader(
    dataset[split],
    batch_size=512,
    shuffle=(split == 'train'),
    num_workers=0,
    pin_memory=True,
  )



class VGG(nn.Module):
  ARCH = [64, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']

  def __init__(self) -> None:
      super().__init__()

      layers = []
      counts = defaultdict(int)

      def add(name: str, layer: nn.Module) -> None:
          layers.append((f"{name}{counts[name]}", layer))
          counts[name] += 1

      in_channels = 3
      for x in self.ARCH:
          if x != 'M':
              # conv-bn-relu
              add("conv", nn.Conv2d(in_channels, x, 3, padding=1, bias=False))
              add("bn", nn.BatchNorm2d(x))
              add("relu", nn.ReLU(True))
              in_channels = x
          else:
              # maxpool
              add("pool", nn.MaxPool2d(2))

      self.backbone = nn.Sequential(OrderedDict(layers))
      self.classifier = nn.Linear(512, 10)

  def forward(self, x: torch.Tensor) -> torch.Tensor:

      # backbone: [N, 3, 32, 32] => [N, 512, 2, 2]
      x = self.backbone(x)

      # avgpool: [N, 512, 2, 2] => [N, 512]
      x = x.mean([2, 3])

      # classifier: [N, 512] => [N, 10]
      x = self.classifier(x)
      return x


# In[ ]:


class VGG(nn.Module):
  ARCH = [64, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']

  def __init__(self) -> None:
    super().__init__()

    layers = []
    counts = defaultdict(int)

    def add(name: str, layer: nn.Module) -> None:
      layers.append((f"{name}{counts[name]}", layer))
      counts[name] += 1

    in_channels = 3
    for x in self.ARCH:
      if x != 'M':
        # conv-bn-relu
        add("conv", nn.Conv2d(in_channels, x, 3, padding=1, bias=False))
        add("bn", nn.BatchNorm2d(x))
        add("relu", nn.ReLU(True))
        in_channels = x
      else:
        # maxpool
        add("pool", nn.MaxPool2d(2))

    self.backbone = nn.Sequential(OrderedDict(layers))
    self.classifier = nn.Linear(512, 10)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    # backbone: [N, 3, 32, 32] => [N, 512, 2, 2]
    x = self.backbone(x)

    # avgpool: [N, 512, 2, 2] => [N, 512]
    x = x.mean([2, 3])

    # classifier: [N, 512] => [N, 10]
    x = self.classifier(x)
    return x




# In[ ]:





# In[ ]:


import copy

class ModelAnalyzer:
    def __init__(self, model):
        self.model = model
        self.mapped_layers = self.layer_mapping(model)
 

    
    def name_fixer(self, names):
        """
        Fix the names by removing the indices in square brackets.
        Args:
        names (list): List of names.

        Returns:
        list: List of fixed names.
        """
        return_list = []
        for string in names:
            matches = re.finditer(r'\.\[(\d+)\]', string)
            pop_list = [m.start(0) for m in matches]
            pop_list.sort(reverse=True)
            if len(pop_list) > 0:
                string = list(string)
                for pop_id in pop_list:
                    string.pop(pop_id)
                string = ''.join(string)
            return_list.append(string)
        return return_list


    def get_all_layers(self, model, parent_name=''):
        layers = []
        for name, module in model.named_children():
            full_name = f"{parent_name}.{name}" if parent_name else name
            test_name = "model." + full_name
            try:
                eval(test_name)
                layers.append((full_name, module))
            except:
                layers.append((self.reformat_layer_name(full_name), module))
            if isinstance(module, nn.Module):
                layers.extend(self.get_all_layers(module, parent_name=full_name))
        return layers


    def reformat_layer_name(self, str_data):
        try:
            split_data = str_data.split('.')
            for ind in range(len(split_data)):
                data = split_data[ind]
                if (data.isdigit()):
                    split_data[ind] = "[" + data + "]"
            final_string = '.'.join(split_data)

            iters_a = re.finditer(r'[a-zA-Z]\.\[', final_string)
            indices = [m.start(0) + 1 for m in iters_a]
            iters = re.finditer(r'\]\.\[', final_string)
            indices.extend([m.start(0) + 1 for m in iters])

            final_string = list(final_string)
            final_string = [final_string[i] for i in range(len(final_string)) if i not in indices]

            str_data = ''.join(final_string)

        except:
            pass

        return str_data


    def summary_string_fixed(self, model, all_layers, input_size, model_name=None, batch_size=-1, dtypes=None):
        if dtypes is None:
            dtypes = [torch.FloatTensor] * len(input_size)

        def register_hook(module, module_idx):
            def hook(module, input, output):
                nonlocal module_idx
                m_key = all_layers[module_idx][0]
                m_key = model_name + "." + m_key

                try:
                    eval(m_key)
                except:
                    m_key = name_fixer([m_key])[0]

                summary[m_key] = OrderedDict()
                summary[m_key]["type"] = str(type(module)).split('.')[-1][:-2]
                summary[m_key]["x"] = input
                summary[m_key]["input_shape"] = list(input[0].size())
                summary[m_key]["input_shape"][0] = batch_size

                if isinstance(output, (list, tuple)):
                    summary[m_key]["y"] = [
                        [-1] + list(o)[1:] for o in output
                    ]
                    summary[m_key]["output_shape"] = [
                        [-1] + list(o.size())[1:] for o in output
                    ]
                else:
                    summary[m_key]["y"] = list(output)


                    summary[m_key]["output_shape"] = list(output.size())
                    summary[m_key]["output_shape"][0] = batch_size

                params = 0
                if hasattr(module, "weight") and hasattr(module.weight, "size"):
                    summary[m_key]["w"] = module.weight
                    params += torch.prod(torch.LongTensor(list(module.weight.size())))
                    summary[m_key]["trainable"] = module.weight.requires_grad
                    summary[m_key]["weight_shape"] = module.weight.shape
                if hasattr(module, "bias") and hasattr(module.bias, "size"):
                    summary[m_key]["b"] = module.bias
                    params += torch.prod(torch.LongTensor(list(module.bias.size())))
                summary[m_key]["nb_params"] = params

            if (
                    not isinstance(module, nn.Sequential)
                    and not isinstance(module, nn.ModuleList)
            ):
                hooks.append(module.register_forward_hook(hook))

        if isinstance(input_size, tuple):
            input_size = [input_size]
            
        model_device = next(iter(model.parameters())).device
        x, _ = next(iter(dataloader['test']))
        x = x.to(model_device)

        summary = OrderedDict()
        hooks = []

        for module_idx, (layer_name, module) in enumerate(all_layers):
            register_hook(module, module_idx)

        model(x)

        for h in hooks:
            h.remove()

        return summary


    def layer_mapping(self, model):
        all_layers = self.get_all_layers(model)
        x,y = next(iter(dataloader['test']))
        model_summary = self.summary_string_fixed(model, all_layers, x.shape, model_name='model')

        name_type_shape = []
        for key in model_summary.keys():
            data = model_summary[key]
            if ("weight_shape" in data.keys()):
                name_type_shape.append([key, data['type'], data['weight_shape'][0]])
            else:
                name_type_shape.append([key, data['type'], 0 ])
        name_type_shape = np.asarray(name_type_shape)

        name_list = name_type_shape[:, 0]

        r_name_list = np.asarray(name_list)
        random_picks = np.random.randint(0, len(r_name_list), 10)
        test_name_list = r_name_list[random_picks]
        eval_hit = False
        for layer in test_name_list:
            try:
                eval(layer)

            except:
                eval_hit = True
                break
        if (eval_hit):
            fixed_name_list = name_fixer(r_name_list)
            name_type_shape[:, 0] = fixed_name_list

        layer_types = name_type_shape[:, 1]
        layer_shapes = name_type_shape[:, 2]
        mapped_layers = {'model_layer': [], 'Conv2d_BatchNorm2d_ReLU': [], 'Conv2d_BatchNorm2d': [], 'Linear_ReLU': [],
                        'Linear_BatchNorm1d': []}

        def detect_sequences(lst):
            fusing_layers = [
                            'Conv2d',
                            'BatchNorm2d',
                            'ReLU',
                            'Linear',
                            'BatchNorm1d',
                        ]

            i = 0
            while i < len(lst):

                if i + 2 < len(lst) and [l for l in lst[i: i + 3]] == [
                    fusing_layers[0],
                    fusing_layers[1],
                    fusing_layers[2],
                ]:
                    
                    mapped_layers['Conv2d_BatchNorm2d_ReLU'].append(
                            np.take(name_list, [i for i in range(i, i + 3)]).tolist()
                        )
                    i += 3

                elif i + 1 < len(lst) and [l for l in lst[i: i + 2]] == [
                    fusing_layers[0],
                    fusing_layers[1],
                ]:

                    mapped_layers['Conv2d_BatchNorm2d'].append(
                            np.take(name_list, [i for i in range(i, i + 2)]).tolist()
                        )
                    i += 2

                elif i + 1 < len(lst) and [l for l in lst[i: i + 2]] == [
                    fusing_layers[3],
                    fusing_layers[2],
                ]:
                    mapped_layers['Linear_ReLU'].append(
                        np.take(name_list, [i for i in range(i, i + 2)]).tolist()
                    )
                    i += 2
                elif i + 1 < len(lst) and [l for l in lst[i: i + 2]] == [
                    fusing_layers[3],
                    fusing_layers[4],
                ]:
                    mapped_layers['Linear_BatchNorm1d'].append(
                        np.take(name_list, [i for i in range(i, i + 2)]).tolist()
                    )
                    i += 2
                else:
                    i += 1

        def detect_sorting_pairs(name_list):
            sorting_pairs = []
            idx = 0
            while idx < len(name_list):
                layer_list = [ layer_name for layer_name in name_list[idx:idx+3]]
                
                if([ type(eval(l)) for l in layer_list] == [nn.Conv2d, nn.BatchNorm2d,nn.Conv2d]):
                    sorting_pairs.append(layer_list)
                    idx+=2
                elif([ type(eval(l)) for l in layer_list[idx:idx+2]] == [nn.Conv2d, nn.Conv2d]):
                    sorting_pairs.append(layer_list[idx:idx+2].insert(1,0))
                    idx+=1
                else:
                    idx+=1
            return sorting_pairs

        detect_sequences(layer_types)

        for keys, value in mapped_layers.items():
            mapped_layers[keys] = np.asarray(mapped_layers[keys])

        mapped_layers['name_type_shape'] = name_type_shape
        mapped_layers['name_list'] = mapped_layers['name_type_shape'][:, 0]
        mapped_layers['type_list'] = mapped_layers['name_type_shape'][:, 1]

        # CWP
        keys_to_lookout = ['Conv2d_BatchNorm2d_ReLU', 'Conv2d_BatchNorm2d']
        pruning_layer_of_interest, qat_layer_of_interest = [], []

        # CWP or QAT Fusion Layers
        for keys in keys_to_lookout:
            data = mapped_layers[keys]
            if (len(data) != 0):
                qat_layer_of_interest.append(data)
        mapped_layers['qat_layers'] = qat_layer_of_interest
        mapped_layers['model_summary'] = model_summary

        name_list = mapped_layers['name_type_shape'][:, 0]
        layer_name_list = []
        w, x, y = [], [], []
        for layer_name in name_list:
            layer = eval(layer_name)
            if(isinstance(layer,(nn.Conv2d,nn.Linear))):
                layer_name_list.append(layer_name)
                x.append(mapped_layers['model_summary'][layer_name]['x'][0])
                w.append(mapped_layers['model_summary'][layer_name]['w'])
                y.append(torch.stack(mapped_layers['model_summary'][layer_name]['y']))
                # b.append(mapped_layers['model_summary'][layer_name]['b'])
        
        mapped_layers['catcher'] = {'name_list':layer_name_list, 'x':x,'w':w,'y':y}


        mapped_layers['conv_list'] = [l_n for l_n in mapped_layers['catcher']['name_list'] if isinstance(eval(l_n), nn.Conv2d)]
        mapped_layers['conv_bn_list'] = name_list[[index for index, layer in enumerate(mapped_layers['type_list']) if layer in ['Conv2d', 'BatchNorm2d']]]
        mapped_layers['sorted_conv_list'] = detect_sorting_pairs(mapped_layers['conv_bn_list'])

        return mapped_layers


    # GMP
    #         layer_of_interest=mapped_layers['name_type_shape'][:,0] # all layers with weights
    #         Check for all with weights
    # Wanda

    def string_fixer(self,name_list):
        for ind in range(len(name_list)):
            modified_string = re.sub(r'\.(\[)', r'\1', name_list[ind])
            name_list[ind] = modified_string



# In[ ]:


# model = VGG().cuda()
# checkpoint = torch.load('vgg.cifar.pretrained.pth')
# model.load_state_dict(checkpoint)


import torch
model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)
model.eval()



# In[2]:


ma = ModelAnalyzer(model)


for layer_name, layer_type in zip(ma.mapped_layers['name_list'], ma.mapped_layers['type_list']):
    print(f"{layer_name} : {layer_type}")


# In[ ]:





# In[1]:


get_ipython().system('jupyter nbconvert --to script Model Analyzer.ipynb')


# In[ ]:




