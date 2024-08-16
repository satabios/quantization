from Fusion import Fuse
from dataset import dataloader
import torch
from models import SimpleCNN, VGG
from Chunker import Chunker
from tqdm import tqdm
from Qop import Qop
import torch.nn.functional as F
from sconce import sconce
import torch.nn as nn
import torch.optim as optim
import torch.fx as fx
from ModelAnalyzer import ModelAnalyzer
resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18')
mbnet = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2')
vgg = VGG()

test_data = torch.rand(1, 3, 128, 128)

import torch.nn as nn

import torch.nn as nn


# def insert_stubs(graph, model):
#     for node in list(graph.nodes):
#         if node.op == 'call_module':  # Check if the node represents a module call
#             layer_name = node.target
#
#             # Split the layer_name to access nested modules
#             module_names = layer_name.split('.')
#
#             layer = model
#             try:
#                 # Navigate through the model hierarchy to find the exact layer
#                 for name in module_names:
#                     layer = getattr(layer, name)
#             except AttributeError:
#                 continue  # Skip if layer is not found, handle gracefully
#
#             # Check if the layer is Conv2d or Linear
#             if isinstance(layer, (nn.Conv2d, nn.Linear)):
#                 # Determine if the current layer is followed by an activation
#                 followed_by_activation = False
#                 activation_user = None
#                 for user in node.users:
#                     if user.op == 'call_module':
#                         next_layer_name = user.target
#                         next_layer = model
#                         try:
#                             # Navigate through the model hierarchy for the next layer
#                             for name in next_layer_name.split('.'):
#                                 next_layer = getattr(next_layer, name)
#                         except AttributeError:
#                             continue
#
#                         if isinstance(next_layer, (nn.ReLU, nn.GELU, nn.Sigmoid, nn.Tanh, nn.Softmax)):
#                             followed_by_activation = True
#                             activation_user = user
#                             break
#                     elif user.op == 'call_function':
#                         if user.target in [torch.relu, torch.nn.functional.gelu]:
#                             followed_by_activation = True
#                             activation_user = user
#                             break
#
#                 # Replace the activation with an Identity layer if followed by activation
#                 if followed_by_activation:
#                     identity_layer = nn.Identity()
#
#                     # Replace the activation in the model with Identity
#                     parent_module = model
#                     *path, last_layer = next_layer_name.split('.')
#                     for attr in path:
#                         parent_module = getattr(parent_module, attr)
#
#                     setattr(parent_module, last_layer, identity_layer)
#
#                     # Update the graph to reflect these changes
#                     activation_users_copy = list(activation_user.users.keys())
#                     for use_node in activation_users_copy:
#                         use_node.replace_input_with(activation_user, node)
#
#                     graph.erase_node(activation_user)
#
#                 # Construct the quantized layer
#                 quantized_layer = nn.Sequential(
#                     torch.ao.quantization.QuantStub(),
#                     layer,
#                     torch.ao.quantization.DeQuantStub()
#                 ) if not followed_by_activation else nn.Sequential(
#                     torch.ao.quantization.QuantStub(),
#                     layer,
#                     next_layer,  # Use the next_layer (activation) if it follows
#                     torch.ao.quantization.DeQuantStub()
#                 )
#
#                 # Replace the original layer in the model
#                 parent_module = model
#                 *path, last_layer = layer_name.split('.')
#                 for attr in path:
#                     parent_module = getattr(parent_module, attr)
#
#                 setattr(parent_module, last_layer, quantized_layer)
#
#     # Ensure the graph is consistent
#     graph.eliminate_dead_code()  # Removes nodes that are no longer used
#     graph.lint()  # Ensures the graph is in a consistent state
from ModelAnalyzer import ModelAnalyzer
from dataset import dataloader
for model in [vgg, resnet, mbnet]:

    fuser = Fuse(model.eval(), dataloader['test'])
    new_mapping = ModelAnalyzer(fuser.fused_model, dataloader['test']).mapped_layers
    # new_mapping['sequences']['']
    break


    # traced_model = fx.symbolic_trace(fused_model)
    # insert_stubs(traced_model.graph, traced_model)
    # traced_model.recompile()
    # print(traced_model)
