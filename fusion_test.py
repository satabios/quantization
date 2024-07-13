import torch
import torch.nn as nn
import torch.nn.functional as F


def fuse_conv_bn(conv, bn):
    # Fuse Conv2d + BatchNorm2d
    with torch.no_grad():
        # Get the scale and bias for the batch normalization
        scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
        bias = bn.bias - bn.running_mean * scale

        # Update conv weight
        conv.weight *= scale.reshape([-1, 1, 1, 1])
        if conv.bias is not None:
            conv.bias *= scale
        else:
            conv.bias = bias

        # Add the bias term
        conv.bias += bias
    return conv

def fuse_conv_bn_relu(conv, bn, relu):
    fused_conv = fuse_conv_bn(conv, bn)
    return nn.Sequential(fused_conv, relu)

def fuse_conv_relu(conv, relu):
    return nn.Sequential(conv, relu)

def fuse_linear_relu(linear, relu):
    return nn.Sequential(linear, relu)

def fuse_bn_relu(bn, relu):
    return nn.Sequential(bn, relu)

def fuse_layers(layers):
    fused_layers = []
    i = 0
    while i < len(layers):
        if i < len(layers) - 2 and isinstance(layers[i], nn.Conv2d) and isinstance(layers[i+1], nn.BatchNorm2d) and isinstance(layers[i+2], nn.ReLU):
            # Fuse conv, bn, relu
            fused_layers.append(fuse_conv_bn_relu(layers[i], layers[i+1], layers[i+2]))
            i += 3
        elif i < len(layers) - 1 and isinstance(layers[i], nn.Conv2d) and isinstance(layers[i+1], nn.BatchNorm2d):
            # Fuse conv and bn
            fused_layers.append(fuse_conv_bn(layers[i], layers[i+1]))
            i += 2
        elif i < len(layers) - 1 and isinstance(layers[i], nn.Conv2d) and isinstance(layers[i+1], nn.ReLU):
            # Fuse conv, relu
            fused_layers.append(fuse_conv_relu(layers[i], layers[i+1]))
            i += 2
        elif i < len(layers) - 1 and isinstance(layers[i], nn.Linear) and isinstance(layers[i+1], nn.ReLU):
            # Fuse linear, relu
            fused_layers.append(fuse_linear_relu(layers[i], layers[i+1]))
            i += 2
        elif i < len(layers) - 1 and isinstance(layers[i], nn.BatchNorm2d) and isinstance(layers[i+1], nn.ReLU):
            # Fuse bn, relu
            fused_layers.append(fuse_bn_relu(layers[i], layers[i+1]))
            i += 2
        else:
            # If no fusing is possible, just append the layer
            fused_layers.append(layers[i])
            i += 1
    return nn.Sequential(*fused_layers)



# Example list of layers
layers = [
    nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(16),
    nn.ReLU(),
    nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(32),
    nn.ReLU(),
    nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
    nn.ReLU(),
    nn.Linear(64 * 8 * 8, 128),
    nn.ReLU(),
    nn.BatchNorm2d(128),
    nn.ReLU()
]

# Fuse the layers
fused_model = fuse_layers(layers)

# Print the fused model
print(fused_model)
