from models import SimpleCNN, VGG
import torch
from ModelAnalyzer import ModelAnalyzer
from Quantizer import Quantizer
from Fusion import Fuse
from dataset import dataloader

class Chunker(ModelAnalyzer):
    def __init__(self, model, calibiration_data):
        self.model = model
        self.calibiration_data = calibiration_data
        self.mapped_layers = ModelAnalyzer(self.model, self.calibiration_data).mapped_layers
        self.chunker()

    def analyze_quantization_potential(self, tensor):

        # Calculate basic statistics
        mean_val = tensor.mean()
        min_val = tensor.min()
        max_val = tensor.max()

        # Calculate the absolute maximum for symmetric quantization comparison
        abs_max = torch.max(torch.abs(min_val), torch.abs(max_val))

        # Determine potential for symmetric quantization
        symmetric_threshold = 0.20 * abs_max  # Threshold can be adjusted based on specific needs
        if (isinstance(symmetric_threshold, torch.Tensor)): symmetric_threshold = symmetric_threshold.item()

        if torch.isclose(mean_val, torch.tensor(0.0), atol=symmetric_threshold):
            return "symmentric"
        else:
            range_positive = max_val - mean_val
            range_negative = mean_val - min_val
            if not torch.isclose(range_positive, range_negative, atol=symmetric_threshold):
                return "asymmentric"

        # If nothing works settle down for "asymmentric"
        return "asymmentric"

    def chunker(self):

        # Run through configs for each layer and compute the error!
        for layer_name, layer_data in self.mapped_layers['calibiration_data'].items():
            # Weights, Activations
            # Per: Group, Channel, Tensor, etc..
            # Dtype: int8, fp8, etc..
            # Symmentric: True, False
            # Compute Error for each qlayer and get the least mse error of qlayer_op and layer_data['output']
            qlayer_wise = ("dim", 0) if layer_data['layer_type'][1] == "Conv2d" else ("tensor", None)
            q_params = {'weights': {'dtype': torch.int8,
                                    'symmentry': self.analyze_quantization_potential(layer_data['weights'].data),
                                    'per': qlayer_wise[0],
                                    'per_dim': qlayer_wise[1]},
                        'activations': {'dtype': torch.int8,
                                        'symmentry': self.analyze_quantization_potential(layer_data['activations']),
                                        'per': qlayer_wise[0],
                                        'per_dim': qlayer_wise[1]},
                        'outputs': {'dtype': None,
                                    'symmentry': None,
                                    'per': None,
                                    'per_dim': None}
                        }

            qlayer = Quantizer.from_float(module=eval('self.'+layer_name),
                                          activations=layer_data['activations'],
                                          data_metry=q_params
                                          )
            setattr(eval('.'.join(layer_name.split('.')[:-1])), layer_name.split('.')[-1], qlayer)

# model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
# model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2')#, pretrained=True)
model = VGG()
# model = SimpleCNN()
test = torch.rand(1, 3, 128, 128)

fuser = Fuse(model.eval(), dataloader['test'])
fused_model = fuser.fused_model.train()

quantized_model = Chunker(fused_model, dataloader['test']).model

print(quantized_model)