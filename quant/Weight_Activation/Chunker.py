import torch
from quant.ModelAnalyzer import ModelAnalyzer
from quant.Quantizer import Quantizer
import torch.nn as nn
from tqdm import tqdm
from Qact import Qact

class Chunker(ModelAnalyzer):

    def __init__(self, model, calibiration_data):
        self.hooks = {}
        self.model = model
        self.calibiration_data = calibiration_data
        self.mapped_layers = ModelAnalyzer(self.model, self.calibiration_data).mapped_layers
        self.interested_layers = []
        self.chunk()

    def analyze_quantization_potential(self, tensor):
        # Calculate basic statistics
        mean_val = tensor.mean()
        min_val = tensor.min()
        max_val = tensor.max()

        # Calculate the absolute maximum for symmetric quantization comparison
        abs_max = torch.max(torch.abs(min_val), torch.abs(max_val))

        # Determine potential for symmetric quantization
        symmetric_threshold = 0.20 * abs_max  # Threshold can be adjusted based on specific needs
        if isinstance(symmetric_threshold, torch.Tensor):
            symmetric_threshold = symmetric_threshold.item()

        if torch.isclose(mean_val, torch.tensor(0.0), atol=symmetric_threshold):
            return "symmentric"

        range_positive = max_val - mean_val
        range_negative = mean_val - min_val
        if torch.isclose(range_positive, range_negative, atol=symmetric_threshold):
            return "asymmentric"

        return "asymmentric"

    def replace_modules(self, module, target_class, look_out_for,  module_name_to_exclude=""):

        def pre_forward_hook(module, input):
            module.input_observer(input[0])

        for name, child in module.named_children():

            if isinstance(child, look_out_for) and not \
                    any([x == name for x in module_name_to_exclude]):

                if(target_class=='weights'):
                    affine = ("channel", 0) if isinstance(child, torch.nn.Conv2d) else ("tensor", None)
                    qdict = {'dtype': torch.int8, 'symmetric': True, 'affine': affine[0], 'affine_dim': affine[1]}
                    q_params = {'weights': qdict }
                    qlayer = Quantizer.from_float(module=child, data_metry=q_params, quantize_output=False)
                    self.hooks[name] = qlayer.register_forward_pre_hook(pre_forward_hook)
                    setattr(module, name, qlayer)
                if(target_class=='activations'):
                    child.input_quantizer.min_val, child.input_quantizer.max_val = child.input_observer.min_val, child.input_observer.max_val
                    child.input_quantizer.scales, child.input_quantizer.zero_point  = child.input_quantizer.calculate_params()
                    child.input_quant = True

            else:
                # Recursively call the function for nested modules
                self.replace_modules(child, target_class, look_out_for, module_name_to_exclude)


    def weight_quantize(self):
        hooks = self.replace_modules(module=self.model, target_class='weights', look_out_for = (torch.nn.Conv2d, torch.nn.Linear))

    def activation_quantize(self):
        self.replace_modules(module=self.model, target_class='activations', look_out_for = (Quantizer))


    def calibirate_model(self):
        print("Calibrating model...")
        device = next(self.model.parameters()).device
        with torch.no_grad():
            for input_data, _ in tqdm(self.calibiration_data):
                _ = self.model(input_data.to(device))
        print("Calibration done!")

    def chunk(self):

        # self.prepare_model()
        self.weight_quantize()
        self.calibirate_model()
        for hook in self.hooks.values():
            hook.remove()
        self.activation_quantize()
        print(self.model.conv1.input_quantizer.scales, self.model.conv1.input_quantizer.zero_point)


        # isinstance(self.model.backbone.conv0[0], Quantizer)

