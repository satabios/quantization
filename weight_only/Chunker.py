import torch
from ModelAnalyzer import ModelAnalyzer
from Quantizer import Quantizer
import torch.nn as nn
from tqdm import tqdm
from Qact import Qact

class Chunker(ModelAnalyzer):

    def __init__(self, model, calibiration_data):
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


    def replace_layer(self, layer_name, qlayer):

        path_parts = layer_name.split('.')
        attr_path = 'self.' + '.'.join(path_parts[:-1])

        # Handle nested modules if there's dictionary access
        if "['" in path_parts[-1]:
            last_part = path_parts[-1].split('[')[0]  # Get the attribute name before the dictionary access
            index = path_parts[-1].split('[')[1].strip("']")  # Extract index/key from the string
            indexed = True
        else:
            last_part = path_parts[-1]
            indexed = False

        # Get the attribute object before the last part
        attr_obj = eval(attr_path)

        # Set the new quantized layer at the correct position
        if indexed:
            # For dictionary-like access within modules
            getattr(attr_obj, last_part)[index] = qlayer
        else:
            # Direct attribute setting
            setattr(attr_obj, last_part, qlayer)


    def prepare_model(self):

        # internal_list = [('Conv2d', 'ReLU'), ('Linear', 'ReLU'), ('Conv2d',), ('Linear',)]
        for keys in self.mapped_layers['sequences'].keys():
            for lyrs_data in self.mapped_layers['sequences'][keys]:
                lyrs = []
                for lr in lyrs_data:
                    lyrs.append(eval('self.' + lr))

                post = torch.ao.quantization.observer.HistogramObserver().to(next(self.model.parameters()).device)
                stubbed_layer = nn.Sequential(
                                      *lyrs,
                                      post
                                     )
                self.interested_layers.append(lyrs_data[0])
                self.replace_layer(lyrs_data[0], stubbed_layer)
                if(len(lyrs) > 1):
                    self.replace_layer(lyrs_data[1], nn.Identity())

        #Update the skeleton
        self.mapped_layers = ModelAnalyzer(self.model, self.calibiration_data).mapped_layers

    def convert_to_modules_notation(self, shorthand):
        splits = shorthand.split('[')
        expanded = '._modules['.join(splits)

        return expanded

    # def quantize (self):
    #     # internal_list = [('Conv2d', 'BatchNorm2d', 'ReLU',), ('Linear',)]
    #     # for keys in internal_list:
    #     for lyr_type, layer_names in self.mapped_layers['w_layers'].items():
    #         for layer_name in layer_names:
    #             layer_name = layer_name[0]
    #             layer = eval('self.'+ layer_name)
    #             affine =  ("channel", 0) if isinstance(layer, torch.nn.Conv2d) else ("tensor",None)
    #             qdict = {'dtype': torch.int8, 'symentric': True, 'affine': 'tensor', 'affine_dim': None}
    #             q_params = {'weights': qdict }
    #             qlayer = Quantizer.from_float(module=layer, data_metry=q_params, quantize_output=False)
    #             self.replace_layer(layer_name, qlayer)

    def replace_modules(self, module, target_class, look_out_for, module_name_to_exclude=""):
        for name, child in module.named_children():
            if isinstance(child, look_out_for) and not \
                    any([x == name for x in module_name_to_exclude]):

                if(target_class=='weights'):
                    affine = ("channel", 0) if isinstance(child, torch.nn.Conv2d) else ("tensor", None)
                    qdict = {'dtype': torch.int8, 'symentric': True, 'affine': affine[0], 'affine_dim': affine[1]}
                    q_params = {'weights': qdict }
                    qlayer = Quantizer.from_float(module=child, data_metry=q_params, quantize_output=False)
                    setattr(module, name, qlayer)
                if(target_class=='activations'):
                    qconfig = {'min_val': child.min_val, 'max_val':child.max_val}
                    qlayer = Qact(qconfig)
                    setattr(module, name, qlayer)

            else:
                # Recursively call the function for nested modules
                self.replace_modules(child, target_class, look_out_for, module_name_to_exclude)

    def weight_quantize(self):
        self.replace_modules(module=self.model, target_class='weights', look_out_for = (torch.nn.Conv2d, torch.nn.Linear))

    def activation_quantize(self):
        self.replace_modules(module=self.model, target_class='activations', look_out_for = (torch.ao.quantization.ObserverBase))


    def calibirate_model(self):
        print("Calibrating model...")
        device = next(self.model.parameters()).device
        with torch.no_grad():
            for input_data, _ in tqdm(self.calibiration_data):
                _ = self.model(input_data.to(device))
        print("Calibration done!")

    def chunk(self):

        self.prepare_model()
        self.weight_quantize()
        self.calibirate_model()
        self.activation_quantize()

        # isinstance(self.model.backbone.conv0[0], Quantizer)

