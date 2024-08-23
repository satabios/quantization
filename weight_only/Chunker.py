import torch
from ModelAnalyzer import ModelAnalyzer
from Quantizer import Quantizer
import torch.nn.functional as F
import torch.quantization.observer as observer
import torch.nn as nn

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

        internal_list = [('Conv2d', 'ReLU'), ('Linear', 'ReLU'), ('Conv2d',), ('Linear',)]
        for keys in internal_list:
            for lyrs_data in self.mapped_layers['sequences'][keys]:
                lyrs = []
                for lr in lyrs_data:
                    lyrs.append(eval('self.' + lr))
                prior = torch.ao.quantization.observer.HistogramObserver()
                post = torch.ao.quantization.observer.HistogramObserver()
                stubbed_layer = nn.Sequential(
                                      prior,
                                      *lyrs,
                                      post
                                     )
                self.interested_layers.append(lyrs_data[0])
                self.replace_layer(lyrs_data[0], stubbed_layer)
                if(len(lyrs) > 1):
                    self.replace_layer(lyrs_data[1], nn.Identity())

        #Update the skeleton
        self.mapped_layers = ModelAnalyzer(self.model, self.calibiration_data).mapped_layers


    def quantize (self):
        internal_list = [('Conv2d',), ('Linear',)]
        for keys in internal_list:
            for layer_name in self.mapped_layers['sequences'][keys]:
                layer_name = layer_name[0]
                layer = eval('self.'+ layer_name)

                dtype = torch.int8

                affine =  ("channel", 0) if isinstance(layer, torch.nn.Conv2d) else ("tensor",None)
                if(isinstance(layer, torch.nn.Conv2d)):
                    qdict= {'dtype':torch.int8, 'symentric':True,'affine':'channel', 'affine_dim':0}
                else:
                    qdict = {'dtype': torch.int8,'symentric': False,'affine': 'tensor','affine_dim':None}
                q_params = {
                    'weights': qdict,
                    # 'activations': {'dtype': dtype},
                    # 'outputs': {'dtype': dtype}
                }
                qlayer = Quantizer.from_float(module=layer, data_metry=q_params, quantize_output=False)
                self.replace_layer(layer_name, qlayer)


    def calibirate_model(self):
        with torch.no_grad():
            for input_data, _ in self.calibiration_data:
                _ = self.model(input_data)

    def chunk(self):

        # self.prepare_model()

        self.quantize()
        # self.calibirate_model()

