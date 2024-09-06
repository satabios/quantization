import torch
from ModelAnalyzer import ModelAnalyzer
from Quantizer import Quantizer
import torch.nn.functional as F
import torch.quantization.observer as observer
import torch.nn as nn
from tqdm import tqdm
from Qop import Qop

class Chunker(ModelAnalyzer):

    def __init__(self, model, calibiration_data):

        self.model = model #QModel(model)

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
        self.model = self.model.to('cuda')


        for key in self.mapped_layers['sequences'].keys():
            for layer_name_list in self.mapped_layers['sequences'][key]:
                # 2 Layer: Weighted Layer, Acitvation (weighted_layer, activation, observer)
                # 1 Layer: Weighted Layer => (weighted_layer, observer)
                if(len(layer_name_list)>1): replace_layer_name = layer_name_list[0].split('[')[0]
                else: replace_layer_name = layer_name_list[0]
                weighted_layer = eval('self.'+ replace_layer_name)

                qdict = {'dtype': torch.int8, 'symentric': True, 'affine': 'channel', 'affine_dim': 0}
                q_params = {
                    'weights': qdict,
                }
                qweighted_layer = Quantizer.from_float(module=weighted_layer, data_metry=q_params, quantize_output=False)
                self.replace_layer( replace_layer_name, qweighted_layer) #Replace the weighted layer with the whole set





    def quantize (self):
        # For Post Training Quantization if Actiation quantizations are active realize those layers as well and replace them wiht Indentitiy block
        #

        #Keeepo progressing these changes and one day you mught be able to get the whole quantizer in place and
        # even take the sensitivity scan into the chunker and realise it into a complete pacakge.

        #If Only targetting weights only Quantization
        for key in list(self.mapped_layers['observers'].keys())[:-1]:
            for observer_layer_name in self.mapped_layers['observers'][key]:
                observer_layer_name = observer_layer_name[0]
                layer = eval('self.'+ observer_layer_name)
                class QObserver(nn.Module):
                    def __init__(self, qdeets):
                        super(QObserver, self).__init__()
                        self.quant =  Qop(
                            dtype=torch.int8,
                            symentric=False,
                            affine='tensor',
                            affine_dim=None
                        )
                        self.quant.scales, self.quant.zero_point = qdeets[0].item(), qdeets[1].item()
                    def forward(self, x):
                        return self.quant.quantize(x)
                    def dequantize(self, x):
                        return self.quant.dequantize(x)

                qdeets = layer.calculate_qparams()
                qobserver = QObserver(qdeets)

                if("[" in observer_layer_name): observer_layer_name = observer_layer_name[:-3]+'._modules[\''+observer_layer_name[-2]+"\']"

                self.replace_layer(observer_layer_name, qobserver)


    def calibirate_model(self):
        print("Calibrating model...")
        with torch.no_grad():
            for input_data, _ in tqdm(self.calibiration_data):
                try:
                    self.model = self.model.to('cuda')
                    _ = self.model(input_data.to('cuda'))
                except:
                    self.model = self.model.to('cpu')
                    _ = self.model(input_data.to('cpu'))#.to('cpu')
        self.mapped_layers = ModelAnalyzer(self.model, self.calibiration_data).mapped_layers
        print("Calibration done!")

    def qlayer_replacement(self, module, module_name_to_exclude=[""]):
        for name, child in module.named_children():
            if ((isinstance(child, nn.Linear) or isinstance(child, nn.Conv2d)) and not any(
                    [x == name for x in module_name_to_exclude])):

                qdict = {'dtype': torch.int8, 'symentric': True, 'affine': 'channel', 'affine_dim': 0}
                q_params = {
                    'weights': qdict,
                }

                qweighted_layer = Quantizer.from_float(module=child, data_metry=q_params, quantize_output=False)
                setattr(module, name, qweighted_layer)

            else:
                self.qlayer_replacement(child, module_name_to_exclude)

    def chunk(self):

        self.qlayer_replacement(self.model)
        # self.prepare_model()
        # self.calibirate_model()
        # self.quantize()


