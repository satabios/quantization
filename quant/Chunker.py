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
        for idx, layer_name in enumerate( self.interested_layers):
            layer = eval('self.'+layer_name)
            input_qparams, output_qparams = layer[0].calculate_qparams(),layer[-1].calculate_qparams()
            input_qparams = {'scale':input_qparams[0].item(), 'zero_point':input_qparams[1].item()}
            output_qparams = {'scale':output_qparams[0].item(), 'zero_point':output_qparams[1].item()}
            dtype = torch.int8
            sym = "asymmentric"
            affine =  ("channel", 0) if isinstance(layer, torch.nn.Conv2d) else ("tensor",None)

            q_params = {
                        'weights': {'dtype': dtype,
                                    'symmentry': sym,
                                    'affine': affine[0],
                                    'affine_dim': affine[1]},
                        'activations': {'dtype': dtype,
                                        'symmentry': "symmentric" if(input_qparams['scale']==0) else "asymmentric",
                                        'scale': input_qparams['scale'],
                                        'zero_point': input_qparams['zero_point']
                                        },
                        'outputs': {'dtype': dtype,
                                    'symmentry': "symmentric" if(output_qparams['scale']==0) else "asymmentric",
                                        'scale': output_qparams['scale'],
                                        'zero_point': output_qparams['zero_point']
                                    }

                        }
            activations = torch.rand(1,3,32,32)

            qlayer = Quantizer.from_float(  module = layer[1],
                                            activations = activations,
                                            data_metry = q_params,
                                            quantize_output = True
                                          )

            self.replace_layer(layer_name, qlayer)


    def calibirate_model(self):
        with torch.no_grad():
            for input_data, _ in self.calibiration_data:
                _ = self.model(input_data)

    def chunk(self):



        self.prepare_model()

        self.quantize()
        self.calibirate_model()



        ###############################
            # out = None
            # # Run through configs for each layer and compute the error!
            # for layer_name, layer_data in self.mapped_layers['calibiration_data'].items():
            #     # Weights, Activations
            #     # Per: Group, Channel, Tensor, etc..
            #     # Dtype: int8, fp8, etc..
            #     # Symmentric: True, False
            #     # Compute Error for each qlayer and get the least mse error of qlayer_op and layer_data['output']
            #
            #     layer_under_test = eval('self.' + layer_name)
            #     if(out is None): activations = layer_data['activations'] #Inital Activations
            #     else:
            #         activations = out.reshape_as(layer_data['activations'])
            #
            #     data_type = torch.int8
            #     sym = "asymmentric"
            #     affine = ("channel", 0) if isinstance(layer_under_test, torch.nn.Conv2d) else ("tensor", None)
            #
            #     q_params = {'weights': {'dtype': data_type,
            #                             'symmentry': sym,
            #                             'affine': affine[0],
            #                             'affine_dim': affine[1]},
            #                 'activations': {'dtype': data_type,
            #                                 'symmentry': sym,
            #                                 'affine': affine[0],
            #                                 'affine_dim': affine[1]
            #                                 },
            #                 'outputs': {'dtype': None,
            #                             'symmentry': None,
            #                             'affine': None,
            #                             'affine_dim': None}
            #                 }
            #
            #
            #     qlayer = Quantizer.from_float(module=layer_under_test,
            #                                   activations=activations,
            #                                   data_metry=q_params
            #                                   )
            #     out = qlayer.forward(activations) #Update layer out with the replaced quantization
            #     self.replace_layer(layer_name, qlayer)
            #
            #
