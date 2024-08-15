import torch
from ModelAnalyzer import ModelAnalyzer
from Quantizer import Quantizer
import torch.nn.functional as F
class Chunker(ModelAnalyzer):

    def __init__(self, model, calibiration_data):
        self.model = model
        self.calibiration_data = calibiration_data
        self.mapped_layers = ModelAnalyzer(self.model, self.calibiration_data).mapped_layers
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


    def chunk(self):

            # Run through configs for each layer and compute the error!
            for layer_name, layer_data in self.mapped_layers['calibiration_data'].items():
                # Weights, Activations
                # Per: Group, Channel, Tensor, etc..
                # Dtype: int8, fp8, etc..
                # Symmentric: True, False
                # Compute Error for each qlayer and get the least mse error of qlayer_op and layer_data['output']

                global_err = float("inf")
                params = None

                layer_under_test = eval('self.' + layer_name)
                activations = layer_data['activations']
                out = layer_data['outputs']

                data_type = torch.int8

                for act_affine in [("dim", 0), ("tensor", None)]:
                    for act_sym in ["asymmentric","symmentric"]:
                        for weight_affine in [("dim", 0), ("tensor", None)]:
                            for weight_sym in ["asymmentric", "symmentric"]:

                                q_params = {'weights': {'dtype': data_type,
                                                        'symmentry': act_sym,
                                                        'per': act_affine[0],
                                                        'per_dim': act_affine[1]},
                                            'activations': {'dtype': data_type,
                                                            'symmentry': weight_sym,
                                                            'per': weight_affine[0],
                                                            'per_dim': weight_affine[1]
                                                            },
                                            'outputs': {'dtype': None,
                                                        'symmentry': None,
                                                        'per': None,
                                                        'per_dim': None}
                                            }

                                qlayer_test = Quantizer.from_float(module=layer_under_test,
                                                              activations=layer_data['activations'],
                                                              data_metry=q_params
                                                              )
                                qout = qlayer_test.forward(activations)
                                mse = F.mse_loss(qout, out)
                                if(mse<global_err):
                                    global_err = mse
                                    params = q_params

                qlayer = Quantizer.from_float(module=layer_under_test,
                                              activations=layer_data['activations'],
                                              data_metry=params
                                              )
                self.replace_layer(layer_name, qlayer)


