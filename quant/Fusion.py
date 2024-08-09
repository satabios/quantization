import torch
from ModelAnalyzer import ModelAnalyzer
import torch.ao.quantization.quantize_fx as quantize_fx

class Fuse(ModelAnalyzer):

    def __init__(self, model, calibiration_data=None):

        self.model = model
        self.fused_model = None
        self.calibiration_data = calibiration_data
        self.fuse_model()

    def fuse_model(self):

        ma = ModelAnalyzer(self.model, self.calibiration_data)
        mapped_layers = ma.mapped_layers
        layer_name_type = [mapped_layers['name_list'],mapped_layers['type_list']]
        self.fused_model = self.fuse_layers(model =self.model, layer_name_type=layer_name_type)

    @staticmethod
    def fuse_layers(model, layer_name_type):

        layer_name, layer_type = layer_name_type
        layer_name, layer_type = list(layer_name), list(layer_type)
        possible_combinations = [
            ["Conv2d", "BatchNorm2d"],
            ["Conv2d", "BatchNorm2d", "ReLU"],
            ["Conv2d", "ReLU"],
            ["Linear", "ReLU"],
            ["BatchNorm2d", "ReLU"]
        ]

        # Initialize containers for the current sequence of layers being analyzed and the final list of fusible layers
        current_streak = ([], [])
        fusable_layers = []

        # Reverse the lists to use pop operation efficiently
        layer_names = list(reversed([name[6:] for name in layer_name]))
        layer_types = list(reversed(layer_type))

        # Process each layer
        while layer_types:
            current_type = layer_types.pop()
            current_name = layer_names.pop()
            current_streak[0].append(current_type)
            current_streak[1].append(current_name)

            # Check if the current sequence is in the possible combinations
            if current_streak[0] in possible_combinations:
                # Check and handle the case when the current streak can potentially be extended
                if len(current_streak[0]) == 2 and layer_types:
                    next_type = layer_types.pop()
                    next_name = layer_names.pop()
                    if current_streak[0] + [next_type] in possible_combinations:
                        fusable_layers.append(current_streak[1] + [next_name])
                        current_streak = ([], [])
                    else:
                        layer_types.append(next_type)
                        layer_names.append(next_name)
                        fusable_layers.append(current_streak[1])
                        current_streak = ([next_type], [next_name])
                else:
                    fusable_layers.append(current_streak[1])
                    current_streak = ([], [])
            elif len(current_streak[0]) > 3:
                # Reset the current streak to the last element if it exceeds the maximum length in combinations
                current_streak = ([current_type], [current_name])

        try: #Try with Custom Fusion
            model_fused = torch.quantization.fuse_modules(model, fusable_layers, inplace=False)
        except:
            model_fused = quantize_fx.fuse_fx(model)

        return model_fused