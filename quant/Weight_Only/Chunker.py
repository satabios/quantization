import torch
from quant.ModelAnalyzer import ModelAnalyzer
from quant.Quantizer import Quantizer
import torch.nn as nn
from tqdm import tqdm

class Chunker(ModelAnalyzer):

    def __init__(self, model, calibiration_data):
        self.model = model
        self.calibiration_data = calibiration_data
        self.mapped_layers = ModelAnalyzer(self.model, self.calibiration_data).mapped_layers
        self.interested_layers = []
        self.chunk()

    def replace_modules(self, module, target_class, look_out_for, module_name_to_exclude=""):
        for name, child in module.named_children():
            if isinstance(child, look_out_for) and not \
                    any([x == name for x in module_name_to_exclude]):

                if(target_class=='weights'):
                    affine = ("channel", 0) if isinstance(child, torch.nn.Conv2d) else ("tensor", None)
                    qdict = {'dtype': torch.int8, 'symmetric': True, 'affine': affine[0], 'affine_dim': affine[1]}
                    q_params = {'weights': qdict }
                    qlayer = Quantizer.from_float(module=child, data_metry=q_params, quantize_output=False)
                    setattr(module, name, qlayer)

            else:
                # Recursively call the function for nested modules
                self.replace_modules(child, target_class, look_out_for, module_name_to_exclude)

    def weight_quantize(self):
        self.replace_modules(module=self.model, target_class='weights', look_out_for = (torch.nn.Conv2d, torch.nn.Linear))


    def chunk(self):

        self.weight_quantize()

