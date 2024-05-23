import torch
import torch.nn as nn
import re
import numpy as np
from collections import OrderedDict
from dataset import dataloader

import copy

import copy


class ModelAnalyzer:
    """
    model_layer: This key holds the layers of the model as per their original order.
    Conv2d_BatchNorm2d_ReLU: Stores sequences of layers where a Conv2d layer is followed by BatchNorm2d and then ReLU activation.
    Conv2d_BatchNorm2d: Stores sequences of layers where a Conv2d layer is followed by BatchNorm2d.
    Linear_ReLU: Stores sequences of layers where a Linear layer is followed by ReLU activation.
    Linear_BatchNorm1d: Stores sequences of layers where a Linear layer is followed by BatchNorm1d.
    name_type_shape: This contains information about each layer, including its name, type, and shape.
    name_list: A list of names of all layers.
    type_list: A list of types of all layers.
    qat_layers: Stores layers that are candidates for Quantization-Aware Training (QAT). Specifically, layers from the keys Conv2d_BatchNorm2d_ReLU and Conv2d_BatchNorm2d are considered for QAT.
    model_summary: Summary information about the model, including layer types, input and output shapes, and parameters.
    catcher: This holds detailed information about layers, including their names, types, input data shapes (x), weights (w), and output data shapes (y).
    fusion_layers: Dictionary containing different types of fusion layers such as Conv2d_BatchNorm2d_ReLU, Conv2d_BatchNorm2d, etc., along with their respective layer sequences.
    sequences: Identified layer sequences based on predefined patterns like 'Conv2d_Linear', 'Linear_Linear', etc. Each sequence is associated with its matching layer names.
    """
    def __init__(self, model):
        self.model = model
        self.mapped_layers = OrderedDict()
        self.layer_mapping(self.model)

    def name_fixer(self, names):
        """
        Fix the names by removing the indices in square brackets.
        Args:
        names (list): List of names.

        Returns:
        list: List of fixed names.
        """
        return_list = []
        for string in names:
            matches = re.finditer(r'\.\[(\d+)\]', string)
            pop_list = [m.start(0) for m in matches]
            pop_list.sort(reverse=True)
            if len(pop_list) > 0:
                string = list(string)
                for pop_id in pop_list:
                    string.pop(pop_id)
                string = ''.join(string)
            return_list.append(string)
        return return_list

    def get_all_layers(self, model, parent_name=''):
        layers = []
        for name, module in model.named_children():
            full_name = f"{parent_name}.{name}" if parent_name else name
            test_name = "model." + full_name
            try:
                eval(test_name)
                layers.append((full_name, module))
            except:
                layers.append((self.reformat_layer_name(full_name), module))
            if isinstance(module, nn.Module):
                layers.extend(self.get_all_layers(module, parent_name=full_name))
        return layers

    def reformat_layer_name(self, str_data):
        try:
            split_data = str_data.split('.')
            for ind in range(len(split_data)):
                data = split_data[ind]
                if (data.isdigit()):
                    split_data[ind] = "[" + data + "]"
            final_string = '.'.join(split_data)

            iters_a = re.finditer(r'[a-zA-Z]\.\[', final_string)
            indices = [m.start(0) + 1 for m in iters_a]
            iters = re.finditer(r'\]\.\[', final_string)
            indices.extend([m.start(0) + 1 for m in iters])

            final_string = list(final_string)
            final_string = [final_string[i] for i in range(len(final_string)) if i not in indices]

            str_data = ''.join(final_string)

        except:
            pass

        return str_data

    def summary_string_fixed(self, model, all_layers, input_size, model_name=None, batch_size=-1, dtypes=None):
        if dtypes is None:
            dtypes = [torch.FloatTensor] * len(input_size)

        def register_hook(module, module_idx):
            def hook(module, input, output):
                nonlocal module_idx
                m_key = all_layers[module_idx][0]
                m_key = model_name + "." + m_key

                try:
                    eval(m_key)
                except:
                    m_key = self.name_fixer([m_key])[0]

                summary[m_key] = OrderedDict()
                summary[m_key]["type"] = str(type(module)).split('.')[-1][:-2]
                summary[m_key]["x"] = input
                summary[m_key]["input_shape"] = list(input[0].size())
                summary[m_key]["input_shape"][0] = batch_size

                if isinstance(output, (list, tuple)):
                    summary[m_key]["y"] = [
                        [-1] + list(o)[1:] for o in output
                    ]
                    summary[m_key]["output_shape"] = [
                        [-1] + list(o.size())[1:] for o in output
                    ]
                else:
                    summary[m_key]["y"] = list(output)

                    summary[m_key]["output_shape"] = list(output.size())
                    summary[m_key]["output_shape"][0] = batch_size

                params = 0
                if hasattr(module, "weight") and hasattr(module.weight, "size"):
                    summary[m_key]["w"] = module.weight
                    params += torch.prod(torch.LongTensor(list(module.weight.size())))
                    summary[m_key]["trainable"] = module.weight.requires_grad
                    summary[m_key]["weight_shape"] = module.weight.shape
                if hasattr(module, "bias") and hasattr(module.bias, "size"):
                    summary[m_key]["b"] = module.bias
                    params += torch.prod(torch.LongTensor(list(module.bias.size())))
                summary[m_key]["nb_params"] = params

            if (
                    not isinstance(module, nn.Sequential)
                    and not isinstance(module, nn.ModuleList)
            ):
                hooks.append(module.register_forward_hook(hook))

        if isinstance(input_size, tuple):
            input_size = [input_size]

        model_device = next(iter(model.parameters())).device
        x, _ = next(iter(dataloader['test']))
        x = x.to(model_device)

        summary = OrderedDict()
        hooks = []

        for module_idx, (layer_name, module) in enumerate(all_layers):
            register_hook(module, module_idx)

        model(x)

        for h in hooks:
            h.remove()

        return summary

    def find_patterns(self, model):

        # name_list = self.mapped_layers['name_type_shape'][:, 0]
        # type_list = self.mapped_layers['name_type_shape'][:, 1]
        #
        # conv_list = []
        # for t_p in self.mapped_layers['catcher']['type_list']:
        #     if (t_p == 'Conv2d'):
        #         conv_list.append(t_p)
        # self.mapped_layers['conv_list'] = conv_list
        #
        # conv_bn_idx = [index for index, layer in enumerate(self.mapped_layers['type_list']) if layer in ['Conv2d', 'BatchNorm2d']]
        #
        # arr = []
        # for l_n, t_p in zip(name_list[conv_bn_idx], type_list[conv_bn_idx]):
        #     arr.append((l_n, t_p))
        # self.mapped_layers['conv_bn_l_n_t_p'] = np.asarray(arr)
        #
        # sorting_pairs = []
        # idx = 0
        # while idx < len(self.mapped_layers['conv_bn_l_n_t_p']):
        #     layer_list = self.mapped_layers['conv_bn_l_n_t_p'][idx:idx + 3][:,1]
        #
        #     if (np.array_equal(layer_list, ['Conv2d', 'BatchNorm2d', 'Conv2d'])):
        #         sorting_pairs.append(self.mapped_layers['conv_bn_l_n_t_p'][idx:idx + 3][:,0].tolist())
        #         idx += 2
        #     elif (np.array_equal(layer_list, ['Conv2d', 'Conv2d'])):
        #         sorting_pairs.append(self.mapped_layers['conv_bn_l_n_t_p'][idx:idx + 2][:,0].tolist())
        #         idx += 1
        #     else:
        #         idx += 1


        sequences = [['Conv2d', 'Linear'], ['Linear', 'Linear'], ['Conv2d', 'Conv2d'], ['Conv2d', 'BatchNorm2d', 'Conv2d']]
        sequences_identifier = ['Conv2d_Linear', 'Linear_Linear', 'Conv2d_Conv2d','Conv2d_BatchNorm2d_Conv2d']
        sek = {}
        layer_types = self.mapped_layers['catcher']['type_list'][:,1]
        layer_names = self.mapped_layers['catcher']['name_list']
        def matches(sublist, seq):
            return sublist == seq

        for idx, seq in enumerate(sequences):
            seq_len = len(seq)
            results = []

            # Loop through the layer types to find matching subsequences
            for i in range(len(layer_types) - seq_len + 1):
                if matches(layer_types[i:i + seq_len].tolist(), seq):
                    matched_names = layer_names[i:i + seq_len]
                    results.append(matched_names)
            sek[sequences_identifier[idx]] = results

        self.mapped_layers['sequences'] = sek


    def layer_mapping(self, model):
        all_layers = self.get_all_layers(model)
        x, y = next(iter(dataloader['test']))
        model_summary = self.summary_string_fixed(model, all_layers, x.shape, model_name='model')

        name_type_shape = []
        for key in model_summary.keys():
            data = model_summary[key]
            if ("weight_shape" in data.keys()):
                name_type_shape.append([key, data['type'], data['weight_shape'][0]])
            else:
                name_type_shape.append([key, data['type'], 0])
        name_type_shape = np.asarray(name_type_shape)

        name_list = name_type_shape[:, 0]

        r_name_list = np.asarray(name_list)
        random_picks = np.random.randint(0, len(r_name_list), 10)
        test_name_list = r_name_list[random_picks]
        eval_hit = False
        for layer in test_name_list:
            try:
                eval(layer)

            except:
                eval_hit = True
                break
        if (eval_hit):
            fixed_name_list = self.name_fixer(r_name_list)
            name_type_shape[:, 0] = fixed_name_list

        layer_types = name_type_shape[:, 1]
        layer_shapes = name_type_shape[:, 2]
        mapped_layers = {'model_layer': [], 'Conv2d_BatchNorm2d_ReLU': [], 'Conv2d_BatchNorm2d': [], 'Linear_ReLU': [],
                         'Linear_BatchNorm1d': []}

        def detect_sequences(lst):
            fusing_layers = [
                'Conv2d',
                'BatchNorm2d',
                'ReLU',
                'Linear',
                'BatchNorm1d',
            ]

            i = 0
            while i < len(lst):

                if i + 2 < len(lst) and [l for l in lst[i: i + 3]] == [
                    fusing_layers[0],
                    fusing_layers[1],
                    fusing_layers[2],
                ]:

                    mapped_layers['Conv2d_BatchNorm2d_ReLU'].append(
                        np.take(name_list, [i for i in range(i, i + 3)]).tolist()
                    )
                    i += 3

                elif i + 1 < len(lst) and [l for l in lst[i: i + 2]] == [
                    fusing_layers[0],
                    fusing_layers[1],
                ]:

                    mapped_layers['Conv2d_BatchNorm2d'].append(
                        np.take(name_list, [i for i in range(i, i + 2)]).tolist()
                    )
                    i += 2
                elif i + 1 < len(lst) and [l for l in lst[i: i + 2]] == [
                    fusing_layers[0],
                    fusing_layers[2],
                ]:

                    mapped_layers['Conv2d_ReLU'].append(
                        np.take(name_list, [i for i in range(i, i + 2)]).tolist()
                    )
                    i += 2

                elif i + 1 < len(lst) and [l for l in lst[i: i + 2]] == [
                    fusing_layers[3],
                    fusing_layers[2],
                ]:
                    mapped_layers['Linear_ReLU'].append(
                        np.take(name_list, [i for i in range(i, i + 2)]).tolist()
                    )
                    i += 2
                elif i + 1 < len(lst) and [l for l in lst[i: i + 2]] == [
                    fusing_layers[3],
                    fusing_layers[4],
                ]:
                    mapped_layers['Linear_BatchNorm1d'].append(
                        np.take(name_list, [i for i in range(i, i + 2)]).tolist()
                    )
                    i += 2
                elif i + 1 < len(lst) and [l for l in lst[i: i + 2]] == [
                    fusing_layers[3],
                    fusing_layers[2],
                ]:
                    mapped_layers['Linear_ReLU'].append(
                        np.take(name_list, [i for i in range(i, i + 2)]).tolist()
                    )
                    i += 2
                else:
                    i += 1



        detect_sequences(layer_types)

        for keys, value in mapped_layers.items():
            mapped_layers[keys] = np.asarray(mapped_layers[keys])

        mapped_layers['name_type_shape'] = name_type_shape
        mapped_layers['name_list'] = mapped_layers['name_type_shape'][:, 0]
        mapped_layers['type_list'] = mapped_layers['name_type_shape'][:, 1]

        # CWP
        keys_to_lookout = ['Conv2d_BatchNorm2d_ReLU', 'Conv2d_BatchNorm2d']
        pruning_layer_of_interest, qat_layer_of_interest = [], []

        # CWP or QAT Fusion Layers
        for keys in keys_to_lookout:
            data = mapped_layers[keys]
            if (len(data) != 0):
                qat_layer_of_interest.append(data)
        mapped_layers['qat_layers'] = qat_layer_of_interest
        mapped_layers['model_summary'] = model_summary

        name_list = mapped_layers['name_type_shape'][:, 0]
        layer_name_list, layer_type_list = [], []
        w, x, y, b = [], [], [], []
        for layer_name in name_list:
            layer = eval(layer_name)
            if (isinstance(layer, (nn.Conv2d, nn.Linear))):
                layer_name_list.append(layer_name)
                layer_type_list.append((type(layer),str(type(layer)).split('.')[-1][:-2]))
                x.append(mapped_layers['model_summary'][layer_name]['x'][0])
                w.append(mapped_layers['model_summary'][layer_name]['w'])
                y.append(torch.stack(mapped_layers['model_summary'][layer_name]['y']))



        mapped_layers['catcher'] = {'name_list': layer_name_list,'type_list':np.asarray(layer_type_list), 'x': x, 'w': w, 'y': y}

        fusion_layers = ['Conv2d_BatchNorm2d_ReLU', 'Conv2d_BatchNorm2d', 'Conv2d_ReLU', 'Linear_BatchNorm1d', 'Linear_ReLU']
        fusion_dict = {}
        for f_l in fusion_layers:
            if(f_l in mapped_layers.keys()):
                fusion_dict.update({f_l: mapped_layers[f_l]})
        mapped_layers['fusion_layers'] = fusion_dict


        self.mapped_layers = mapped_layers
        self.find_patterns(model)

        layers_to_remove  = ['model_layer', 'Conv2d_BatchNorm2d_ReLU', 'Conv2d_BatchNorm2d', 'Linear_ReLU', 'Linear_BatchNorm1d',
                   'name_type_shape']
        for key in layers_to_remove:
            self.mapped_layers.pop(key, None)



