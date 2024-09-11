# Symentric Quantization
from quant.Qop import Qop
import torch
import torch.nn.functional as F


# dtypes = [torch.int8,torch.int16, torch.int32, torch.int64, torch.bfloat16, torch.float16]
# symmentricies = [True, False]
#
#
# original_tensor = [torch.randn(16, 3, 32, 32), torch.randn(2048,512) ]
#
# for symentric in symmentricies:
#     for dtype in dtypes:
#         for tensor in original_tensor:
#
#             print("\n ======================================================== \n")
#             print(f"Quantization to {dtype}, Symmentric: {symentric}")
#             quantizer = Quantizer(dtype=dtype, symentric=symentric)
#             quantized_tensor = quantizer.quantize(tensor)
#             dequantized_tensor = quantizer.dequantize(quantized_tensor)
#             print(f"quantized_tensor  dtype: {quantized_tensor.dtype}")
#
#             print("MSE:", (dequantized_tensor - tensor).square().mean())
#             print("Scale:", quantizer.scales, "Zero:", quantizer.zero_point)


from models import VGG


def asymmetric_channel_wise_quantization(tensor):
    quantizer = Qop(dtype=torch.int8, symentric=False, affine='channel', affine_dim=0)
    quantized_tensor = quantizer.quantize(tensor)
    dequantized_tensor = quantizer.dequantize(quantized_tensor)
    mse = F.mse_loss(tensor, dequantized_tensor)
    return mse


def symmetric_channel_wise_quantization(tensor):
    quantizer = Qop(dtype=torch.int8, symentric=True, affine='channel', affine_dim=0)
    quantized_tensor = quantizer.quantize(tensor)
    dequantized_tensor = quantizer.dequantize(quantized_tensor)
    mse = F.mse_loss(tensor, dequantized_tensor)
    return mse


def symmetric_tensor_wise_quantization(tensor):
    quantizer = Qop(dtype=torch.int8, symentric=True, affine='tensor')
    quantized_tensor = quantizer.quantize(tensor)
    dequantized_tensor = quantizer.dequantize(quantized_tensor)
    mse = F.mse_loss(tensor, dequantized_tensor)
    return mse


def asymmetric_tensor_wise_quantization(tensor):
    quantizer = Qop(dtype=torch.int8, symentric=False, affine='tensor')
    quantized_tensor = quantizer.quantize(tensor)
    dequantized_tensor = quantizer.dequantize(quantized_tensor)
    mse = F.mse_loss(tensor, dequantized_tensor)
    return mse


def compare_quantization_methods(tensor):
    methods = [
        ("Asymmetric Channel-wise", asymmetric_channel_wise_quantization),
        ("Symmetric Channel-wise", symmetric_channel_wise_quantization),
        ("Symmetric Tensor-wise", symmetric_tensor_wise_quantization),
        ("Asymmetric Tensor-wise", asymmetric_tensor_wise_quantization)
    ]

    mse_results = [(name, func(tensor)) for name, func in methods]
    best_method = min(mse_results, key=lambda x: x[1])

    return best_method


# Load the VGG model
vgg = VGG()
vgg.load_state_dict(torch.load("../vgg.cifar.pretrained.pth"))

# Iterate through the modules and quantize weights
for module_name, module in vgg.named_modules():
    if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
        tensor = module.weight
        best_method, mse = compare_quantization_methods(tensor)
        print(f"Layer: {module_name}")
        print(f"Best quantization method: {best_method}")
        print(f"Lowest MSE: {mse}")
        print("-" * 50)

