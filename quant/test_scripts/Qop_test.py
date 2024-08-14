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
vgg = VGG()
vgg.load_state_dict(torch.load("../vgg.cifar.pretrained.pth"))

tensor = vgg.backbone.conv0.weight
quantizer = Qop(dtype=torch.int8, symentric=False, per='dim', per_dim=0)
quantized_tensor = quantizer.quantize(tensor)
dequantized_tensor = quantizer.dequantize(quantized_tensor)
mse = F.mse_loss(tensor, dequantized_tensor)
print(mse)


quantizer = Qop(dtype=torch.int8, symentric=True, per='dim', per_dim=0)
quantized_tensor = quantizer.quantize(tensor)
dequantized_tensor = quantizer.dequantize(quantized_tensor)
mse = F.mse_loss(tensor, dequantized_tensor)
print(mse)

quantizer = Qop(dtype=torch.int8, symentric=True, per='tensor')
quantized_tensor = quantizer.quantize(tensor)
dequantized_tensor = quantizer.dequantize(quantized_tensor)
mse = F.mse_loss(tensor, dequantized_tensor)
print(mse)