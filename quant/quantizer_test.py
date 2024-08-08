# Symentric Quantization
from quantization import Quantizer
import torch



dtypes = [torch.int8,torch.int16, torch.int32, torch.int64, torch.bfloat16, torch.float16]
symmentricies = [True, False]


original_tensor = [torch.randn(16, 3, 32, 32), torch.randn(2048,512) ]

for symentric in symmentricies:
    for dtype in dtypes:
        for tensor in original_tensor:

            print("\n ======================================================== \n")
            print(f"Quantization to {dtype}, Symmentric: {symentric}")
            quantizer = Quantizer(dtype=dtype, symentric=symentric)
            quantized_tensor = quantizer.quantize(tensor)
            dequantized_tensor = quantizer.dequantize(quantized_tensor)
            print(f"quantized_tensor  dtype: {quantized_tensor.dtype}")

            print("MSE:", (dequantized_tensor - tensor).square().mean())
            print("Scale:", quantizer.scales, "Zero:", quantizer.zero_point)
