
from quantization import Quantizer
import torch



original_tensor = torch.tensor([ [ 0.6967,  0.1568,  0.1024, -0.9279],
        [ 1.8730,  0.8987,  0.8966,  0.1578],
        [-0.4760,  0.6169, -1.6372,  0.2544],
        [-0.1727,  0.7768,  0.0392,  0.2127] ])


quantizer = Quantizer(tensor=original_tensor, dtype=torch.int8)
print(original_tensor)
quantized_tensor = quantizer.quantize()
print(quantized_tensor)
dequantized_tensor = quantizer.dequantize(quantized_tensor)
print(dequantized_tensor)
print("MSE:", (dequantized_tensor - original_tensor).square().mean())
print("Scale:", quantizer.scales, "Zero:", quantizer.zero_point)