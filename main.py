import torch
from quantization import Quantizer
from visualizer import visualizer

###################################### L2###############################################################


original_tensor = torch.tensor([[-0.6250, -1.6061,  2.4067,  0.2539],
        [-1.3833, -0.8534, -0.4792, -3.2985],
        [-0.1950,  0.9677,  0.5905,  0.5262],
        [-0.2198,  0.0532,  1.1031, -1.2016]])


quantizer = Quantizer(tensor=original_tensor, dtype=torch.int8, symentric=False)
print(original_tensor)
quantized_tensor = quantizer.quantize()
print(quantized_tensor)
dequantized_tensor = quantizer.dequantize(quantized_tensor)
print(dequantized_tensor)
print("MSE:", (dequantized_tensor - original_tensor).square().mean())
print("Scale:", quantizer.scales, "Zero:", quantizer.zero_point)


#########################################L3 ############################################################

test_tensor=torch.tensor(
    [[191.6, -13.5, 728.6],
     [92.14, 295.5,  -184],
     [0,     684.6, 245.5]]
)

dtype=torch.int8
quantizer = Quantizer(tensor=test_tensor, per='dim',per_dim=0, dtype=dtype, symentric=True)
print(test_tensor)
print("scales:", quantizer.scales)
quantized_tensor = quantizer.quantize()
print(quantized_tensor)
dequantized_tensor = quantizer.dequantize(quantized_tensor)

print(dequantized_tensor)
print("MSE:", (dequantized_tensor - test_tensor).square().mean())
print("Scale:", quantizer.scales, "Zero:", quantizer.zero_point)


dtype=torch.int8
quantizer = Quantizer(tensor=test_tensor, per='dim',per_dim=1, dtype=dtype, symentric=True)
print(test_tensor)
print("scales:", quantizer.scales)
quantized_tensor = quantizer.quantize()
print(quantized_tensor)
dequantized_tensor = quantizer.dequantize(quantized_tensor)

print(dequantized_tensor)
print("MSE:", (dequantized_tensor - test_tensor).square().mean())
print("Scale:", quantizer.scales, "Zero:", quantizer.zero_point)

