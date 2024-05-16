import torch
from quantization import Quantizer
from visualizer import visualizer

###################################### L2###############################################################

# Symentric Quantization
# print("Symentric Quantization")
# original_tensor = torch.tensor([[-0.6250, -1.6061,  2.4067,  0.2539],
#         [-1.3833, -0.8534, -0.4792, -3.2985],
#         [-0.1950,  0.9677,  0.5905,  0.5262],
#         [-0.2198,  0.0532,  1.1031, -1.2016]])


# quantizer = Quantizer(tensor=original_tensor, dtype=torch.int8, symentric=True)
# print(original_tensor)
# quantized_tensor = quantizer.quantize()
# print(quantized_tensor)
# dequantized_tensor = quantizer.dequantize(quantized_tensor)
# print(dequantized_tensor)
# print("MSE:", (dequantized_tensor - original_tensor).square().mean())
# print("Scale:", quantizer.scales, "Zero:", quantizer.zero_point)

# # Affie/ASymentric Quantization
# print("Asymentric Quantization")

# original_tensor = torch.tensor([[-0.6250, -1.6061,  2.4067,  0.2539],
#         [-1.3833, -0.8534, -0.4792, -3.2985],
#         [-0.1950,  0.9677,  0.5905,  0.5262],
#         [-0.2198,  0.0532,  1.1031, -1.2016]])


# quantizer = Quantizer(tensor=original_tensor, dtype=torch.int8, symentric=False)
# print(original_tensor)
# quantized_tensor = quantizer.quantize()
# print(quantized_tensor)
# dequantized_tensor = quantizer.dequantize(quantized_tensor)
# print(dequantized_tensor)
# print("MSE:", (dequantized_tensor - original_tensor).square().mean())
# print("Scale:", quantizer.scales, "Zero:", quantizer.zero_point)



#########################################L3 ############################################################

test_tensor=torch.tensor(
    [[191.6, -13.5, 728.6],
     [92.14, 295.5,  -184],
     [0,     684.6, 245.5]]
)

# dtype=torch.int8
# quantizer = Quantizer(tensor=test_tensor, per='dim',per_dim=0, dtype=dtype, symentric=True)
# print(test_tensor)
# print("scales:", quantizer.scales)
# quantized_tensor = quantizer.quantize()
# print(quantized_tensor)
# dequantized_tensor = quantizer.dequantize(quantized_tensor)

# print(dequantized_tensor)
# print("MSE:", (dequantized_tensor - test_tensor).square().mean())
# print("Scale:", quantizer.scales, "Zero:", quantizer.zero_point)


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




######################################## L4 ########################################


# test_tensor = torch.tensor([[ 0.7734, -1.5156,  0.0620,  0.4531,  0.5469, -0.8047, -1.4062, -0.8750],
#         [-1.9297, -0.9219, -0.3164, -0.9531, -0.1055, -1.3984,  0.8281,  1.9844],
#         [ 0.4746,  0.0513,  0.4395,  0.1953, -1.2344, -0.1777, -2.0000, -0.9961],
#         [ 0.7500,  2.1406, -0.0564,  1.2969,  0.0591,  0.8008, -1.0156, -0.8672]],
#        dtype=torch.bfloat16)

# dtype=torch.int8
# quantizer = Quantizer(tensor=test_tensor, per='dim',per_dim=0, dtype=dtype, symentric=True)
# print(test_tensor)
# print("scales:", quantizer.scales)
# quantized_tensor = quantizer.quantize()
# print(quantized_tensor)
# dequantized_tensor = quantizer.dequantize(quantized_tensor)

# print(dequantized_tensor)
# print("MSE:", (dequantized_tensor - test_tensor).square().mean())
# print("Scale:", quantizer.scales, "Zero:", quantizer.zero_point)
