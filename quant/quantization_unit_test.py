# import unittest
# import torch
from quantization import Quantizer
import unittest
import torch
from quantization import Quantizer
import torch.nn.functional as F

class TestQuantizer(unittest.TestCase):

    def setUp(self):
        self.array_size = (2048*2, 2048*4)
        self.dtype = torch.int8

    def test_per_tensor_symmetric(self):
        tensor = torch.randn(self.array_size)
        quantizer = Quantizer(tensor, self.dtype, symentric=True)
        quantized_tensor = quantizer.quantize()
        dequantized_tensor = quantizer.dequantize(quantized_tensor)
        mse = F.mse_loss(tensor, dequantized_tensor)
        self.assertLess(mse.item(), 0.1)  # Assuming an acceptable error threshold

    def test_per_tensor_asymmetric(self):
        tensor = torch.randn(self.array_size)
        quantizer = Quantizer(tensor, self.dtype, symentric=False)
        quantized_tensor = quantizer.quantize()
        dequantized_tensor = quantizer.dequantize(quantized_tensor)
        mse = F.mse_loss(tensor, dequantized_tensor)
        self.assertLess(mse.item(), 0.1)

    def test_per_dimension_symmetric(self):
        tensor = torch.randn(self.array_size)
        quantizer = Quantizer(tensor, self.dtype, symentric=True, per='dim', per_dim=1)
        quantized_tensor = quantizer.quantize()
        dequantized_tensor = quantizer.dequantize(quantized_tensor)
        mse = F.mse_loss(tensor, dequantized_tensor)
        self.assertLess(mse.item(), 0.1)

    def test_per_dimension_asymmetric(self):
        tensor = torch.randn(self.array_size)
        quantizer = Quantizer(tensor, self.dtype, symentric=False, per='dim', per_dim=1)
        quantized_tensor = quantizer.quantize()
        dequantized_tensor = quantizer.dequantize(quantized_tensor)
        mse = F.mse_loss(tensor, dequantized_tensor)
        self.assertLess(mse.item(), 0.1)


    def test_per_group_asymmetric(self):
        tensor = torch.randn(self.array_size)
        quantizer = Quantizer(tensor, self.dtype, symentric=False, per='group', group_size=4)
        quantized_tensor = quantizer.quantize()
        dequantized_tensor = quantizer.dequantize(quantized_tensor)
        mse = F.mse_loss(tensor, dequantized_tensor)
        self.assertLess(mse.item(), 0.1)

if __name__ == '__main__':
    unittest.main()

