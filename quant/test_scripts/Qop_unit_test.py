# import unittest
# import torch
from Qop import Qop
import unittest
import torch
import torch.nn.functional as F

class TestQuantizer(unittest.TestCase):

    def setUp(self):
        self.dtype = torch.int8

    def test_per_tensor_symmetric(self):
        tensor = torch.randn(4, 4)
        quantizer = Qop(tensor, self.dtype, symentric=True)
        quantized_tensor = quantizer.quantize()
        dequantized_tensor = quantizer.dequantize(quantized_tensor)
        mse = F.mse_loss(tensor, dequantized_tensor)
        self.assertLess(mse.item(), 0.1)  # Assuming an acceptable error threshold

    def test_per_tensor_asymmetric(self):
        tensor = torch.randn(4, 4)
        quantizer = Qop(tensor, self.dtype, symentric=False)
        quantized_tensor = quantizer.quantize()
        dequantized_tensor = quantizer.dequantize(quantized_tensor)
        mse = F.mse_loss(tensor, dequantized_tensor)
        self.assertLess(mse.item(), 0.1)

    def test_per_dimension_symmetric(self):
        tensor = torch.randn(4, 4)
        quantizer = Qop(tensor, self.dtype, symentric=True, per='dim', per_dim=1)
        quantized_tensor = quantizer.quantize()
        dequantized_tensor = quantizer.dequantize(quantized_tensor)
        mse = F.mse_loss(tensor, dequantized_tensor)
        self.assertLess(mse.item(), 0.1)

    def test_per_dimension_asymmetric(self):
        tensor = torch.randn(4, 4)
        quantizer = Qop(tensor, self.dtype, symentric=False, per='dim', per_dim=1)
        quantized_tensor = quantizer.quantize()
        dequantized_tensor = quantizer.dequantize(quantized_tensor)
        mse = F.mse_loss(tensor, dequantized_tensor)
        self.assertLess(mse.item(), 0.1)

    def test_per_group_symmetric(self):
        tensor = torch.randn(4, 8)
        quantizer = Qop(tensor, self.dtype, symentric=True, per='group', group_size=8)
        quantized_tensor = quantizer.quantize()
        dequantized_tensor = quantizer.dequantize(quantized_tensor)
        mse = F.mse_loss(tensor, dequantized_tensor)
        self.assertLess(mse.item(), 0.1)

    def test_per_group_asymmetric(self):
        tensor = torch.randn(4, 8)
        quantizer = Qop(tensor, self.dtype, symentric=False, per='group', group_size=4)
        quantized_tensor = quantizer.quantize()
        dequantized_tensor = quantizer.dequantize(quantized_tensor)
        mse = F.mse_loss(tensor, dequantized_tensor)
        self.assertLess(mse.item(), 0.1)

if __name__ == '__main__':
    unittest.main()
