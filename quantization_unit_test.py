import unittest
import torch
from quantization import Quantizer

class TestQuantizer(unittest.TestCase):

    def setUp(self):
        self.dtype = torch.int8

    def test_per_tensor_symmetric(self):
        tensor = torch.randn(4, 4)
        quantizer = Quantizer(tensor, self.dtype, symentric=True)
        quantized_tensor = quantizer.quantize()
        dequantized_tensor = quantizer.dequantize(quantized_tensor)
        self.assertEqual(tensor.shape, quantized_tensor.shape)
        self.assertEqual(tensor.shape, dequantized_tensor.shape)
        self.assertAlmostEqual(tensor.mean().item(), dequantized_tensor.mean().item(), places=1)

    def test_per_tensor_asymmetric(self):
        tensor = torch.randn(4, 4)
        quantizer = Quantizer(tensor, self.dtype, symentric=False)
        quantized_tensor = quantizer.quantize()
        dequantized_tensor = quantizer.dequantize(quantized_tensor)
        self.assertEqual(tensor.shape, quantized_tensor.shape)
        self.assertEqual(tensor.shape, dequantized_tensor.shape)
        self.assertAlmostEqual(tensor.mean().item(), dequantized_tensor.mean().item(), places=1)

    def test_per_dimension_symmetric(self):
        tensor = torch.randn(4, 4)
        quantizer = Quantizer(tensor, self.dtype, symentric=True, per='dim', per_dim=1)
        quantized_tensor = quantizer.quantize()
        dequantized_tensor = quantizer.dequantize(quantized_tensor)
        self.assertEqual(tensor.shape, quantized_tensor.shape)
        self.assertEqual(tensor.shape, dequantized_tensor.shape)
        self.assertAlmostEqual(tensor.mean().item(), dequantized_tensor.mean().item(), places=1)

    def test_per_dimension_asymmetric(self):
        tensor = torch.randn(4, 4)
        quantizer = Quantizer(tensor, self.dtype, symentric=False, per='dim', per_dim=1)
        quantized_tensor = quantizer.quantize()
        dequantized_tensor = quantizer.dequantize(quantized_tensor)
        self.assertEqual(tensor.shape, quantized_tensor.shape)
        self.assertEqual(tensor.shape, dequantized_tensor.shape)
        self.assertAlmostEqual(tensor.mean().item(), dequantized_tensor.mean().item(), places=1)

    def test_per_group_symmetric(self):
        tensor = torch.randn(4, 8)
        quantizer = Quantizer(tensor, self.dtype, symentric=True, per='group', group_size=8)
        quantized_tensor = quantizer.quantize()
        dequantized_tensor = quantizer.dequantize(quantized_tensor)
        self.assertEqual(tensor.shape, quantized_tensor.shape)
        self.assertEqual(tensor.shape, dequantized_tensor.shape)
        self.assertAlmostEqual(tensor.mean().item(), dequantized_tensor.mean().item(), places=1)

    def test_per_group_asymmetric(self):
        tensor = torch.randn(4, 8)
        quantizer = Quantizer(tensor, self.dtype, symentric=False, per='group', group_size=2)
        quantized_tensor = quantizer.quantize()
        dequantized_tensor = quantizer.dequantize(quantized_tensor)
        self.assertEqual(tensor.shape, quantized_tensor.shape)
        self.assertEqual(tensor.shape, dequantized_tensor.shape)
        self.assertAlmostEqual(tensor.mean().item(), dequantized_tensor.mean().item(), places=1)

if __name__ == '__main__':
    unittest.main()
