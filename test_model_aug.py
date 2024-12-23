import unittest
import torch
import torch.nn as nn
from model import Network

class TestModel(unittest.TestCase):
    def setUp(self):
        self.model = Ne()

    def test_total_parameters(self):
        """Test if total parameters are less than 20000"""
        total_params = sum(p.numel() for p in self.model.parameters())
        self.assertLess(total_params, 20000, f"Total parameters ({total_params}) exceed 20000")

    def test_batch_normalization(self):
        """Test if batch normalization is used in the model"""
        has_batchnorm = any(isinstance(module, nn.BatchNorm2d) for module in self.model.modules())
        self.assertTrue(has_batchnorm, "Model should use Batch Normalization")

    def test_dropout(self):
        """Test if dropout is used in the model"""
        has_dropout = any(isinstance(module, nn.Dropout) for module in self.model.modules())
        self.assertTrue(has_dropout, "Model should use Dropout")

    def test_gap(self):
        """Test if model uses Fully Connected layer (not GAP)"""
        has_linear = any(isinstance(module, nn.Linear) for module in self.model.modules())
        self.assertTrue(has_linear, "Model should use Fully Connected layer")

if __name__ == '__main__':
    unittest.main() 
