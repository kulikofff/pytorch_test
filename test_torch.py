import torch
import unittest

class TestTorch(unittest.TestCase):
    def test_tensor_creation(self):
        \"\"\"Проверка создания тензора.\"\"\"
        x = torch.tensor([1.0, 2.0, 3.0])
        self.assertEqual(x.shape, (3,))

    def test_matmul(self):
        \"\"\"Проверка матричного умножения.\"\"\"
        a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        b = torch.tensor([[5.0], [6.0]])
        res = torch.matmul(a, b)
        expected = torch.tensor([[17.0], [22.0]])
        self.assertTrue(torch.allclose(res, expected))

    def test_cuda_availability(self):
        \"\"\"Проверка доступности CUDA (если поддерживается).\"\"\"
        if torch.cuda.is_available():
            gpu = torch.cuda.current_device()
            self.assertIsNotNone(gpu)

if __name__ == '__main__':
    unittest.main()
