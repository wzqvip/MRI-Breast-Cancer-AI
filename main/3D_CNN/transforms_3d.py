import torch

class Normalize3D:
    """
    对 3D 张量 (C, D, H, W) 在每个通道做 (x - mean[c]) / std[c].
    """
    def __init__(self, mean, std):
        self.mean = mean  # list of floats, 长度=通道数
        self.std = std    # list of floats, 长度=通道数

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        # tensor 形状: (C, D, H, W)
        for c in range(tensor.shape[0]):
            tensor[c] = (tensor[c] - self.mean[c]) / (self.std[c])
        return tensor
