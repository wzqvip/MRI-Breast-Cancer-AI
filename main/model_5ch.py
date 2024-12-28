import torch
import torch.nn as nn
import torch.nn.functional as F

class Simple3DCNN_5ch(nn.Module):
    """
    输入: (B,5,64,64,64)
    输出: 二分类 logits (B,2)
    """
    def __init__(self, in_channels=5, num_classes=2):
        super(Simple3DCNN_5ch, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool3d(2, 2)
        # 两次pool => 空间 (64->32->16)
        # 通道 => 32
        # Flatten大小 = 32 * 16 * 16 * 16 = 32 * 4096 = 131072
        self.fc1 = nn.Linear(32*16*16*16, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # x: (B,5,64,64,64)
        x = F.relu(self.conv1(x))  # (B,16,64,64,64)
        x = self.pool(x)           # (B,16,32,32,32)
        x = F.relu(self.conv2(x))  # (B,32,32,32,32)
        x = self.pool(x)           # (B,32,16,16,16)

        x = x.view(x.size(0), -1)  # flatten
        x = F.relu(self.fc1(x))
        out = self.fc2(x)
        return out
