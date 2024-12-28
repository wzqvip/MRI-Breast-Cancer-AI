import torch
import torch.nn as nn
import torch.nn.functional as F

class SlowR50_5ch(nn.Module):
    """
    使用 `torch.hub.load('facebookresearch/pytorchvideo', 'slow_r50', ...)`
    加载得到的 Net，对其第一层(3通道)和最后一层(400类)进行替换：
      - 第一层 -> in_channels
      - 最后一层 -> num_classes
    """
    def __init__(self, in_channels=5, num_classes=2, pretrained=True):
        super().__init__()
        # 1) 加载 slow_r50
        self.model = torch.hub.load(
            'facebookresearch/pytorchvideo',
            'slow_r50',
            pretrained=pretrained
        )
        
        # 查看网络结构 (可选调试)
        # print(self.model)  
        # for name, module in self.model.named_modules():
        #     print(name, module)

        # 2) 替换第一层卷积
        #    blocks[0] 是 ResNetBasicStem, 其中有个 conv: Conv3d(3,64,...)
        stem = self.model.blocks[0]
        old_conv = stem.conv  # Conv3d(3, 64, ...)
        
        new_conv = nn.Conv3d(
            in_channels=in_channels,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=(old_conv.bias is not None),
            dilation=old_conv.dilation,
            groups=old_conv.groups
        )

        # 初始化新 conv 的权重
        with torch.no_grad():
            if in_channels > 3:
                # 拷贝前3个通道的预训练权重
                new_conv.weight[:, :3, ...] = old_conv.weight
                # 其余通道用前3通道的均值(或其他策略)
                new_conv.weight[:, 3:, ...] = torch.mean(old_conv.weight, dim=1, keepdim=True)
            else:
                # 若 in_channels < 3, 可只用前 in_channels 个通道
                new_conv.weight = old_conv.weight

        # 替换到模型
        stem.conv = new_conv
        
        # 3) 替换最后一层
        #    blocks[5] 是 ResNetBasicHead, 其中 (proj): Linear(2048,400)
        head = self.model.blocks[5]
        old_fc = head.proj
        in_features = old_fc.in_features
        
        new_fc = nn.Linear(in_features, num_classes)
        head.proj = new_fc

         # 3) **最关键：替换 pool => 自适应池化**
        # 原: AvgPool3d(kernel_size=(8,7,7)) => 改: AdaptiveAvgPool3d(1)
        head.pool = nn.AdaptiveAvgPool3d(output_size=(1,1,1))

    
    def forward(self, x):
        # x shape: (B, in_channels, T, H, W)
        return self.model(x)
