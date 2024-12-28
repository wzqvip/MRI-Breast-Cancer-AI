import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

from dataset_5ch import MultiModal3DDataset
from model_5ch import Simple3DCNN_5ch
from transforms_3d import Normalize3D

def train_model(model, dataloader, criterion, optimizer, device='cuda', num_epochs=10):
    model.to(device)
    best_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)  # (B,2)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {epoch_loss:.4f}")

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), "best_3dcnn_5ch.pth")
            print(f"  Saved best model at epoch {epoch+1} with loss {best_loss:.4f}")

if __name__ == "__main__":
    csv_path = "./sample_list_small_avg.csv"  # 上一步生成的CSV (含5种MR信息)
    
    # 数据变换示例: 统一对5通道做 Normalize
    # transform = transforms.Compose([
    #     transforms.Normalize(mean=[0.5]*5, std=[0.5]*5)
    # ])
    transform = None
    transform = transforms.Compose([
        # 5 通道分别的均值方差，如果都相同就 [0.5,0.5,0.5,0.5,0.5] 等
        Normalize3D(mean=[0.5]*5, std=[0.5]*5)
    ])
    
    # 构建Dataset & DataLoader
    dataset = MultiModal3DDataset(csv_path, transform=transform, output_shape=(64,64,64))

    # 为了快速测试，这里只取前10个样本
    # dataset = Subset(dataset, range(50))

    # dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
    dataloader = DataLoader(dataset, batch_size=20, shuffle=True, num_workers=0)

    # 初始化网络, in_channels=5
    model = Simple3DCNN_5ch(in_channels=5, num_classes=2)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    train_model(model, dataloader, criterion, optimizer, device='cuda', num_epochs=10)
