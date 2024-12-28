# train_5ch.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset_5ch import MultiModal3DDataset
from model_5ch import Simple3DCNN_5ch

def train_model(model, dataloader, criterion, optimizer, device='cuda', num_epochs=10):
    model.to(device)
    best_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)  # (B,2)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), "best_3dcnn_5ch.pth")
            print(f"  Saved best model at epoch {epoch+1} with loss {best_loss:.4f}")


if __name__ == "__main__":
    # 1) 先假设您已经生成了 "sample_list.csv"
    #    里面包含 ct_path, pet_path, mr_dwi, mr_t1, mr_dynamic, label
    csv_path = "./sample_list.csv"

    # 2) 可选 transforms (去掉 Normalize2D, 或自定义3D)
    #    如果您仅做 simple [0,1] 归一化，则这里可不再处理
    transform = transforms.Compose([])  # 不做额外 transform

    dataset = MultiModal3DDataset(csv_path, transform=transform, output_shape=(64,64,64))
    dataloader = DataLoader(dataset, batch_size=30, shuffle=True, num_workers=0)

    # 3) 初始化模型, loss, optimizer
    model = Simple3DCNN_5ch(in_channels=5, num_classes=2)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # 4) 训练
    train_model(model, dataloader, criterion, optimizer, device='cuda', num_epochs=10)
