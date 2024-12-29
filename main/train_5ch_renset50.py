import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset_5ch import MultiModal3DDataset, my_collate_fn
from model_5ch_resnet50 import SlowR50_5ch


def train_model(model, dataloader, criterion, optimizer, device='cuda', num_epochs=10):
    model.to(device)
    best_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels, _ in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)  # => (B, 2)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

        # 保存最优
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), "best_slow_r50_5ch.pth")
            print(f"  Saved best model at epoch {epoch+1} with loss {best_loss:.4f}")

if __name__ == "__main__":
    csv_path = "./sample_list.csv"
    dataset = MultiModal3DDataset(csv_path, transform=None, output_shape=(64,64,64))
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4, collate_fn=my_collate_fn)

    model = SlowR50_5ch(in_channels=5, num_classes=2, pretrained=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=3e-4)

    train_model(model, dataloader, criterion, optimizer, device='cuda', num_epochs=50)
