import os
import csv
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader

# ==========            自定义模块           ========== #
from dataset_5ch import MultiModal3DDataset, my_collate_fn
from model_5ch_resnet50 import SlowR50_5ch
# ==================================================== #

def train_one_epoch(model, dataloader, criterion, optimizer, device='cuda'):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc="Train", leave=False)
    for batch_idx, (inputs, labels, _) in enumerate(pbar):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

        preds = torch.argmax(outputs, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        batch_loss = loss.item()
        batch_acc = (preds == labels).float().mean().item()
        pbar.set_postfix({
            'batch_loss': f"{batch_loss:.4f}",
            'batch_acc':  f"{batch_acc:.4f}"
        })

    epoch_loss = running_loss / total
    epoch_acc  = correct / total
    return epoch_loss, epoch_acc

def validate_one_epoch(model, dataloader, criterion, device='cuda'):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc="Valid", leave=False)
    with torch.no_grad():
        for batch_idx, (inputs, labels, _) in enumerate(pbar):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)

            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            batch_loss = loss.item()
            batch_acc = (preds == labels).float().mean().item()
            pbar.set_postfix({
                'batch_loss': f"{batch_loss:.4f}",
                'batch_acc':  f"{batch_acc:.4f}"
            })

    epoch_loss = running_loss / total
    epoch_acc  = correct / total
    return epoch_loss, epoch_acc


def train_and_eval(model, 
                   train_dataloader, 
                   val_dataloader, 
                   criterion, 
                   optimizer, 
                   device='cuda', 
                   num_epochs=10,
                   save_path="best_slow_r50_5ch.pth",
                   csv_log_path="train_log.csv"):
    """
    在训练过程中，将每个 epoch 的 Train/Val Loss 和 Acc 写入 CSV 文件。
    同时返回 loss/acc 历史列表以供可视化。
    """
    model.to(device)

    train_loss_history = []
    train_acc_history  = []
    val_loss_history   = []
    val_acc_history    = []

    best_val_loss = float('inf')

    # 打开 CSV 文件进行写入（若文件已存在则覆盖）
    with open(csv_log_path, 'w', newline='') as f:
        writer = csv.writer(f)
        # 写表头
        writer.writerow(["Epoch", "Train_Loss", "Train_Acc", "Val_Loss", "Val_Acc"])

        for epoch in range(num_epochs):
            print(f"Epoch [{epoch+1}/{num_epochs}]")

            # ---------- 训练阶段 ----------
            train_loss, train_acc = train_one_epoch(model, train_dataloader, criterion, optimizer, device)
            # ---------- 验证阶段 ----------
            val_loss, val_acc = validate_one_epoch(model, val_dataloader, criterion, device)

            # 保存到列表
            train_loss_history.append(train_loss)
            train_acc_history.append(train_acc)
            val_loss_history.append(val_loss)
            val_acc_history.append(val_acc)

            # 打印本epoch结果
            print(f"[Epoch {epoch+1}]  Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f} | "
                  f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")

            # 写入当前epoch的训练/验证结果到CSV
            writer.writerow([epoch+1, f"{train_loss:.4f}", f"{train_acc:.4f}",
                             f"{val_loss:.4f}", f"{val_acc:.4f}"])
            # 立即 flush，确保数据实时写入文件
            f.flush()

            # 如果在验证集上 Loss 更低，就保存模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), save_path)
                print(f"  [*] Saved best model (val_loss={val_loss:.4f})")

    return (train_loss_history, train_acc_history,
            val_loss_history, val_acc_history)

def plot_history(train_loss, train_acc, val_loss, val_acc, out_dir="./plots"):
    os.makedirs(out_dir, exist_ok=True)

    # 绘制 Loss 曲线
    plt.figure(figsize=(8, 4))
    plt.plot(train_loss, label='Train Loss', color='blue')
    plt.plot(val_loss,   label='Validation Loss', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()
    plt.savefig(os.path.join(out_dir, "loss_curve.png"))
    plt.close()

    # 绘制 Accuracy 曲线
    plt.figure(figsize=(8, 4))
    plt.plot(train_acc, label='Train Accuracy', color='blue')
    plt.plot(val_acc,   label='Validation Accuracy', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Curve')
    plt.legend()
    plt.savefig(os.path.join(out_dir, "accuracy_curve.png"))
    plt.close()

    print(f"[Info] Plots saved in {out_dir}")


if __name__ == "__main__":
    # ========== 1) 数据集准备 ========== #
    train_csv_path = "./only_petct_train.csv"
    val_csv_path   = "./only_petct_val.csv"

    train_dataset = MultiModal3DDataset(csv_path=train_csv_path, transform=None, output_shape=(64,64,64))
    val_dataset   = MultiModal3DDataset(csv_path=val_csv_path,   transform=None, output_shape=(64,64,64))

    train_loader = DataLoader(
        train_dataset, batch_size=2, shuffle=True, num_workers=4, collate_fn=my_collate_fn
    )
    val_loader   = DataLoader(
        val_dataset,   batch_size=2, shuffle=False, num_workers=4, collate_fn=my_collate_fn
    )

    # ========== 2) 初始化模型/损失/优化器 ========== #
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SlowR50_5ch(in_channels=5, num_classes=2, pretrained=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-6)

    # ========== 3) 训练 + 验证 + 保存 & CSV日志 ========== #
    log_csv_path = "train_log.csv"   # 训练日志保存的CSV文件
    (train_loss_hist, train_acc_hist,
     val_loss_hist,   val_acc_hist) = train_and_eval(
         model=model,
         train_dataloader=train_loader,
         val_dataloader=val_loader,
         criterion=criterion,
         optimizer=optimizer,
         device=device,
         num_epochs=20,
         save_path="best_slow_r50_5ch.pth",
         csv_log_path=log_csv_path
    )

    # ========== 4) 绘图 ========== #
    plot_history(train_loss_hist, train_acc_hist, val_loss_hist, val_acc_hist, out_dir="./plots")
