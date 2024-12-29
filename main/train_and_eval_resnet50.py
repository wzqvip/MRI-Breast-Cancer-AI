import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

# ========== 自定义模块/函数：请根据实际位置修改 ========== #
from dataset_5ch import MultiModal3DDataset, my_collate_fn
from model_5ch_resnet50 import SlowR50_5ch
# ======================================================== #

def train_one_epoch(model, dataloader, criterion, optimizer, device='cuda'):
    """
    单个epoch的训练循环，使用tqdm显示进度条和当前batch的loss/acc。
    返回该epoch的平均loss和准确率。
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    # tqdm包装 dataloader，显示进度条
    pbar = tqdm(dataloader, desc="Train", leave=False)
    for batch_idx, (inputs, labels, _) in enumerate(pbar):
        inputs, labels = inputs.to(device), labels.to(device)

        # 前向传播 + 反向传播 + 更新参数
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # 累计loss
        running_loss += loss.item() * inputs.size(0)

        # 计算当前批次准确率
        preds = torch.argmax(outputs, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        # 动态更新tqdm的后缀信息
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
    """
    单个epoch的验证/测试循环，使用tqdm显示进度条和当前batch的loss/acc。
    返回该epoch的平均loss和准确率。
    """
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
                   save_path="best_slow_r50_5ch.pth"):
    """
    进行完整的训练 + 验证流程，每个epoch结束后记录Loss和Accuracy，
    同时保存在验证集上表现最好的模型。
    返回 (train_loss_history, train_acc_history, val_loss_history, val_acc_history) 
    以便后续绘图。
    """
    model.to(device)

    train_loss_history = []
    train_acc_history  = []
    val_loss_history   = []
    val_acc_history    = []

    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        print(f"Epoch [{epoch+1}/{num_epochs}]")

        # ---------- 训练阶段 ----------
        train_loss, train_acc = train_one_epoch(model, train_dataloader, criterion, optimizer, device)
        # ---------- 验证阶段 ----------
        val_loss, val_acc = validate_one_epoch(model, val_dataloader, criterion, device)

        train_loss_history.append(train_loss)
        train_acc_history.append(train_acc)
        val_loss_history.append(val_loss)
        val_acc_history.append(val_acc)

        # 打印本epoch结果
        print(f"[Epoch {epoch+1}]  Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f} | "
              f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")

        # 如果在验证集上的Loss更低，则保存当前模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print(f"  [*] Saved best model with val_loss={best_val_loss:.4f}")

    return (train_loss_history, train_acc_history,
            val_loss_history, val_acc_history)


def plot_history(train_loss, train_acc, val_loss, val_acc, out_dir="./plots"):
    """
    绘制并保存 训练/验证 的 Loss 与 Accuracy 曲线图。
    """
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
    train_csv_path = "./sample_list.csv"          # 训练集 CSV
    val_csv_path   = "./eval_list_small_avg.csv"  # 验证或测试集 CSV

    train_dataset = MultiModal3DDataset(csv_path=train_csv_path, transform=None, output_shape=(64,64,64))
    val_dataset   = MultiModal3DDataset(csv_path=val_csv_path,   transform=None, output_shape=(64,64,64))

    train_loader = DataLoader(
        train_dataset, 
        batch_size=4, 
        shuffle=True, 
        num_workers=4, 
        collate_fn=my_collate_fn
    )
    val_loader   = DataLoader(
        val_dataset, 
        batch_size=4, 
        shuffle=False, 
        num_workers=4, 
        collate_fn=my_collate_fn
    )

    # ========== 2) 初始化模型/损失/优化器 ========== #
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SlowR50_5ch(in_channels=5, num_classes=2, pretrained=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # ========== 3) 训练 + 验证 + 保存 ========== #
    (train_loss_hist, train_acc_hist,
     val_loss_hist,   val_acc_hist) = train_and_eval(
         model=model,
         train_dataloader=train_loader,
         val_dataloader=val_loader,
         criterion=criterion,
         optimizer=optimizer,
         device=device,
         num_epochs=50,
         save_path="best_slow_r50_5ch.pth"
    )

    # ========== 4) 绘图 ========== #
    plot_history(train_loss_hist, train_acc_hist, val_loss_hist, val_acc_hist, out_dir="./plots")
