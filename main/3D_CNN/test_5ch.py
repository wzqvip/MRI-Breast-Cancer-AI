import torch
from torch.utils.data import DataLoader
import numpy as np

# 依旧用您之前的5通道Dataset
from dataset_5ch import MultiModal3DDataset
# 以及5通道网络
from model_5ch import Simple3DCNN_5ch

def test_model(model, dataloader, device='cuda'):
    model.eval()  # 切换到推理模式
    model.to(device)

    all_preds = []
    all_labels = []  # 若没有标签可不存
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            # labels = labels.to(device)  # 有标签再用

            # 前向传播
            outputs = model(inputs)  # shape: (batch_size, num_classes)

            # 假设是二分类，取argmax(二分类 logits)
            preds = torch.argmax(outputs, dim=1)  # shape: (batch_size,)

            # 存结果到列表
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())  # 若无 label，可省略

    return np.array(all_preds), np.array(all_labels)

if __name__ == "__main__":
    # 1) 加载训练好的模型
    model = Simple3DCNN_5ch(in_channels=5, num_classes=2)
    model.load_state_dict(torch.load("best_3dcnn_5ch.pth", map_location="cuda"))
    
    # 2) 构造测试集 Dataset / DataLoader
    #    假设您准备了 test_list.csv (格式和训练csv类似),
    #    但里面是测试病人的 CT/PET/MR 路径, 以及 label=0/1（若没有则都填0）
    test_csv_path = "./test_list.csv"
    test_dataset = MultiModal3DDataset(test_csv_path, transform=None, output_shape=(64,64,64))
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

    # 3) 调用测试函数
    preds, labels = test_model(model, test_loader, device='cuda')

    # 4) 若测试集有真实标签，则可计算准确率
    #    如果没有标签，这一步可以省略或注释
    accuracy = (preds == labels).mean()
    print(f"Test Accuracy = {accuracy:.4f}")

    # 如果需要更深入的指标(如F1-score、混淆矩阵等),可借助 sklearn:
    # from sklearn.metrics import classification_report, confusion_matrix
    # print(classification_report(labels, preds, target_names=["Negative", "Positive"]))
    # print(confusion_matrix(labels, preds))
