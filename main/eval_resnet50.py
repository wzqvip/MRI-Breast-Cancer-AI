import torch
from torch.utils.data import DataLoader
import numpy as np

from model_5ch_resnet50 import SlowR50_5ch  
from dataset_5ch import MultiModal3DDataset ,my_collate_fn

def evaluate_model(model, dataloader, device='cuda'):
    """
    对 dataloader 中所有样本进行推理，计算准确率/其他指标。
    返回: preds, labels, probs, used_modalities(列表)
    """
    model.eval()
    model.to(device)

    all_preds = []
    all_probs = []
    all_labels = []
    all_used_mods = []  # 存储每个sample使用的模态列表

    with torch.no_grad():
        for inputs, labels, used_mods in dataloader:
            # inputs.shape => (B, 5, D, H, W)
            # labels.shape => (B,)
            # used_mods => 列表长度=B，每个元素是一个包含若干字符串的list

            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)  # (B, num_classes)
            # 转为概率
            probs = torch.softmax(outputs, dim=1)  # (B, num_classes)
            # 取 argmax
            preds = torch.argmax(probs, dim=1)     # (B,)

            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # 记录 used_mods (需把它从 batch list合并到all_used_mods)
            # used_mods 是一个长度=B的列表
            # 例: [["CT","PET"], ["CT","MR-T1","MR-dynamic"]]
            all_used_mods.extend(used_mods)

    return np.array(all_preds), np.array(all_probs), np.array(all_labels), all_used_mods

if __name__ == "__main__":
    # 1) 加载已训练好的模型
    model = SlowR50_5ch(in_channels=5, num_classes=2, pretrained=False)
    model.load_state_dict(torch.load("best_slow_r50_5ch.pth", map_location="cuda", weights_only=True))

    print(model)

    # 2) 构建测试/验证集 DataLoader
    test_csv_path = "./eval_list_small_avg.csv"  # 您的评估CSV
    test_dataset = MultiModal3DDataset(
        csv_path=test_csv_path,
        transform=None,         # 或者与训练相同的 transform
        output_shape=(64,64,64) # 与训练时保持一致
    )

    test_loader = DataLoader(
        test_dataset, 
        batch_size=2, 
        shuffle=False, 
        num_workers=4,
        collate_fn=my_collate_fn  # <--- 指定自定义collate
        )


    # 3) 调用评估函数
    preds, probs, labels, used_mods_list = evaluate_model(model, test_loader, device='cuda')

    # 4) 计算准确率（若您有真实标签）
    accuracy = (preds == labels).mean()
    print(f"Test Accuracy = {accuracy:.4f}")

    # 5) 打印每个样本的预测结果
    #    used_mods_list[i] 对应第 i 个样本使用的模态
    #    probs[i] 是 [prob_class0, prob_class1], preds[i] 是 argmax
    for i in range(len(preds)):
        mod_info = ",".join(used_mods_list[i]) if len(used_mods_list[i])>0 else "No Modality"
        print(f"Sample {i}: label={labels[i]}, pred={preds[i]}, used=[{mod_info}], prob={probs.tolist()[i][1] * 100 :.2f}%")

    # 6) 如果需要更多指标: sklearn.metrics
    from sklearn.metrics import classification_report, confusion_matrix
    print("\nClassification Report:")
    print(classification_report(labels, preds, digits=4))
    print("Confusion Matrix:")
    print(confusion_matrix(labels, preds))
