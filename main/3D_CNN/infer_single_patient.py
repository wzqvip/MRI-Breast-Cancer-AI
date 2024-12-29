import os
import torch
import numpy as np

# 复用 dataset_5ch 里已有的函数:
from dataset_5ch import load_dicom_series, normalize_and_resize
from model_5ch import Simple3DCNN_5ch

def infer_single_patient(model, patient_folder, device='cuda', output_shape=(64,64,64)):
    """
    对单个病人文件夹进行推理，可能包含多个会话。
    每个会话下有:
      - CT/CTAC (可选)
      - PT/PET (可选)
      - MR下的子文件夹(可选): 可能包括DWI, T1, dynamic (多个或没有)
    
    参数:
      model: 训练好的5通道模型 (Simple3DCNN_5ch)
      patient_folder: 单个病人的文件夹路径 (e.g. .../QIN-BREAST-01-0067)
      device: 'cuda' 或 'cpu'
      output_shape: 预处理后的形状, 如 (64,64,64)
    
    返回:
      results: list, 其中每个元素是:
         {
           "session_id": <str>,
           "prediction": <0或1>
         }
      如果您需要 logits 或概率，也可保存 "logits" 或 "prob"。
    """
    model.eval()
    model.to(device)

    results = []

    # 遍历该病人文件夹下的所有会话子文件夹
    # 例如 06-20-1995-38955, 07-14-1995-83383 等
    for session_id in os.listdir(patient_folder):
        session_path = os.path.join(patient_folder, session_id)
        if not os.path.isdir(session_path):
            continue

        # 1) 检查 CT 路径
        ct_path = None
        possible_ct_path = os.path.join(session_path, "CT", "CTAC")
        if os.path.isdir(possible_ct_path):
            ct_path = possible_ct_path
            print(f"Found CT at: {ct_path}")
        
        # 2) 检查 PET 路径
        pet_path = None
        possible_pet_path = os.path.join(session_path, "PT", "PET")
        if os.path.isdir(possible_pet_path):
            pet_path = possible_pet_path
            print(f"Found PET at: {pet_path}")

        # 3) 细分 MR
        mr_dwi_folders = []
        mr_t1_folders = []
        mr_dynamic_folders = []

        mr_root = os.path.join(session_path, "MR")
        if os.path.isdir(mr_root):
            for subf in os.listdir(mr_root):
                subf_path = os.path.join(mr_root, subf)
                if not os.path.isdir(subf_path):
                    continue
                subf_lower = subf.lower()

                if "dwi" in subf_lower:
                    mr_dwi_folders.append(subf_path)
                    print(f"Found MRI DWI at: {subf_path}")
                elif "t1" in subf_lower:
                    mr_t1_folders.append(subf_path)
                    print(f"Found MRI T1 at: {subf_path}")
                elif "dynamic" in subf_lower:
                    mr_dynamic_folders.append(subf_path)
                    print(f"Found MRI Dynamic at: {subf_path}")
                else:
                    # 如果有其他子文件夹,可根据需要再分类
                    pass

        # ==========================
        # 构建 5 通道 (CT, PET, DWI, T1, dynamic)
        # ==========================
        
        # --- CT ---
        ct_data = load_dicom_series(ct_path)
        ct_data = normalize_and_resize(ct_data, output_shape)

        # --- PET ---
        pet_data = load_dicom_series(pet_path)
        pet_data = normalize_and_resize(pet_data, output_shape)

        # --- MR_DWI ---
        mr_dwi_data = np.zeros(output_shape, dtype=np.float32)
        if len(mr_dwi_folders) > 0:
            dwi_arr = load_dicom_series(mr_dwi_folders[0])
            mr_dwi_data = normalize_and_resize(dwi_arr, output_shape)

        # --- MR_T1 ---
        mr_t1_data = np.zeros(output_shape, dtype=np.float32)
        if len(mr_t1_folders) > 0:
            t1_arr = load_dicom_series(mr_t1_folders[0])
            mr_t1_data = normalize_and_resize(t1_arr, output_shape)

        # --- MR_dynamic ---
        mr_dynamic_data = np.zeros(output_shape, dtype=np.float32)
        if len(mr_dynamic_folders) > 0:
            dyn_arr = load_dicom_series(mr_dynamic_folders[0])
            mr_dynamic_data = normalize_and_resize(dyn_arr, output_shape)

        # 拼接: shape = (5, D, H, W)
        combined_5ch = np.stack([ct_data, pet_data,
                                 mr_dwi_data, mr_t1_data, mr_dynamic_data], axis=0)

        # 转 Tensor
        input_tensor = torch.tensor(combined_5ch, dtype=torch.float32).unsqueeze(0).to(device)
        # shape => (1, 5, D, H, W)

        with torch.no_grad():
            outputs = model(input_tensor)   # shape: (1, 2)  (假设二分类)
            
            # 1) 用 softmax 将 logits 转为概率，dim=1 表示在类别维度上做归一化
            probabilities = torch.softmax(outputs, dim=1)  # shape: (1,2)
            
            # 2) 取第0、1类的概率
            prob_class0 = probabilities[0, 0].item()  # 第一个维度是batch=0，第二个维度是类=0
            prob_class1 = probabilities[0, 1].item()
            
            # 3) 用 argmax 获取最终预测标签
            pred = torch.argmax(probabilities, dim=1)  # (1,)
            pred_class = pred.item()  # 0 或 1


        results.append({
            "session_id": session_id,
            "prediction": pred_class,
            "prob_class0": prob_class0,
            "prob_class1": prob_class1
        })


    return results

# ---- 以下是演示用的 main 函数 ----
if __name__ == "__main__":
    # 1) 加载已经训练好的模型
    model = Simple3DCNN_5ch(in_channels=5, num_classes=2)
    model.load_state_dict(torch.load("best_3dcnn_5ch.pth", map_location="cuda", weights_only=True))

    # 2) 指定某个病人的文件夹(如 QIN-BREAST-01-0067)
    patient_folder = "..\datasets\PyDownloader\QIN-BREAST\QIN-BREAST-01-0024"
    
    # 3) 调用推理函数
    results = infer_single_patient(model, patient_folder, device='cuda', output_shape=(64,64,64))

    # 4) 查看/打印结果
    print("Inference results for single patient:")
    for r in results:
        print(f"Session: {r['session_id']}, Prediction: {r['prediction']}, Probability: {r['prob_class1']:.4f}")
