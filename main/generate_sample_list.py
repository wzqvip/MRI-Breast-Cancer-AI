# generate_sample_list.py
import os
import pandas as pd

# config 
data_dir = "../datasets/manifest-1542731172463/QIN-BREAST"          # 修改为您实际的数据根目录
label_csv = "./QIN_labels.csv"         # 包含 ["PatientID","Label"] 的 CSV
output_csv = "./sample_list.csv"  # 输出


def generate_sample_list(data_dir, label_csv, output_csv):
    """
    扫描 data_dir 下的 QIN-BREAST-01-xxxx 病人文件夹，解析多会话文件夹，
    并细分出 CTAC / PET AC / DWI / T1 / dynamic 五个通道路径。
    
    最终输出 CSV 列: patient_id, session_id, ct_path, pet_path, mr_dwi, mr_t1, mr_dynamic, label
    """

    # 读取 label.csv (如果有). 
    # 如果您没有 label，就可以给个缺省值(比如0)，或者只存 patient_id 不存 label
    df_label = pd.read_csv(label_csv)  # 假设有列: ["PatientID","Label"]
    label_dict = dict(zip(df_label["PatientID"], df_label["Label"]))

    rows = []

    # 遍历 data_dir 下的所有 "QIN-BREAST-01-xxxx"
    for patient_id in os.listdir(data_dir):
        patient_folder = os.path.join(data_dir, patient_id)
        if not os.path.isdir(patient_folder):
            continue

        # 查看是否有标签
        if patient_id not in label_dict:
            print(f"[Warning] {patient_id} not found in label.csv, skip or set label=0.")
            patient_label = 0  # 或者 continue
        else:
            patient_label = label_dict[patient_id]

        # 遍历 "09-12-1995-NA-BREAST PRONE-48232" 之类会话文件夹
        for session_id in os.listdir(patient_folder):
            session_path = os.path.join(patient_folder, session_id)
            if not os.path.isdir(session_path):
                continue

            # 准备记录 5 条路径
            ct_path = ""
            pet_path = ""
            mr_dwi = ""
            mr_t1 = ""
            mr_dynamic = ""

            # 遍历子文件夹 (e.g. "2.000000-CTAC-98431", "3.000000-PET AC 3DWB-27349", "501.000000-DWIEPIMPSsmartTX-xxxx", ...)
            for subf in os.listdir(session_path):
                subf_path = os.path.join(session_path, subf)
                if not os.path.isdir(subf_path):
                    continue

                subf_lower = subf.lower()

                if "ctac" in subf_lower:
                    ct_path = subf_path
                elif "pet ac" in subf_lower:
                    pet_path = subf_path
                elif "dwiep" in subf_lower or "dwimp" in subf_lower or "dwiepimp" in subf_lower:
                    # 根据实际文件夹名称匹配
                    mr_dwi = subf_path
                elif "flip" in subf_lower or "multi-flipt1" in subf_lower or "t1-map" in subf_lower:
                    mr_t1 = subf_path
                elif "dynamic" in subf_lower:
                    mr_dynamic = subf_path
                else:
                    # 若还有其他子文件夹，但不属于以上任何模态，可忽略或打印提示
                    pass

            row = {
                "patient_id": patient_id,
                "session_id": session_id,
                "ct_path": ct_path,
                "pet_path": pet_path,
                "mr_dwi": mr_dwi,
                "mr_t1": mr_t1,
                "mr_dynamic": mr_dynamic,
                "label": patient_label
            }
            rows.append(row)

    df_samples = pd.DataFrame(rows)
    df_samples.to_csv(output_csv, index=False)
    print(f"Sample list saved to {output_csv}")


if __name__ == "__main__":

    generate_sample_list(data_dir, label_csv, output_csv)
