import os
import pandas as pd

# config 
data_dir = "../datasets/manifest-1542731172463/QIN-BREAST"  # 修改为您实际的数据根目录
label_csv = "./QIN_labels.csv"                              # 包含 ["PatientID","Label"] 的 CSV
output_csv = "./sample_list.csv"                            # 输出的CSV

def generate_sample_list(data_dir, label_csv, output_csv):
    """
    扫描 data_dir 下的 QIN-BREAST-01-xxxx 病人文件夹，解析多会话文件夹，
    并细分出 CTAC / PET AC / DWI / T1 / dynamic 五个通道路径。
    
    针对同一天多次扫描 (比如 CT/PET 和 MR 分开session) 的情况，
    以“日期”作为key进行合并，最终让同一天同一病人的CT、PET、MR信息写在同一行。
    
    最终输出 CSV 列: patient_id, session_date, ct_path, pet_path, mr_dwi, mr_t1, mr_dynamic, label
    """
    # 1) 读取 label.csv 
    df_label = pd.read_csv(label_csv)  # 假设有列: ["PatientID","Label"]
    label_dict = dict(zip(df_label["PatientID"], df_label["Label"]))

    # 2) sessions_dict 用来合并同一天扫描, key=(patient_id, date_key)
    sessions_dict = {}

    # 3) 遍历 data_dir 下的所有 "QIN-BREAST-01-xxxx" 病人
    for patient_id in os.listdir(data_dir):
        patient_folder = os.path.join(data_dir, patient_id)
        if not os.path.isdir(patient_folder):
            continue

        # 查看是否有标签
        if patient_id not in label_dict:
            print(f"[Warning] {patient_id} not in label.csv, set label=0.")
            patient_label = 0
        else:
            patient_label = label_dict[patient_id]

        # 遍历该病人的所有 session 文件夹
        # e.g. "09-12-1995-NA-BREAST PRONE-48232"
        for session_id in os.listdir(patient_folder):
            session_path = os.path.join(patient_folder, session_id)
            if not os.path.isdir(session_path):
                continue

            # 从 session_id 中提取日期, 假设前10字符就是 "MM-DD-YYYY"
            # 若命名规律不同, 您需要调整 parse 逻辑.
            date_key = session_id[:10]  # e.g. "09-12-1995"

            # 在 sessions_dict 中找 (patient_id, date_key) 没有就新建
            if (patient_id, date_key) not in sessions_dict:
                sessions_dict[(patient_id, date_key)] = {
                    "patient_id": patient_id,
                    "session_date": date_key,
                    "ct_path": "",
                    "pet_path": "",
                    "mr_dwi": "",
                    "mr_t1": "",
                    "mr_dynamic": "",
                    "label": patient_label
                }

            # 准备临时指针
            record = sessions_dict[(patient_id, date_key)]

            # 遍历子文件夹 (e.g. "2.000000-CTAC-98431", "501.000000-DWIEPIMPSsmartTX-xxxx", ...)
            for subf in os.listdir(session_path):
                subf_path = os.path.join(session_path, subf)
                if not os.path.isdir(subf_path):
                    continue

                subf_lower = subf.lower()

                if "ctac" in subf_lower:
                    # 若已有存过 ct_path, 这里可能要决定是否覆盖.
                    record["ct_path"] = subf_path
                elif "pet ac" in subf_lower:
                    record["pet_path"] = subf_path
                elif ("dwiep" in subf_lower 
                      or "dwimp" in subf_lower 
                      or "dwiepimp" in subf_lower):
                    record["mr_dwi"] = subf_path
                elif ("flip" in subf_lower 
                      or "multi-flipt1" in subf_lower 
                      or "t1-map" in subf_lower):
                    record["mr_t1"] = subf_path
                elif "dynamic" in subf_lower:
                    record["mr_dynamic"] = subf_path
                else:
                    # 其他未分类的文件夹, 可按需忽略或打印
                    pass

    # 4) 将合并后的 sessions_dict 展开写入 DataFrame
    rows = []
    for (pid, dkey), info in sessions_dict.items():
        rows.append(info)
    df_samples = pd.DataFrame(rows)

    # 5) 输出 CSV
    df_samples.to_csv(output_csv, index=False)
    print(f"Sample list saved to {output_csv} with {len(df_samples)} rows.")


if __name__ == "__main__":
    generate_sample_list(data_dir, label_csv, output_csv)
