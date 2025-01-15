# split_and_filter_dataset.py

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def filter_pet_ct_only(df):
    """
    仅筛选 PET/CT 数据：
      - pet_path 和 ct_path 均不为空
      - MRI 序列全部为空 (mr_dwi, mr_t1, mr_dynamic) 或者不强制？
        这里示例：强制 MRI 均为空，表示真正“只有 PET/CT”
    """
    mask = (df['pet_path'].str.strip() != "") & (df['ct_path'].str.strip() != "") & \
           (df['mr_dwi'].str.strip() == "") & (df['mr_t1'].str.strip() == "") & (df['mr_dynamic'].str.strip() == "")
    return df[mask].copy()

def filter_mri_only(df):
    """
    仅筛选 MRI 数据：
      - PET 和 CT 都为空
      - 至少有一个 MRI 序列不为空 (mr_dwi / mr_t1 / mr_dynamic)
    """
    mask_mri = (
        ((df['mr_dwi'].str.strip() != "") | (df['mr_t1'].str.strip() != "") | (df['mr_dynamic'].str.strip() != ""))
    )
    mask_no_petct = ((df['pet_path'].str.strip() == "") & (df['ct_path'].str.strip() == ""))
    return df[mask_mri & mask_no_petct].copy()

def filter_petct_plus_mri(df):
    """
    同时包含 PET/CT 和 MRI:
      - pet_path != "" and ct_path != ""
      - 至少有一个 MRI 序列 != ""
    """
    mask_petct = ((df['pet_path'].str.strip() != "") & (df['ct_path'].str.strip() != ""))
    mask_mri   = ((df['mr_dwi'].str.strip() != "") | (df['mr_t1'].str.strip() != "") | (df['mr_dynamic'].str.strip() != ""))
    return df[mask_petct & mask_mri].copy()

def filter_all(df):
    """
    筛选所有数据，不做额外限制
    """
    return df.copy()

def random_split_save(df, subset_name, train_ratio=0.8, seed=42):
    """
    随机划分 train/val 并存储成 CSV
    subset_name: 用于输出文件名前缀
    """
    # 如果 df 太小可根据需求
    if len(df) == 0:
        print(f"[Warning] No samples found for subset = {subset_name}, skip splitting.")
        return

    train_df, val_df = train_test_split(df, test_size=(1 - train_ratio), random_state=seed, shuffle=True)
    
    train_csv = f"{subset_name}_train.csv"
    val_csv   = f"{subset_name}_val.csv"

    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)

    print(f"{subset_name}: total={len(df)}, train={len(train_df)}, val={len(val_df)}")
    print(f"  => saved to {train_csv} and {val_csv}")

if __name__ == "__main__":
    # 1) 读取合并好的CSV（包含pet_path, ct_path, mr_dwi, mr_t1, mr_dynamic等）
    merged_csv = "sample_list.csv"  # 你最终合并后的CSV
    df = pd.read_csv(merged_csv)

    # 2) 确保空字符串，而不是NaN，方便比较
    # 如果列里有NaN，可以填充为空字符串
    for col in ['ct_path','pet_path','mr_dwi','mr_t1','mr_dynamic']:
        df[col] = df[col].fillna("")

    # 3) 分别筛选3种子集
    df_petct_only = filter_pet_ct_only(df)
    df_mri_only   = filter_mri_only(df)
    df_petct_mri  = filter_petct_plus_mri(df)
    all = filter_all(df)

    # 4) 对每个子集做train/val 划分 并保存
    random_split_save(df_petct_only, "only_petct", train_ratio=0.8, seed=42)
    random_split_save(df_mri_only,   "only_mri",   train_ratio=0.8, seed=42)
    random_split_save(df_petct_mri,  "petct_mri",  train_ratio=0.8, seed=42)
    random_split_save(all,           "all",        train_ratio=0.8, seed=42)
