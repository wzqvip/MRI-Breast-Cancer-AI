import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import SimpleITK as sitk
from skimage.transform import resize

def load_dicom_series(folder_path):
    """
    读取DICOM序列并返回 (D, H, W) 的 numpy array, 若失败则返回None
    """
    if not folder_path or not os.path.isdir(folder_path):
        return None

    reader = sitk.ImageSeriesReader()
    dicom_files = reader.GetGDCMSeriesFileNames(folder_path)
    if len(dicom_files) == 0:
        return None

    reader.SetFileNames(dicom_files)
    image = reader.Execute()  # SimpleITK image
    array = sitk.GetArrayFromImage(image)  # (D, H, W)
    return array

def normalize_and_resize(image_np, output_shape=(64, 64, 64)):
    """
    将 (D, H, W) 归一化到[0,1], 并 resize 到 output_shape.
    若 image_np=None 或全0, 返回全0占位.
    """
    if image_np is None or np.all(image_np == 0):
        return np.zeros(output_shape, dtype=np.float32)

    min_val, max_val = np.min(image_np), np.max(image_np)
    if max_val - min_val < 1e-8:
        image_np = np.zeros_like(image_np, dtype=np.float32)
    else:
        image_np = (image_np - min_val) / (max_val - min_val + 1e-8)

    image_resized = resize(image_np, output_shape, mode='constant', anti_aliasing=True)
    return image_resized.astype(np.float32)

class MultiModal3DDataset(Dataset):
    """
    5 通道: [CT, PET, MR_DWI, MR_T1, MR_DYNAMIC]
    """
    def __init__(self, csv_path, transform=None, output_shape=(64,64,64)):
        self.df = pd.read_csv(csv_path)
        self.transform = transform
        self.output_shape = output_shape

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        label = row["label"]

        # 1) CT
        ct_path = row["ct_path"] if isinstance(row["ct_path"], str) and row["ct_path"] else None
        ct_data = load_dicom_series(ct_path)
        ct_data = normalize_and_resize(ct_data, self.output_shape)

        # 2) PET
        pet_path = row["pet_path"] if isinstance(row["pet_path"], str) and row["pet_path"] else None
        pet_data = load_dicom_series(pet_path)
        pet_data = normalize_and_resize(pet_data, self.output_shape)

        # 3) MR_DWI
        mr_dwi_str = row["mr_dwi"] if isinstance(row["mr_dwi"], str) else ""
        mr_dwi_data = np.zeros(self.output_shape, dtype=np.float32)
        if mr_dwi_str:
            dwi_list = mr_dwi_str.split(";")
            if len(dwi_list) > 0:  # 这里只示范取第一个文件夹
                dwi_arr = load_dicom_series(dwi_list[0])
                mr_dwi_data = normalize_and_resize(dwi_arr, self.output_shape)

        # 4) MR_T1
        mr_t1_str = row["mr_t1"] if isinstance(row["mr_t1"], str) else ""
        mr_t1_data = np.zeros(self.output_shape, dtype=np.float32)
        if mr_t1_str:
            t1_list = mr_t1_str.split(";")
            if len(t1_list) > 0:
                t1_arr = load_dicom_series(t1_list[0])
                mr_t1_data = normalize_and_resize(t1_arr, self.output_shape)

        # 5) MR_dynamic
        mr_dynamic_str = row["mr_dynamic"] if isinstance(row["mr_dynamic"], str) else ""
        mr_dynamic_data = np.zeros(self.output_shape, dtype=np.float32)
        if mr_dynamic_str:
            dyn_list = mr_dynamic_str.split(";")
            if len(dyn_list) > 0:
                dyn_arr = load_dicom_series(dyn_list[0])
                mr_dynamic_data = normalize_and_resize(dyn_arr, self.output_shape)

        # 拼接 5 通道: (C=5, D, H, W)
        combined = np.stack([ct_data, pet_data,
                             mr_dwi_data, mr_t1_data, mr_dynamic_data], axis=0)

        combined_tensor = torch.tensor(combined, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.long)

        if self.transform:
            combined_tensor = self.transform(combined_tensor)

        return combined_tensor, label_tensor
