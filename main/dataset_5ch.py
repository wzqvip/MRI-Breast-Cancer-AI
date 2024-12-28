# dataset_5ch.py
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import SimpleITK as sitk
from skimage.transform import resize

def load_dicom_series(folder_path):
    if not folder_path or not os.path.isdir(folder_path):
        return None
    sitk.ProcessObject_SetGlobalWarningDisplay(False)
    reader = sitk.ImageSeriesReader()
    dicom_files = reader.GetGDCMSeriesFileNames(folder_path)
    if len(dicom_files) == 0:
        return None
    reader.SetFileNames(dicom_files)
    image = reader.Execute()
    array = sitk.GetArrayFromImage(image)  # (D, H, W)
    return array

def normalize_and_resize(image_np, output_shape=(64,64,64)):
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
    5 通道: [CT, PET, DWI, T1, dynamic]
    CSV字段: patient_id, session_id, ct_path, pet_path, mr_dwi, mr_t1, mr_dynamic, label
    """
    def __init__(self, csv_path, transform=None, output_shape=(64,64,64)):
        self.df = pd.read_csv(csv_path)
        self.transform = transform
        self.output_shape = output_shape

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        ct_data = self.load_and_preprocess(row["ct_path"])
        pet_data = self.load_and_preprocess(row["pet_path"])
        dwi_data = self.load_and_preprocess(row["mr_dwi"])
        t1_data = self.load_and_preprocess(row["mr_t1"])
        dyn_data = self.load_and_preprocess(row["mr_dynamic"])

        # 拼接 5通道 => (5, D, H, W)
        combined = np.stack([ct_data, pet_data, dwi_data, t1_data, dyn_data], axis=0)

        combined_tensor = torch.tensor(combined, dtype=torch.float32)
        label = torch.tensor(row["label"], dtype=torch.long)

        if self.transform:
            combined_tensor = self.transform(combined_tensor)

        return combined_tensor, label

    def load_and_preprocess(self, path_str):
        if not isinstance(path_str, str) or path_str.strip() == "":
            # 空 => 全0
            return np.zeros(self.output_shape, dtype=np.float32)

        arr = load_dicom_series(path_str)
        arr = normalize_and_resize(arr, self.output_shape)
        return arr
