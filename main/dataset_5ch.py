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
    reader = sitk.ImageSeriesReader()
    sitk.ProcessObject_SetGlobalWarningDisplay(False)
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


def my_collate_fn(batch):
    """
    batch: List of tuples: [(inputs, labels, used_mods), (inputs, labels, used_mods), ...]
    这里 used_mods 是可变长的list。
    """
    inputs_list, labels_list, used_mods_list = [], [], []

    for (inp, lab, mods) in batch:
        inputs_list.append(inp)       # inp shape => (5, D, H, W)
        labels_list.append(lab)      # lab => scalar
        used_mods_list.append(mods)  # mods => list of strings

    # 把 inputs 堆叠成 (B, 5, D, H, W)
    inputs_batch = torch.stack(inputs_list, dim=0)
    # 把 labels 堆叠成 (B,) 
    labels_batch = torch.tensor(labels_list, dtype=torch.long)

    # used_mods_list 就保持 list of list，原样返回
    return inputs_batch, labels_batch, used_mods_list



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
        label = row["label"]

        # 记录哪些模态被实际使用
        used_mods = []

        # 1) CT
        ct_path = row["ct_path"] if isinstance(row["ct_path"], str) else ""
        ct_data = self.load_and_preprocess(ct_path)
        if ct_path.strip() != "":
            used_mods.append("CT")

        # 2) PET
        pet_path = row["pet_path"] if isinstance(row["pet_path"], str) else ""
        pet_data = self.load_and_preprocess(pet_path)
        if pet_path.strip() != "":
            used_mods.append("PET")

        # 3) MR DWI
        dwi_path = row["mr_dwi"] if isinstance(row["mr_dwi"], str) else ""
        dwi_data = self.load_and_preprocess(dwi_path)
        if dwi_path.strip() != "":
            used_mods.append("MR-DWI")

        # 4) MR T1
        t1_path = row["mr_t1"] if isinstance(row["mr_t1"], str) else ""
        t1_data = self.load_and_preprocess(t1_path)
        if t1_path.strip() != "":
            used_mods.append("MR-T1")

        # 5) MR dynamic
        dyn_path = row["mr_dynamic"] if isinstance(row["mr_dynamic"], str) else ""
        dyn_data = self.load_and_preprocess(dyn_path)
        if dyn_path.strip() != "":
            used_mods.append("MR-dynamic")

        # 拼接 5通道 => (5, D, H, W)
        combined = np.stack([ct_data, pet_data, dwi_data, t1_data, dyn_data], axis=0)

        combined_tensor = torch.tensor(combined, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.long)

        if self.transform:
            combined_tensor = self.transform(combined_tensor)

        # 返回时，除了 (inputs, label)，再返回一个 used_mods
        return combined_tensor, label_tensor, used_mods

    def load_and_preprocess(self, path_str):
        if not isinstance(path_str, str) or path_str.strip() == "":
            return np.zeros(self.output_shape, dtype=np.float32)
        arr = load_dicom_series(path_str)
        arr = normalize_and_resize(arr, self.output_shape)
        return arr
