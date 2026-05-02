import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class DistillDataset(Dataset):
    def __init__(self, img_dir, img_size=640):
        self.img_dir = img_dir
        self.img_size = img_size
        self.img_files = [f for f in os.listdir(img_dir)
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        print(f"[Dataset] Loaded {len(self.img_files)} images from {img_dir}")

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        # 读取图像
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Letterbox resize (和YOLO训练时保持一致)
        h, w = img.shape[:2]
        scale = min(self.img_size / h, self.img_size / w)
        new_w, new_h = int(w * scale), int(h * scale)
        img_resized = cv2.resize(img, (new_w, new_h))

        # 填充
        pad_w = self.img_size - new_w
        pad_h = self.img_size - new_h
        top, bottom = pad_h // 2, pad_h - (pad_h // 2)
        left, right = pad_w // 2, pad_w - (pad_w // 2)
        img_padded = cv2.copyMakeBorder(img_resized, top, bottom, left, right,
                                        cv2.BORDER_CONSTANT, value=(114, 114, 114))

        # 转Tensor
        img_tensor = torch.from_numpy(img_padded).float() / 255.0
        img_tensor = img_tensor.permute(2, 0, 1)

        return img_tensor