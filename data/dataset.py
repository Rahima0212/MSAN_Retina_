import os
import cv2
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

def imread_safe(file_path, flags=cv2.IMREAD_COLOR):
    """Reads image from a path potentially containing Unicode characters."""
    if not os.path.exists(file_path):
        return None
    try:
        with open(file_path, 'rb') as f:
            file_buffer = np.frombuffer(f.read(), np.uint8)
        img = cv2.imdecode(file_buffer, flags)
        return img
    except Exception:
        return None

class MultiModalDataset(Dataset):
    """
    Loads Fundus, OCT, and ROI images from the CSVs created by prepare_data.py.
    """
    def __init__(self, csv_file, image_size=300, is_training=True):
        self.data_frame = pd.read_csv(csv_file)
        self.is_training = is_training
        
        # Define transforms
        # For training, include augmentations
        if self.is_training:
            self.fundus_oct_transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else: # For validation/testing, just resize, convert, and normalize
            self.fundus_oct_transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        
        # ROI Mask transform is simpler (no normalization or color augmentation)
        self.roi_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor() # Converts to [0, 1] range
        ])

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        row = self.data_frame.iloc[idx]
        
        # 1. Load Fundus Image
        fundus_img = imread_safe(row['fundus_path'], cv2.IMREAD_COLOR)
        fundus_img = cv2.cvtColor(fundus_img, cv2.COLOR_BGR2RGB)
        fundus_pil = Image.fromarray(fundus_img)
        
        # 2. Load OCT Image
        oct_img = imread_safe(row['oct_path'], cv2.IMREAD_GRAYSCALE)
        oct_pil = Image.fromarray(oct_img).convert("RGB") # Convert to 3 channels
        
        # 3. Load ROI Mask
        roi_img = imread_safe(row['roi_path'], cv2.IMREAD_GRAYSCALE)
        roi_pil = Image.fromarray(roi_img)

        # Apply transforms
        fundus_tensor = self.fundus_oct_transform(fundus_pil)
        oct_tensor = self.fundus_oct_transform(oct_pil)
        roi_tensor = self.roi_transform(roi_pil)
        
        # Get label
        label = torch.tensor(row['label'], dtype=torch.long)

        return fundus_tensor, oct_tensor, roi_tensor, label