# File    : step3_dataset.py
# Purpose : PyTorch Dataset class that loads processed PNG images
#           and matches them with biomarker and severity labels
#           from ct_biomarker.xlsx. Splits data into train and
#           validation sets at image level to prevent data leakage.

import os
import pandas as pd
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split

EXCEL_PATH     = r"D:\COVID\ct_biomarker.xlsx"
IMAGES_DIR     = r"D:\COVID\processed_images"
BIOMARKER_COLS = ["CRP", "NLR", "D_dimer", "LDH"]
SEVERITY_COL   = "Severity"
IMAGE_COL      = "Image_Name"


def get_transforms(train=True):
    if train:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])


class CTDataset(Dataset):
    def __init__(self, df, images_dir, bm_means, bm_stds, train=True):
        self.df         = df.reset_index(drop=True)
        self.images_dir = images_dir
        self.bm_means   = bm_means
        self.bm_stds    = bm_stds
        self.transform  = get_transforms(train)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row      = self.df.iloc[idx]
        img_name = row[IMAGE_COL]
        img_path = os.path.join(self.images_dir, img_name)
        image    = Image.open(img_path).convert("RGB")
        image    = self.transform(image)

        bm_raw    = row[BIOMARKER_COLS].values.astype(np.float32)
        bm_norm   = (bm_raw - self.bm_means) / self.bm_stds
        biomarker = torch.tensor(bm_norm, dtype=torch.float32)
        severity  = torch.tensor(int(row[SEVERITY_COL]), dtype=torch.long)

        return image, biomarker, severity


def get_dataloaders(excel_path=EXCEL_PATH,
                    images_dir=IMAGES_DIR,
                    val_ratio=0.2,
                    batch_size=8,
                    seed=42):
    df = pd.read_excel(excel_path)

    bm_means = df[BIOMARKER_COLS].mean().values.astype(np.float32)
    bm_stds  = df[BIOMARKER_COLS].std().values.astype(np.float32)
    bm_stds  = np.where(bm_stds == 0, 1, bm_stds)

    train_df, val_df = train_test_split(
        df,
        test_size=val_ratio,
        stratify=df[SEVERITY_COL],
        random_state=seed
    )

    print("=" * 55)
    print("  Dataset Split  -  step3_dataset.py")
    print("=" * 55)
    print(f"  Total images : {len(df)}")
    print(f"  Train        : {len(train_df)}")
    print(f"  Val          : {len(val_df)}")
    print(f"\n  Train severity distribution:")
    print(train_df[SEVERITY_COL].value_counts().sort_index().to_string())
    print(f"\n  Val severity distribution:")
    print(val_df[SEVERITY_COL].value_counts().sort_index().to_string())
    print("=" * 55)

    train_dataset = CTDataset(train_df, images_dir, bm_means, bm_stds, train=True)
    val_dataset   = CTDataset(val_df,   images_dir, bm_means, bm_stds, train=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, bm_means, bm_stds


if __name__ == "__main__":
    train_loader, val_loader, bm_means, bm_stds = get_dataloaders()

    print("\n  Testing one batch...")
    for images, biomarkers, severities in train_loader:
        print(f"  Image shape    : {images.shape}")
        print(f"  Biomarker shape: {biomarkers.shape}")
        print(f"  Severity shape : {severities.shape}")
        print(f"  Severity values: {severities.tolist()}")
        break

    print("\n  step3_dataset.py OK")