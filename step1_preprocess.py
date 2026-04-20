# File    : step1_preprocess.py
# Purpose : Load .nii CT scans from MosMedData (CT-0 to CT-4),
#           extract one middle slice per scan,
#           normalize intensity, resize to 224x224,
#           save as PNG in D:\COVID\processed_images\
#           CT-4 has only 2 real scans — augments to 10

import os
import nibabel as nib
import numpy as np
from PIL import Image, ImageEnhance
from tqdm import tqdm

STUDIES_DIR      = r"D:\COVID\archive (1)\MosMedData Chest CT Scans with COVID-19 Related Findings COVID19_1110 1.0\studies"
OUTPUT_DIR       = r"D:\COVID\processed_images"
SCANS_PER_FOLDER = 10
IMG_SIZE         = (224, 224)
CT_FOLDERS       = ["CT-0", "CT-1", "CT-2", "CT-3", "CT-4"]


def normalize_slice(slice_2d):
    slice_2d = np.clip(slice_2d, -1000, 400)
    slice_2d = slice_2d - slice_2d.min()
    max_val  = slice_2d.max()
    if max_val > 0:
        slice_2d = slice_2d / max_val
    return (slice_2d * 255).astype(np.uint8)


def extract_middle_slice(volume):
    mid = volume.shape[2] // 2
    return volume[:, :, mid]


def process_scan(nii_path, out_path):
    img    = nib.load(nii_path)
    volume = img.get_fdata()
    if volume.ndim == 4:
        volume = volume[..., 0]
    slice_2d = extract_middle_slice(volume)
    slice_2d = normalize_slice(slice_2d)
    pil_img  = Image.fromarray(slice_2d).convert("L")
    pil_img  = pil_img.resize(IMG_SIZE, Image.BILINEAR)
    pil_img.save(out_path)
    return pil_img


def augment_image(pil_img, aug_id):
    img = pil_img.copy()
    if aug_id == 1:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    elif aug_id == 2:
        img = img.rotate(5)
    elif aug_id == 3:
        img = img.rotate(-5)
    elif aug_id == 4:
        img = img.rotate(10)
    elif aug_id == 5:
        img = img.rotate(-10)
    elif aug_id == 6:
        img = img.transpose(Image.FLIP_LEFT_RIGHT).rotate(5)
    elif aug_id == 7:
        img = ImageEnhance.Brightness(img).enhance(1.1)
    elif aug_id == 8:
        img = ImageEnhance.Brightness(img).enhance(0.9)
    return img


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 55)
    print("  CT Scan Preprocessing  -  step1_preprocess.py")
    print("=" * 55)
    print(f"  Source : {STUDIES_DIR}")
    print(f"  Output : {OUTPUT_DIR}")
    print(f"  Target : {SCANS_PER_FOLDER} per folder x {len(CT_FOLDERS)} = {SCANS_PER_FOLDER * len(CT_FOLDERS)} total")
    print(f"  CT-4   : 2 real + 8 augmented = 10")
    print("=" * 55)

    success = 0
    failed  = []

    for folder in CT_FOLDERS:
        folder_path = os.path.join(STUDIES_DIR, folder)

        if not os.path.isdir(folder_path):
            print(f"\n  ERROR: Folder not found -> {folder_path}")
            continue

        nii_files = sorted([
            f for f in os.listdir(folder_path) if f.endswith(".nii")
        ])[:SCANS_PER_FOLDER]

        if not nii_files:
            print(f"\n  WARNING: No .nii files in {folder}")
            continue

        print(f"\n  Processing {folder}  ({len(nii_files)} real scans)")

        saved_images = []

        for fname in tqdm(nii_files, desc=f"  {folder}", ncols=55):
            nii_path = os.path.join(folder_path, fname)
            # Output: study_0001.nii -> study_0001.png (no folder prefix)
            png_name = fname.replace(".nii", ".png")
            out_path = os.path.join(OUTPUT_DIR, png_name)
            try:
                pil_img = process_scan(nii_path, out_path)
                # Store just the base name without extension
                base_name = fname.replace(".nii", "")
                saved_images.append((base_name, pil_img))
                success += 1
            except Exception as e:
                failed.append(fname)
                print(f"\n    ERROR: {fname} -> {e}")

        # Augment CT-4 to reach 10 images
        if folder == "CT-4" and len(saved_images) < SCANS_PER_FOLDER:
            needed   = SCANS_PER_FOLDER - len(saved_images)
            aug_id   = 1
            aug_done = 0
            print(f"\n  Augmenting CT-4: generating {needed} extra images...")
            while aug_done < needed:
                base_name, base_img = saved_images[aug_done % len(saved_images)]
                aug_img  = augment_image(base_img, aug_id)
                # Output: study_1109_aug1.png (matches Excel Image_Name exactly)
                aug_name = f"{base_name}_aug{aug_id}.png"
                aug_path = os.path.join(OUTPUT_DIR, aug_name)
                aug_img.save(aug_path)
                print(f"    Saved: {aug_name}")
                aug_id   += 1
                aug_done += 1
                success  += 1

    print("\n" + "=" * 55)
    print(f"  Done!")
    print(f"  Saved  : {success} PNG images -> {OUTPUT_DIR}")
    if failed:
        print(f"  Failed : {len(failed)} -> {failed}")
    else:
        print(f"  Failed : 0")
    print("=" * 55)


if __name__ == "__main__":
    main()