# File    : step2_load_excel.py
# Purpose : Load existing ct_biomarker.xlsx,
#           verify image names match processed_images folder,
#           display dataset summary and severity distribution.

import os
import pandas as pd

EXCEL_PATH  = r"D:\COVID\ct_biomarker.xlsx"
IMAGES_DIR  = r"D:\COVID\processed_images"


def main():
    print("=" * 55)
    print("  Load Excel Dataset  -  step2_load_excel.py")
    print("=" * 55)

    # Load Excel
    df = pd.read_excel(EXCEL_PATH)
    print(f"\n  Excel loaded : {EXCEL_PATH}")
    print(f"  Rows         : {len(df)}")
    print(f"  Columns      : {df.columns.tolist()}")

    # Severity distribution
    print(f"\n  Severity distribution:")
    print(df["Severity"].value_counts().sort_index().to_string())

    # Check image names match processed folder
    print(f"\n  Checking image names against {IMAGES_DIR}...")
    missing = []
    for img_name in df["Image_Name"]:
        img_path = os.path.join(IMAGES_DIR, img_name)
        if not os.path.exists(img_path):
            missing.append(img_name)

    if missing:
        print(f"\n  Missing images: {len(missing)}")
        for m in missing:
            print(f"    {m}")
    else:
        print(f"  All {len(df)} images found in processed folder.")

    print("\n" + "=" * 55)
    print(f"  Done!")
    print("=" * 55)


if __name__ == "__main__":
    main()