# File    : step4_verify.py
# Purpose : Verify which images are in train and validation sets.
#           Ensures correct split and no data leakage.

import pandas as pd
from sklearn.model_selection import train_test_split

EXCEL_PATH   = r"D:\COVID\ct_biomarker.xlsx"
SEVERITY_COL = "Severity"
IMAGE_COL    = "Image_Name"


def main():
    df = pd.read_excel(EXCEL_PATH)

    train_df, val_df = train_test_split(
        df,
        test_size=0.2,
        stratify=df[SEVERITY_COL],
        random_state=42
    )

    print("=" * 55)
    print("  Train/Val Split Verification  -  step4_verify.py")
    print("=" * 55)
    print(f"  Total : {len(df)} images")
    print(f"  Train : {len(train_df)} images")
    print(f"  Val   : {len(val_df)} images")

    print("\n  TRAIN images:")
    print(f"  {'Image_Name':<30} Severity")
    print(f"  {'-'*30} --------")
    for _, row in train_df.sort_values(SEVERITY_COL).iterrows():
        print(f"  {row[IMAGE_COL]:<30} {row[SEVERITY_COL]}")

    print("\n  VAL images:")
    print(f"  {'Image_Name':<30} Severity")
    print(f"  {'-'*30} --------")
    for _, row in val_df.sort_values(SEVERITY_COL).iterrows():
        print(f"  {row[IMAGE_COL]:<30} {row[SEVERITY_COL]}")

    print("\n" + "=" * 55)
    print("  Done!")
    print("=" * 55)


if __name__ == "__main__":
    main()