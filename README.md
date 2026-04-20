# COVID-19-CT-Scan-Severity-Predictor-with-Biomarker-Estimation-using-Multi-Task-ResNet
An end-to-end multi-task deep learning system that predicts COVID-19 severity and clinical biomarkers from lung CT scans with GradCAM explainability and PDF report generation.
# COVID Severity AI — Multi-Task Deep Learning Pipeline

> An end-to-end multi-task deep learning system that predicts COVID-19 severity and clinical biomarkers from lung CT scans with GradCAM explainability and PDF report generation.

---

## Live Demo

Hugging Face Space: [https://huggingface.co/spaces/Ranjith445/covid-severity](https://huggingface.co/spaces/Ranjith445/covid-severity)

---

## Project Overview

This project builds a multi-task AI system that takes a lung CT scan (.nii format) as input and simultaneously predicts:

- **Severity Level** — CT-0 (No findings) to CT-4 (Critical, more than 75% lung involvement)
- **Clinical Biomarkers** — CRP, NLR, D-dimer, LDH

The model uses a ResNet18 backbone with a shared feature layer that branches into two output heads. GradCAM highlights the lung regions that influenced the prediction. A PDF report is automatically generated with the CT slice, heatmap, severity prediction, and biomarker values.

---

## Pipeline

```
CT Scan (.nii)
      |
step1_preprocess.py   -- Extract middle slice, normalize, resize to 224x224
      |
step2_load_excel.py   -- Load and verify ct_biomarker.xlsx
      |
step3_dataset.py      -- PyTorch dataset loader, train/val split
      |
step4_verify.py       -- Verify train/val image names
      |
step5_model.py        -- ResNet18 multi-task model architecture
      |
step6_train.py        -- Train 50 epochs, save best model
      |
step7_inference.py    -- Inference with GradCAM + PDF report
      |
app.py                -- Flask / Streamlit frontend
```

---

## Model Architecture

```
CT Slice [3 x 224 x 224]
        |
ResNet18 Backbone (pretrained ImageNet)
        |
Global Average Pool -> [512]
        |
Shared FC Layer (512 -> 256) + ReLU + Dropout(0.4)
        |              |
Biomarker Head    Severity Head
FC(256->128->4)   FC(256->128->5)
        |              |
CRP, NLR,         CT-0 to CT-4
D-dimer, LDH
```

---

## Dataset

| Item | Detail |
|---|---|
| Source | MosMedData (Kaggle) |
| Format | .nii (3D volumetric CT scans) |
| Total scans | 50 (10 per severity class) |
| Severity classes | 5 (CT-0 to CT-4) |
| Biomarkers | Synthetically generated based on clinical ranges |
| CT-4 | 2 real scans + 8 augmented = 10 total |
| Train split | 40 images (8 per class) |
| Val split | 10 images (2 per class) |

### Biomarker Clinical Ranges

| Severity | CRP (mg/L) | NLR | D-dimer (mg/L) | LDH (U/L) |
|---|---|---|---|---|
| CT-0 | 2 - 15 | 1.0 - 3.0 | 0.1 - 0.5 | 140 - 250 |
| CT-1 | 15 - 40 | 3.0 - 5.0 | 0.5 - 1.0 | 250 - 350 |
| CT-2 | 40 - 80 | 5.0 - 8.0 | 1.0 - 2.0 | 350 - 500 |
| CT-3 | 80 - 150 | 8.0 - 13.0 | 2.0 - 4.0 | 500 - 700 |
| CT-4 | 150 - 200 | 13.0 - 20.0 | 4.0 - 6.0 | 700 - 900 |

---

## Training Results

| Metric | Value |
|---|---|
| Best epoch | 17 |
| Best val loss | 1.79 |
| Best val accuracy | 60% |
| Total parameters | 11,374,793 |
| Training device | CPU |
| Epochs | 50 |
| Batch size | 8 |
| Learning rate | 0.001 |
| Optimizer | Adam |

---

## Project Structure

```
D:\COVID\
|-- step1_preprocess.py     <- Extract CT slices, augment CT-4
|-- step2_load_excel.py     <- Load and verify Excel dataset
|-- step3_dataset.py        <- PyTorch dataset loader
|-- step4_verify.py         <- Train/val split verification
|-- step5_model.py          <- ResNet18 multi-task model
|-- step6_train.py          <- Training script
|-- step7_inference.py      <- Local inference + PDF report
|-- app.py                  <- Flask local frontend
|-- ct_biomarker.xlsx       <- Biomarker dataset (50 rows)
|-- templates\
|   |-- index.html          <- UI layout
|-- static\
|   |-- style.css           <- Dark clinical theme
|   |-- script.js           <- Frontend logic
|-- processed_images\       <- 50 preprocessed PNG slices
|-- saved_model\
|   |-- best_model.pth      <- Trained model weights
|-- upload\                 <- CT scans for inference
```

---

## Installation

```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install torch torchvision nibabel numpy pillow
pip install pandas openpyxl scikit-learn tqdm
pip install flask reportlab streamlit
```

---

## How to Run

```bash
# Step 1 - Preprocess CT scans
python step1_preprocess.py

# Step 2 - Load and verify Excel
python step2_load_excel.py

# Step 3 - Test dataset loader
python step3_dataset.py

# Step 4 - Verify train/val split
python step4_verify.py

# Step 5 - Test model architecture
python step5_model.py

# Step 6 - Train the model
python step6_train.py

# Step 7 - Run inference
python step7_inference.py

# Run local Flask app
python app.py
# Open http://localhost:5000

# Run Streamlit app
streamlit run app.py
# Open http://localhost:8501
```

---

## Severity Levels

| Level | Meaning | Lung Involvement |
|---|---|---|
| CT-0 | No findings | 0% |
| CT-1 | Minimal | Less than 25% |
| CT-2 | Moderate | 25 to 50% |
| CT-3 | Significant | 50 to 75% |
| CT-4 | Critical | More than 75% |

---

## Tech Stack

| Category | Tools |
|---|---|
| Deep Learning | PyTorch, TorchVision |
| CT Processing | nibabel, numpy |
| Image Processing | Pillow |
| Explainability | GradCAM |
| PDF Generation | ReportLab |
| Local Frontend | Flask, HTML, CSS, JS |
| Cloud Deployment | Streamlit, Hugging Face Spaces |
| Version Control | Git, Git LFS |

---

## Limitations

| Limitation | Impact |
|---|---|
| Synthetic biomarkers | Model learns simulation, not real physiology |
| Small dataset (50 scans) | Overfitting |
| 2D slice only | Misses 3D spatial context |
| CPU training | Slow convergence |

---

## Future Work

- Replace synthetic biomarkers with real clinical data
- Upgrade to 3D CNN for full volumetric input
- Increase dataset size
- Add SHAP values for additional explainability
- Deploy as DICOM-compatible clinical tool

---

## Disclaimer

This project is for **research and demonstration purposes only**. The biomarker values are synthetically generated and not from real patients. This tool is **not intended for clinical diagnosis**.

---

## Author

**M. Ranjith Kumar**
Hugging Face: [Ranjith445](https://huggingface.co/Ranjith445)
