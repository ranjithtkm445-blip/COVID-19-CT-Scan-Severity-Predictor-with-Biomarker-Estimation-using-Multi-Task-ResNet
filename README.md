Here is your **fully updated layman-friendly README**, rewritten cleanly while keeping all important details, but making it easy for anyone to understand.

---

# COVID-19 Severity Detection using CT Scans and AI

**Live Demo:** [https://huggingface.co/spaces/Ranjith445/covid-severity](https://huggingface.co/spaces/Ranjith445/covid-severity) 

---

## What is this project about?

Doctors use CT scans of the lungs to understand how serious a COVID-19 infection is.

But analyzing these scans:

* Takes time
* Requires medical expertise
* Can vary between doctors

This project builds an **AI system that automatically analyzes CT scans and gives a simple report**.

---

## What does this system do?

When you upload a CT scan, the system:

* Predicts how severe the infection is
* Estimates important health indicators (biomarkers)
* Highlights affected areas in the lungs
* Generates a detailed report

---

## Understanding Severity Levels

The system classifies COVID severity into 5 levels:

| Level | Meaning      | Lung Involvement |
| ----- | ------------ | ---------------- |
| CT-0  | No infection | 0%               |
| CT-1  | Mild         | < 25%            |
| CT-2  | Moderate     | 25–50%           |
| CT-3  | Severe       | 50–75%           |
| CT-4  | Critical     | > 75%            |

---

## What are Biomarkers?

Biomarkers are medical values that help doctors understand the condition of a patient.

This system estimates:

* **CRP** → inflammation level
* **NLR** → immune system response
* **D-dimer** → blood clot risk
* **LDH** → tissue damage

---

## How does it work (simple explanation)

The system works step by step:

### 1. Reads the CT scan

* CT scans are 3D images of the lungs
* The system selects an important slice for analysis

---

### 2. Prepares the image

* Cleans and resizes the image
* Makes it suitable for AI processing

---

### 3. Uses AI to analyze the lungs

* Detects patterns related to infection
* Identifies affected regions

---

### 4. Makes two predictions at once

* Severity level (CT-0 to CT-4)
* Biomarker values

---

### 5. Explains the result

* Highlights areas in the lungs that influenced the prediction
* Helps understand what the model focused on

---

### 6. Generates a report

* CT image
* Heatmap (highlighted regions)
* Severity level
* Biomarker values

---

## What data was used?

* Dataset: MosMedData (COVID-19 CT scans)
* Format: 3D CT scans

For this project:

* **50 scans used** (10 per severity level)
* Original dataset contains more than 1000 scans

---

## What results does it give?

* Model accuracy: **60%**
* Best validation achieved after 17 training epochs

This is a **demonstration-level model**, not optimized for clinical performance.

---

## Features of the application

* Upload CT scan
* Predict severity level
* Estimate biomarker values
* View highlighted lung regions (GradCAM)
* Download PDF report

---

## Technologies used

* Deep Learning: PyTorch
* Image Processing: NumPy, Pillow, nibabel
* Explainability: GradCAM
* Web App: Streamlit, Flask
* Reporting: ReportLab
* Deployment: Docker, Hugging Face

---

## Important Note

* Trained on a **small dataset (50 scans)**
* Biomarker values are **synthetically generated**

This project is built for **learning and demonstration purposes**.

---

## Limitations

* Small dataset may reduce accuracy
* Uses only 2D slices (not full 3D scans)
* Biomarkers are simulated, not real patient data
* Not tested in real clinical environments

---

## Future Improvements

* Use larger real-world datasets
* Replace synthetic biomarkers with real data
* Upgrade to 3D deep learning models
* Improve accuracy and reliability

---

## Disclaimer

This project is for educational and research purposes only.
It should not be used for medical diagnosis or treatment decisions.

---

## One-Line Summary

An AI system that analyzes lung CT scans to estimate COVID severity and health indicators while showing how it made its decision.

---

## Author

M. Ranjith Kumar
Hugging Face: [https://huggingface.co/Ranjith445](https://huggingface.co/Ranjith445)

---

