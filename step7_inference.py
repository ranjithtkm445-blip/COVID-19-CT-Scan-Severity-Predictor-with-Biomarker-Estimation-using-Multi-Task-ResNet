# File    : step7_inference.py
# Purpose : Load trained model, preprocess a selected .nii CT scan,
#           run prediction, generate GradCAM heatmap,
#           display predicted severity and biomarkers,
#           save result as PDF report.

import os
import torch
import torch.nn.functional as F
import numpy as np
import nibabel as nib
from PIL import Image
from torchvision import transforms
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas as rl_canvas
from reportlab.lib import colors

from step5_model import MultiTaskCT

MODEL_PATH  = r"D:\COVID\saved_model\best_model.pth"
UPLOAD_DIR  = r"D:\COVID\upload"
RESULTS_DIR = r"D:\COVID\results"
IMG_SIZE    = (224, 224)

SEVERITY_LABELS = {
    0: "CT-0 - No Findings",
    1: "CT-1 - Minimal (less than 25% lung involvement)",
    2: "CT-2 - Moderate (25 to 50% lung involvement)",
    3: "CT-3 - Significant (50 to 75% lung involvement)",
    4: "CT-4 - Critical (more than 75% lung involvement)",
}

SEVERITY_COLORS_RL = {
    0: colors.HexColor("#2ECC71"),
    1: colors.HexColor("#3498DB"),
    2: colors.HexColor("#F1C40F"),
    3: colors.HexColor("#E67E22"),
    4: colors.HexColor("#E74C3C"),
}


class GradCAM:
    def __init__(self, model):
        self.model       = model
        self.gradients   = None
        self.activations = None
        target = list(model.backbone.children())[-3][-1].conv2
        target.register_forward_hook(self._save_activation)
        target.register_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, tensor, class_idx):
        self.model.zero_grad()
        _, sev_logits = self.model(tensor)
        sev_logits[0, class_idx].backward()
        w   = self.gradients.mean(dim=[0, 2, 3])
        cam = self.activations[0].clone()
        for i, wi in enumerate(w):
            cam[i] *= wi
        cam = F.relu(cam.mean(dim=0))
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        cam = F.interpolate(cam.unsqueeze(0).unsqueeze(0),
                            size=IMG_SIZE, mode="bilinear", align_corners=False)
        return cam.squeeze().numpy()


def apply_heatmap(cam, pil_img):
    c = (cam * 255).astype(np.uint8)
    h = np.zeros((*c.shape, 3), dtype=np.uint8)
    h[:, :, 0] = c
    h[:, :, 1] = (c * 0.3).astype(np.uint8)
    return Image.blend(pil_img.convert("RGB"), Image.fromarray(h), alpha=0.5)


def preprocess_nii(nii_path):
    vol = nib.load(nii_path).get_fdata()
    if vol.ndim == 4:
        vol = vol[..., 0]
    s = vol[:, :, vol.shape[2] // 2]
    s = np.clip(s, -1000, 400)
    s = s - s.min()
    if s.max() > 0:
        s = s / s.max()
    s   = (s * 255).astype(np.uint8)
    pil = Image.fromarray(s).convert("RGB").resize(IMG_SIZE, Image.BILINEAR)
    t   = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])(pil).unsqueeze(0)
    t.requires_grad_(True)
    return pil, t


def denormalize(bm_norm, bm_means, bm_stds):
    return bm_norm * bm_stds + bm_means


def save_pdf(pil_slice, cam_img, fname, sc, probs,
             crp, nlr, dd, ldh, pdf_path):
    w, h = letter
    c    = rl_canvas.Canvas(pdf_path, pagesize=letter)

    c.setFillColor(colors.HexColor("#0D0D14"))
    c.rect(0, 0, w, h, fill=1, stroke=0)

    c.setFillColor(colors.white)
    c.setFont("Helvetica-Bold", 20)
    c.drawString(40, h-55, "COVID Severity Analysis Report")
    c.setFont("Helvetica", 11)
    c.setFillColor(colors.HexColor("#888888"))
    c.drawString(40, h-75, "File: " + fname)
    c.setStrokeColor(colors.HexColor("#333333"))
    c.line(40, h-88, w-40, h-88)

    ct_tmp  = os.path.join(RESULTS_DIR, "_ct.png")
    cam_tmp = os.path.join(RESULTS_DIR, "_cam.png")
    pil_slice.save(ct_tmp)
    cam_img.save(cam_tmp)

    c.setFillColor(colors.HexColor("#888888"))
    c.setFont("Helvetica", 10)
    c.drawString(40,  h-108, "Original CT Slice")
    c.drawString(260, h-108, "GradCAM Heatmap")
    c.drawImage(ct_tmp,  40,  h-320, width=200, height=200)
    c.drawImage(cam_tmp, 260, h-320, width=200, height=200)

    sc_col = SEVERITY_COLORS_RL[sc]
    c.setFillColor(colors.HexColor("#888888"))
    c.setFont("Helvetica-Bold", 12)
    c.drawString(480, h-108, "SEVERITY")
    c.setFillColor(sc_col)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(480, h-130, "CT-" + str(sc))
    c.setFont("Helvetica", 10)
    c.drawString(480, h-148, SEVERITY_LABELS[sc].split("-")[1].strip())

    c.setFont("Helvetica", 10)
    c.setFillColor(colors.HexColor("#888888"))
    c.drawString(480, h-170, "Confidence:")
    for i, p in enumerate(probs):
        y  = h-188-i*22
        bw = int(p*140)
        c.setFillColor(colors.HexColor("#222233"))
        c.rect(480, y, 140, 14, fill=1, stroke=0)
        if bw > 0:
            c.setFillColor(SEVERITY_COLORS_RL[i])
            c.rect(480, y, bw, 14, fill=1, stroke=0)
        c.setFillColor(colors.white)
        c.setFont("Helvetica", 9)
        c.drawString(625, y+2, "CT-{}: {:.0f}%".format(i, p*100))

    c.line(40, h-340, w-40, h-340)
    c.setFillColor(colors.white)
    c.setFont("Helvetica-Bold", 13)
    c.drawString(40, h-365, "Predicted Biomarkers")

    col_x = [40, 200, 360, 480]
    hdrs  = ["Biomarker", "Predicted", "Normal Range", "Status"]
    ry    = h-390
    c.setFillColor(colors.HexColor("#1E1E2E"))
    c.rect(38, ry-4, w-80, 20, fill=1, stroke=0)
    c.setFillColor(colors.HexColor("#888888"))
    c.setFont("Helvetica-Bold", 10)
    for i, hdr in enumerate(hdrs):
        c.drawString(col_x[i], ry, hdr)

    rows = [
        ("CRP",     "{:.1f} mg/L".format(crp),  "< 10 mg/L",   crp > 10),
        ("NLR",     "{:.1f}".format(nlr),         "1.0 - 3.0",  nlr > 3),
        ("D-dimer", "{:.2f} mg/L".format(dd),    "< 0.5 mg/L",  dd > 0.5),
        ("LDH",     "{} U/L".format(int(ldh)),   "140-280 U/L", ldh > 280),
    ]
    for ri, (n, v, nr, high) in enumerate(rows):
        y2 = ry-22-ri*22
        if ri % 2 == 0:
            c.setFillColor(colors.HexColor("#16161E"))
            c.rect(38, y2-4, w-80, 20, fill=1, stroke=0)
        c.setFillColor(colors.white)
        c.setFont("Helvetica", 10)
        c.drawString(col_x[0], y2, n)
        c.drawString(col_x[1], y2, v)
        c.setFillColor(colors.HexColor("#888888"))
        c.drawString(col_x[2], y2, nr)
        c.setFillColor(colors.HexColor("#E74C3C") if high else colors.HexColor("#2ECC71"))
        c.setFont("Helvetica-Bold", 10)
        c.drawString(col_x[3], y2, "HIGH" if high else "Normal")

    c.line(40, 60, w-40, 60)
    c.setFillColor(colors.HexColor("#555555"))
    c.setFont("Helvetica", 9)
    c.drawString(40, 45, "COVID Severity AI Pipeline  |  For research use only")
    c.save()

    for tmp in [ct_tmp, cam_tmp]:
        if os.path.exists(tmp):
            os.remove(tmp)


def predict(nii_path):
    fname = os.path.basename(nii_path)

    print("=" * 55)
    print("  CT Scan Inference  -  step7_inference.py")
    print("=" * 55)
    print(f"  File : {fname}")

    ckpt  = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    model = MultiTaskCT()
    model.load_state_dict(ckpt["model_state"])
    bm_means = ckpt["bm_means"]
    bm_stds  = ckpt["bm_stds"]

    print("  Preprocessing...")
    pil_slice, tensor = preprocess_nii(nii_path)

    print("  Generating GradCAM...")
    model.train()
    gc            = GradCAM(model)
    _, sev_logits = model(tensor)
    sc_tmp        = int(torch.argmax(torch.softmax(sev_logits, dim=1)).item())
    cam           = gc.generate(tensor, sc_tmp)
    cam_img       = apply_heatmap(cam, pil_slice)

    print("  Running prediction...")
    model.eval()
    with torch.no_grad():
        pred_bm, pred_sev = model(tensor)

    probs        = torch.softmax(pred_sev, dim=1).squeeze().numpy()
    sc           = int(np.argmax(probs))
    bm           = denormalize(pred_bm.squeeze().numpy(), bm_means, bm_stds)
    crp, nlr, dd, ldh = bm

    print("\n" + "=" * 55)
    print("  PREDICTION RESULTS")
    print("=" * 55)
    print(f"\n  Severity  : {SEVERITY_LABELS[sc]}")
    print(f"\n  Confidence per class:")
    for i, prob in enumerate(probs):
        bar = "#" * int(prob * 20)
        print("    CT-{} : {:<22} {:.1f}%".format(i, bar, prob * 100))
    print(f"\n  Predicted Biomarkers:")
    print(f"    CRP     : {crp:.1f}  mg/L   (normal < 10)")
    print(f"    NLR     : {nlr:.1f}         (normal 1-3)")
    print(f"    D-dimer : {dd:.2f} mg/L   (normal < 0.5)")
    print(f"    LDH     : {int(ldh)}   U/L    (normal 140-280)")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    pdf_path = os.path.join(RESULTS_DIR, fname.replace(".nii", "_result.pdf"))
    save_pdf(pil_slice, cam_img, fname, sc, probs,
             crp, nlr, dd, ldh, pdf_path)

    print(f"\n  PDF saved -> {pdf_path}")
    print("=" * 55)
    os.startfile(pdf_path)


def main():
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    nii_files = sorted([f for f in os.listdir(UPLOAD_DIR) if f.endswith(".nii")])

    if not nii_files:
        print("No .nii file found in", UPLOAD_DIR)
        print("Please copy your CT scan into:", UPLOAD_DIR)
        return

    print("=" * 55)
    print("  Select a CT scan to analyze:")
    print("=" * 55)
    for i, fname in enumerate(nii_files):
        print(f"  {i+1}. {fname}")
    print("=" * 55)

    while True:
        try:
            choice = int(input("\n  Enter number (1-{}): ".format(len(nii_files))))
            if 1 <= choice <= len(nii_files):
                break
            else:
                print("  Invalid choice. Try again.")
        except ValueError:
            print("  Please enter a number.")

    selected = nii_files[choice - 1]
    print(f"\n  Selected: {selected}")
    predict(os.path.join(UPLOAD_DIR, selected))


if __name__ == "__main__":
    main()