# File    : app.py
# Purpose : Flask backend for COVID Severity AI frontend.
#           Serves index.html, loads .nii files from upload folder,
#           runs prediction, returns CT slice, GradCAM, severity,
#           biomarkers and PDF report.

import os
import torch
import torch.nn.functional as F
import numpy as np
import nibabel as nib
from PIL import Image
from torchvision import transforms
from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas as rl_canvas
from reportlab.lib import colors
import base64
import io

from step5_model import MultiTaskCT

MODEL_PATH  = r"D:\COVID\saved_model\best_model.pth"
UPLOAD_DIR  = r"D:\COVID\upload"
RESULTS_DIR = r"D:\COVID\results"
IMG_SIZE    = (224, 224)

app = Flask(__name__)

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


def load_model():
    ckpt  = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    model = MultiTaskCT()
    model.load_state_dict(ckpt["model_state"])
    return model, ckpt["bm_means"], ckpt["bm_stds"]


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


def pil_to_base64(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


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


def run_prediction(nii_path):
    model, bm_means, bm_stds = load_model()
    pil_slice, tensor        = preprocess_nii(nii_path)

    model.train()
    gc            = GradCAM(model)
    _, sev_logits = model(tensor)
    sc_tmp        = int(torch.argmax(torch.softmax(sev_logits, dim=1)).item())
    cam           = gc.generate(tensor, sc_tmp)
    cam_img       = apply_heatmap(cam, pil_slice)

    model.eval()
    with torch.no_grad():
        pred_bm, pred_sev = model(tensor)

    probs        = torch.softmax(pred_sev, dim=1).squeeze().numpy()
    sc           = int(np.argmax(probs))
    bm           = denormalize(pred_bm.squeeze().numpy(), bm_means, bm_stds)
    crp, nlr, dd, ldh = bm

    os.makedirs(RESULTS_DIR, exist_ok=True)
    fname    = os.path.basename(nii_path)
    pdf_name = fname.replace(".nii", "_result.pdf")
    pdf_path = os.path.join(RESULTS_DIR, pdf_name)
    save_pdf(pil_slice, cam_img, fname, sc, probs,
             crp, nlr, dd, ldh, pdf_path)

    return {
        "severity_class": int(sc),
        "probabilities" : [float(p) for p in probs],
        "biomarkers"    : {
            "CRP"    : float(crp),
            "NLR"    : float(nlr),
            "D_dimer": float(dd),
            "LDH"    : float(ldh),
        },
        "ct_image" : pil_to_base64(pil_slice),
        "cam_image": pil_to_base64(cam_img),
        "pdf_name" : pdf_name,
    }


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/folder_files")
def folder_files():
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    files = sorted([f for f in os.listdir(UPLOAD_DIR) if f.endswith(".nii")])
    return jsonify({"files": files})


@app.route("/predict_folder", methods=["POST"])
def predict_folder():
    try:
        data     = request.get_json()
        fname    = secure_filename(data.get("filename", ""))
        nii_path = os.path.join(UPLOAD_DIR, fname)
        if not os.path.exists(nii_path):
            return jsonify({"error": "File not found: " + fname}), 400
        return jsonify(run_prediction(nii_path))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/download/<filename>")
def download(filename):
    path = os.path.join(RESULTS_DIR, secure_filename(filename))
    if os.path.exists(path):
        return send_file(path, as_attachment=True)
    return "File not found", 404


if __name__ == "__main__":
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print("=" * 55)
    print("  COVID Severity AI  -  Flask Frontend")
    print("=" * 55)
    print("  Open: http://localhost:5000")
    print("  Upload folder:", UPLOAD_DIR)
    print("=" * 55)
    app.run(debug=False, port=5000)