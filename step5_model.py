# File    : step5_model.py
# Purpose : Multi-task CNN model architecture.
#           ResNet18 backbone with shared feature layer
#           and two output heads — one for biomarker
#           regression and one for severity classification.

import torch
import torch.nn as nn
from torchvision import models


class MultiTaskCT(nn.Module):
    def __init__(self, num_biomarkers=4, num_severity_classes=5, dropout=0.4):
        super(MultiTaskCT, self).__init__()

        # ResNet18 backbone — pretrained on ImageNet
        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.backbone    = nn.Sequential(*list(backbone.children())[:-1])
        backbone_out_dim = 512

        # Shared layer
        self.shared = nn.Sequential(
            nn.Linear(backbone_out_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Head 1 — Biomarker regression (CRP, NLR, D_dimer, LDH)
        self.biomarker_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_biomarkers),
        )

        # Head 2 — Severity classification (CT-0 to CT-4)
        self.severity_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_severity_classes),
        )

    def forward(self, x):
        features = self.backbone(x)
        features = features.view(features.size(0), -1)
        shared   = self.shared(features)

        biomarkers = self.biomarker_head(shared)
        severity   = self.severity_head(shared)

        return biomarkers, severity


class MultiTaskLoss(nn.Module):
    # Combined loss using learnable uncertainty weights
    # Total = w1 * MSE(biomarkers) + w2 * CrossEntropy(severity)
    def __init__(self):
        super(MultiTaskLoss, self).__init__()
        self.log_var_bm  = nn.Parameter(torch.zeros(1))
        self.log_var_sev = nn.Parameter(torch.zeros(1))
        self.mse = nn.MSELoss()
        self.ce  = nn.CrossEntropyLoss()

    def forward(self, pred_bm, true_bm, pred_sev, true_sev):
        loss_bm  = self.mse(pred_bm, true_bm)
        loss_sev = self.ce(pred_sev, true_sev)

        precision_bm  = torch.exp(-self.log_var_bm)
        precision_sev = torch.exp(-self.log_var_sev)

        total = (precision_bm  * loss_bm  + self.log_var_bm +
                 precision_sev * loss_sev + self.log_var_sev)

        return total, loss_bm.item(), loss_sev.item()


if __name__ == "__main__":
    print("=" * 55)
    print("  Model Architecture Test  -  step5_model.py")
    print("=" * 55)

    model   = MultiTaskCT()
    loss_fn = MultiTaskLoss()

    total_params = sum(p.numel() for p in model.parameters())
    train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters    : {total_params:,}")
    print(f"  Trainable parameters: {train_params:,}")

    dummy_img = torch.randn(4, 3, 224, 224)
    dummy_bm  = torch.randn(4, 4)
    dummy_sev = torch.randint(0, 5, (4,))

    bm_pred, sev_pred       = model(dummy_img)
    total, l_bm, l_sev      = loss_fn(bm_pred, dummy_bm, sev_pred, dummy_sev)

    print(f"\n  Forward pass:")
    print(f"  Input shape        : {dummy_img.shape}")
    print(f"  Biomarker output   : {bm_pred.shape}")
    print(f"  Severity output    : {sev_pred.shape}")
    print(f"\n  Loss:")
    print(f"  Biomarker (MSE)    : {l_bm:.4f}")
    print(f"  Severity  (CE)     : {l_sev:.4f}")
    print(f"  Total              : {total.item():.4f}")
    print("\n  step5_model.py OK")
    print("=" * 55)