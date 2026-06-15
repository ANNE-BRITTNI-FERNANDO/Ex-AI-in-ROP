"""Corrected high-level architecture diagram (matches the leakage-free pipeline)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path

fig, ax = plt.subplots(figsize=(11, 14))
ax.set_xlim(0, 10); ax.set_ylim(0, 23); ax.axis("off")

bands = [
    ("LAYER 1 — INPUT", 20.3, "#DCEAF7"),
    ("LAYER 2 — DATA PROCESSING (split-first, leakage-free)", 16.7, "#DDF2E3"),
    ("LAYER 3 — MODEL", 12.8, "#FBE9D0"),
    ("LAYER 4 — XAI GENERATION (4 methods)", 9.0, "#EAD9F2"),
    ("LAYER 5 — QUANTITATIVE VALIDATION & OUTPUT", 5.0, "#FCF3CF"),
    ("LAYER 6 — CLINICAL INTERFACE", 1.4, "#D6DBDF"),
]
for title, y, color in bands:
    ax.add_patch(FancyBboxPatch((0.2, y-0.1), 9.6, 1.0 if False else 0.0,
                 boxstyle="round,pad=0.02", fc=color, ec="none"))
    ax.text(5, y+1.15, title, ha="center", va="center", fontsize=11, fontweight="bold")

def box(x, y, w, h, text, fc="white"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04",
                 fc=fc, ec="#444", lw=1.0))
    ax.text(x+w/2, y+h/2, text, ha="center", va="center", fontsize=8.2)

def band_bg(y0, y1, color):
    ax.add_patch(plt.Rectangle((0.2, y0), 9.6, y1-y0, fc=color, ec="none", alpha=0.45, zorder=0))

# band backgrounds
band_bg(19.4, 21.4, "#DCEAF7")
band_bg(15.4, 18.6, "#DDF2E3")
band_bg(11.4, 14.6, "#FBE9D0")
band_bg(7.4, 10.6, "#EAD9F2")
band_bg(3.0, 6.6, "#FCF3CF")
band_bg(0.4, 2.6, "#D6DBDF")

# L1
box(1.0, 19.7, 3.6, 1.2, "HVDROPDB\n185 fundus images\n(97 with expert pixel masks)", "#fff")
box(5.4, 19.7, 3.6, 1.2, "Czech ROP\n6,004 images\n(architecture comparison)", "#fff")
# L2
box(0.7, 15.6, 2.7, 1.2, "Preprocessing\nQuality filter · Resize 224×224\n· ImageNet normalisation", "#fff")
box(3.7, 15.6, 2.7, 1.2, "Stratified split FIRST\n70/15/15 → 129/27/29\n(seed 42)", "#eaffea")
box(6.7, 15.6, 2.6, 1.2, "Augment TRAIN ONLY (online)\nHFlip · Rot ±15° · Bright/Contrast\n±20% · CLAHE  (val/test original)", "#eaffea")
# L3
box(0.9, 11.7, 3.4, 1.4, "ResNet50 (transfer learning)\nImageNet-pretrained · layer4 fine-tuned\nSelected vs EfficientNet-B0 & DenseNet121", "#fff")
box(4.6, 11.7, 2.6, 1.4, "Training config\nAdam LR=1e-3 · CE loss\nReduceLROnPlateau\nEarly stop (pat=10) · batch 16", "#fff")
box(7.5, 11.7, 1.9, 1.4, "Outputs\nDiagnosis +\nLayer4 maps\n7×7×2048", "#fff")
# L4
box(0.7, 7.7, 2.0, 1.3, "Grad-CAM\ngradient-based\n~0.5 s/img", "#fff")
box(2.9, 7.7, 2.0, 1.3, "Integrated\nGradients\npath-integral", "#fff")
box(5.1, 7.7, 2.0, 1.3, "SHAP\nGradientExplainer\ngame-theoretic", "#fff")
box(7.3, 7.7, 2.0, 1.3, "LIME\nsuperpixel\nperturbation", "#fff")
# L5
box(0.7, 3.3, 2.7, 1.5, "Quantitative validation\nOracle threshold (0.1–0.9)\nIoU & Dice vs expert masks\nFaithfulness (insertion/deletion)", "#fff")
box(3.7, 3.3, 2.7, 1.5, "Experiments\n• Multi-arch comparison\n• 4-method XAI comparison\n• Lesion-specific · Augmentation", "#fff")
box(6.7, 3.3, 2.6, 1.5, "Statistics & output\nANOVA + pairwise t-tests\nMetrics · plots · reports", "#fff")
# L6
box(1.6, 0.6, 3.2, 1.2, "Gradio web interface\nImage upload · inference", "#fff")
box(5.2, 0.6, 3.2, 1.2, "Diagnosis + confidence\n+ Grad-CAM heatmap", "#fff")

# arrows between layers (centre)
for y0, y1 in [(19.6,18.7),(15.5,14.7),(11.6,10.7),(7.6,6.7),(3.2,2.7)]:
    ax.add_patch(FancyArrowPatch((5,y0),(5,y1), arrowstyle="-|>", mutation_scale=16, color="#555", lw=1.4))

ax.text(5, 22.5, "High-Level System Architecture — Quantitative XAI Validation for ROP Detection",
        ha="center", fontsize=12.5, fontweight="bold")

Path("results/canonical_figures").mkdir(parents=True, exist_ok=True)
fig.savefig("results/canonical_figures/architecture_diagram_corrected.png", dpi=200, bbox_inches="tight")
print("Saved results/canonical_figures/architecture_diagram_corrected.png")
