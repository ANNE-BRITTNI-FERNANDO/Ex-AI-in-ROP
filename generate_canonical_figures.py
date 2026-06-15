"""
Generate corrected, canonical figures for Chapter 6 directly from raw result files.
Outputs to results/canonical_figures/. Numbers match CANONICAL_METRICS.md exactly.
"""
import json, numpy as np, itertools
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

OUT = Path("results/canonical_figures"); OUT.mkdir(parents=True, exist_ok=True)

# ---- recompute canonical IoU/Dice from raw per-pair data ----
def load(method):
    if method == "Grad-CAM":
        d = json.load(open("results/iou_dice_metrics/detailed_results.json"))
        return np.array([x["iou_optimal"] for x in d]), np.array([x["dice_optimal"] for x in d])
    if method == "Integrated\nGradients":
        d = json.load(open("results/ig_visualizations/ig_results.json"))["pair_results"]
        return np.array([x["iou"] for x in d]), np.array([x["dice"] for x in d])
    if method == "SHAP":
        d = json.load(open("results/shap_visualizations/shap_results.json"))["images"]
        return np.array([x["iou"] for x in d]), np.array([x["dice"] for x in d])
    if method == "LIME":
        d = json.load(open("results/lime_visualizations/lime_results.json"))["images"]
        return np.array([x["iou"] for x in d]), np.array([x["dice"] for x in d])

methods = ["Grad-CAM", "Integrated\nGradients", "LIME", "SHAP"]
iou = {m: load(m)[0] for m in methods}
dice = {m: load(m)[1] for m in methods}

# ---- Figure 6.3: IoU/Dice bar chart (corrected) ----
fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
x = np.arange(len(methods)); w = 0.6
colors = ["#2c7fb8", "#41b6c4", "#a1dab4", "#fec44f"]
for a, vals, title in [(ax[0], iou, "Mean IoU"), (ax[1], dice, "Mean Dice")]:
    means = [vals[m].mean() for m in methods]
    sds = [vals[m].std(ddof=1) for m in methods]
    a.bar(x, means, w, yerr=sds, capsize=4, color=colors, edgecolor="black", linewidth=0.6)
    for i, mn in enumerate(means):
        a.text(i, mn + (sds[i] if sds[i] else 0) + 0.003, f"{mn:.4f}", ha="center", fontsize=9)
    a.set_xticks(x); a.set_xticklabels([m.replace("\n", " ") for m in methods], fontsize=9)
    a.set_title(f"{title} (N=111 pairs, oracle threshold)", fontsize=10)
    a.set_ylabel(title); a.grid(axis="y", alpha=0.3)
fig.suptitle("XAI Spatial Overlap with Expert Masks — Grad-CAM leads (p<1e-18)", fontsize=11, y=1.0)
fig.tight_layout(); fig.savefig(OUT / "fig6_3_iou_dice_corrected.png", dpi=200, bbox_inches="tight"); plt.close(fig)

# ---- Figure 6.1: multi-architecture comparison ----
comp = json.load(open("results/multi_arch_comparison/comparison_summary.json"))["models"]
archs = ["ResNet50", "EfficientNet-B0", "DenseNet121"]
metr = ["test_accuracy", "f1_score", "auc_roc", "recall_sensitivity", "specificity"]
labels = ["Accuracy", "F1", "AUC-ROC", "Sensitivity", "Specificity"]
fig, ax = plt.subplots(figsize=(9, 4.8))
xa = np.arange(len(labels)); bw = 0.25
for i, arch in enumerate(archs):
    vals = [comp[arch][m] for m in metr]
    ax.bar(xa + (i - 1) * bw, vals, bw, label=arch, color=colors[i], edgecolor="black", linewidth=0.5)
ax.set_xticks(xa); ax.set_xticklabels(labels); ax.set_ylim(0.90, 1.0)
ax.set_ylabel("Score"); ax.set_title("Multi-Architecture Comparison — Czech ROP test set (n=1,466)")
ax.legend(); ax.grid(axis="y", alpha=0.3)
fig.tight_layout(); fig.savefig(OUT / "fig6_1_multiarch.png", dpi=200, bbox_inches="tight"); plt.close(fig)

# ---- Figure: early-stopping / training curves (augmented model) ----
h = json.load(open("models/augmented_results.json"))["history"]
ep = np.arange(1, len(h["train_acc"]) + 1)
best = int(np.argmax(h["val_acc"])) + 1
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
ax[0].plot(ep, h["train_acc"], "-o", ms=3, label="Train acc")
ax[0].plot(ep, h["val_acc"], "-s", ms=3, label="Val acc")
ax[0].axvline(best, color="red", ls="--", label=f"Best epoch {best} ({max(h['val_acc']):.3f})")
ax[0].set_xlabel("Epoch"); ax[0].set_ylabel("Accuracy"); ax[0].set_title("Augmented model — accuracy"); ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)
ax[1].plot(ep, h["train_loss"], "-o", ms=3, label="Train loss")
ax[1].plot(ep, h["val_loss"], "-s", ms=3, label="Val loss")
ax[1].axvline(best, color="red", ls="--")
ax[1].set_xlabel("Epoch"); ax[1].set_ylabel("Loss"); ax[1].set_title("Augmented model — loss"); ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)
fig.suptitle("Training convergence & early stopping (augmented HVDROPDB ResNet50)", y=1.01)
fig.tight_layout(); fig.savefig(OUT / "fig6_aug_training_curves.png", dpi=200, bbox_inches="tight"); plt.close(fig)

# ---- non-augmented overfitting curve (the "why stop at epoch 8" proof) ----
h2 = json.load(open("models/training_history.json"))
ep2 = np.arange(1, len(h2["train_acc"]) + 1)
best2 = h2["best_epoch"]
fig, ax = plt.subplots(figsize=(7, 4.2))
ax.plot(ep2, h2["train_acc"], "-o", ms=4, label="Train acc")
ax.plot(ep2, h2["val_acc"], "-s", ms=4, label="Val acc")
ax.axvline(best2, color="red", ls="--", label=f"Best/early-stop epoch {best2}")
ax.fill_between(ep2, h2["train_acc"], h2["val_acc"], color="orange", alpha=0.15, label="Overfitting gap")
ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy")
ax.set_title("Non-augmented model: train→100% while val plateaus = overfitting")
ax.legend(fontsize=8); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(OUT / "fig6_overfitting_proof.png", dpi=200, bbox_inches="tight"); plt.close(fig)

print("Saved figures to", OUT)
for p in sorted(OUT.glob("*.png")):
    print(" -", p)
