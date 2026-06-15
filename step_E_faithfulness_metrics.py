"""
STEP E: FAITHFULNESS METRICS — Insertion & Deletion AUC
Evaluates XAI methods by measuring how much the model's confidence
changes when important pixels are progressively revealed (insertion)
or removed (deletion).

- Deletion AUC: Start with full image, progressively blank out pixels
  ordered by importance (highest attribution first).
  Good XAI: confidence drops sharply → low AUC (blanketing important pixels hurts).

- Insertion AUC: Start with blank image, progressively reveal pixels
  ordered by importance (highest attribution first).
  Good XAI: confidence rises sharply → high AUC (revealing important pixels helps).

A faithful explanation = high Insertion AUC + low Deletion AUC.
The gap (Insertion - Deletion) is the Faithfulness Score.

Uses: stored heatmaps (.npy) + model + Czech test images (no masks needed)
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from pathlib import Path
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

# ─── CONFIG ──────────────────────────────────────────────────────────────────
MODEL_PATH    = Path("models/augmented_best_model.pth")
GRADCAM_DIR   = Path("results/gradcam_visualizations/heatmaps")
IG_DIR        = Path("results/ig_visualizations/heatmaps")
GRADCAM_META  = Path("results/gradcam_visualizations/gradcam_results.json")
IG_META       = Path("results/ig_visualizations/ig_results.json")
OUTPUT_DIR    = Path("results/faithfulness_metrics")
N_STEPS       = 50     # number of pixel-reveal steps (50 points = smoother AUC curve)
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

print("=" * 70)
print("STEP E: FAITHFULNESS METRICS (Insertion & Deletion AUC)")
print("=" * 70)
print(f"Device: {DEVICE} | Steps: {N_STEPS}")

# ─── LOAD MODEL ──────────────────────────────────────────────────────────────
def load_model():
    m = models.resnet50(weights=None)
    m.fc = nn.Linear(m.fc.in_features, 2)
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    sd = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    m.load_state_dict(sd)
    m.eval()
    return m.to(DEVICE)

model = load_model()
print("Model loaded.")

def get_confidence(tensor, target_class):
    with torch.no_grad():
        out   = model(tensor)
        probs = torch.softmax(out, dim=1)
    return probs[0, target_class].item()

def load_image(path):
    img = Image.open(path).convert("RGB").resize((224, 224))
    t   = preprocess(img).unsqueeze(0).to(DEVICE)
    return t, np.array(img) / 255.0

# ─── INSERTION & DELETION CURVES ─────────────────────────────────────────────
def compute_insertion_deletion(img_tensor, img_arr, heatmap, target_class, n_steps=20):
    """
    Compute insertion and deletion AUC scores.

    Deletion: mask out the top-k% most important pixels progressively.
    Insertion: reveal only the top-k% most important pixels progressively.

    Returns: (insertion_auc, deletion_auc, insertion_curve, deletion_curve)
    """
    h, w, c = img_arr.shape   # 224, 224, 3
    total_pixels = h * w

    # Flatten heatmap and get indices sorted by importance (descending)
    flat_heatmap = heatmap.flatten()
    sorted_indices = np.argsort(flat_heatmap)[::-1]  # most important first

    # Step sizes: 0%, 5%, 10%, ..., 100% of pixels
    percentages = np.linspace(0, 1, n_steps + 1)
    insertion_scores  = []
    deletion_scores   = []

    # Baseline for insertion: blurred full image (less informative than black)
    # Use Gaussian-blurred image as the "uninformative" baseline
    from scipy.ndimage import gaussian_filter
    blurred = gaussian_filter(img_arr, sigma=10)  # strong blur

    for pct in percentages:
        k = int(pct * total_pixels)

        # ── DELETION: full image with top-k pixels blanked (set to mean) ──
        del_img = img_arr.copy()
        if k > 0:
            top_k_flat = sorted_indices[:k]
            rows, cols  = np.unravel_index(top_k_flat, (h, w))
            del_img[rows, cols, :] = 0.5  # replace with gray (mean)
        del_pil  = Image.fromarray((del_img * 255).astype(np.uint8))
        del_t    = preprocess(del_pil).unsqueeze(0).to(DEVICE)
        deletion_scores.append(get_confidence(del_t, target_class))

        # ── INSERTION: blurred image with top-k pixels revealed ──
        ins_img = blurred.copy()
        if k > 0:
            top_k_flat = sorted_indices[:k]
            rows, cols  = np.unravel_index(top_k_flat, (h, w))
            ins_img[rows, cols, :] = img_arr[rows, cols, :]
        ins_pil  = Image.fromarray((np.clip(ins_img, 0, 1) * 255).astype(np.uint8))
        ins_t    = preprocess(ins_pil).unsqueeze(0).to(DEVICE)
        insertion_scores.append(get_confidence(ins_t, target_class))

    # AUC via trapezoidal rule (normalized to [0,1] x-axis)
    insertion_auc = float(np.trapz(insertion_scores, percentages))
    deletion_auc  = float(np.trapz(deletion_scores,  percentages))

    return insertion_auc, deletion_auc, insertion_scores, deletion_scores


# ─── PROCESS METHOD ──────────────────────────────────────────────────────────
def evaluate_method(name, heatmap_dir, meta_path, max_images=20):
    if not meta_path.exists():
        print(f"  {name}: metadata not found, skipping.")
        return None

    with open(meta_path) as f:
        meta = json.load(f)

    # Build list of (image_path, heatmap_path, predicted_class)
    items = []
    if name == "Grad-CAM":
        for img_key, entry in meta.items():
            # img_key is like "Normal_Neo_Normal_1.png" (unique)
            npy_name = img_key.replace(".png", ".npy")
            hm_path  = heatmap_dir / npy_name
            if not hm_path.exists():
                continue
            img_path = Path(entry.get("classification_path", ""))
            if img_path.exists():
                pred_cls = 1 if entry.get("class", "ROP") == "ROP" else 0
                items.append((img_path, hm_path, pred_cls))

    elif name == "Integ. Grad.":
        # Build key→img_path lookup from overlapping_images_map.json
        # (same image set IG uses — same unique key scheme)
        overlap_map_path = Path("results/overlapping_images_map.json")
        if not overlap_map_path.exists():
            print(f"  {name}: overlapping_images_map.json not found.")
            return None
        with open(overlap_map_path) as f:
            omap = json.load(f)["images"]
        key_to_path = {}
        for d in omap:
            p = Path(d["classification_path"])
            parts = p.parts
            key = "_".join(parts[-3:]).replace(" ", "_")
            key_to_path[key] = (p, 1 if d["classification_label"] == "ROP" else 0)

        # Deduplicate by image_key (one faithfulness eval per image)
        pair_results = meta.get("pair_results", [])
        seen_keys = set()
        for entry in pair_results:
            img_key = entry.get("image_key", "")
            if img_key in seen_keys:
                continue
            seen_keys.add(img_key)
            npy_name = img_key.replace(".png", ".npy")
            hm_path  = heatmap_dir / npy_name
            if not hm_path.exists():
                continue
            if img_key in key_to_path:
                img_path, pred_cls = key_to_path[img_key]
                if img_path.exists():
                    items.append((img_path, hm_path, pred_cls))

    if not items:
        print(f"  {name}: no valid image-heatmap pairs found.")
        return None

    items = items[:max_images]  # cap per method
    print(f"  {name}: evaluating {len(items)} images...")

    all_ins, all_del = [], []
    all_ins_curves, all_del_curves = [], []

    for img_path, hm_path, pred_cls in tqdm(items, desc=f"  {name}", leave=False):
        try:
            img_tensor, img_arr = load_image(img_path)
            heatmap = np.load(hm_path)
            # Resize heatmap to 224x224 if needed
            if heatmap.shape != (224, 224):
                from PIL import Image as PILImage
                hm_pil = PILImage.fromarray((heatmap * 255).astype(np.uint8))
                hm_pil = hm_pil.resize((224, 224), PILImage.BILINEAR)
                heatmap = np.array(hm_pil) / 255.0
            # Normalize heatmap
            if heatmap.max() > heatmap.min():
                heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())

            ins_auc, del_auc, ins_c, del_c = compute_insertion_deletion(
                img_tensor, img_arr, heatmap, pred_cls, N_STEPS)
            all_ins.append(ins_auc)
            all_del.append(del_auc)
            all_ins_curves.append(ins_c)
            all_del_curves.append(del_c)
        except Exception as e:
            print(f"    Error: {e}")
            continue

    if not all_ins:
        return None

    result = {
        "method": name,
        "n_images": len(all_ins),
        "insertion_auc_mean": round(float(np.mean(all_ins)), 4),
        "insertion_auc_std":  round(float(np.std(all_ins)),  4),
        "deletion_auc_mean":  round(float(np.mean(all_del)), 4),
        "deletion_auc_std":   round(float(np.std(all_del)),  4),
        "faithfulness_score": round(float(np.mean(all_ins)) - float(np.mean(all_del)), 4),
        "avg_ins_curve":  [round(float(np.mean([c[i] for c in all_ins_curves])), 4)
                           for i in range(len(all_ins_curves[0]))],
        "avg_del_curve":  [round(float(np.mean([c[i] for c in all_del_curves])), 4)
                           for i in range(len(all_del_curves[0]))],
    }
    print(f"    Insertion AUC: {result['insertion_auc_mean']:.4f}  "
          f"Deletion AUC: {result['deletion_auc_mean']:.4f}  "
          f"Faithfulness: {result['faithfulness_score']:.4f}")
    return result


# ─── RUN EVALUATION ──────────────────────────────────────────────────────────
method_configs = [
    ("Grad-CAM",    GRADCAM_DIR, GRADCAM_META),
    ("Integ. Grad.", IG_DIR,     IG_META),
]

all_results = {}
for name, hm_dir, meta in method_configs:
    print(f"\nEvaluating {name}...")
    res = evaluate_method(name, hm_dir, meta, max_images=97)
    if res:
        all_results[name] = res

# ─── FIGURE: Insertion & Deletion Curves ─────────────────────────────────────
if all_results:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Faithfulness Evaluation: Insertion & Deletion Curves",
                 fontsize=14, fontweight='bold')

    colors = {"Grad-CAM": "#e74c3c", "SHAP": "#3498db",
              "LIME": "#2ecc71", "Integ. Grad.": "#9b59b6"}

    percentages = np.linspace(0, 1, N_STEPS + 1)

    for name, res in all_results.items():
        c = colors.get(name, "#95a5a6")
        if res.get("avg_ins_curve"):
            axes[0].plot(percentages, res["avg_ins_curve"], 'o-', label=name,
                         color=c, linewidth=2, markersize=4)
        if res.get("avg_del_curve"):
            axes[1].plot(percentages, res["avg_del_curve"], 'o-', label=name,
                         color=c, linewidth=2, markersize=4)

    axes[0].set_title("Insertion Curve\n(Higher is better — adding important pixels → confidence rises)",
                      fontweight='bold')
    axes[0].set_xlabel("Fraction of pixels revealed")
    axes[0].set_ylabel("Model Confidence (predicted class)")
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0, 1)
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].set_title("Deletion Curve\n(Lower is better — removing important pixels → confidence drops)",
                      fontweight='bold')
    axes[1].set_xlabel("Fraction of pixels removed")
    axes[1].set_ylabel("Model Confidence (predicted class)")
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(0, 1)
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "faithfulness_curves.png", dpi=200, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: faithfulness_curves.png")

    # Bar chart: Faithfulness scores
    fig2, ax = plt.subplots(figsize=(10, 6))
    names = list(all_results.keys())
    ins_scores = [all_results[n]["insertion_auc_mean"] for n in names]
    del_scores = [all_results[n]["deletion_auc_mean"]  for n in names]
    faith_scores = [all_results[n]["faithfulness_score"] for n in names]

    x = np.arange(len(names))
    w = 0.25
    bars_i = ax.bar(x - w, ins_scores, w, label="Insertion AUC (↑)", color="#2ecc71", alpha=0.85)
    bars_d = ax.bar(x,     del_scores, w, label="Deletion AUC (↓)",  color="#e74c3c", alpha=0.85)
    bars_f = ax.bar(x + w, faith_scores, w, label="Faithfulness Score (Ins−Del, ↑)",
                    color="#9b59b6", alpha=0.85)

    ax.set_title("Faithfulness Metrics by XAI Method", fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=12)
    ax.set_ylabel("Score")
    ax.legend(fontsize=10)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    for bar in [*bars_i, *bars_d, *bars_f]:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{bar.get_height():.3f}", ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "faithfulness_bar.png", dpi=200, bbox_inches='tight')
    plt.close()
    print("Saved: faithfulness_bar.png")

    # Save JSON
    with open(OUTPUT_DIR / "faithfulness_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # ── Summary ──
    print("\n" + "=" * 70)
    print("FAITHFULNESS SUMMARY")
    print("=" * 70)
    print(f"{'Method':<20} {'N':>4} {'Ins AUC':>10} {'Del AUC':>10} {'Faithfulness':>14}")
    print("-" * 62)
    for n, r in all_results.items():
        print(f"{n:<20} {r['n_images']:>4} {r['insertion_auc_mean']:>10.4f} "
              f"{r['deletion_auc_mean']:>10.4f} {r['faithfulness_score']:>14.4f}")

    best = max(all_results, key=lambda k: all_results[k]["faithfulness_score"])
    print(f"\nMost faithful method: {best} "
          f"(faithfulness={all_results[best]['faithfulness_score']:.4f})")

print(f"\nResults saved to: {OUTPUT_DIR}")
print("=" * 70)
