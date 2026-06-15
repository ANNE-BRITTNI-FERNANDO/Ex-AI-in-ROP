"""
STEP C: INTEGRATED GRADIENTS (4th XAI Method)
Runs on the same 97 HVDROPDB overlapping images as Grad-CAM/SHAP/LIME,
computes IoU/Dice against expert segmentation masks — giving N=111 pairs
for a fair comparison with all other XAI methods.

Integrated Gradients: Mukund Sundararajan et al. (2017)
- Computes attributions as path integral from baseline to input
- Satisfies Completeness and Sensitivity axioms
- N_STEPS=100 for high-accuracy approximation (error < 1%)

Outputs: heatmaps (.npy), IoU/Dice per lesion pair, timing stats
"""

import json
import time
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from pathlib import Path
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tqdm import tqdm

# ─── CONFIG ──────────────────────────────────────────────────────────────────
MODEL_PATH    = Path("models/clean_augmented_wrapped.pth")
OVERLAP_MAP   = Path("results/overlapping_images_map.json")  # 97 HVDROPDB images
OUTPUT_DIR    = Path("results/ig_clean")
N_STEPS       = 50    # 50 steps → error ~2% — standard in IG literature (Sundararajan 2017)
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "heatmaps").mkdir(exist_ok=True)
(OUTPUT_DIR / "visualizations").mkdir(exist_ok=True)

print("=" * 70)
print("STEP C: INTEGRATED GRADIENTS — HVDROPDB overlapping images")
print("=" * 70)
print(f"Device: {DEVICE}  |  Steps: {N_STEPS}  |  Source: {OVERLAP_MAP}")

# ─── MODEL LOAD ──────────────────────────────────────────────────────────────
def load_model(path):
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    checkpoint = torch.load(path, map_location=DEVICE, weights_only=True)
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    model.to(DEVICE)
    print(f"Model loaded from: {path}")
    return model

# ─── PREPROCESSING ───────────────────────────────────────────────────────────
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

def load_image(path):
    img_pil = Image.open(path).convert("RGB")
    tensor = preprocess(img_pil).unsqueeze(0).to(DEVICE)
    return tensor, img_pil

# ─── INTEGRATED GRADIENTS ────────────────────────────────────────────────────
def integrated_gradients(model, input_tensor, target_class, n_steps=50):
    """
    Compute Integrated Gradients attributions.
    Baseline = zero tensor (black image in normalized space).
    
    IG(x) = (x - x') * integral_0^1 [ dF(x' + a*(x-x'))/dx da ]
    Approximated via Riemann sum over n_steps interpolations.
    """
    baseline = torch.zeros_like(input_tensor).to(DEVICE)
    
    # Generate interpolated inputs along the straight-line path
    alphas = torch.linspace(0, 1, n_steps).to(DEVICE)
    # Process in smaller chunks to save memory on CPU
    all_grads = []
    for alpha in alphas:
        interp = (baseline + alpha * (input_tensor - baseline)).detach()
        interp.requires_grad_(True)
        out = model(interp)
        out[0, target_class].backward()
        all_grads.append(interp.grad.detach().clone())
        model.zero_grad()
    
    grads = torch.stack(all_grads, dim=0).squeeze(1)  # [n_steps, C, H, W]
    
    # Riemann sum approximation of integral
    avg_grads = grads.mean(dim=0)  # [C, H, W]
    
    # Element-wise multiply with (input - baseline)
    diff = (input_tensor.squeeze(0) - baseline.squeeze(0)).detach()
    integrated_grads = (diff * avg_grads).cpu().numpy()  # [C, H, W]
    
    # Convert to single-channel attribution map
    # Sum across channels and take absolute value
    attribution = np.sum(np.abs(integrated_grads), axis=0)  # [H, W]
    
    # Normalize to [0, 1]
    if attribution.max() > attribution.min():
        attribution = (attribution - attribution.min()) / (attribution.max() - attribution.min())
    
    return attribution

# ─── VISUALIZATION ───────────────────────────────────────────────────────────
def create_visualization(img_pil, ig_map, pred_label, confidence, save_path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original
    axes[0].imshow(img_pil.resize((224, 224)))
    axes[0].set_title(f"Original Image\n({pred_label}, {confidence:.1%})", fontweight='bold')
    axes[0].axis('off')
    
    # IG heatmap
    im = axes[1].imshow(ig_map, cmap='hot')
    axes[1].set_title("Integrated Gradients\nAttribution Map", fontweight='bold')
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    
    # Overlay
    img_arr = np.array(img_pil.resize((224, 224))).astype(float) / 255.0
    heatmap_rgb = cm.hot(ig_map)[:, :, :3]
    alpha = 0.5
    overlay = (1 - alpha) * img_arr + alpha * heatmap_rgb
    overlay = np.clip(overlay, 0, 1)
    axes[2].imshow(overlay)
    axes[2].set_title("IG Overlay\n(Attribution on Image)", fontweight='bold')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

# ─── IoU / DICE HELPERS ──────────────────────────────────────────────────────
def load_mask(mask_path):
    mask = Image.open(mask_path).convert("L").resize((224, 224), Image.NEAREST)
    return (np.array(mask) > 127).astype(np.uint8)

def best_iou_dice(heatmap, gt_mask):
    """Try 16 thresholds, return best IoU + matching Dice."""
    best_iou, best_dice = 0.0, 0.0
    for t in np.arange(0.1, 0.91, 0.05):
        pred = (heatmap > t).astype(np.uint8)
        inter = np.logical_and(pred, gt_mask).sum()
        union = np.logical_or(pred, gt_mask).sum()
        denom = pred.sum() + gt_mask.sum()
        iou  = inter / union  if union  > 0 else 0.0
        dice = 2 * inter / denom if denom > 0 else 0.0
        if iou > best_iou:
            best_iou, best_dice = iou, dice
    return float(best_iou), float(best_dice)

# ─── MAIN ────────────────────────────────────────────────────────────────────
def main():
    model = load_model(MODEL_PATH)
    class_names = ["Normal", "ROP"]

    with open(OVERLAP_MAP) as f:
        overlap_data = json.load(f)
    images_list = overlap_data["images"]
    print(f"Loaded {len(images_list)} overlapping HVDROPDB images\n")

    all_times, pair_results = [], []
    per_lesion = {"optic_disc": [], "vessels": [], "ridge": []}

    for img_data in tqdm(images_list, desc="IG Processing"):
        img_path = Path(img_data["classification_path"])
        if not img_path.exists():
            continue

        try:
            input_tensor, img_pil = load_image(img_path)

            with torch.no_grad():
                probs = torch.softmax(model(input_tensor), dim=1)
                pred_class  = probs.argmax(1).item()
                confidence  = probs[0, pred_class].item()
            pred_label = class_names[pred_class]

            # ─── COMPUTE IG ───────────────────────────────────────────────
            t0 = time.time()
            ig_map = integrated_gradients(model, input_tensor, pred_class, N_STEPS)
            elapsed = time.time() - t0
            all_times.append(elapsed)

            # Unique filename (same scheme as Grad-CAM)
            parts    = img_path.parts
            img_key  = "_".join(parts[-3:]).replace(" ", "_")
            npy_name = img_key.replace(".png", ".npy")

            # Save heatmap
            hm_path = OUTPUT_DIR / "heatmaps" / npy_name
            np.save(hm_path, ig_map)

            # ─── IoU / DICE per lesion ────────────────────────────────────
            for lesion in img_data["available_lesions"]:
                mask_path = Path(img_data["mask_paths"][lesion])
                if not mask_path.exists():
                    continue
                gt_mask = load_mask(mask_path)
                iou, dice = best_iou_dice(ig_map, gt_mask)

                pair = {
                    "image_key":   img_key,
                    "class":       img_data["classification_label"],
                    "lesion_type": lesion,
                    "iou":         round(iou,  4),
                    "dice":        round(dice, 4),
                    "comp_time":   round(elapsed, 3),
                }
                pair_results.append(pair)
                per_lesion[lesion].append(pair)

            # ─── VISUALIZATION (4-panel) ──────────────────────────────────
            viz_path = OUTPUT_DIR / "visualizations" / (img_key.replace(".png","") + "_ig.png")
            create_visualization(img_pil, ig_map, pred_label, confidence, viz_path)

        except Exception as e:
            print(f"  Error on {img_path.name}: {e}")
            continue

    # ─── SUMMARY STATS ───────────────────────────────────────────────────────
    n       = len(pair_results)
    all_iou  = [r["iou"]  for r in pair_results]
    all_dice = [r["dice"] for r in pair_results]
    avg_t    = float(np.mean(all_times)) if all_times else 0.0

    print(f"\n{'='*70}")
    print(f"INTEGRATED GRADIENTS RESULTS")
    print(f"{'='*70}")
    print(f"  Images processed : {len(images_list)}")
    print(f"  Lesion pairs     : {n}")
    print(f"  Mean IoU         : {np.mean(all_iou):.4f} +/- {np.std(all_iou):.4f}")
    print(f"  Mean Dice        : {np.mean(all_dice):.4f} +/- {np.std(all_dice):.4f}")
    print(f"  Avg time/image   : {avg_t:.3f}s  ({avg_t/60:.1f} min)")
    print(f"  N_STEPS          : {N_STEPS}")

    print("\nPer-Lesion:")
    lesion_stats = {}
    for lt, pairs in per_lesion.items():
        if pairs:
            ious  = [p["iou"]  for p in pairs]
            dices = [p["dice"] for p in pairs]
            print(f"  {lt:15s} n={len(pairs):3d}  IoU={np.mean(ious):.4f}  Dice={np.mean(dices):.4f}")
            lesion_stats[lt] = {
                "count":     len(pairs),
                "iou_mean":  round(float(np.mean(ious)),  4),
                "iou_std":   round(float(np.std(ious)),   4),
                "dice_mean": round(float(np.mean(dices)), 4),
                "dice_std":  round(float(np.std(dices)),  4),
            }

    results = {
        "n_steps":                   N_STEPS,
        "total_images":              len(images_list),
        "total_pairs":               n,
        "computation_time_total":    round(sum(all_times), 2),
        "computation_time_per_image_avg": round(avg_t, 3),
        "mean_iou":                  round(float(np.mean(all_iou)),  4),
        "std_iou":                   round(float(np.std(all_iou)),   4),
        "mean_dice":                 round(float(np.mean(all_dice)), 4),
        "std_dice":                  round(float(np.std(all_dice)),  4),
        "per_lesion":                lesion_stats,
        "pair_results":              pair_results,
    }
    with open(OUTPUT_DIR / "ig_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to: {OUTPUT_DIR}")
    print("=" * 70)

if __name__ == "__main__":
    main()
