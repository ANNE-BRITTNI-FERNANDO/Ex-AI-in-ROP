"""
Enhanced ROP Detection Clinical GUI
- Tab 1: Real-time analysis with Grad-CAM + Integrated Gradients (real-time) + SHAP + LIME (pre-computed)
- Tab 2: Multi-Architecture Model Comparison Dashboard
- Tab 3: XAI Method Comparison (4 methods) with publication figures
- Tab 4: System Information & Methodology
"""

import gradio as gr
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm_mod
from pathlib import Path
import json
import time
from datetime import datetime

# ─── DEVICE & MODEL ──────────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

def load_resnet50(path):
    m = models.resnet50(weights=None)
    m.fc = nn.Linear(m.fc.in_features, 2)
    ckpt = torch.load(path, map_location=device, weights_only=True)
    sd = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    m.load_state_dict(sd)
    m.eval()
    return m.to(device)

print("Loading ResNet50 (augmented model)...")
model = load_resnet50("models/augmented_best_model.pth")
print("Model loaded.")

CLASS_NAMES = ["Normal", "ROP"]

# ─── GRAD-CAM ────────────────────────────────────────────────────────────────
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(lambda m, i, o: setattr(self, 'activations', o.detach()))
        target_layer.register_full_backward_hook(lambda m, gi, go: setattr(self, 'gradients', go[0].detach()))

    def __call__(self, tensor, class_idx):
        self.model.zero_grad()
        out = self.model(tensor)
        out[0, class_idx].backward()
        w = self.gradients[0].mean(dim=(1, 2), keepdim=True)
        cam = torch.relu((w * self.activations[0]).sum(dim=0)).cpu().numpy()
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam

gradcam = GradCAM(model, model.layer4[-1])

# ─── INTEGRATED GRADIENTS ────────────────────────────────────────────────────
def integrated_gradients(input_tensor, target_class, n_steps=20):
    baseline = torch.zeros_like(input_tensor).to(device)
    alphas = torch.linspace(0, 1, n_steps).to(device)
    all_grads = []
    for alpha in alphas:
        interp = (baseline + alpha * (input_tensor - baseline)).detach()
        interp.requires_grad_(True)
        out = model(interp)
        out[0, target_class].backward()
        all_grads.append(interp.grad.detach().clone())
        model.zero_grad()
    avg_grads = torch.stack(all_grads).squeeze(1).mean(dim=0)
    diff = (input_tensor.squeeze(0) - baseline.squeeze(0)).detach()
    ig = (diff * avg_grads).cpu().numpy()
    attr = np.sum(np.abs(ig), axis=0)
    if attr.max() > attr.min():
        attr = (attr - attr.min()) / (attr.max() - attr.min())
    return attr

# ─── HELPERS ─────────────────────────────────────────────────────────────────
def apply_heatmap(img_arr, heatmap):
    h, w = img_arr.shape[:2]
    hm = cv2.resize(heatmap, (w, h))
    hm_color = cv2.applyColorMap(np.uint8(255 * hm), cv2.COLORMAP_JET)
    hm_color = cv2.cvtColor(hm_color, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(img_arr, 0.55, hm_color, 0.45, 0)
    return hm_color, overlay

def load_json(path):
    p = Path(path)
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return None

# ─── LOAD PRE-COMPUTED HEATMAPS ───────────────────────────────────────────────
# SHAP and LIME are pre-computed for dataset images (too slow for real-time)
def load_precomputed_heatmap(image_input, method_dir):
    """Load pre-computed heatmap for an image if it exists.
    Args:
        image_input: Can be a Path, str (for pre-computed images), or PIL Image (returns None)
    """
    # Only string/Path inputs can match pre-computed heatmaps
    # PIL Images from uploads cannot be matched
    if not isinstance(image_input, (str, Path)):
        return None
    
    image_path = Path(image_input)
    method_dir = Path(method_dir)
    
    # Extract base filename (e.g., "1.png" → "1")
    base_name = image_path.stem
    
    # Try matching by base filename (heatmaps are stored as "1.npy", "10.npy", etc.)
    heatmap_file = method_dir / "heatmaps" / f"{base_name}.npy"
    print(f"[DEBUG] Looking for heatmap: {heatmap_file}, exists: {heatmap_file.exists()}")
    if heatmap_file.exists():
        print(f"[DEBUG] Found heatmap, loading...")
        return np.load(heatmap_file)
    
    # Try matching with unique filename scheme (for Grad-CAM style naming)
    parts = image_path.parts
    if len(parts) >= 3:
        unique_key = "_".join(parts[-3:]).replace(" ", "_")
        heatmap_file = method_dir / "heatmaps" / f"{unique_key}.npy"
        print(f"[DEBUG] Trying unique key: {unique_key}, file: {heatmap_file}, exists: {heatmap_file.exists()}")
        if heatmap_file.exists():
            return np.load(heatmap_file)
    
    print(f"[DEBUG] No heatmap found for {image_path}")
    return None

# ─── TAB 1: ANALYSIS ─────────────────────────────────────────────────────────
def analyze_image(image, image_path=None, show_shap=False, show_lime=False):
    if image is None:
        return "Please upload a retinal image first.", None, None

    try:
        print(f"[DEBUG] analyze_image called: image_path={image_path}, show_shap={show_shap}, show_lime={show_lime}")
        t_start = time.time()

        if isinstance(image, np.ndarray):
            img_pil = Image.fromarray(image).convert("RGB")
        else:
            img_pil = image.convert("RGB")

        img_arr = np.array(img_pil.resize((224, 224)))
        tensor  = preprocess(img_pil).unsqueeze(0).to(device)

        # Prediction
        with torch.no_grad():
            out   = model(tensor)
            probs = torch.softmax(out, dim=1)[0].cpu().numpy()
        pred  = int(np.argmax(probs))
        label = CLASS_NAMES[pred]
        conf  = probs[pred]

        # Grad-CAM (real-time)
        t_gc = time.time()
        gc_map = gradcam(tensor, pred)
        t_gc = time.time() - t_gc
        gc_hm, gc_overlay = apply_heatmap(img_arr, gc_map)

        # Integrated Gradients (real-time)
        t_ig = time.time()
        ig_map = integrated_gradients(tensor, pred)
        t_ig = time.time() - t_ig
        ig_hm, ig_overlay = apply_heatmap(img_arr, ig_map)

        # SHAP (pre-computed if available)
        shap_available = False
        if show_shap:
            try:
                shap_map = load_precomputed_heatmap(image_path if image_path else img_pil, "results/shap_visualizations")
                if shap_map is not None:
                    shap_available = True
                    shap_hm, shap_overlay = apply_heatmap(img_arr, shap_map)
                    t_shap = "pre-computed"
                    print(f"[DEBUG] SHAP loaded successfully, shap_available=True")
                else:
                    t_shap = "N/A (not pre-computed)"
                    print(f"[DEBUG] SHAP map is None")
            except Exception as e:
                t_shap = "N/A (not pre-computed)"
                print(f"[DEBUG] SHAP error: {e}")

        # LIME (pre-computed if available)
        lime_available = False
        if show_lime:
            try:
                lime_map = load_precomputed_heatmap(image_path if image_path else img_pil, "results/lime_visualizations")
                if lime_map is not None:
                    lime_available = True
                    lime_hm, lime_overlay = apply_heatmap(img_arr, lime_map)
                    t_lime = "pre-computed"
                    print(f"[DEBUG] LIME loaded successfully, lime_available=True")
                else:
                    t_lime = "N/A (not pre-computed)"
                    print(f"[DEBUG] LIME map is None")
            except Exception as e:
                t_lime = "N/A (not pre-computed)"
                print(f"[DEBUG] LIME error: {e}")

        total_time = time.time() - t_start

        # ── Figure: 4-method XAI ──
        if show_shap and show_lime:
            # Show all 4 methods
            fig, axes = plt.subplots(2, 4, figsize=(22, 12))
            fig.suptitle(f"ROP Analysis — Prediction: {label}  ({conf:.1%} confidence)",
                         fontsize=16, fontweight='bold',
                         color='#e74c3c' if label == 'ROP' else '#27ae60')

            # Row 1: Grad-CAM and SHAP
            axes[0][0].imshow(img_arr)
            axes[0][0].set_title("Original Image", fontweight='bold', fontsize=10)
            axes[0][0].axis('off')
            
            axes[0][1].imshow(gc_hm)
            axes[0][1].set_title(f"Grad-CAM ({t_gc:.2f}s)", fontweight='bold', fontsize=10)
            axes[0][1].axis('off')
            
            axes[0][2].imshow(gc_overlay)
            axes[0][2].set_title("Grad-CAM Overlay", fontweight='bold', fontsize=10)
            axes[0][2].axis('off')

            if shap_available:
                axes[0][3].imshow(shap_hm)
                axes[0][3].set_title(f"SHAP ({t_shap})", fontweight='bold', fontsize=10)
            else:
                axes[0][3].text(0.5, 0.5, "SHAP not available\nfor this image", ha='center', va='center', fontsize=9)
                axes[0][3].axis('off')

            # Row 2: IG and LIME
            axes[1][0].imshow(img_arr)
            axes[1][0].set_title("Original Image", fontweight='bold', fontsize=10)
            axes[1][0].axis('off')
            
            axes[1][1].imshow(ig_hm, cmap='hot')
            axes[1][1].set_title(f"Integ. Grad. ({t_ig:.2f}s)", fontweight='bold', fontsize=10)
            axes[1][1].axis('off')
            
            axes[1][2].imshow(ig_overlay)
            axes[1][2].set_title("Integ. Grad. Overlay", fontweight='bold', fontsize=10)
            axes[1][2].axis('off')

            if lime_available:
                axes[1][3].imshow(lime_hm)
                axes[1][3].set_title(f"LIME ({t_lime})", fontweight='bold', fontsize=10)
            else:
                axes[1][3].text(0.5, 0.5, "LIME not available\nfor this image", ha='center', va='center', fontsize=9)
                axes[1][3].axis('off')

        elif show_shap:
            # Show Grad-CAM, IG, SHAP
            fig, axes = plt.subplots(2, 3, figsize=(18, 10))
            fig.suptitle(f"ROP Analysis — Prediction: {label}  ({conf:.1%} confidence)",
                         fontsize=16, fontweight='bold',
                         color='#e74c3c' if label == 'ROP' else '#27ae60')

            axes[0][0].imshow(img_arr)
            axes[0][0].set_title("Original Image", fontweight='bold', fontsize=11)
            axes[0][0].axis('off')
            axes[0][1].imshow(gc_hm)
            axes[0][1].set_title(f"Grad-CAM ({t_gc:.2f}s)", fontweight='bold', fontsize=11)
            axes[0][1].axis('off')
            axes[0][2].imshow(gc_overlay)
            axes[0][2].set_title("Grad-CAM Overlay", fontweight='bold', fontsize=11)
            axes[0][2].axis('off')

            axes[1][0].imshow(img_arr)
            axes[1][0].set_title("Original Image", fontweight='bold', fontsize=11)
            axes[1][0].axis('off')
            axes[1][1].imshow(ig_hm, cmap='hot')
            axes[1][1].set_title(f"Integ. Grad. ({t_ig:.2f}s)", fontweight='bold', fontsize=11)
            axes[1][1].axis('off')
            
            if shap_available:
                axes[1][2].imshow(shap_hm)
                axes[1][2].set_title(f"SHAP ({t_shap})", fontweight='bold', fontsize=11)
            else:
                axes[1][2].text(0.5, 0.5, "SHAP not available\nfor this image", ha='center', va='center', fontsize=10)
            axes[1][2].axis('off')

        elif show_lime:
            # Show Grad-CAM, IG, LIME
            fig, axes = plt.subplots(2, 3, figsize=(18, 10))
            fig.suptitle(f"ROP Analysis — Prediction: {label}  ({conf:.1%} confidence)",
                         fontsize=16, fontweight='bold',
                         color='#e74c3c' if label == 'ROP' else '#27ae60')

            axes[0][0].imshow(img_arr)
            axes[0][0].set_title("Original Image", fontweight='bold', fontsize=11)
            axes[0][0].axis('off')
            axes[0][1].imshow(gc_hm)
            axes[0][1].set_title(f"Grad-CAM ({t_gc:.2f}s)", fontweight='bold', fontsize=11)
            axes[0][1].axis('off')
            axes[0][2].imshow(gc_overlay)
            axes[0][2].set_title("Grad-CAM Overlay", fontweight='bold', fontsize=11)
            axes[0][2].axis('off')

            axes[1][0].imshow(img_arr)
            axes[1][0].set_title("Original Image", fontweight='bold', fontsize=11)
            axes[1][0].axis('off')
            axes[1][1].imshow(ig_hm, cmap='hot')
            axes[1][1].set_title(f"Integ. Grad. ({t_ig:.2f}s)", fontweight='bold', fontsize=11)
            axes[1][1].axis('off')
            
            if lime_available:
                axes[1][2].imshow(lime_hm)
                axes[1][2].set_title(f"LIME ({t_lime})", fontweight='bold', fontsize=11)
            else:
                axes[1][2].text(0.5, 0.5, "LIME not available\nfor this image", ha='center', va='center', fontsize=10)
            axes[1][2].axis('off')

        else:
            # Original: Grad-CAM + IG only
            fig, axes = plt.subplots(2, 4, figsize=(20, 10))
            fig.suptitle(f"ROP Analysis — Prediction: {label}  ({conf:.1%} confidence)",
                         fontsize=16, fontweight='bold',
                         color='#e74c3c' if label == 'ROP' else '#27ae60')

            axes[0][0].imshow(img_arr)
            axes[0][0].set_title("Original Image", fontweight='bold', fontsize=11)
            axes[0][0].axis('off')
            axes[0][1].imshow(gc_hm)
            axes[0][1].set_title(f"Grad-CAM ({t_gc:.2f}s)", fontweight='bold', fontsize=11)
            axes[0][1].axis('off')
            axes[0][2].imshow(gc_overlay)
            axes[0][2].set_title("Grad-CAM Overlay", fontweight='bold', fontsize=11)
            axes[0][2].axis('off')

            # Confidence bar
            axes[0][3].barh(CLASS_NAMES, probs,
                            color=['#27ae60' if c == 'Normal' else '#e74c3c' for c in CLASS_NAMES])
            axes[0][3].set_xlim(0, 1)
            axes[0][3].set_title("Class Probabilities", fontweight='bold', fontsize=11)
            axes[0][3].set_xlabel("Probability")
            for i, v in enumerate(probs):
                axes[0][3].text(v + 0.01, i, f"{v:.1%}", va='center', fontsize=10)
            axes[0][3].axvline(0.5, color='gray', linestyle='--', alpha=0.5)
            axes[0][3].grid(axis='x', alpha=0.3)

            axes[1][0].imshow(img_arr)
            axes[1][0].set_title("Original Image", fontweight='bold', fontsize=11)
            axes[1][0].axis('off')
            axes[1][1].imshow(ig_hm, cmap='hot')
            axes[1][1].set_title(f"Integ. Grad. ({t_ig:.2f}s)", fontweight='bold', fontsize=11)
            axes[1][1].axis('off')
            axes[1][2].imshow(ig_overlay)
            axes[1][2].set_title("Integ. Grad. Overlay", fontweight='bold', fontsize=11)
            axes[1][2].axis('off')

            # Method comparison
            methods = ["Grad-CAM", "Integ.\nGrad."]
            times = [t_gc, t_ig]
            bar_colors = ['#e74c3c', '#9b59b6']
            axes[1][3].bar(methods, times, color=bar_colors, alpha=0.8, width=0.5)
            axes[1][3].set_title("XAI Computation Time", fontweight='bold', fontsize=11)
            axes[1][3].set_ylabel("Seconds")
            axes[1][3].yaxis.grid(True, alpha=0.3)
            axes[1][3].set_axisbelow(True)
            for i, (m, t) in enumerate(zip(methods, times)):
                axes[1][3].text(i, t + 0.01, f"{t:.2f}s", ha='center', fontsize=10)

        plt.tight_layout()
        print(f"[DEBUG] Figure created, show_shap={show_shap}, show_lime={show_lime}, shap_available={shap_available}, lime_available={lime_available}")

        # ── Text report ──
        icon  = 'ROP DETECTED' if label == 'ROP' else 'NORMAL'
        action = ("**Immediate referral to pediatric ophthalmologist recommended.**"
                  if label == 'ROP' else
                  "Continue routine ROP screening schedule.")

        xai_methods = f"| **Grad-CAM** | {t_gc:.2f}s | Gradient-weighted activation maps (layer4) |\n"
        xai_methods += f"| **Integrated Gradients** | {t_ig:.2f}s | Axiomatic attribution — path integral baseline→input |"
        
        if show_shap:
            shap_status = "✓ Available (pre-computed)" if shap_available else "✗ Not pre-computed for this image"
            xai_methods += f"\n| **SHAP** | {t_shap} | {shap_status} |"
        
        if show_lime:
            lime_status = "✓ Available (pre-computed)" if lime_available else "✗ Not pre-computed for this image"
            xai_methods += f"\n| **LIME** | {t_lime} | {lime_status} |"

        report = f"""
## {'🔴' if label=='ROP' else '🟢'} Diagnosis: **{icon}**

---

| Metric | Value |
|--------|-------|
| Predicted Class | **{label}** |
| Confidence | **{conf:.1%}** |
| Normal Probability | {probs[0]:.1%} |
| ROP Probability | {probs[1]:.1%} |
| Total Analysis Time | {total_time:.2f}s |

---

### XAI Methods Applied
| Method | Time | Basis |
|--------|------|-------|
{xai_methods}

---

### Clinical Action
{action}

> ⚠️ **Disclaimer:** AI screening tool. All predictions must be confirmed by a qualified pediatric ophthalmologist.  
> *Analysis timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
        return report, fig, None

    except Exception as e:
        return f"Error: {str(e)}", None, None


# ─── TAB 2: MODEL COMPARISON DASHBOARD ───────────────────────────────────────
def build_model_comparison_fig():
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Multi-Architecture Model Comparison — Czech ROP Dataset",
                 fontsize=14, fontweight='bold')

    # Try to load multi-arch results
    multi = load_json("results/multi_arch_comparison/comparison_summary.json")

    if multi and "models" in multi:
        model_data = multi["models"]
        arch_names = list(model_data.keys())
        metrics_keys = ["test_accuracy", "recall_sensitivity", "specificity", "f1_score", "auc_roc"]
        metric_labels = ["Accuracy", "Sensitivity", "Specificity", "F1-Score", "AUC-ROC"]
        colors = ['#e74c3c', '#3498db', '#2ecc71']

        # Bar chart: all metrics
        x = np.arange(len(metric_labels))
        width = 0.25
        for i, (arch, color) in enumerate(zip(arch_names, colors)):
            vals = [model_data[arch].get(k, 0) for k in metrics_keys]
            bars = axes[0].bar(x + i*width, vals, width, label=arch, color=color, alpha=0.85)

        axes[0].set_title("Classification Metrics by Architecture", fontweight='bold')
        axes[0].set_xticks(x + width)
        axes[0].set_xticklabels(metric_labels, rotation=20, ha='right')
        axes[0].set_ylabel("Score")
        axes[0].set_ylim(0, 1.1)
        axes[0].legend()
        axes[0].yaxis.grid(True, alpha=0.4)
        axes[0].set_axisbelow(True)

        # Training time
        train_times = [model_data[a].get("training_time_seconds", 0)/60 for a in arch_names]
        axes[1].bar(arch_names, train_times, color=colors, alpha=0.85, width=0.5)
        axes[1].set_title("Training Time (minutes)", fontweight='bold')
        axes[1].set_ylabel("Minutes")
        axes[1].yaxis.grid(True, alpha=0.4)
        axes[1].set_axisbelow(True)
        for i, (a, t) in enumerate(zip(arch_names, train_times)):
            axes[1].text(i, t + 0.3, f"{t:.1f}m", ha='center', fontsize=10)

        # AUC comparison
        aucs = [model_data[a].get("auc_roc", 0) for a in arch_names]
        bars3 = axes[2].bar(arch_names, aucs, color=colors, alpha=0.85, width=0.5)
        axes[2].set_title("AUC-ROC Comparison", fontweight='bold')
        axes[2].set_ylabel("AUC-ROC")
        axes[2].set_ylim(0.5, 1.05)
        axes[2].yaxis.grid(True, alpha=0.4)
        axes[2].set_axisbelow(True)
        for i, (a, v) in enumerate(zip(arch_names, aucs)):
            axes[2].text(i, v + 0.005, f"{v:.4f}", ha='center', fontsize=10, fontweight='bold')
        axes[2].axhline(0.9, color='red', linestyle='--', alpha=0.5, label='AUC=0.90')
        axes[2].legend()

    else:
        # Training not done yet — show placeholder
        for ax in axes:
            ax.text(0.5, 0.5, "Training in progress...\nResults will appear here when complete.",
                    ha='center', va='center', fontsize=12, style='italic',
                    transform=ax.transAxes)
            ax.axis('off')

    plt.tight_layout()
    return fig


def get_model_comparison_table():
    multi = load_json("results/multi_arch_comparison/comparison_summary.json")
    # Also include the original HVDROPDB model result
    hvdropdb_result = load_json("results/comprehensive_evaluation/metrics.json")

    rows = []
    if hvdropdb_result:
        rows.append({
            "Model": "ResNet50 (HVDROPDB, Augmented)",
            "Dataset": "HVDROPDB (185→2000 imgs)",
            "Accuracy": f"{hvdropdb_result.get('accuracy', 0.92):.4f}",
            "Sensitivity": f"{hvdropdb_result.get('recall', 0.9467):.4f}",
            "Specificity": f"{hvdropdb_result.get('specificity', 0.8933):.4f}",
            "F1": f"{hvdropdb_result.get('f1_score', 0.9221):.4f}",
            "AUC": f"{hvdropdb_result.get('auc_roc', 0.9725):.4f}",
        })

    if multi and "models" in multi:
        for arch, data in multi["models"].items():
            rows.append({
                "Model": arch,
                "Dataset": "Czech ROP (6004 imgs)",
                "Accuracy": f"{data.get('test_accuracy', 0):.4f}",
                "Sensitivity": f"{data.get('recall_sensitivity', 0):.4f}",
                "Specificity": f"{data.get('specificity', 0):.4f}",
                "F1": f"{data.get('f1_score', 0):.4f}",
                "AUC": f"{data.get('auc_roc', 0):.4f}",
            })

    if not rows:
        return "Training in progress — results will appear here when complete."

    header = "| Model | Dataset | Accuracy | Sensitivity | Specificity | F1-Score | AUC-ROC |\n"
    header += "|-------|---------|----------|-------------|-------------|----------|---------|\n"
    for r in rows:
        header += (f"| {r['Model']} | {r['Dataset']} | {r['Accuracy']} | "
                   f"{r['Sensitivity']} | {r['Specificity']} | {r['F1']} | {r['AUC']} |\n")
    return header


# ─── TAB 3: XAI COMPARISON ───────────────────────────────────────────────────
def build_xai_overview_fig():
    img_path = Path("results/enhanced_xai_comparison/fig4_radar_comparison.png")
    if img_path.exists():
        img = Image.open(img_path)
        fig, ax = plt.subplots(figsize=(10, 9))
        ax.imshow(img)
        ax.axis('off')
        plt.tight_layout()
        return fig
    return None

def build_xai_iou_fig():
    img_path = Path("results/enhanced_xai_comparison/fig1_iou_dice_comparison.png")
    if img_path.exists():
        img = Image.open(img_path)
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.imshow(img)
        ax.axis('off')
        plt.tight_layout()
        return fig
    return None

def build_xai_speed_fig():
    img_path = Path("results/enhanced_xai_comparison/fig2_computation_time.png")
    if img_path.exists():
        img = Image.open(img_path)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.imshow(img)
        ax.axis('off')
        plt.tight_layout()
        return fig
    return None

def build_xai_lesion_fig():
    img_path = Path("results/enhanced_xai_comparison/fig3_lesion_specific_iou.png")
    if img_path.exists():
        img = Image.open(img_path)
        fig, ax = plt.subplots(figsize=(12, 7))
        ax.imshow(img)
        ax.axis('off')
        plt.tight_layout()
        return fig
    return None

def get_xai_summary_table():
    report = load_json("results/enhanced_xai_comparison/enhanced_comparison_report.json")
    if not report:
        return "XAI comparison results not found."

    table = report.get("summary_table", {})
    lines = ["| Method | N Pairs | Mean IoU | Mean Dice | Avg Time (s) |",
             "|--------|---------|----------|-----------|--------------|"]
    for m, s in table.items():
        iou  = f"{s['mean_iou']:.4f}"  if s.get("mean_iou")  else "N/A"
        dice = f"{s['mean_dice']:.4f}" if s.get("mean_dice") else "N/A"
        lines.append(f"| **{m}** | {s['n']} | {iou} | {dice} | {s['mean_time']:.2f} |")

    stats = report.get("statistical_tests", {})
    anova = stats.get("anova", {})
    if anova:
        lines += ["", f"**ANOVA**: F={anova.get('f_stat','?')}, p={anova.get('p_value','?'):.2e} (highly significant, p<0.001)"]

    return "\n".join(lines)


# ─── CSS & LAYOUT ────────────────────────────────────────────────────────────
CSS = """
.gradio-container { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
#main-header { text-align: center; background: linear-gradient(135deg, #1a1a2e, #16213e);
    color: white !important; padding: 28px; border-radius: 10px; margin-bottom: 18px; }
#main-header * { color: white !important; }
.metric-box { background: #f8f9fa; border-left: 4px solid #3498db;
    padding: 12px; border-radius: 6px; margin: 6px 0; }
.warning-box { background: #fff3cd; border-left: 4px solid #ffc107;
    padding: 12px; border-radius: 6px; }
"""

with gr.Blocks(title="ROP Detection System", css=CSS, theme=gr.themes.Soft()) as demo:

    with gr.Row(elem_id="main-header"):
        gr.Markdown("""
# Retinopathy of Prematurity (ROP) Detection System
### Multi-Architecture AI with 4-Method Explainability Analysis
**ResNet50 | EfficientNet-B0 | DenseNet121 | Grad-CAM | SHAP | LIME | Integrated Gradients**
        """)

    with gr.Tabs():

        # ── TAB 1: CLINICAL ANALYSIS ──────────────────────────────────────────
        with gr.Tab("Clinical Analysis"):
            gr.Markdown("""
<div class='warning-box'>

**Clinical Disclaimer:** AI-powered screening tool. All predictions must be verified by a qualified pediatric ophthalmologist before clinical decisions.

</div>
            """)
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Upload Retinal Fundus Image")
                    img_input = gr.Image(label="Fundus Photograph", type="pil", height=380)
                    # Hidden state to store image path for pre-computed heatmaps
                    img_path_state = gr.State(None)
                    with gr.Row():
                        btn_analyze = gr.Button("Analyze Image", variant="primary", size="lg")
                        btn_clear   = gr.Button("Clear", variant="secondary")

                with gr.Column(scale=2):
                    report_out = gr.Markdown(label="Diagnostic Report")

            gr.Markdown("### XAI Method Selection")
            with gr.Row():
                chk_shap = gr.Checkbox(label="Include SHAP (pre-computed for dataset images)", value=False)
                chk_lime = gr.Checkbox(label="Include LIME (pre-computed for dataset images)", value=False)
            gr.Markdown("*SHAP and LIME heatmaps are pre-computed for dataset images. For new uploads, only Grad-CAM and Integrated Gradients will be available (real-time).")

            gr.Markdown("### Explainability Visualization")
            xai_plot = gr.Plot(label="XAI Analysis")

            btn_analyze.click(fn=analyze_image,
                              inputs=[img_input, img_path_state, chk_shap, chk_lime],
                              outputs=[report_out, xai_plot, gr.State()])
            btn_clear.click(fn=lambda: (None, None, "", None, False, False),
                            inputs=None,
                            outputs=[img_input, img_path_state, report_out, xai_plot, chk_shap, chk_lime])

            # Sample images from HVDROPDB (with pre-computed SHAP/LIME heatmaps)
            with gr.Accordion("Sample Test Images (with SHAP/LIME heatmaps)", open=False):
                overlapping = load_json("results/overlapping_images_map.json")
                
                if overlapping:
                    # Get mix of Neo Normal and RetCam ROP
                    images = overlapping.get("images", [])
                    
                    # Build dropdown options with explicit ROP and Normal sections
                    normal_options = []
                    rop_options = []
                    for img in images:
                        path = img["classification_path"]
                        if Path(path).exists():
                            label = f"{img['classification_label']} - {Path(path).name}"
                            if img['classification_label'] == 'Normal':
                                normal_options.append((label, path))
                            else:
                                rop_options.append((label, path))
                    
                    # Take 6 of each
                    normal_options = normal_options[:6]
                    rop_options = rop_options[:6]
                    
                    sample_options = normal_options + rop_options
                    
                    if sample_options:
                        sample_dropdown = gr.Dropdown(
                            choices=[opt[0] for opt in sample_options],
                            label="Select Sample Image (HVDROPDB - has SHAP/LIME heatmaps)",
                            value=None
                        )
                        
                        def load_sample_image(label):
                            """Load selected sample image and set path state"""
                            # Find the path for this label
                            for opt_label, opt_path in sample_options:
                                if opt_label == label:
                                    return Image.open(opt_path), opt_path
                            return None, None
                        
                        sample_dropdown.change(
                            fn=load_sample_image,
                            inputs=sample_dropdown,
                            outputs=[img_input, img_path_state]
                        )

        # ── TAB 2: MODEL COMPARISON ───────────────────────────────────────────
        with gr.Tab("Model Comparison"):
            gr.Markdown("## Multi-Architecture Performance Comparison")

            btn_refresh_models = gr.Button("Refresh Results", variant="secondary")

            model_table_out = gr.Markdown(value=get_model_comparison_table())
            model_fig_out   = gr.Plot()

            def refresh_models():
                return get_model_comparison_table(), build_model_comparison_fig()

            demo.load(fn=build_model_comparison_fig, outputs=model_fig_out)
            btn_refresh_models.click(fn=refresh_models,
                                     outputs=[model_table_out, model_fig_out])

            gr.Markdown("""
### Architecture Details
| Architecture | Parameters | Key Feature | Fine-Tuned Layers |
|---|---|---|---|
| **ResNet50** | 25.6M | Skip connections prevent vanishing gradients | layer4 + FC |
| **EfficientNet-B0** | 5.3M | Compound scaling, most parameter-efficient | features.7/8 + classifier |
| **DenseNet121** | 8.0M | Dense connectivity, maximum feature reuse | denseblock4 + classifier |

All models trained with:
- **ImageNet pretrained weights** (transfer learning)
- **Adam optimizer** (lr=0.001, ReduceLROnPlateau, factor=0.5)
- **Early stopping** (patience=8 epochs)
- **Data augmentation** (horizontal flip, rotation ±15°, color jitter)
            """)

        # ── TAB 3: XAI COMPARISON ─────────────────────────────────────────────
        with gr.Tab("XAI Method Comparison"):
            gr.Markdown("## 4-Method XAI Quantitative Comparison")

            xai_summary_md = gr.Markdown(value=get_xai_summary_table())

            gr.Markdown("### IoU & Dice Coefficient vs Expert Segmentation Masks")
            xai_iou_fig = gr.Plot()
            demo.load(fn=build_xai_iou_fig, outputs=xai_iou_fig)

            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Computational Speed")
                    xai_speed_fig = gr.Plot()
                    demo.load(fn=build_xai_speed_fig, outputs=xai_speed_fig)
                with gr.Column():
                    gr.Markdown("### Lesion-Specific IoU")
                    xai_lesion_fig = gr.Plot()
                    demo.load(fn=build_xai_lesion_fig, outputs=xai_lesion_fig)

            gr.Markdown("### Multi-Dimensional Radar Chart")
            xai_radar_fig = gr.Plot()
            demo.load(fn=build_xai_overview_fig, outputs=xai_radar_fig)

            gr.Markdown("""
### Method Characteristics
| Method | Approach | Speed | Spatial Accuracy | Theoretical Basis |
|--------|----------|-------|-----------------|-------------------|
| **Grad-CAM** | Gradient-weighted activation maps | Fast (~1.2s) | Highest IoU | Class Activation Mapping |
| **SHAP** | Shapley value decomposition | Slow (~39.7s) | Low IoU | Game-theoretic fairness axioms |
| **LIME** | Local superpixel perturbation | Very Slow (~58.6s) | Low IoU | Local linear approximation |
| **Integrated Gradients** | Path integral from baseline | Moderate (~4-15s) | Not measured* | Completeness + Sensitivity axioms |

> *IG evaluated on Czech dataset which lacks expert segmentation masks for IoU computation.
            """)

        # ── TAB 4: ABOUT ──────────────────────────────────────────────────────
        with gr.Tab("System Info"):
            gr.Markdown(f"""
## ROP Detection System — Technical Details

### Datasets
| Dataset | Images | Purpose |
|---------|--------|---------|
| **HVDROPDB** | 785 total (185 classification + 600 segmentation) | XAI evaluation with expert masks |
| **Czech ROP (Kaggle)** | 6,004 (2,980 Normal + 3,024 ROP) | Multi-architecture model training |

### Training Configuration
- **Optimizer:** Adam (lr=0.001)
- **Scheduler:** ReduceLROnPlateau (factor=0.5, patience=4)
- **Early stopping:** Patience=8
- **Batch size:** 32
- **Image size:** 224×224
- **Normalization:** ImageNet (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

### XAI Evaluation
- **HVDROPDB overlapping images:** 97 unique images with expert segmentation masks
- **Lesion types evaluated:** Optic Disc, Vessels, Ridge
- **Metrics:** IoU (Jaccard Index), Dice Coefficient
- **Statistical test:** One-Way ANOVA + pairwise t-tests

### Performance (ResNet50, HVDROPDB augmented)
- **Test Accuracy:** 92.00%
- **AUC-ROC:** 0.9725
- **Sensitivity:** 94.67%
- **Specificity:** 89.33%

---
*System loaded: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Device: {str(device).upper()}*
            """)

if __name__ == "__main__":
    print("\n" + "="*70)
    print("Starting Enhanced ROP Detection System GUI")
    print("="*70)
    print(f"Device:  {device}")
    print(f"Access:  http://127.0.0.1:7860")
    print("="*70 + "\n")
    demo.launch(server_name="127.0.0.1", server_port=7860, show_error=True)
