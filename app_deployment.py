"""
ROP Detection System - Hugging Face Spaces Deployment Version
Full-featured with graceful fallbacks for optional components
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
from pathlib import Path
import json
import time
from datetime import datetime

# ─── DEVICE & MODEL ──────────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

def load_resnet50(path):
    """Load ResNet50 model with error handling"""
    m = models.resnet50(weights=None)
    m.fc = nn.Linear(m.fc.in_features, 2)
    
    try:
        ckpt = torch.load(path, map_location=device, weights_only=True)
        sd = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        m.load_state_dict(sd)
        print(f"✓ Model loaded from {path}")
    except Exception as e:
        print(f"⚠️ Warning: Could not load checkpoint - {e}")
    
    m.eval()
    return m.to(device)

# Load model
print("Loading ResNet50 (augmented model)...")
model = load_resnet50("models/augmented_best_model.pth")
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
    """Compute integrated gradients attribution"""
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
    """Apply heatmap to image"""
    h, w = img_arr.shape[:2]
    hm = cv2.resize(heatmap, (w, h))
    hm_color = cv2.applyColorMap(np.uint8(255 * hm), cv2.COLORMAP_JET)
    hm_color = cv2.cvtColor(hm_color, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(img_arr, 0.55, hm_color, 0.45, 0)
    return hm_color, overlay

def load_json(path):
    """Safely load JSON file"""
    p = Path(path)
    if p.exists():
        try:
            with open(p) as f:
                return json.load(f)
        except:
            return None
    return None

def load_precomputed_heatmap(image_path, method_dir):
    """Load pre-computed heatmap (SHAP/LIME)"""
    if not isinstance(image_path, (str, Path)):
        return None
    
    try:
        image_path = Path(image_path)
        method_dir = Path(method_dir)
        base_name = image_path.stem
        
        heatmap_file = method_dir / "heatmaps" / f"{base_name}.npy"
        if heatmap_file.exists():
            return np.load(heatmap_file)
        
        # Try unique key
        parts = image_path.parts
        if len(parts) >= 3:
            unique_key = "_".join(parts[-3:]).replace(" ", "_")
            heatmap_file = method_dir / "heatmaps" / f"{unique_key}.npy"
            if heatmap_file.exists():
                return np.load(heatmap_file)
    except:
        pass
    
    return None

# ─── CHECK OPTIONAL COMPONENTS ───────────────────────────────────────────────
SHAP_AVAILABLE = Path("results/shap_visualizations/heatmaps").exists()
LIME_AVAILABLE = Path("results/lime_visualizations/heatmaps").exists()
XAI_FIGS_AVAILABLE = Path("results/enhanced_xai_comparison").exists()
MULTI_ARCH_AVAILABLE = Path("results/multi_arch_comparison/comparison_summary.json").exists()

print(f"✓ SHAP available: {SHAP_AVAILABLE}")
print(f"✓ LIME available: {LIME_AVAILABLE}")
print(f"✓ XAI figures available: {XAI_FIGS_AVAILABLE}")
print(f"✓ Multi-arch comparison available: {MULTI_ARCH_AVAILABLE}")

# ─── TAB 1: CLINICAL ANALYSIS ────────────────────────────────────────────────
def analyze_image(image, show_shap=False, show_lime=False):
    """Main prediction and XAI analysis"""
    if image is None:
        return "Please upload a retinal image first.", None
    
    try:
        t_start = time.time()
        
        if isinstance(image, np.ndarray):
            img_pil = Image.fromarray(image).convert("RGB")
        else:
            img_pil = image.convert("RGB")
        
        img_arr = np.array(img_pil.resize((224, 224)))
        tensor = preprocess(img_pil).unsqueeze(0).to(device)
        
        # Prediction
        with torch.no_grad():
            out = model(tensor)
            probs = torch.softmax(out, dim=1)[0].cpu().numpy()
        
        pred = int(np.argmax(probs))
        label = CLASS_NAMES[pred]
        conf = probs[pred]
        
        # Grad-CAM
        t_gc = time.time()
        gc_map = gradcam(tensor, pred)
        t_gc = time.time() - t_gc
        gc_hm, gc_overlay = apply_heatmap(img_arr, gc_map)
        
        # Integrated Gradients
        t_ig = time.time()
        ig_map = integrated_gradients(tensor, pred)
        t_ig = time.time() - t_ig
        ig_hm, ig_overlay = apply_heatmap(img_arr, ig_map)
        
        # Try SHAP & LIME (pre-computed only)
        shap_available = False
        lime_available = False
        
        if show_shap and SHAP_AVAILABLE:
            try:
                shap_map = load_precomputed_heatmap(str(img_pil), "results/shap_visualizations")
                if shap_map is not None:
                    shap_available = True
                    shap_hm, shap_overlay = apply_heatmap(img_arr, shap_map)
            except:
                pass
        
        if show_lime and LIME_AVAILABLE:
            try:
                lime_map = load_precomputed_heatmap(str(img_pil), "results/lime_visualizations")
                if lime_map is not None:
                    lime_available = True
                    lime_hm, lime_overlay = apply_heatmap(img_arr, lime_map)
            except:
                pass
        
        total_time = time.time() - t_start
        
        # Create figure
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f"ROP Detection Analysis\nPrediction: **{label}** ({conf:.1%} confidence)",
                     fontsize=14, fontweight='bold',
                     color='#e74c3c' if label == 'ROP' else '#27ae60')
        
        # Grad-CAM
        axes[0, 0].imshow(img_arr)
        axes[0, 0].set_title("Original Image", fontweight='bold')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(gc_hm)
        axes[0, 1].set_title(f"Grad-CAM ({t_gc:.2f}s)", fontweight='bold')
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(gc_overlay)
        axes[0, 2].set_title("Grad-CAM Overlay", fontweight='bold')
        axes[0, 2].axis('off')
        
        # Integrated Gradients
        axes[1, 0].imshow(img_arr)
        axes[1, 0].set_title("Original Image", fontweight='bold')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(ig_hm, cmap='hot')
        axes[1, 1].set_title(f"Integrated Gradients ({t_ig:.2f}s)", fontweight='bold')
        axes[1, 1].axis('off')
        
        axes[1, 2].imshow(ig_overlay)
        axes[1, 2].set_title("IG Overlay", fontweight='bold')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        
        # Report
        icon = '🔴' if label == 'ROP' else '🟢'
        action = ("**Refer to pediatric ophthalmologist immediately**" 
                  if label == 'ROP' 
                  else "Continue routine screening schedule")
        
        xai_info = f"""
| Method | Time | Status |
|--------|------|--------|
| Grad-CAM | {t_gc:.2f}s | ✅ Real-time |
| Integrated Gradients | {t_ig:.2f}s | ✅ Real-time |
| SHAP | {'✅ Pre-computed' if shap_available else '⚠️ Not available'} | For dataset images |
| LIME | {'✅ Pre-computed' if lime_available else '⚠️ Not available'} | For dataset images |
"""
        
        report = f"""
## {icon} **Diagnosis: {label}**

### Classification
- **Predicted:** {label}
- **Confidence:** {conf:.1%}
- **Normal Prob:** {probs[0]:.1%}
- **ROP Prob:** {probs[1]:.1%}

### XAI Methods Applied
{xai_info}

### Clinical Action
{action}

### Analysis Summary
- Total time: {total_time:.2f}s
- Model: ResNet50 (Augmented Training)
- Device: {device}

⚠️ **Disclaimer:** AI screening tool only. All predictions must be confirmed by a qualified pediatric ophthalmologist.
"""
        
        return report, fig
    
    except Exception as e:
        return f"❌ Error during analysis: {str(e)}", None

# ─── BUILD COMPARISON FIGURES ────────────────────────────────────────────────
def build_model_comparison_fig():
    """Load model comparison if available"""
    try:
        multi = load_json("results/multi_arch_comparison/comparison_summary.json")
        
        if not multi or "models" not in multi:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "Multi-architecture comparison not available.\nTraining in progress...",
                    ha='center', va='center', fontsize=12)
            ax.axis('off')
            return fig
        
        model_data = multi["models"]
        arch_names = list(model_data.keys())
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle("Multi-Architecture Comparison", fontsize=14, fontweight='bold')
        
        # Accuracy
        accs = [model_data[a].get("test_accuracy", 0) for a in arch_names]
        axes[0].bar(arch_names, accs, color=['#e74c3c', '#3498db', '#2ecc71'], alpha=0.8)
        axes[0].set_ylabel("Test Accuracy")
        axes[0].set_ylim(0, 1)
        axes[0].set_title("Accuracy Comparison")
        axes[0].grid(axis='y', alpha=0.3)
        
        # F1-Score
        f1s = [model_data[a].get("f1_score", 0) for a in arch_names]
        axes[1].bar(arch_names, f1s, color=['#e74c3c', '#3498db', '#2ecc71'], alpha=0.8)
        axes[1].set_ylabel("F1-Score")
        axes[1].set_ylim(0, 1)
        axes[1].set_title("F1-Score Comparison")
        axes[1].grid(axis='y', alpha=0.3)
        
        # AUC-ROC
        aucs = [model_data[a].get("auc_roc", 0) for a in arch_names]
        axes[2].bar(arch_names, aucs, color=['#e74c3c', '#3498db', '#2ecc71'], alpha=0.8)
        axes[2].set_ylabel("AUC-ROC")
        axes[2].set_ylim(0.8, 1)
        axes[2].set_title("AUC-ROC Comparison")
        axes[2].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        return fig
    except Exception as e:
        print(f"Error building comparison fig: {e}")
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, f"Error loading comparison: {e}", ha='center', va='center')
        ax.axis('off')
        return fig

def build_xai_overview():
    """Load XAI overview figure if available"""
    try:
        img_path = Path("results/enhanced_xai_comparison/fig4_radar_comparison.png")
        if img_path.exists():
            img = Image.open(img_path)
            fig, ax = plt.subplots(figsize=(10, 9))
            ax.imshow(img)
            ax.axis('off')
            return fig
    except:
        pass
    
    fig, ax = plt.subplots()
    ax.text(0.5, 0.5, "XAI comparison figures not available", ha='center', va='center')
    ax.axis('off')
    return fig

# ─── GRADIO INTERFACE ────────────────────────────────────────────────────────
CSS = """
.gradio-container { font-family: 'Segoe UI', sans-serif; }
#main-header { text-align: center; background: linear-gradient(135deg, #1a1a2e, #16213e);
    color: white !important; padding: 20px; border-radius: 10px; }
#main-header * { color: white !important; }
"""

with gr.Blocks(title="ROP Detection System", css=CSS, theme=gr.themes.Soft()) as demo:
    
    with gr.Row(elem_id="main-header"):
        gr.Markdown("""
# 🔍 Retinopathy of Prematurity (ROP) Detection
## AI-Powered Diagnosis with Explainable AI
**ResNet50 | Grad-CAM | SHAP | LIME | Integrated Gradients**
        """)
    
    with gr.Tabs():
        
        # Tab 1: Clinical Analysis
        with gr.Tab("📋 Clinical Analysis"):
            gr.Markdown("""
⚠️ **CLINICAL DISCLAIMER**
This is an AI screening tool. All predictions must be verified by a qualified pediatric ophthalmologist.
            """)
            
            with gr.Row():
                with gr.Column(scale=1):
                    img_input = gr.Image(label="Upload Retinal Image", type="pil", height=350)
                    btn_analyze = gr.Button("🔍 Analyze", variant="primary", size="lg")
                    btn_clear = gr.Button("Clear", variant="secondary")
                
                with gr.Column(scale=2):
                    report_out = gr.Markdown(label="Diagnostic Report")
            
            gr.Markdown("### XAI Options")
            chk_shap = gr.Checkbox(label="Include SHAP", value=False, 
                                  info="Pre-computed for dataset images only")
            chk_lime = gr.Checkbox(label="Include LIME", value=False,
                                  info="Pre-computed for dataset images only")
            
            gr.Markdown("### Visualization")
            xai_plot = gr.Plot(label="XAI Analysis")
            
            btn_analyze.click(fn=analyze_image, inputs=[img_input, chk_shap, chk_lime],
                            outputs=[report_out, xai_plot])
            btn_clear.click(fn=lambda: (None, "", None, False, False),
                          outputs=[img_input, report_out, xai_plot, chk_shap, chk_lime])
        
        # Tab 2: Model Comparison (if available)
        if MULTI_ARCH_AVAILABLE:
            with gr.Tab("🏆 Model Comparison"):
                gr.Markdown("## Multi-Architecture Performance")
                model_fig = gr.Plot()
                demo.load(fn=build_model_comparison_fig, outputs=model_fig)
        
        # Tab 3: XAI Comparison (if available)
        if XAI_FIGS_AVAILABLE:
            with gr.Tab("📊 XAI Comparison"):
                gr.Markdown("## XAI Method Evaluation")
                xai_fig = gr.Plot()
                demo.load(fn=build_xai_overview, outputs=xai_fig)
        
        # Tab 4: About
        with gr.Tab("ℹ️ About"):
            gr.Markdown(f"""
## System Information
- **Model:** ResNet50 (Augmented Training)
- **Dataset:** HVDROPDB + Czech ROP (6,004 images)
- **Test Accuracy:** 92.00%
- **Device:** {device}
- **Loaded:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## XAI Methods
- **Grad-CAM:** Fast gradient-based attention (1-2s)
- **Integrated Gradients:** Path integral attribution (4-15s)
- **SHAP:** Shapley value decomposition (pre-computed)
- **LIME:** Local linear approximation (pre-computed)

## Citation
If you use this system, please cite the underlying research.
            """)

if __name__ == "__main__":
    print("\n" + "="*70)
    print("ROP Detection System - Starting")
    print("="*70)
    print(f"Device: {device}")
    print(f"SHAP available: {SHAP_AVAILABLE}")
    print(f"LIME available: {LIME_AVAILABLE}")
    print(f"XAI comparison available: {XAI_FIGS_AVAILABLE}")
    print(f"Multi-arch available: {MULTI_ARCH_AVAILABLE}")
    print("="*70 + "\n")
    
    demo.launch(share=False)
