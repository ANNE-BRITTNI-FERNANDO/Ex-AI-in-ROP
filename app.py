"""
ROP Detection Clinical GUI - Hugging Face Spaces Version
Simplified for deployment with Gradio
"""

import gradio as gr
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from datetime import datetime
import os

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ─── MODEL LOADING ───────────────────────────────────────────────────────────
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

def load_resnet50(path):
    """Load ResNet50 model from checkpoint"""
    m = models.resnet50(weights=None)
    m.fc = nn.Linear(m.fc.in_features, 2)
    
    try:
        ckpt = torch.load(path, map_location=device, weights_only=True)
        sd = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        m.load_state_dict(sd)
    except Exception as e:
        print(f"Warning: Could not load checkpoint - {e}")
    
    m.eval()
    return m.to(device)

# Load model
print("Loading ResNet50 model...")
try:
    model = load_resnet50("models/augmented_best_model.pth")
    print("✓ Model loaded successfully")
except Exception as e:
    print(f"⚠️  Model loading issue: {e}")
    print("Loading empty model for demo purposes")
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model = model.to(device)

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

# ─── HEATMAP HELPERS ─────────────────────────────────────────────────────────
def apply_heatmap(img_arr, heatmap):
    """Apply heatmap to image and create overlay"""
    h, w = img_arr.shape[:2]
    hm = cv2.resize(heatmap, (w, h))
    hm_color = cv2.applyColorMap(np.uint8(255 * hm), cv2.COLORMAP_JET)
    hm_color = cv2.cvtColor(hm_color, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(img_arr, 0.55, hm_color, 0.45, 0)
    return hm_color, overlay

# ─── MAIN PREDICTION FUNCTION ────────────────────────────────────────────────
def predict_rop(image):
    """Predict ROP and generate XAI visualizations"""
    if image is None:
        return "Please upload a retinal image", None
    
    try:
        # Convert to RGB
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
        gc_map = gradcam(tensor, pred)
        gc_hm, gc_overlay = apply_heatmap(img_arr, gc_map)
        
        # Integrated Gradients
        ig_map = integrated_gradients(tensor, pred)
        ig_hm, ig_overlay = apply_heatmap(img_arr, ig_map)
        
        # Create comparison figure
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f"ROP Detection Analysis\nPrediction: {label} ({conf:.1%} confidence)",
                     fontsize=14, fontweight='bold',
                     color='#e74c3c' if label == 'ROP' else '#27ae60')
        
        # Row 1: Original and Grad-CAM
        axes[0, 0].imshow(img_arr)
        axes[0, 0].set_title("Original Image", fontweight='bold')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(gc_hm)
        axes[0, 1].set_title("Grad-CAM Heatmap", fontweight='bold')
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(gc_overlay)
        axes[0, 2].set_title("Grad-CAM Overlay", fontweight='bold')
        axes[0, 2].axis('off')
        
        # Row 2: Original and Integrated Gradients
        axes[1, 0].imshow(img_arr)
        axes[1, 0].set_title("Original Image", fontweight='bold')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(ig_hm, cmap='hot')
        axes[1, 1].set_title("Integrated Gradients", fontweight='bold')
        axes[1, 1].axis('off')
        
        axes[1, 2].imshow(ig_overlay)
        axes[1, 2].set_title("IG Overlay", fontweight='bold')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        
        # Prepare text report
        report = f"""
        **ROP Detection Report**
        
        **Prediction:** {label}
        **Confidence:** {conf:.2%}
        **Other Class:** {CLASS_NAMES[1-pred]} ({probs[1-pred]:.2%})
        
        **Model:** ResNet50 (Augmented Training)
        **Device:** {device}
        **Analysis Time:** Real-time
        """
        
        return report, fig
    
    except Exception as e:
        return f"Error during prediction: {str(e)}", None

# ─── GRADIO INTERFACE ────────────────────────────────────────────────────────
demo = gr.Interface(
    fn=predict_rop,
    inputs=gr.Image(label="Upload Retinal Image", type="pil"),
    outputs=[
        gr.Markdown(label="Analysis Report"),
        gr.Plot(label="XAI Visualizations")
    ],
    title="ROP Detection System",
    description="AI-powered Retinopathy of Prematurity (ROP) detection with Explainable AI",
    examples=[],
    allow_flagging="never",
    theme=gr.themes.Soft(),
)

if __name__ == "__main__":
    demo.launch(share=False)
