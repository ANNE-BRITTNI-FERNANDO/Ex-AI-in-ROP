"""
Gradio GUI for ROP Detection with Explainable AI
For clinician interaction - simple and focused on clinical utility
"""

import gradio as gr
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import io
from datetime import datetime

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the trained model
print("Loading model...")
model = models.resnet50(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)  # 2 classes: Normal, ROP
model = model.to(device)

# Load trained weights
checkpoint_path = 'models/augmented_best_model.pth'
checkpoint = torch.load(checkpoint_path, map_location=device)
if 'model_state_dict' in checkpoint:
    model.load_state_dict(checkpoint['model_state_dict'])
else:
    model.load_state_dict(checkpoint)
model.eval()
print("✓ Model loaded successfully")

# Image preprocessing
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])

# Grad-CAM class
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks (using full_backward_hook to avoid FutureWarning)
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        self.activations = output.detach()
    
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def generate_cam(self, input_tensor, class_idx):
        # Forward pass
        output = self.model(input_tensor)
        
        # Backward pass
        self.model.zero_grad()
        class_score = output[0, class_idx]
        class_score.backward()
        
        # Generate CAM
        gradients = self.gradients[0]
        activations = self.activations[0]
        weights = gradients.mean(dim=(1, 2), keepdim=True)
        cam = (weights * activations).sum(dim=0)
        cam = torch.relu(cam)
        cam = cam.cpu().numpy()
        
        # Normalize
        if cam.max() > 0:
            cam = cam / cam.max()
        
        return cam

# Initialize Grad-CAM
gradcam = GradCAM(model, model.layer4)

def predict_rop(image):
    """
    Main prediction function for the GUI
    
    Args:
        image: PIL Image or numpy array
        
    Returns:
        prediction: str - "Normal" or "ROP"
        confidence: str - Percentage confidence
        heatmap_fig: matplotlib figure with heatmap overlay
    """
    try:
        # Check if image is provided
        if image is None:
            error_text = "⚠️ Please upload an image first"
            fig, ax = plt.subplots(figsize=(5, 3))
            ax.text(0.5, 0.5, 'No image uploaded', 
                    ha='center', va='center', fontsize=12)
            ax.axis('off')
            return error_text, fig
        
        # Convert to PIL if needed
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Store original for visualization
        original_image = image.copy()
        original_array = np.array(original_image)
        
        # Preprocess for model
        input_tensor = preprocess(image).unsqueeze(0).to(device)
        
        # Get prediction
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence_scores = probabilities[0].cpu().numpy()
            predicted_class = torch.argmax(probabilities, dim=1).item()
        
        # Class labels
        class_names = ['Normal', 'ROP']
        prediction = class_names[predicted_class]
        confidence = confidence_scores[predicted_class] * 100
        
        # Generate Grad-CAM
        cam = gradcam.generate_cam(input_tensor, predicted_class)
        
        # Resize CAM to match original image
        cam_resized = cv2.resize(cam, (original_array.shape[1], original_array.shape[0]))
        
        # Create heatmap overlay
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Overlay on original image
        overlay = cv2.addWeighted(original_array, 0.6, heatmap, 0.4, 0)
        
        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(original_array)
        axes[0].set_title('Original Image', fontsize=14, fontweight='bold')
        axes[0].axis('off')
        
        # Heatmap only
        axes[1].imshow(cam_resized, cmap='jet')
        axes[1].set_title('Attention Heatmap', fontsize=14, fontweight='bold')
        axes[1].axis('off')
        
        # Overlay
        axes[2].imshow(overlay)
        axes[2].set_title('Overlay (AI Focus Areas)', fontsize=14, fontweight='bold')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        # Format output text with improved styling
        if prediction == "ROP":
            result_icon = "🔴"
            result_color = "#e74c3c"
        else:
            result_icon = "🟢"
            result_color = "#2ecc71"
        
        result_text = f"""
### {result_icon} **Diagnosis Prediction: {prediction}**

---

#### 📊 **Analysis Results**

<div style="background-color: #f8f9fa; padding: 15px; border-radius: 8px; border-left: 4px solid {result_color};">

**Predicted Diagnosis:** <span style="color: {result_color}; font-weight: bold; font-size: 18px;">{prediction}</span>  
**Confidence Level:** <span style="font-weight: bold; font-size: 18px;">{confidence:.1f}%</span>

**Detailed Probability:**
- 🟢 Normal: `{confidence_scores[0]*100:.2f}%`
- 🔴 ROP: `{confidence_scores[1]*100:.2f}%`

</div>

---

####  **Understanding the Heatmap**

The attention heatmap visualizes which regions of the retina the AI model analyzed when making this prediction:

- **Red/Yellow areas:** High attention - regions most important for the diagnosis
- **Green/Blue areas:** Low attention - less relevant regions
- **Overlay image:** Shows AI focus areas superimposed on the original retinal image

**Clinical Interpretation:**  
For ROP cases, the model typically focuses on vascular abnormalities, ridge formations, and peripheral retinal changes.

---

---

**Important Clinical Disclaimer:**  
This system is designed as a **screening and decision support tool**. All predictions must be verified by qualified pediatric ophthalmologists before clinical decisions are made.

**Timestamp:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        """
        
        return result_text, fig
        
    except Exception as e:
        error_text = f"❌ Error: {str(e)}"
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.text(0.5, 0.5, 'Error processing image', 
                ha='center', va='center', fontsize=12)
        ax.axis('off')
        return error_text, fig

# Create Gradio interface with custom theme
custom_css = """
.gradio-container {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}
#title {
    text-align: center;
    background-color: #000000;
    color: #ffffff !important;
    padding: 25px;
    border-radius: 8px;
    margin-bottom: 20px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}
#title h1, #title h2, #title h3, #title p, #title * {
    color: #ffffff !important;
}
.instruction-box {
    background-color: #f8f9fa;
    padding: 15px;
    border-radius: 8px;
    border-left: 4px solid #2c3e50;
    margin: 10px 0;
}
"""

with gr.Blocks(title="ROP Detection System") as demo:
    
    with gr.Row(elem_id="title"):
        gr.Markdown("""
        #  Retinopathy of Prematurity (ROP) Detection System
        ### AI-Powered Diagnostic Assistance with Explainable AI
        """)
    
    gr.Markdown("""
    <div class="instruction-box">
    
    **Instructions for Clinicians:**
    1. Upload a retinal fundus photograph (RetCam or NeoScop images supported)
    2. Click the **"Analyze Image"** button to run AI analysis
    3. Review the diagnosis prediction, confidence scores, and explainability heatmap
    4. Verify findings with clinical examination
    
    **Clinical Disclaimer:** This system is a decision support tool for ROP screening. 
    All predictions must be verified by qualified pediatric ophthalmologists before clinical decisions.
    
    </div>
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📤 Upload Retinal Image")
            
            # Input
            image_input = gr.Image(
                label="Fundus Photograph", 
                type="pil",
                sources=["upload", "clipboard"],
                height=400
            )
            
            analyze_btn = gr.Button(
                "🔍 Analyze Image", 
                variant="primary",
                size="lg",
                scale=1
            )
            
            clear_btn = gr.Button(
                "🗑️ Clear",
                variant="secondary",
                size="sm"
            )
            
        
        with gr.Column(scale=2):
            gr.Markdown("###  Analysis Results")
            
            # Outputs
            result_text = gr.Markdown(label="Diagnostic Report")
            
            gr.Markdown("### Explainability Visualization")
            heatmap_output = gr.Plot(label="Grad-CAM Attention Heatmap")
    
    # Connect button to function
    analyze_btn.click(
        fn=predict_rop,
        inputs=image_input,
        outputs=[result_text, heatmap_output]
    )
    
    # Clear button functionality
    clear_btn.click(
        fn=lambda: (None, "", None),
        inputs=None,
        outputs=[image_input, result_text, heatmap_output]
    )
    
    # Example images section
    gr.Markdown("---")
    
    with gr.Accordion("📋 Example Test Images (Click to expand)", open=False):
        gr.Markdown("""
        **Test the system with sample images from the dataset:**
        - Click on an example image to automatically load it
        - Then click "Analyze Image" to see the prediction
        - These examples show varied confidence scores (not just 100%)
        """)
        
        # Provide multiple examples from different categories for varied results
        examples_list = []
        
        # Normal examples
        for i in [1, 3, 5]:
            normal_neo = f"data/raw_data/classification/Normal/Neo_Normal/{i}.png"
            if Path(normal_neo).exists():
                examples_list.append([normal_neo])
        
        for i in [1, 2]:
            normal_retcam = f"data/raw_data/classification/Normal/RetCam_Normal/{i}.png"
            if Path(normal_retcam).exists():
                examples_list.append([normal_retcam])
        
        # ROP examples
        for i in [1, 2, 3]:
            rop_neo = f"data/raw_data/classification/ROP/Neo_ROP/{i}.png"
            if Path(rop_neo).exists():
                examples_list.append([rop_neo])
        
        for i in [1, 2]:
            rop_retcam = f"data/raw_data/classification/ROP/RetCam_ROP/{i}.png"
            if Path(rop_retcam).exists():
                examples_list.append([rop_retcam])
        
        if examples_list:
            gr.Examples(
                examples=examples_list,
                inputs=image_input,
                label="Sample Images (Normal & ROP cases)"
            )
    


# Launch the app
if __name__ == "__main__":
    print("\n" + "="*80)
    print("🚀 Starting ROP Detection System GUI...")
    print("="*80)
    print(f" Launch Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Device: {device}")
    print(f" Model: ResNet50 (Transfer Learning)")
    print(f"Test Accuracy: 82.8%")
    print(f" XAI Validation: 97 expert-annotated images")
    print("="*80)
    print("\n Access the interface at: http://127.0.0.1:7860")
    print(" Press Ctrl+C to stop the server\n")
    
    demo.launch(
        share=False,  # Set to True if you want a public shareable link
        server_name="127.0.0.1",
        server_port=7860,
        show_error=True,
        css=custom_css,
        theme=gr.themes.Soft()
    )
