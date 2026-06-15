"""
Retinopathy of Prematurity - Clinical Decision-Support Interface (v2)
Quantitative Validation of Explainable AI Methods for ROP Detection (CB012565)

Tabs:
  1. Screening        - upload/select fundus image -> diagnosis, confidence, 4-method explanations
  2. XAI Comparison   - Grad-CAM vs Integrated Gradients vs LIME vs SHAP (leakage-free numbers)
  3. Model Comparison - ResNet50 vs EfficientNet-B0 vs DenseNet121
  4. Validation       - metrics, faithfulness, methodology, limitations

All comparison numbers are read live from results/*.json (the clean canonical files).
Run with the project venv:  .\.venv\Scripts\python.exe clinical_gui_v2.py
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from matplotlib import colormaps
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms
import gradio as gr

ROOT = Path(__file__).parent
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
TF = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                         transforms.Normalize(MEAN, STD)])
_JET = colormaps["jet"]

# ---- model -------------------------------------------------------------------
def load_model():
    for p in ["models/clean_augmented_model.pth", "models/augmented_best_model.pth"]:
        fp = ROOT / p
        if fp.exists():
            m = models.resnet50(weights=None); m.fc = nn.Linear(2048, 2)
            sd = torch.load(fp, map_location=DEVICE)
            if isinstance(sd, dict) and "model_state_dict" in sd:
                sd = sd["model_state_dict"]
            m.load_state_dict(sd); m.eval().to(DEVICE)
            print(f"Loaded {p} on {DEVICE}")
            return m, p
    raise FileNotFoundError("No model checkpoint found")

MODEL, MODEL_NAME = load_model()
_act, _grad = {}, {}
MODEL.layer4.register_forward_hook(lambda m, i, o: _act.__setitem__("v", o))
MODEL.layer4.register_full_backward_hook(lambda m, gi, go: _grad.__setitem__("v", go[0].detach()))

def fit_temperature():
    """Temperature scaling: calibrate softmax confidence on the held-out validation set
    (reduces the over-confidence typical of CNNs so demo probabilities are realistic)."""
    val = ROOT / "data/classification/val"
    if not val.exists():
        return 1.0
    logits, labels = [], []
    for ci, c in enumerate(["Normal", "ROP"]):
        for f in (val / c).glob("*.png"):
            x = TF(Image.open(f).convert("RGB")).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                logits.append(MODEL(x)[0].cpu())
            labels.append(ci)
    if not logits:
        return 1.0
    logits = torch.stack(logits); labels = torch.tensor(labels)
    T = torch.nn.Parameter(torch.ones(1) * 1.5)
    opt = torch.optim.LBFGS([T], lr=0.05, max_iter=80)
    nll = torch.nn.CrossEntropyLoss()
    def closure():
        opt.zero_grad(); loss = nll(logits / T.clamp(min=0.05), labels); loss.backward(); return loss
    opt.step(closure)
    return float(T.detach().clamp(min=1.0))

TEMP = fit_temperature()
print(f"Calibration temperature T = {TEMP:.2f}")

def load_background(n=8):
    """Real-image background for SHAP (expected gradients) so it differs from IG's zero baseline."""
    tr = ROOT / "data/classification/train"
    paths = []
    for c in ["Normal", "ROP"]:
        paths += sorted((tr / c).glob("*.png"))[: n // 2]
    imgs = [TF(Image.open(p).convert("RGB")) for p in paths[:n]]
    return torch.stack(imgs).to(DEVICE) if imgs else torch.zeros(1, 3, 224, 224, device=DEVICE)

SHAP_BG = load_background()

def _norm(a):
    return (a - a.min()) / (a.max() - a.min() + 1e-8)

def _smooth(a, sigma=7):
    """Render sparse pixel-attributions (IG, SHAP) as a readable heat-map."""
    a = gaussian_filter(a.astype(float), sigma=sigma)
    hi = np.percentile(a, 99) + 1e-8
    return _norm(np.clip(a, 0, hi))

def overlay(orig, cam):
    heat = (_JET(cam)[..., :3] * 255).astype(np.uint8)
    return (0.55 * orig + 0.45 * heat).astype(np.uint8)

def predict_probs(x):
    with torch.no_grad():
        return torch.softmax(MODEL(x) / TEMP, 1)[0].cpu().numpy()   # temperature-calibrated

def grad_cam(x, cls):
    MODEL.zero_grad()
    out = MODEL(x); out[0, cls].backward()
    w = _grad["v"].mean(dim=(2, 3), keepdim=True)
    cam_t = F.relu((w * _act["v"]).sum(1, keepdim=True))
    cam_t = F.interpolate(cam_t, size=(224, 224), mode="bilinear", align_corners=False)
    return _norm(cam_t[0, 0].detach().cpu().numpy())

def integrated_gradients(x, cls, steps=32):
    base = torch.zeros_like(x)
    total = torch.zeros_like(x)
    for a in torch.linspace(0, 1, steps, device=DEVICE):
        xi = (base + a * (x - base)).clone().requires_grad_(True)
        MODEL.zero_grad()
        MODEL(xi)[0, cls].backward()
        total = total + xi.grad
    ig = (x - base) * total / steps
    return _smooth(ig[0].abs().sum(0).detach().cpu().numpy())

def shap_heatmap(x, cls):
    import shap
    sv = shap.GradientExplainer(MODEL, SHAP_BG).shap_values(x)
    arr = sv[cls] if isinstance(sv, list) else sv
    arr = np.abs(np.squeeze(np.array(arr)))     # drop size-1 (batch) dims
    while arr.ndim > 2:                          # collapse channel/class dims (smaller than 224)
        arr = arr.sum(axis=int(np.argmin(arr.shape)))
    return _smooth(arr)

def lime_heatmap(orig_uint8, cls):
    from lime import lime_image
    def predict_fn(imgs):
        batch = torch.stack([TF(Image.fromarray(im.astype(np.uint8))) for im in imgs]).to(DEVICE)
        with torch.no_grad():
            return torch.softmax(MODEL(batch), 1).cpu().numpy()
    expl = lime_image.LimeImageExplainer().explain_instance(
        orig_uint8, predict_fn, labels=(cls,), hide_color=0, num_samples=400)
    weights = dict(expl.local_exp[cls]); seg = expl.segments
    h = np.zeros(seg.shape, dtype=float)
    for s, w in weights.items():
        h[seg == s] = max(w, 0)
    return _norm(h)

BLANK = "<placeholder>"
def analyze(image, include_slow):
    if image is None:
        neutral = "<div class='card neutral'>Select or upload a fundus image, then choose Analyze.</div>"
        return (None, neutral, None, None, None,
                gr.update(visible=False), gr.update(visible=False), "")
    pil = image.convert("RGB")
    x = TF(pil).unsqueeze(0).to(DEVICE)
    probs = predict_probs(x)
    p_norm, p_rop = float(probs[0]), float(probs[1])
    cls = int(np.argmax(probs))
    # cap displayed confidence at 99.9% (a model is never absolutely certain; avoids a misleading "100%")
    top = min(max(p_norm, p_rop), 0.999)
    if cls == 1:
        p_rop, p_norm = top, 1 - top
    else:
        p_norm, p_rop = top, 1 - top
    orig = np.array(pil.resize((224, 224)))

    def safe(fn):
        try:
            return overlay(orig, fn())
        except Exception as e:
            print("xai err", e); return None
    gc_im = safe(lambda: grad_cam(x, cls))
    ig_im = safe(lambda: integrated_gradients(x, cls))
    if include_slow:
        shap_im = safe(lambda: shap_heatmap(x, cls))
        lime_im = safe(lambda: lime_heatmap(orig, cls))
        shap_u = gr.update(value=shap_im, visible=True)
        lime_u = gr.update(value=lime_im, visible=True)
    else:
        shap_u = gr.update(value=None, visible=False)
        lime_u = gr.update(value=None, visible=False)

    if cls == 1:
        card = (f"<div class='card rop'><div class='dx'>ROP DETECTED</div>"
                f"<div class='conf'>Confidence {p_rop*100:.1f}%</div>"
                f"<div class='sub'>Recommend referral to a paediatric ophthalmologist for confirmation.</div></div>")
    else:
        card = (f"<div class='card normal'><div class='dx'>NORMAL</div>"
                f"<div class='conf'>Confidence {p_norm*100:.1f}%</div>"
                f"<div class='sub'>No retinopathy features detected by the model.</div></div>")
    extra = " SHAP and LIME shown." if include_slow else " Tick the box and re-analyze to add SHAP and LIME."
    note = (f"**Probabilities** &mdash; Normal {p_norm*100:.1f}% &middot; ROP {p_rop*100:.1f}% "
            f"(temperature-calibrated, T = {TEMP:.2f}).\n\n"
            f"**Reading the overlays** &mdash; warm colours mark the regions that most influenced the "
            f"model's decision. Confirm the model focused on clinically relevant structures "
            f"(optic disc, vessels, demarcation ridge).{extra}\n\n"
            f"Note: decision-support tool only. Final clinical decisions rest with a qualified clinician.")
    return {"Normal": p_norm, "ROP": p_rop}, card, orig, gc_im, ig_im, shap_u, lime_u, note

# ---- comparison tables (from clean canonical JSON) ---------------------------
def load_json(p):
    fp = ROOT / p
    return json.load(open(fp)) if fp.exists() else {}

def xai_table():
    d = load_json("results/clean_xai_comparison.json").get("summary", {})
    times = {"Grad-CAM": "0.5 s", "Integrated Gradients": "28 s", "LIME": "59 s", "SHAP": "40 s"}
    rows = []
    for m in ["Grad-CAM", "Integrated Gradients", "LIME", "SHAP"]:
        s = d.get(m, {})
        rows.append([m, f"{s.get('iou_mean',0):.4f}", f"{s.get('dice_mean',0):.4f}",
                     times.get(m, "-"), "Best" if m == "Grad-CAM" else ""])
    return pd.DataFrame(rows, columns=["XAI Method", "Mean IoU", "Mean Dice", "Time / image", "Verdict"])

def model_table():
    d = load_json("results/multi_arch_comparison/comparison_summary.json").get("models", {})
    rows = []
    for m in ["ResNet50", "EfficientNet-B0", "DenseNet121"]:
        s = d.get(m, {})
        rows.append([m, f"{s.get('test_accuracy',0)*100:.2f}%", f"{s.get('f1_score',0):.4f}",
                     f"{s.get('auc_roc',0):.4f}", f"{s.get('recall_sensitivity',0):.4f}",
                     "Selected" if m == "ResNet50" else ""])
    return pd.DataFrame(rows, columns=["Architecture", "Accuracy", "F1", "AUC-ROC", "Sensitivity", "Backbone"])

def faith_table():
    d = load_json("results/faithfulness_metrics/faithfulness_results.json")
    rows = []
    for m in ["Grad-CAM", "Integ. Grad."]:
        s = d.get(m, {})
        rows.append([m, f"{s.get('insertion_auc_mean','-')}", f"{s.get('deletion_auc_mean','-')}",
                     f"{s.get('faithfulness_score','-')}"])
    return pd.DataFrame(rows, columns=["Method", "Insertion AUC (higher better)",
                                       "Deletion AUC (lower better)", "Faithfulness (higher better)"])

EXAMPLES = [[str(p)] for p in sorted((ROOT / "gui_examples").glob("*.png"))]

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
*, .gradio-container, .gradio-container * { font-family:'Inter','Segoe UI',system-ui,sans-serif !important; }
.gradio-container { max-width:1180px !important; margin:auto; background:#f4f6f8; }
#hdr { background:#13293d; padding:20px 26px; border-radius:0 0 6px 6px; margin-bottom:6px; }
#hdr h1 { font-size:21px; font-weight:700; margin:0; letter-spacing:.2px; color:#ffffff !important; }
#hdr p { font-size:13px; margin:6px 0 0 0; color:#c7d3de !important; }
.card { border-radius:8px; padding:20px; text-align:center; }
.card .dx { font-size:26px; font-weight:700; letter-spacing:1px; color:#ffffff !important; }
.card .conf { font-size:16px; margin-top:6px; color:#ffffff !important; }
.card .sub { font-size:12.5px; margin-top:10px; font-weight:400; color:#eef3f7 !important; }
.card div { color:#ffffff !important; }
.card.rop { background:#b3261e; }
.card.normal { background:#1b5e20; }
.card.neutral { background:#5f6b76; }
.section-title { font-size:15px; font-weight:600; color:#13293d; border-left:3px solid #0e7c86;
                 padding-left:10px; margin:14px 0 6px 0; }
button.primary, .primary { background:#0e7c86 !important; border:none !important; }
footer { visibility:hidden; }
"""

with gr.Blocks(title="ROP Clinical Decision-Support", theme=gr.themes.Base(primary_hue="teal"), css=CSS) as demo:
    gr.HTML("<div id='hdr'><h1>Retinopathy of Prematurity &ndash; Clinical Decision-Support</h1>"
            f"<p>ResNet50 screening with explainability quantitatively validated against expert annotations "
            f"&middot; compute device: {DEVICE.upper()}</p></div>")

    with gr.Tabs():
        # ---- TAB 1: SCREENING ------------------------------------------------
        with gr.Tab("Screening"):
            with gr.Row():
                with gr.Column(scale=1):
                    img_in = gr.Image(label="Retinal fundus image", type="pil", height=300)
                    chk_slow = gr.Checkbox(label="Include SHAP and LIME (slower, about one to two minutes)", value=False)
                    with gr.Row():
                        btn = gr.Button("Analyze", variant="primary", size="lg")
                        clr = gr.Button("Clear", size="lg")
                    gr.HTML("<div class='section-title'>Demo cases (6 normal, 6 ROP)</div>")
                    gr.Examples(examples=EXAMPLES, inputs=img_in, examples_per_page=12)
                with gr.Column(scale=1):
                    dx_card = gr.HTML("<div class='card neutral'>Select or upload a fundus image, then choose Analyze.</div>")
                    probs = gr.Label(num_top_classes=2, label="Class probabilities")
                    orig_view = gr.Image(label="Analyzed image (224 x 224)", height=220)
            gr.HTML("<div class='section-title'>Model attention &ndash; explanation overlays</div>")
            with gr.Row(equal_height=True):
                gc_im = gr.Image(label="Grad-CAM (gradient-based)", height=240)
                ig_im = gr.Image(label="Integrated Gradients", height=240)
                shap_im = gr.Image(label="SHAP (game-theoretic)", height=240, visible=False)
                lime_im = gr.Image(label="LIME (superpixel)", height=240, visible=False)
            note = gr.Markdown()
            outs = [probs, dx_card, orig_view, gc_im, ig_im, shap_im, lime_im, note]
            btn.click(analyze, [img_in, chk_slow], outs)
            clr.click(lambda: (None, "<div class='card neutral'>Cleared.</div>", None, None, None,
                               gr.update(visible=False), gr.update(visible=False), ""), None, outs)

        # ---- TAB 2: XAI COMPARISON -------------------------------------------
        with gr.Tab("XAI Method Comparison"):
            gr.HTML("<div class='section-title'>Which explanation method is most trustworthy?</div>")
            gr.Markdown("Four XAI methods evaluated on the leakage-free model against expert pixel masks "
                        "(N = 111 image-lesion pairs). Higher IoU / Dice indicates closer agreement with expert annotations.")
            gr.Dataframe(value=xai_table(), interactive=False, wrap=True)
            gr.Markdown("**Verdict:** Grad-CAM leads on spatial overlap and speed. One-way ANOVA across the four "
                        "methods: F = 54.07, p = 9 x 10^-30 (all pairwise differences significant). Grad-CAM is also "
                        "the most faithful method (Validation tab) and the only one fast enough for real-time use.")
            with gr.Row():
                gr.Image(value="results/canonical_figures/fig6_3_iou_dice_corrected.png",
                         label="IoU / Dice comparison (clean model)", height=300)
                gr.Image(value="results/enhanced_xai_comparison/fig3_lesion_specific_iou.png",
                         label="Per-lesion IoU", height=300)

        # ---- TAB 3: MODEL COMPARISON -----------------------------------------
        with gr.Tab("Model Comparison"):
            gr.HTML("<div class='section-title'>Backbone selection</div>")
            gr.Markdown("Three CNN backbones trained on the Czech ROP dataset (test n = 1,466). ResNet50 was "
                        "selected for the highest AUC-ROC and F1, with the cleanest Grad-CAM compatibility.")
            gr.Dataframe(value=model_table(), interactive=False, wrap=True)
            gr.Image(value="results/canonical_figures/fig6_1_multiarch.png",
                     label="Multi-architecture comparison", height=330)

        # ---- TAB 4: VALIDATION -----------------------------------------------
        with gr.Tab("Validation and Methodology"):
            gr.HTML("<div class='section-title'>Faithfulness</div>")
            gr.Markdown("Does the explanation reflect the model's reasoning? Insertion AUC (adding important pixels "
                        "restores the prediction) should be high; Deletion AUC (removing them) should be low. A larger "
                        "gap means a more faithful explanation.")
            gr.Dataframe(value=faith_table(), interactive=False, wrap=True)
            gr.HTML("<div class='section-title'>Methodology (leakage-free)</div>")
            gr.Markdown(
                "- Dataset: HVDROPDB (185 images; 97 with expert masks, giving 111 image-lesion pairs) and Czech ROP (6,004).\n"
                "- Split-first, augment-training-only: the 185 images were split (129/27/29) before augmentation, which "
                "was applied online to the training set only, removing the augment-then-split leakage of the original pipeline.\n"
                "- Model: ResNet50, ImageNet-pretrained, layer4 fine-tuned, Adam (LR 1e-3), early stopping.\n"
                "- XAI: Grad-CAM, Integrated Gradients, SHAP, LIME, binarised at an oracle threshold, scored by IoU / Dice against masks.\n"
                "- Statistics: one-way ANOVA and pairwise t-tests.")
            gr.HTML("<div class='section-title'>Limitations</div>")
            gr.Markdown(
                "- Absolute IoU (0.05-0.15) is modest because the classifier was trained on image-level labels, not pixel "
                "masks; this is expected for weakly-supervised attention.\n"
                "- The clean HVDROPDB test set is small (29 images); the robust accuracy benchmark is the Czech ResNet50 "
                "(94.47%, AUC 0.9839).\n"
                "- An augment-then-split data-leakage flaw was identified, corrected by retraining split-first, and the XAI "
                "ranking confirmed unchanged on the clean model.\n\n"
                "Research prototype, not a certified medical device.")

    gr.HTML("<center style='color:#7a8794;font-size:12px;padding:8px;'>CB012565 &middot; "
            "Quantitative Validation of Explainable AI Methods for Retinopathy of Prematurity Detection</center>")

if __name__ == "__main__":
    demo.launch(inbrowser=True)
