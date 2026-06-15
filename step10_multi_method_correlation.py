"""Multi-method spatial correlation between the four XAI heatmaps (leakage-free model).

Mirrors the pair alignment used in final_clean_comparison.py: the 111 (image, lesion)
pairs are taken in the same order across Grad-CAM / IG / SHAP / LIME. For each pair we
load each method's 224x224 heatmap, flatten it, and compute the pairwise Pearson
correlation between methods; results are averaged over all valid pairs.

Interpretation: high correlation => methods highlight the same pixels (methodological
convergence); low correlation => methods disagree on where the model "looks".
"""
import json, itertools, numpy as np, cv2, torch, torch.nn as nn, torch.nn.functional as F
from pathlib import Path
from PIL import Image
from torchvision import models, transforms
from scipy import stats

dev = "cuda" if torch.cuda.is_available() else "cpu"
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                         transforms.Normalize(MEAN, STD)])

# ---- Grad-CAM on the clean model (same as final_clean_comparison.py) ----
m = models.resnet50(weights=None); m.fc = nn.Linear(2048, 2)
m.load_state_dict(torch.load("models/clean_augmented_model.pth", map_location=dev)); m.eval().to(dev)
act, grad = {}, {}
m.layer4.register_forward_hook(lambda mo, i, o: act.__setitem__("v", o.detach()))
m.layer4.register_full_backward_hook(lambda mo, gi, go: grad.__setitem__("v", go[0].detach()))

def gradcam(x):
    m.zero_grad(); out = m(x); c = int(out.argmax(1)); out[0, c].backward()
    w = grad["v"].mean(dim=(2, 3), keepdim=True)
    cam = F.relu((w * act["v"]).sum(1, keepdim=True))[0, 0].cpu().numpy()
    cam = cv2.resize(cam, (224, 224))
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam.astype(np.float32)

def load_hm(p):
    """Load a saved heatmap (.npy), squeeze, resize to 224x224, min-max normalise."""
    a = np.load(p)
    a = np.asarray(a, dtype=np.float32)
    if a.ndim == 3:                      # (H,W,C) or (C,H,W) -> single channel
        a = a.mean(axis=0) if a.shape[0] in (1, 3) else a.mean(axis=2)
    if a.shape != (224, 224):
        a = cv2.resize(a, (224, 224))
    rng = a.max() - a.min()
    return (a - a.min()) / (rng + 1e-8) if rng > 0 else a * 0.0

# ---- build the 111-pair-aligned Grad-CAM list (same nested order as the comparison) ----
omap = json.load(open("results/overlapping_images_map.json"))["images"]
gc_cams = []
for it in omap:
    x = tf(Image.open(it["classification_path"]).convert("RGB")).unsqueeze(0).to(dev)
    cam = gradcam(x)
    for lt, mp in it.get("mask_paths", {}).items():
        if Path(mp).exists():
            gc_cams.append(cam)          # per-image cam repeated per lesion (matches gc list)

ig = json.load(open("results/ig_clean/ig_results.json"))["pair_results"]
sh = json.load(open("results/shap_clean/shap_results.json"))["images"]
li = json.load(open("results/lime_clean/lime_results.json"))["images"]
assert len(gc_cams) == len(ig) == len(sh) == len(li), (len(gc_cams), len(ig), len(sh), len(li))
N = len(gc_cams)

# ---- per-pair heatmaps for every method ----
def ig_path(rec):
    return Path("results/ig_clean/heatmaps") / (rec["image_key"].replace(".png", ".npy"))

methods = ["Grad-CAM", "Integrated Gradients", "SHAP", "LIME"]
pair_maps = []            # list of dicts: method -> flattened heatmap
for i in range(N):
    maps = {}
    try:
        maps["Grad-CAM"] = gc_cams[i].ravel()
        maps["Integrated Gradients"] = load_hm(ig_path(ig[i])).ravel()
        maps["SHAP"] = load_hm(sh[i]["heatmap_path"]).ravel()
        maps["LIME"] = load_hm(li[i]["heatmap_path"]).ravel()
        pair_maps.append(maps)
    except FileNotFoundError as e:
        print("  skip pair", i, "->", e)

# ---- pairwise Pearson per pair, averaged ----
pairwise = {f"{a} vs {b}": [] for a, b in itertools.combinations(methods, 2)}
for maps in pair_maps:
    for a, b in itertools.combinations(methods, 2):
        va, vb = maps[a], maps[b]
        if va.std() > 0 and vb.std() > 0:
            r, _ = stats.pearsonr(va, vb)
            if np.isfinite(r):
                pairwise[f"{a} vs {b}"].append(r)

summary = {k: round(float(np.mean(v)), 4) for k, v in pairwise.items() if v}
mean_r = round(float(np.mean([np.mean(v) for v in pairwise.values() if v])), 4)

# build symmetric mean-correlation matrix
mat = {a: {b: 1.0 for b in methods} for a in methods}
for a, b in itertools.combinations(methods, 2):
    v = pairwise[f"{a} vs {b}"]
    r = round(float(np.mean(v)), 4) if v else None
    mat[a][b] = mat[b][a] = r

out = {"model": "clean_augmented_model.pth (leakage-free)",
       "n_pairs_used": len(pair_maps),
       "pairwise_mean_pearson": summary,
       "mean_correlation_overall": mean_r,
       "correlation_matrix": mat}
json.dump(out, open("results/multi_method_correlation.json", "w"), indent=2)

print(f"\nPairs used: {len(pair_maps)} / {N}")
print("Pairwise mean Pearson correlation between heatmaps:")
for k, v in summary.items():
    print(f"  {k:38s} r = {v:+.4f}")
print(f"\nOverall mean correlation: r = {mean_r:+.4f}")
print("Saved -> results/multi_method_correlation.json")
