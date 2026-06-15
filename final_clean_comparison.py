"""Final clean 4-way XAI comparison (all methods on the leakage-free model) + statistics."""
import json, numpy as np, itertools, cv2, torch, torch.nn as nn, torch.nn.functional as F
from pathlib import Path
from PIL import Image
from torchvision import models, transforms
from scipy import stats

dev = "cuda" if torch.cuda.is_available() else "cpu"
MEAN,STD=[0.485,0.456,0.406],[0.229,0.224,0.225]
tf=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize(MEAN,STD)])

# ---- recompute Grad-CAM per-pair on clean model ----
m=models.resnet50(weights=None); m.fc=nn.Linear(2048,2)
m.load_state_dict(torch.load("models/clean_augmented_model.pth",map_location=dev)); m.eval().to(dev)
act={}; grad={}
m.layer4.register_forward_hook(lambda mo,i,o: act.__setitem__("v",o.detach()))
m.layer4.register_full_backward_hook(lambda mo,gi,go: grad.__setitem__("v",go[0].detach()))
def gradcam(x):
    m.zero_grad(); out=m(x); c=int(out.argmax(1)); out[0,c].backward()
    w=grad["v"].mean(dim=(2,3),keepdim=True)
    cam=F.relu((w*act["v"]).sum(1,keepdim=True))[0,0].cpu().numpy()
    cam=cv2.resize(cam,(224,224)); cam=(cam-cam.min())/(cam.max()-cam.min()+1e-8); return cam
def loadmask(p):
    a=np.array(Image.open(p).convert("L").resize((224,224),Image.NEAREST)); return (a>127).astype(np.uint8)
def best_iou_dice(cam,mask):
    b=(0,0)
    for t in np.arange(0.1,0.9,0.05):
        pm=(cam>t).astype(np.uint8); inter=np.logical_and(pm,mask).sum(); union=np.logical_or(pm,mask).sum()
        iou=inter/union if union else 0
        if iou>b[0]: b=(iou, 2*inter/(pm.sum()+mask.sum()) if (pm.sum()+mask.sum()) else 0)
    return b
omap=json.load(open("results/overlapping_images_map.json"))["images"]
gc=[]
for it in omap:
    x=tf(Image.open(it["classification_path"]).convert("RGB")).unsqueeze(0).to(dev); cam=gradcam(x)
    for lt,mp in it.get("mask_paths",{}).items():
        if Path(mp).exists():
            iou,dice=best_iou_dice(cam,loadmask(mp)); gc.append((lt,iou,dice))

# ---- load IG / SHAP / LIME per-pair ----
ig=json.load(open("results/ig_clean/ig_results.json"))["pair_results"]
sh=json.load(open("results/shap_clean/shap_results.json"))["images"]
li=json.load(open("results/lime_clean/lime_results.json"))["images"]
methods={
 "Grad-CAM":([r[1] for r in gc],[r[2] for r in gc],[(r[0],r[1]) for r in gc]),
 "Integrated Gradients":([x["iou"] for x in ig],[x["dice"] for x in ig],[(x["lesion_type"],x["iou"]) for x in ig]),
 "LIME":([x["iou"] for x in li],[x["dice"] for x in li],[(x["lesion_type"],x["iou"]) for x in li]),
 "SHAP":([x["iou"] for x in sh],[x["dice"] for x in sh],[(x["lesion_type"],x["iou"]) for x in sh]),
}
out={"model":"clean_augmented_model.pth (leakage-free)","n_pairs":111,"summary":{}}
print(f"{'METHOD':22s} N  IoU mean±std     Dice mean±std    optic vessels ridge")
for name,(iou,dice,pl) in methods.items():
    iou=np.array(iou); dice=np.array(dice)
    per={lt:float(np.mean([v for l,v in pl if l==lt])) for lt in ["optic_disc","vessels","ridge"]}
    out["summary"][name]=dict(n=len(iou),iou_mean=round(float(iou.mean()),4),iou_std=round(float(iou.std(ddof=1)),4),
        dice_mean=round(float(dice.mean()),4),dice_std=round(float(dice.std(ddof=1)),4),per_lesion_iou={k:round(v,4) for k,v in per.items()})
    print(f"{name:22s}{len(iou):3d} {iou.mean():.4f}±{iou.std(ddof=1):.4f}  {dice.mean():.4f}±{dice.std(ddof=1):.4f}  {per['optic_disc']:.3f} {per['vessels']:.3f} {per['ridge']:.3f}")
ious={n:np.array(v[0]) for n,v in methods.items()}
F,p=stats.f_oneway(*ious.values())
out["anova_iou"]={"F":round(float(F),3),"p":float(p)}
print(f"\nANOVA(IoU): F={F:.3f}, p={p:.3e}")
out["pairwise_ttests_iou"]={}
for a,b in itertools.combinations(methods,2):
    t,pp=stats.ttest_ind(ious[a],ious[b]); out["pairwise_ttests_iou"][f"{a} vs {b}"]={"t":round(float(t),3),"p":float(pp)}
    print(f"  {a} vs {b}: t={t:.3f}, p={pp:.3e}")
out["ranking_by_iou"]=sorted(out["summary"],key=lambda k:-out["summary"][k]["iou_mean"])
json.dump(out,open("results/clean_xai_comparison.json","w"),indent=2)
print("\nRanking:",out["ranking_by_iou"]); print("Saved results/clean_xai_comparison.json")
