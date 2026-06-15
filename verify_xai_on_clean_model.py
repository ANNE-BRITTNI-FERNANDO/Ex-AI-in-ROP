"""
Robustness check: regenerate Grad-CAM IoU/Dice on the CLEAN (leakage-free) model
for all 111 image-lesion pairs, and compare to the original (leaked-model) result.
If the IoU is similar, the XAI conclusions are robust to the classification leakage.
"""
import json, numpy as np, torch, torch.nn as nn, torch.nn.functional as F, cv2
from pathlib import Path
from PIL import Image
from torchvision import models, transforms

MEAN, STD = [0.485,0.456,0.406], [0.229,0.224,0.225]
tf = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), transforms.Normalize(MEAN,STD)])

def load_model(p):
    m = models.resnet50(weights=None); m.fc = nn.Linear(2048,2)
    m.load_state_dict(torch.load(p, map_location="cpu")); m.eval(); return m

class GradCAM:
    def __init__(self, model):
        self.model=model; self.act=None; self.grad=None
        model.layer4.register_forward_hook(lambda m,i,o: setattr(self,"act",o.detach()))
        model.layer4.register_full_backward_hook(lambda m,gi,go: setattr(self,"grad",go[0].detach()))
    def __call__(self, x):
        self.model.zero_grad()
        out = self.model(x); c = int(out.argmax(1))
        out[0,c].backward()
        w = self.grad.mean(dim=(2,3), keepdim=True)
        cam = F.relu((w*self.act).sum(1, keepdim=True))[0,0].numpy()
        cam = cv2.resize(cam, (224,224))
        cam = (cam - cam.min())/(cam.max()-cam.min()+1e-8)
        return cam

def load_mask(p):
    m = np.array(Image.open(p).convert("L").resize((224,224), Image.NEAREST))
    return (m>127).astype(np.uint8)

def iou_dice(cam, mask):
    best=(0,0)
    for t in np.arange(0.1,0.9,0.05):
        pm=(cam>t).astype(np.uint8)
        inter=np.logical_and(pm,mask).sum(); union=np.logical_or(pm,mask).sum()
        iou=inter/union if union else 0
        if iou>best[0]:
            dice=2*inter/(pm.sum()+mask.sum()) if (pm.sum()+mask.sum()) else 0
            best=(iou,dice)
    return best

m = load_model("models/clean_augmented_model.pth")
cam = GradCAM(m)
omap = json.load(open("results/overlapping_images_map.json"))["images"]
rows=[]
for it in omap:
    img = tf(Image.open(it["classification_path"]).convert("RGB")).unsqueeze(0)
    heat = cam(img)
    for lesion, mpath in it.get("mask_paths", {}).items():
        if not Path(mpath).exists(): continue
        mask = load_mask(mpath)
        iou,dice = iou_dice(heat, mask)
        rows.append((lesion, iou, dice))

iou=np.array([r[1] for r in rows]); dice=np.array([r[2] for r in rows])
print(f"CLEAN model Grad-CAM: N={len(rows)} pairs  IoU={iou.mean():.4f}±{iou.std(ddof=1):.4f}  Dice={dice.mean():.4f}±{dice.std(ddof=1):.4f}")
for lt in ["optic_disc","vessels","ridge"]:
    v=[r[1] for r in rows if r[0]==lt]
    if v: print(f"  {lt:11s} n={len(v):3d}  IoU={np.mean(v):.4f}")
print(f"\nCompare to leaked-model Grad-CAM: IoU=0.0957, Dice=0.1692 (N=111)")
json.dump({"n":len(rows),"iou_mean":float(iou.mean()),"iou_std":float(iou.std(ddof=1)),
           "dice_mean":float(dice.mean()),"dice_std":float(dice.std(ddof=1)),
           "per_lesion":{lt:float(np.mean([r[1] for r in rows if r[0]==lt])) for lt in ["optic_disc","vessels","ridge"] if any(r[0]==lt for r in rows)}},
          open("results/clean_model_gradcam_iou.json","w"), indent=2)
print("Saved results/clean_model_gradcam_iou.json")
