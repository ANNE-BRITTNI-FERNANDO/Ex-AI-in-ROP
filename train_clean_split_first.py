"""
CLEAN, leakage-free training (split-first, augment-training-only).
- Splits already done: data/classification/{train,val,test} are ORIGINAL images (129/27/29).
- Augmentation is applied ONLINE to the training set only; val/test stay original.
- This eliminates the augment-then-split leakage in the old pipeline.
Outputs: models/clean_augmented_model.pth + results/clean_training_results.json
"""
import json, time, numpy as np, torch, torch.nn as nn
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
import albumentations as A
from sklearn.metrics import roc_auc_score

torch.manual_seed(42); np.random.seed(42)
ROOT = Path("data/classification")
MEAN, STD = [0.485,0.456,0.406], [0.229,0.224,0.225]

train_aug = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=15, p=0.7),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.CLAHE(clip_limit=2.0, tile_grid_size=(8,8), p=0.3),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=0, p=0.3),
])
to_tensor = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])

class DS(Dataset):
    def __init__(self, split, augment):
        self.items=[]; self.augment=augment
        for ci,c in enumerate(["Normal","ROP"]):
            for f in sorted((ROOT/split/c).glob("*.png")):
                self.items.append((f, ci))
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        f, y = self.items[i]
        img = np.array(Image.open(f).convert("RGB").resize((224,224)))
        if self.augment:
            img = train_aug(image=img)["image"]
        return to_tensor(Image.fromarray(img)), y

def make_model():
    m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    for p in m.parameters(): p.requires_grad=False
    for p in m.layer4.parameters(): p.requires_grad=True   # match methodology: train layer4 + fc
    m.fc = nn.Linear(2048, 2)
    return m

def run(name, augment, epochs=40):
    tr = DataLoader(DS("train", augment), batch_size=16, shuffle=True)
    va = DataLoader(DS("val", False), batch_size=16)
    te = DataLoader(DS("test", False), batch_size=16)
    m = make_model()
    opt = torch.optim.Adam([p for p in m.parameters() if p.requires_grad], lr=1e-3)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.1, patience=5)
    crit = nn.CrossEntropyLoss()
    best_val=0; best_state=None; patience=10; bad=0; hist=[]; t0=time.time()
    for ep in range(epochs):
        m.train()
        for x,y in tr:
            opt.zero_grad(); loss=crit(m(x),y); loss.backward(); opt.step()
        # val
        m.eval(); vc=vt=0
        with torch.no_grad():
            for x,y in va:
                vc+=(m(x).argmax(1)==y).sum().item(); vt+=len(y)
        vacc=vc/vt; sched.step(1-vacc); hist.append(vacc)
        if vacc>best_val: best_val=vacc; best_state={k:v.clone() for k,v in m.state_dict().items()}; bad=0
        else: bad+=1
        if bad>=patience: print(f"  early stop @ {ep+1}"); break
    m.load_state_dict(best_state)
    # test
    m.eval(); yt=[]; yp=[]; pr=[]
    with torch.no_grad():
        for x,y in te:
            p=torch.softmax(m(x),1)
            yp+=p.argmax(1).tolist(); yt+=y.tolist(); pr+=p[:,1].tolist()
    yt=np.array(yt); yp=np.array(yp)
    acc=(yt==yp).mean()
    tp=int(((yp==1)&(yt==1)).sum()); tn=int(((yp==0)&(yt==0)).sum())
    fp=int(((yp==1)&(yt==0)).sum()); fn=int(((yp==0)&(yt==1)).sum())
    sens=tp/(tp+fn) if tp+fn else 0; spec=tn/(tn+fp) if tn+fp else 0
    f1=2*tp/(2*tp+fp+fn) if (2*tp+fp+fn) else 0
    auc=roc_auc_score(yt,pr) if len(set(yt))>1 else float("nan")
    res=dict(name=name, augment=augment, best_val=round(best_val,4), epochs_ran=len(hist),
             test_acc=round(float(acc),4), sensitivity=round(sens,4), specificity=round(spec,4),
             f1=round(f1,4), auc=round(float(auc),4), cm=dict(tn=tn,fp=fp,fn=fn,tp=tp),
             time_s=round(time.time()-t0,1))
    print(f"  {name}: test_acc={acc:.4f} sens={sens:.4f} spec={spec:.4f} auc={auc:.4f} CM={res['cm']} val={best_val:.4f} t={res['time_s']}s")
    return m, res

print("Training CLEAN augmented model (split-first, online aug)...")
m_aug, r_aug = run("clean_augmented", augment=True, epochs=40)
torch.save(m_aug.state_dict(), "models/clean_augmented_model.pth")
print("Training CLEAN no-aug model (control)...")
m_noaug, r_noaug = run("clean_no_aug", augment=False, epochs=40)

out=dict(clean_augmented=r_aug, clean_no_aug=r_noaug,
         note="Split-first; augmentation applied online to training set only; val/test original. Leakage-free.")
Path("results").mkdir(exist_ok=True)
json.dump(out, open("results/clean_training_results.json","w"), indent=2)
print("\nSaved models/clean_augmented_model.pth and results/clean_training_results.json")
print(json.dumps(out, indent=2))
