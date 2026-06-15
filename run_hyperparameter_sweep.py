"""
Genuine hyperparameter sweep on the clean split-first setup.
Sweeps learning rate and batch size; records validation and test accuracy per setting.
Output: results/hyperparameter_tuning.json
"""
import json, time, numpy as np, torch, torch.nn as nn
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
import albumentations as A

torch.manual_seed(42); np.random.seed(42)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
ROOT = Path("data/classification")
MEAN, STD = [0.485,0.456,0.406], [0.229,0.224,0.225]
train_aug = A.Compose([A.HorizontalFlip(p=0.5), A.Rotate(limit=15, p=0.7),
                       A.RandomBrightnessContrast(0.2,0.2,p=0.5), A.CLAHE(2.0,(8,8),p=0.3)])
to_tensor = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN,STD)])

class DS(Dataset):
    def __init__(self, split, aug):
        self.items=[]; self.aug=aug
        for ci,c in enumerate(["Normal","ROP"]):
            for f in sorted((ROOT/split/c).glob("*.png")): self.items.append((f,ci))
    def __len__(self): return len(self.items)
    def __getitem__(self,i):
        f,y=self.items[i]; img=np.array(Image.open(f).convert("RGB").resize((224,224)))
        if self.aug: img=train_aug(image=img)["image"]
        return to_tensor(Image.fromarray(img)), y

def make_model():
    m=models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    for p in m.parameters(): p.requires_grad=False
    for p in m.layer4.parameters(): p.requires_grad=True
    m.fc=nn.Linear(2048,2); return m.to(DEV)

def evaluate(m, loader):
    m.eval(); c=t=0
    with torch.no_grad():
        for x,y in loader:
            x,y=x.to(DEV),y.to(DEV); c+=(m(x).argmax(1)==y).sum().item(); t+=len(y)
    return c/t

def train_once(lr, batch, epochs=25, patience=8):
    try:
        tr=DataLoader(DS("train",True), batch_size=batch, shuffle=True)
        va=DataLoader(DS("val",False), batch_size=16); te=DataLoader(DS("test",False), batch_size=16)
        m=make_model()
        opt=torch.optim.Adam([p for p in m.parameters() if p.requires_grad], lr=lr)
        sch=torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.1, patience=5)
        crit=nn.CrossEntropyLoss(); best=0; best_state=None; bad=0
        for ep in range(epochs):
            m.train()
            for x,y in tr:
                x,y=x.to(DEV),y.to(DEV); opt.zero_grad(); crit(m(x),y).backward(); opt.step()
            v=evaluate(m,va); sch.step(1-v)
            if v>best: best=v; best_state={k:val.clone() for k,val in m.state_dict().items()}; bad=0
            else: bad+=1
            if bad>=patience: break
        m.load_state_dict(best_state)
        return round(best,4), round(evaluate(m,te),4), "ok"
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            torch.cuda.empty_cache(); return None, None, "OOM"
        raise

results={"learning_rate_sweep":[], "batch_size_sweep":[], "device":DEV}
print("=== Learning-rate sweep (batch=16) ===")
for lr in [1e-4, 5e-4, 1e-3]:
    t=time.time(); v,te,st=train_once(lr,16); dt=round(time.time()-t,1)
    print(f"  LR={lr:<7} val={v} test={te} [{st}] {dt}s")
    results["learning_rate_sweep"].append({"lr":lr,"val_acc":v,"test_acc":te,"status":st,"time_s":dt})
print("=== Batch-size sweep (LR=1e-3) ===")
for b in [8,16,32]:
    t=time.time(); v,te,st=train_once(1e-3,b); dt=round(time.time()-t,1)
    print(f"  batch={b:<3} val={v} test={te} [{st}] {dt}s")
    results["batch_size_sweep"].append({"batch":b,"val_acc":v,"test_acc":te,"status":st,"time_s":dt})

Path("results").mkdir(exist_ok=True)
json.dump(results, open("results/hyperparameter_tuning.json","w"), indent=2)
print("\nSaved results/hyperparameter_tuning.json")
print(json.dumps(results, indent=2))
