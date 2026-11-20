"""Train the ResNet-50 teacher on CIFAR-100."""
import argparse, json, torch
from common import *
from models import Teacher

ap = argparse.ArgumentParser(); ap.add_argument("--epochs", type=int, default=30); ap.add_argument("--bs", type=int, default=128)
ap.add_argument("--lr", type=float, default=0.1); ap.add_argument("--limit", type=int, default=None)
a = ap.parse_args()
seed_all(0)
data = Data("cifar100")
m = Teacher(100)
h = fit(m, data, a.epochs, a.bs, a.lr, 5e-4, tag="teacher", log=f"{RES}/logs/teacher.json", limit=a.limit)
torch.save(m.state_dict(), f"{CKPT}/teacher.pt")
json.dump({"model": "resnet50_teacher", "params": n_params(m), "top1": h[-1]["acc"], "epochs": a.epochs, "train_time_s": h[-1]["time_s"]}, open(f"{RES}/teacher.json", "w"))
print("FINAL teacher top1", h[-1]["acc"])
