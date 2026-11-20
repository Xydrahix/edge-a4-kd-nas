"""Train the 3 student architectures on CIFAR-100: baseline (no KD), logit KD (soft+hard), and KD + feature distillation."""
import argparse, json, sys, torch
from common import *
from models import Teacher, build_student, STUDENTS
from kd import kd_loss_fn, FeatureHint

ap = argparse.ArgumentParser()
ap.add_argument("--mode", choices=["baseline", "kd", "kd_feat"], required=True)
ap.add_argument("--students", nargs="+", default=list(STUDENTS))
ap.add_argument("--epochs", type=int, default=30); ap.add_argument("--bs", type=int, default=128)
ap.add_argument("--lr", type=float, default=0.1); ap.add_argument("--T", type=float, default=4.0)
ap.add_argument("--alpha", type=float, default=0.7); ap.add_argument("--beta", type=float, default=1.0)
ap.add_argument("--limit", type=int, default=None)
a = ap.parse_args()
data = Data("cifar100")
teacher = None
if a.mode != "baseline":
    teacher = Teacher(100); teacher.load_state_dict(torch.load(f"{CKPT}/teacher.pt", map_location="cpu")); teacher.to(DEV).to(memory_format=torch.channels_last).eval()
    for p in teacher.parameters(): p.requires_grad_(False)
for name in a.students:
    seed_all(0)
    m = build_student(name, 100)
    tag = f"{name}_{a.mode}"
    hint = None; extra = ()
    if a.mode == "kd_feat":
        hint = FeatureHint(m.feat_ch, Teacher.FEAT_CH).to(DEV).to(memory_format=torch.channels_last); extra = list(hint.parameters())
    lf = None if a.mode == "baseline" else kd_loss_fn(teacher, a.T, a.alpha, hint, a.beta)
    h = fit(m, data, a.epochs, a.bs, a.lr, 5e-4, loss_fn=lf, extra_params=extra, tag=tag, log=f"{RES}/logs/{tag}.json", limit=a.limit)
    torch.save(m.state_dict(), f"{CKPT}/{tag}.pt")
    json.dump({"model": name, "mode": a.mode, "params": n_params(m), "top1": h[-1]["acc"], "best_top1": max(x["acc"] for x in h),
               "epochs": a.epochs, "T": a.T, "alpha": a.alpha, "train_time_s": h[-1]["time_s"]}, open(f"{RES}/{tag}.json", "w"))
    print("FINAL", tag, h[-1]["acc"], flush=True)
