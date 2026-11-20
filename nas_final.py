"""Retrain the best NAS architecture from scratch on the full CIFAR-10 train set, evaluate on the test set, export to ONNX."""
import argparse, json, torch
from common import *
import hw
from nas_space import build

ap = argparse.ArgumentParser(); ap.add_argument("--epochs", type=int, default=40)
a = ap.parse_args()
best = json.load(open(f"{RES}/nas_best.json"))["best"]; arch = best["arch"]
seed_all(0); data = Data("cifar10"); m = build(arch)
h = fit(m, data, a.epochs, 128, 0.1, 5e-4, res=arch["res"], tag="nas_final", log=f"{RES}/logs/nas_final.json")
torch.save(m.state_dict(), f"{CKPT}/nas_final.pt")
hw.export_bytes(m, arch["res"], path=f"{ONNX}/nas_final.onnx")
meas = hw.measure(m, arch["res"], iters=100)
json.dump({"model": "nas_best", "arch": arch, "params": n_params(m), "top1": h[-1]["acc"], "epochs": a.epochs, **meas}, open(f"{RES}/nas_final.json", "w"), indent=1)
print("FINAL nas", h[-1]["acc"], meas)
