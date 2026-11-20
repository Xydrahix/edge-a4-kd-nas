"""Figures for the report, built from results/*.json|csv (nothing hand-made)."""
import json, os, glob, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
R = "results"; F = f"{R}/figures"; os.makedirs(F, exist_ok=True)
S = ["mobilenetv2", "squeezenet", "efficientnet_lite0"]; MODES = [("baseline", "baseline (CE only)"), ("kd", "KD (soft+hard)"), ("kd_feat", "KD + feature hint")]
col = {"baseline": "#888888", "kd": "#1f77b4", "kd_feat": "#d62728"}
J = lambda p: json.load(open(p))

# 1 training curves (CIFAR-100)
fig, ax = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
for a, s in zip(ax, S):
    for m, lab in MODES:
        p = f"{R}/logs/{s}_{m}.json"
        if os.path.exists(p): h = J(p); a.plot([x["epoch"] for x in h], [x["acc"] for x in h], label=lab, color=col[m])
    if os.path.exists(f"{R}/logs/teacher.json"): h = J(f"{R}/logs/teacher.json"); a.plot([x["epoch"] for x in h], [x["acc"] for x in h], "k--", lw=1, label="teacher ResNet-50")
    a.set_title(s); a.set_xlabel("epoch"); a.grid(alpha=.3)
ax[0].set_ylabel("CIFAR-100 test top-1 (%)"); ax[0].legend(fontsize=8); plt.tight_layout(); plt.savefig(f"{F}/kd_training_curves.png", dpi=130); plt.close()

# 2 KD vs baseline bars
b = pd.read_csv(f"{R}/benchmark.csv")
fig, ax = plt.subplots(figsize=(8, 4.2)); w = .26
for k, (m, lab) in enumerate(MODES):
    v = [b[b.model == f"{s}_{m}"]["top1_%"].values[0] if (b.model == f"{s}_{m}").any() else 0 for s in S]
    bars = ax.bar(np.arange(3) + (k - 1) * w, v, w, label=lab, color=col[m])
    for r_, y in zip(bars, v): ax.text(r_.get_x() + r_.get_width() / 2, y + .3, f"{y:.1f}", ha="center", fontsize=8)
t = b[b.model == "teacher"]["top1_%"]
if len(t): ax.axhline(t.values[0], color="k", ls="--", lw=1, label=f"teacher {t.values[0]:.1f}")
ax.set_xticks(range(3)); ax.set_xticklabels(S); ax.set_ylabel("CIFAR-100 top-1 (%)"); ax.set_ylim(max(0, b["top1_%"].min() - 8), None); ax.legend(fontsize=8); ax.grid(axis="y", alpha=.3)
plt.tight_layout(); plt.savefig(f"{F}/kd_accuracy.png", dpi=130); plt.close()

# 3 accuracy vs CPU latency / size
c = b[b.dataset == "cifar100"]
fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
for a, xk, xl in ((ax[0], "lat_1thr_ms", "CPU latency, batch 1, 1 thread (ms)"), (ax[1], "onnx_size_MB", "ONNX size (MB)")):
    for _, r in c.iterrows():
        m = r["mode"]; a.scatter(r[xk], r["top1_%"], color=col.get(m, "k"), marker="*" if r.model == "teacher" else "o", s=120 if r.model == "teacher" else 50)
        a.annotate(r.model.replace("efficientnet_lite0", "effnet-l0").replace("mobilenetv2", "mnv2").replace("squeezenet", "sqz"), (r[xk], r["top1_%"]), fontsize=6, xytext=(3, 3), textcoords="offset points")
    a.set_xlabel(xl); a.set_ylabel("top-1 (%)"); a.set_xscale("log"); a.grid(alpha=.3)
plt.tight_layout(); plt.savefig(f"{F}/accuracy_vs_cost.png", dpi=130); plt.close()

# 4 NAS
if os.path.exists(f"{R}/nas_trials.json"):
    T = J(f"{R}/nas_trials.json"); ok = [t for t in T if t["status"] == "ok"]; rej = [t for t in T if t["status"] != "ok"]
    best = J(f"{R}/nas_best.json"); lim = best["constraints"]
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
    ax[0].scatter([t["latency_ms"] for t in ok], [t["val_acc"] for t in ok], c=[t["size_mb"] for t in ok], cmap="viridis", label="trained")
    ax[0].scatter([t["latency_ms"] for t in rej], [min(t2["val_acc"] for t2 in ok) - 2] * len(rej), marker="x", color="r", label=f"rejected ({len(rej)})")
    ax[0].axvline(lim["max_latency_ms"], color="r", ls="--", lw=1); ax[0].set_xlabel("ORT-CPU latency (ms)"); ax[0].set_ylabel("val acc after short training (%)"); ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
    ax[1].scatter([t["size_mb"] for t in ok], [t["val_acc"] for t in ok], color="C0"); ax[1].scatter([t["size_mb"] for t in rej], [min(t2["val_acc"] for t2 in ok) - 2] * len(rej), marker="x", color="r")
    ax[1].axvline(lim["max_size_mb"], color="r", ls="--", lw=1); ax[1].set_xlabel("ONNX size (MB)"); ax[1].grid(alpha=.3)
    sc = [t.get("score", np.nan) for t in T]; bs = np.fmax.accumulate(np.nan_to_num(sc, nan=-1))
    ax[2].plot([t["trial"] for t in T], sc, "o", ms=4, label="trial score"); ax[2].plot([t["trial"] for t in T], bs, "r-", label="best so far"); ax[2].set_xlabel("trial"); ax[2].set_ylabel("score"); ax[2].legend(); ax[2].grid(alpha=.3)
    plt.tight_layout(); plt.savefig(f"{F}/nas_search.png", dpi=130); plt.close()
if os.path.exists(f"{R}/logs/nas_final.json"):
    h = J(f"{R}/logs/nas_final.json"); plt.figure(figsize=(5.5, 3.8)); plt.plot([x["epoch"] for x in h], [x["acc"] for x in h]); plt.xlabel("epoch"); plt.ylabel("CIFAR-10 test top-1 (%)"); plt.grid(alpha=.3); plt.title("Best NAS architecture, full training"); plt.tight_layout(); plt.savefig(f"{F}/nas_final_curve.png", dpi=130); plt.close()
print("figures:", sorted(os.listdir(F)))
