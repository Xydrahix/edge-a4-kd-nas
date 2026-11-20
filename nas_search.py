"""Hardware-aware NAS with Optuna (TPE). Each trial: sample arch -> build -> measure CPU latency / size / peak memory with
ONNX Runtime -> REJECT immediately (TrialPruned, no training) if any hardware constraint is violated -> otherwise train briefly
and report validation accuracy. Objective = validation accuracy with a soft latency penalty."""
import argparse, json, time, optuna, torch
from common import *
import hw
from nas_space import suggest, build

ap = argparse.ArgumentParser()
ap.add_argument("--trials", type=int, default=40); ap.add_argument("--epochs", type=int, default=4)
ap.add_argument("--subset", type=int, default=20000); ap.add_argument("--max_latency_ms", type=float, default=1.2)
ap.add_argument("--max_size_mb", type=float, default=2.0); ap.add_argument("--max_mem_mb", type=float, default=25.0)
a = ap.parse_args()
optuna.logging.set_verbosity(optuna.logging.WARNING)
data = Data("cifar10", val_split=5000)
rows = []


def objective(trial):
    seed_all(trial.number)
    arch = suggest(trial)
    model = build(arch)
    m = hw.measure(model, arch["res"], iters=40)
    rec = {"trial": trial.number, "arch": arch, "params": n_params(model), **m}
    viol = [k for k, lim in (("latency_ms", a.max_latency_ms), ("size_mb", a.max_size_mb), ("extra_mem_mb", a.max_mem_mb)) if m[k] > lim]
    if viol:
        rec.update(status="rejected", violated=viol); rows.append(rec); json.dump(rows, open(f"{RES}/nas_trials.json", "w"), indent=1)
        print(f"trial {trial.number}: REJECT {viol} lat {m['latency_ms']:.2f}ms size {m['size_mb']:.2f}MB mem {m['extra_mem_mb']:.0f}MB", flush=True)
        raise optuna.TrialPruned()
    h = fit(model, data, a.epochs, 128, 0.1, 5e-4, res=arch["res"], tag=f"nas{trial.number}", eval_split="val", limit=a.subset, verbose=False)
    acc = h[-1]["acc"]
    score = acc - 2.0 * max(0.0, m["latency_ms"] / a.max_latency_ms - 0.5)   # mild penalty when using >50% of latency budget
    rec.update(status="ok", val_acc=acc, score=score); rows.append(rec); json.dump(rows, open(f"{RES}/nas_trials.json", "w"), indent=1)
    print(f"trial {trial.number}: val {acc:.2f} score {score:.2f} lat {m['latency_ms']:.2f}ms size {m['size_mb']:.2f}MB mem {m['extra_mem_mb']:.0f}MB params {rec['params']/1e6:.2f}M", flush=True)
    return score


study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=0, n_startup_trials=10))
t0 = time.time(); study.optimize(objective, n_trials=a.trials)
ok = [r for r in rows if r["status"] == "ok"]
best = max(ok, key=lambda r: r["score"])
json.dump({"best": best, "constraints": {"max_latency_ms": a.max_latency_ms, "max_size_mb": a.max_size_mb, "max_extra_mem_mb": a.max_mem_mb},
           "n_trials": len(rows), "n_rejected": len(rows) - len(ok), "search_time_s": time.time() - t0}, open(f"{RES}/nas_best.json", "w"), indent=1)
print("BEST", json.dumps(best))
