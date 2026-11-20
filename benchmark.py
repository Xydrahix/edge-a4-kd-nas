"""CPU benchmark of every exported ONNX model with ONNX Runtime: accuracy (full test set), params, MACs, size, latency
(batch 1; 1 and 4 threads), throughput (batch 64), peak memory. Writes results/benchmark.csv."""
import json, os, subprocess, sys, time, numpy as np, pandas as pd, torch, onnxruntime as ort
from common import *
from models import Teacher, build_student, STUDENTS
from nas_space import build as nas_build
import hw

CORES = os.environ.get("BENCH_CPUS")  # optional taskset core list for quieter timings


def macs(model, res):
    tot = [0]
    def hk(m, i, o):
        if isinstance(m, torch.nn.Conv2d): tot[0] += o.numel() * (m.in_channels // m.groups) * m.kernel_size[0] * m.kernel_size[1]
        elif isinstance(m, torch.nn.Linear): tot[0] += m.in_features * m.out_features
    hs = [m.register_forward_hook(hk) for m in model.modules() if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear))]
    with torch.no_grad(): model.eval()(torch.randn(1, 3, res, res))
    [h.remove() for h in hs]; return tot[0]


def run(sess, x, iters, warm=10):
    nm = sess.get_inputs()[0].name
    for _ in range(warm): sess.run(None, {nm: x})
    ts = []
    for _ in range(iters):
        t = time.perf_counter(); sess.run(None, {nm: x}); ts.append((time.perf_counter() - t) * 1e3)
    return np.array(ts)


def session(path, threads):
    so = ort.SessionOptions(); so.intra_op_num_threads = threads; so.inter_op_num_threads = 1
    return ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])


def test_acc(path, ds, res):
    d = np.load(f"{DATA}/{ds}_u8.npz"); mean, std = (np.array(v, dtype=np.float32).reshape(1, 3, 1, 1) for v in STATS[ds])
    s = session(path, 4); ok = 0
    for k in range(0, 10000, 500):
        x = d["xte"][k:k + 500].astype(np.float32).transpose(0, 3, 1, 2) / 255.0
        x = (x - mean) / std
        if res != 32: x = torch.nn.functional.interpolate(torch.from_numpy(x), size=(res, res), mode="bilinear", align_corners=False, antialias=res < 32).numpy()
        ok += (s.run(None, {"input": np.ascontiguousarray(x)})[0].argmax(1) == d["yte"][k:k + 500]).sum()
    return ok / 100.0


rows = []
specs = [("teacher", "resnet50 (teacher)", "-", "cifar100", Teacher(100), 32)]
for s in STUDENTS:
    for mode in ("baseline", "kd", "kd_feat"):
        specs.append((f"{s}_{mode}", s, mode, "cifar100", build_student(s, 100), 32))
nb = json.load(open(f"{RES}/nas_best.json"))["best"]["arch"] if os.path.exists(f"{RES}/nas_best.json") else None
if nb: specs.append(("nas_final", "nas_best", "-", "cifar10", nas_build(nb), nb["res"]))
for name, arch, mode, ds, m, res in specs:
    p = f"{ONNX}/{name}.onnx"
    if not os.path.exists(p): continue
    x1 = np.random.rand(1, 3, res, res).astype(np.float32); x64 = np.random.rand(64, 3, res, res).astype(np.float32)
    l1 = run(session(p, 1), x1, 100); l4 = run(session(p, 4), x1, 100); lb = run(session(p, 4), x64, 10, 3)
    peak = subprocess.check_output([sys.executable, "-I", "-c", hw._PEAK_SNIPPET, p, str(res)]).decode().strip()
    r = {"model": name, "architecture": arch, "mode": mode, "dataset": ds, "input_res": res, "params_M": n_params(m) / 1e6, "MMACs": macs(m, res) / 1e6,
         "onnx_size_MB": os.path.getsize(p) / 1e6, "top1_%": test_acc(p, ds, res), "lat_1thr_ms": float(np.median(l1)), "lat_1thr_p90_ms": float(np.percentile(l1, 90)),
         "lat_4thr_ms": float(np.median(l4)), "throughput_img_s": 64 / (np.median(lb) / 1e3), "peak_mem_MB": float(peak), "extra_mem_MB": float(peak) - hw.baseline_mb()}
    rows.append(r); print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
pd.DataFrame(rows).to_csv(f"{RES}/benchmark.csv", index=False)
print("DONE benchmark", len(rows))
