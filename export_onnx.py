"""Export every trained PyTorch checkpoint to ONNX (opset 17, dynamic batch) and verify ORT-CPU logits match PyTorch."""
import json, os, numpy as np, torch, onnxruntime as ort
from common import *
from models import Teacher, build_student, STUDENTS
from nas_space import build as nas_build
import hw

jobs = [("teacher", lambda: Teacher(100), 32, "teacher")]
for s in STUDENTS:
    for mode in ("baseline", "kd", "kd_feat"):
        jobs.append((f"{s}_{mode}", (lambda s=s: build_student(s, 100)), 32, f"{s}_{mode}"))
nas = json.load(open(f"{RES}/nas_best.json"))["best"]["arch"] if os.path.exists(f"{RES}/nas_best.json") else None
if nas:
    jobs.append(("nas_final", lambda: nas_build(nas), nas["res"], "nas_final"))
report = {}
for name, mk, res, ck in jobs:
    p = f"{CKPT}/{ck}.pt"
    if not os.path.exists(p):
        continue
    m = mk(); m.load_state_dict(torch.load(p, map_location="cpu")); m.eval()
    out = f"{ONNX}/{name}.onnx"; hw.export_bytes(m, res, path=out)
    x = torch.randn(4, 3, res, res)
    with torch.no_grad():
        ref = m(x).numpy()
    sess = ort.InferenceSession(out, providers=["CPUExecutionProvider"])
    got = sess.run(None, {"input": x.numpy()})[0]
    report[name] = {"onnx_mb": os.path.getsize(out) / 1e6, "max_abs_diff": float(np.abs(ref - got).max()), "argmax_match": bool((ref.argmax(1) == got.argmax(1)).all())}
    print(name, report[name], flush=True)
json.dump(report, open(f"{RES}/onnx_check.json", "w"), indent=1)
