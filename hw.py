"""Hardware measurement on CPU with ONNX Runtime: latency (batch 1, 1 thread), model size, and peak process memory."""
import io, json, os, subprocess, sys, tempfile
import numpy as np
import torch

_PEAK_SNIPPET = r"""
import sys, json, resource, numpy as np
import onnxruntime as ort
so = ort.SessionOptions(); so.intra_op_num_threads = 1; so.inter_op_num_threads = 1
s = ort.InferenceSession(sys.argv[1], so, providers=["CPUExecutionProvider"])
x = np.random.rand(1, 3, int(sys.argv[2]), int(sys.argv[2])).astype(np.float32)
for _ in range(5): s.run(None, {s.get_inputs()[0].name: x})
print([int(l.split()[1]) for l in open("/proc/self/status") if l.startswith("VmHWM")][0] / 1024.0)
"""
_BASE_SNIPPET = r"""
import resource, numpy as np, onnxruntime as ort
print([int(l.split()[1]) for l in open("/proc/self/status") if l.startswith("VmHWM")][0] / 1024.0)
"""
_baseline = None


def export_bytes(model, res, path=None, opset=17):
    model = model.eval().cpu().float()
    x = torch.randn(1, 3, res, res)
    buf = io.BytesIO() if path is None else path
    with torch.no_grad():
        torch.onnx.export(model, x, buf, input_names=["input"], output_names=["logits"], opset_version=opset,
                          dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}}, dynamo=False)
    return buf.getvalue() if path is None else None


def baseline_mb():
    global _baseline
    if _baseline is None:
        _baseline = float(subprocess.check_output([sys.executable, "-I", "-c", _BASE_SNIPPET]).decode().strip())
    return _baseline


def measure(model, res, iters=60, warmup=10, with_memory=True):
    """Return dict(latency_ms, latency_p90_ms, size_mb, peak_mem_mb, extra_mem_mb) for a CPU ORT run (batch 1, 1 thread)."""
    import onnxruntime as ort, time
    raw = export_bytes(model, res)
    so = ort.SessionOptions(); so.intra_op_num_threads = 1; so.inter_op_num_threads = 1
    sess = ort.InferenceSession(raw, so, providers=["CPUExecutionProvider"])
    x = np.random.rand(1, 3, res, res).astype(np.float32); nm = sess.get_inputs()[0].name
    for _ in range(warmup):
        sess.run(None, {nm: x})
    ts = []
    for _ in range(iters):
        t = time.perf_counter(); sess.run(None, {nm: x}); ts.append((time.perf_counter() - t) * 1e3)
    out = {"latency_ms": float(np.median(ts)), "latency_p90_ms": float(np.percentile(ts, 90)), "size_mb": len(raw) / 1e6}
    if with_memory:
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            f.write(raw); p = f.name
        try:
            peak = float(subprocess.check_output([sys.executable, "-I", "-c", _PEAK_SNIPPET, p, str(res)]).decode().strip())
        finally:
            os.unlink(p)
        out["peak_mem_mb"] = peak; out["extra_mem_mb"] = peak - baseline_mb()
    return out
