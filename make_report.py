"""Builds report.md / report.pdf (headless Chrome) from results/*.json|csv. All numbers come from the files."""
import json, base64, subprocess, pandas as pd
from pathlib import Path
R = Path("results"); B = pd.read_csv(R/"benchmark.csv").set_index("model")
T = json.load(open(R/"teacher.json")); NB = json.load(open(R/"nas_best.json")); NF = json.load(open(R/"nas_final.json"))
trials = json.load(open(R/"nas_trials.json")); chk = json.load(open(R/"onnx_check.json")) if (R/"onnx_check.json").exists() else {}
S = ["mobilenetv2", "squeezenet", "efficientnet_lite0"]; M = ["baseline", "kd", "kd_feat"]
a = lambda s, m: B.loc[f"{s}_{m}", "top1_%"]
rows = []
for s in S:
    b, k, f = a(s, "baseline"), a(s, "kd"), a(s, "kd_feat")
    rows.append(f"| {s} | {B.loc[s+'_baseline','params_M']:.2f} | {b:.2f} | {k:.2f} ({k-b:+.2f}) | {f:.2f} ({f-b:+.2f}) |")
t1 = "| Student | Params (M) | Baseline | KD (soft+hard) | KD + feature hint |\n|---|---|---|---|---|\n" + "\n".join(rows)
cols = ["architecture", "mode", "input_res", "params_M", "MMACs", "onnx_size_MB", "top1_%", "lat_1thr_ms", "lat_4thr_ms", "throughput_img_s", "peak_mem_MB"]
t2 = B[cols].round(2).reset_index().to_markdown(index=False)
rej = [t for t in trials if t.get("status") != "ok"]; ok = [t for t in trials if t.get("status") == "ok"]
nb = NB["best"]; c = NB["constraints"]
figs = {"kd_training_curves": "Fig 1: training curves of teacher and students", "kd_accuracy": "Fig 2: student accuracy, baseline vs KD vs KD+feature",
        "accuracy_vs_cost": "Fig 3: accuracy vs ORT-CPU latency / size", "nas_search": "Fig 4: NAS search history", "nas_final_curve": "Fig 5: retraining of the best NAS model"}
md = f"""# Edge Computing Assignment 4 - Technical Report
Roll no: i230139 | Section: AI-B | Q1 knowledge distillation (CIFAR-100) | Q2 hardware-aware NAS (CIFAR-10)

## 1. Setup
- Hardware: HP Victus 15 (RTX 4050 6 GB for training, 11-core CPU). All latency / memory numbers are **ONNX Runtime CPU** (CPUExecutionProvider), batch 1, 1 thread unless stated.
- Data: CIFAR-100 (Q1) and CIFAR-10 (Q2), 32x32, standard train/test split, GPU-side flip + 4px random-crop augmentation, bf16 autocast.
- Because of the single-day deadline and one 6 GB GPU, training budgets are short (teacher {T.get('epochs','30')} epochs, students 15 epochs, NAS final 30 epochs). Absolute accuracies are therefore lower than long-schedule literature values; comparisons between methods use identical budgets and seeds.

## 2. Q1 - Knowledge distillation
**Teacher.** ResNet-50 with a CIFAR stem (3x3 conv, no max-pool), {B.loc['teacher','params_M']:.1f} M params, trained from scratch: **{B.loc['teacher','top1_%']:.2f}%** top-1 (ONNX/ORT).

**Students.** MobileNetV2, SqueezeNet (Fire modules + BN) and EfficientNet-Lite0 (MBConv, ReLU6, no SE), all re-implemented from scratch with strides adapted for 32x32 input.

**Distillation loss.** L = alpha * T^2 * KL(softmax(z_t/T) || softmax(z_s/T)) + (1-alpha) * CE(z_s, y), T = 4, alpha = 0.7. The T^2 factor keeps gradient magnitudes comparable across temperatures. **Feature distillation** adds beta * MSE between the L2-normalised channels of the student's 8x8 intermediate feature map (after a 1x1 conv+BN hint regressor) and the teacher's layer3 output (1024x8x8), beta = 1.

{t1}

Notes: single seed per cell (seed 0), so differences below ~0.5 pt are within noise on a 10k test set. Where KD is not better than baseline, that is what was measured and is reported as such.

**Unified inference.** `inference.py` loads any exported `.onnx` through ONNX Runtime CPU and runs the same preprocessing for every model (`--model name --dataset --image/--n`).

**Benchmark (all models, ORT CPU).**

{t2}

## 3. Q2 - Hardware-aware NAS (CIFAR-10)
- Search space (`nas_space.py`): input resolution {{24,28,32}}, stem width, global width multiplier {{0.5..1.25}}, head width, dropout and, for 4 stages, depth 1-3, kernel {{3,5}}, expansion {{2,3,4,6}} (MBConv blocks).
- Search: Optuna TPE, {NB['n_trials']} trials, {NB['search_time_s']:.0f} s. Each candidate is exported to ONNX and **measured on CPU first** (latency, model size, peak process memory above the runtime baseline). Constraints: latency <= {c['max_latency_ms']} ms, size <= {c['max_size_mb']} MB, extra memory <= {c['max_extra_mem_mb']:.0f} MB. A violating candidate is **rejected before training** (`TrialPruned`): **{NB['n_rejected']} of {NB['n_trials']}** trials were rejected, {len(ok)} were trained (4 epochs on a 20k subset, scored on a held-out 5k validation split; score = val acc - 2*max(0, latency/max_latency - 0.5)).
- Best trial #{nb['trial']}: val acc {nb['val_acc']:.2f}% (short proxy training), {nb['params']/1e3:.0f}k params, {nb['latency_ms']:.2f} ms, {nb['size_mb']:.2f} MB, extra mem {nb['extra_mem_mb']:.1f} MB.
- Architecture: `{json.dumps(nb['arch'])}`
- Retrained {NF['epochs']} epochs on the full training set: **{NF['top1']:.2f}%** CIFAR-10 test top-1 (torch), {B.loc['nas_final','top1_%']:.2f}% via ORT; ONNX size {B.loc['nas_final','onnx_size_MB']:.2f} MB, latency {B.loc['nas_final','lat_1thr_ms']:.2f} ms, peak memory {B.loc['nas_final','peak_mem_MB']:.0f} MB.

## 4. Honest limitations
- Short schedules and one seed per configuration; no confidence intervals.
- Latency numbers are on a desktop-class laptop CPU, not an actual edge device; ratios between models matter more than absolutes. ORT 1-thread latency for these tiny nets is close to the framework's per-call overhead.
- NAS proxy training (4 epochs) ranks candidates imperfectly; the search is small (32 trials).

## 5. Figures
""" + "\n".join(f"![{v}](results/figures/{k}.png)\n*{v}*\n" for k, v in figs.items()) + """
## 6. Reproduction
See `README.md` (`bash run_all.sh`). Code was written with AI assistance (Claude Code); all numbers above come from running the included scripts.
"""
Path("report.md").write_text(md)
import markdown
html = markdown.markdown(md, extensions=["tables"])
for k in figs:
    d = base64.b64encode(open(R/"figures"/f"{k}.png", "rb").read()).decode()
    html = html.replace(f"results/figures/{k}.png", "data:image/png;base64," + d)
css = "body{font-family:sans-serif;font-size:10.5px;margin:18px;line-height:1.35}table{border-collapse:collapse;font-size:9px}td,th{border:1px solid #888;padding:2px 4px}img{max-width:85%}code{font-size:9px;word-break:break-all}"
Path("report.html").write_text(f"<html><meta charset=utf-8><style>{css}</style><body>{html}</body></html>")
subprocess.run(["google-chrome-stable", "--headless=new", "--no-sandbox", "--user-data-dir=/tmp/chr-a4", "--disable-gpu", "--print-to-pdf=report.pdf", "--no-pdf-header-footer", "file://" + str(Path("report.html").resolve())], check=True, capture_output=True)
print("report.pdf written")
