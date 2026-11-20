#!/usr/bin/env bash
# Full pipeline. GPU recommended for training; data/*.npz (CIFAR-10/100 as uint8 arrays, not shipped in the zip to save space) are built from the official CIFAR archives.
set -euo pipefail
cd "$(dirname "$0")"
python train_teacher.py --epochs 30
for m in baseline kd kd_feat; do python train_students.py --mode $m --epochs 15; done
python nas_search.py --trials 32 --epochs 4 --subset 20000 --max_latency_ms 0.5 --max_size_mb 1.0 --max_mem_mb 18
python nas_final.py --epochs 30
python export_onnx.py
python benchmark.py
python make_plots.py
python make_report.py
