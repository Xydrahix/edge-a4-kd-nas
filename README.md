# Edge Computing - Assignment 4 (i230139, AI-B)
Q1: knowledge distillation on CIFAR-100 (ResNet-50 teacher -> MobileNetV2 / SqueezeNet / EfficientNet-Lite0). Q2: hardware-aware NAS on CIFAR-10 with measured ONNX-Runtime-CPU latency, size and peak memory.

## Layout
- `common.py` data (GPU-resident), augmentation, training loop; `models.py` teacher + students; `kd.py` soft-target and feature-hint losses
- `train_teacher.py`, `train_students.py --mode baseline|kd|kd_feat` (flags: `--T --alpha --beta --epochs`)
- `hw.py` ONNX export + CPU latency/size/peak-memory measurement; `nas_space.py`, `nas_search.py` (Optuna TPE, constraint rejection before training), `nas_final.py` (retrain best)
- `export_onnx.py`, `inference.py` (unified ORT-CPU inference: `python inference.py --model mobilenetv2_kd --dataset cifar100 --n 1000`), `benchmark.py`, `make_plots.py`, `make_report.py`
- `results/` JSON/CSV logs, `results/figures/`, `onnx/` exported models, `checkpoints/` weights, `report.pdf`

## Run
`pip install -r requirements.txt && bash run_all.sh` (about 4 h on an RTX 4050). Seeds are fixed (seed 0).
## Notes
Short training budgets (teacher 30 epochs, students 15, NAS final 30) because of a one-day deadline; see the report's limitations section.
