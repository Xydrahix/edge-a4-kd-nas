"""Hardware-aware NAS search space: a 4-stage MBConv network on CIFAR-10 (macro-architecture fixed, micro choices searched).

Searched per trial (13 variables): input resolution, stem width, global width multiplier, and for each of the 4 stages the
number of blocks (1-3), kernel size (3/5) and expansion ratio (2/3/4/6) -- plus final head width and dropout.
"""
import torch.nn as nn
from models import cbn, MBConv, _Net

STAGE_STRIDES = [1, 2, 2, 2]
BASE_CH = [16, 32, 64, 96]


def round_ch(c, m=8):
    return max(m, int(c + m / 2) // m * m)


def suggest(trial):
    arch = {"res": trial.suggest_categorical("res", [24, 28, 32]),
            "stem": trial.suggest_categorical("stem", [16, 24, 32]),
            "width": trial.suggest_categorical("width", [0.5, 0.75, 1.0, 1.25]),
            "head": trial.suggest_categorical("head", [128, 256, 512]),
            "drop": trial.suggest_categorical("drop", [0.0, 0.1, 0.2])}
    for i in range(4):
        arch[f"d{i}"] = trial.suggest_int(f"d{i}", 1, 3)
        arch[f"k{i}"] = trial.suggest_categorical(f"k{i}", [3, 5])
        arch[f"e{i}"] = trial.suggest_categorical(f"e{i}", [2, 3, 4, 6])
    return arch


def build(arch, num_classes=10):
    stages, ch = [], round_ch(arch["stem"] * arch["width"])
    stem = cbn(3, ch, 3, 1)
    for i in range(4):
        out = round_ch(BASE_CH[i] * arch["width"]); blocks = []
        for b in range(arch[f"d{i}"]):
            blocks.append(MBConv(ch, out, arch[f"k{i}"], STAGE_STRIDES[i] if b == 0 else 1, arch[f"e{i}"])); ch = out
        stages.append(nn.Sequential(*blocks))
    return _Net(stem, stages, 2, cbn(ch, arch["head"], 1), num_classes, arch["head"], arch["drop"])
