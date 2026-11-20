"""Teacher (ResNet-50) and the three student architectures, all adapted to 32x32 inputs.

Every model's forward(x, return_feat=False) returns logits, or (logits, mid_feature) where mid_feature is the
8x8 intermediate map used by feature-based distillation (teacher: layer3, 1024ch; students: see FEAT_CH).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


# ------------------------------------------------------------------------------------------------ teacher
class Teacher(nn.Module):
    """torchvision ResNet-50 with a CIFAR stem (3x3 stride-1 conv, no max-pool), trained from scratch."""
    FEAT_CH = 1024

    def __init__(self, num_classes=100):
        super().__init__()
        r = torchvision.models.resnet50(weights=None, num_classes=num_classes)
        r.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        r.maxpool = nn.Identity()
        self.r = r

    def forward(self, x, return_feat=False):
        r = self.r
        x = r.relu(r.bn1(r.conv1(x)))
        x = r.layer2(r.layer1(x))
        f = r.layer3(x)                                  # 1024 x 8 x 8
        out = r.fc(torch.flatten(r.avgpool(r.layer4(f)), 1))
        return (out, f) if return_feat else out


# ------------------------------------------------------------------------------------------------ building blocks
def cbn(i, o, k=1, s=1, groups=1, act=True):
    layers = [nn.Conv2d(i, o, k, s, k // 2, groups=groups, bias=False), nn.BatchNorm2d(o)]
    if act:
        layers.append(nn.ReLU6(inplace=True))
    return nn.Sequential(*layers)


class MBConv(nn.Module):
    """Inverted residual: 1x1 expand -> kxk depthwise -> 1x1 linear project (+ skip). Used by MobileNetV2 and EfficientNet-Lite."""
    def __init__(self, i, o, k, s, expand):
        super().__init__()
        mid = i * expand
        self.use_res = s == 1 and i == o
        layers = [cbn(i, mid, 1)] if expand != 1 else []
        layers += [cbn(mid, mid, k, s, groups=mid), cbn(mid, o, 1, act=False)]
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        y = self.block(x)
        return x + y if self.use_res else y


class _Net(nn.Module):
    """stem -> stages(list) -> head. `mid_idx` = index of the stage whose output is the 8x8 feature-distillation tap."""
    def __init__(self, stem, stages, mid_idx, head, num_classes, last_ch, drop):
        super().__init__()
        self.stem, self.stages, self.mid_idx = stem, nn.ModuleList(stages), mid_idx
        self.head = head
        self.drop = nn.Dropout(drop)
        self.fc = nn.Linear(last_ch, num_classes)

    def forward(self, x, return_feat=False):
        x = self.stem(x); feat = None
        for i, st in enumerate(self.stages):
            x = st(x)
            if i == self.mid_idx:
                feat = x
        x = self.head(x)
        out = self.fc(self.drop(torch.flatten(F.adaptive_avg_pool2d(x, 1), 1)))
        return (out, feat) if return_feat else out


def _stack(cfg, in_ch):
    stages = []
    for t, c, n, s, k in cfg:
        blocks = []
        for b in range(n):
            blocks.append(MBConv(in_ch, c, k, s if b == 0 else 1, t)); in_ch = c
        stages.append(nn.Sequential(*blocks))
    return stages, in_ch


def mobilenet_v2(num_classes=100):
    # (expand, channels, repeats, stride, kernel); stem stride 1 for 32x32 inputs, total stride 16 (feature tap = 8x8 after stage 3)
    cfg = [(1, 16, 1, 1, 3), (6, 24, 2, 2, 3), (6, 32, 3, 2, 3), (6, 64, 4, 2, 3), (6, 96, 3, 1, 3), (6, 160, 3, 2, 3), (6, 320, 1, 1, 3)]
    stages, ch = _stack(cfg, 32)
    m = _Net(cbn(3, 32, 3, 1), stages, 2, cbn(ch, 1280, 1), num_classes, 1280, 0.2)
    m.feat_ch = 32
    return m


def efficientnet_lite0(num_classes=100):
    # EfficientNet-B0 layout without squeeze-excite, ReLU6 and fixed stem/head (the "Lite" recipe), CIFAR strides
    cfg = [(1, 16, 1, 1, 3), (6, 24, 2, 2, 3), (6, 40, 2, 2, 5), (6, 80, 3, 2, 3), (6, 112, 3, 1, 5), (6, 192, 4, 2, 5), (6, 320, 1, 1, 3)]
    stages, ch = _stack(cfg, 32)
    m = _Net(cbn(3, 32, 3, 1), stages, 2, cbn(ch, 1280, 1), num_classes, 1280, 0.2)
    m.feat_ch = 40
    return m


class Fire(nn.Module):
    def __init__(self, i, s, e1, e3):
        super().__init__()
        self.squeeze = nn.Sequential(nn.Conv2d(i, s, 1, bias=False), nn.BatchNorm2d(s), nn.ReLU(inplace=True))
        self.e1 = nn.Sequential(nn.Conv2d(s, e1, 1, bias=False), nn.BatchNorm2d(e1), nn.ReLU(inplace=True))
        self.e3 = nn.Sequential(nn.Conv2d(s, e3, 3, 1, 1, bias=False), nn.BatchNorm2d(e3), nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.squeeze(x)
        return torch.cat([self.e1(x), self.e3(x)], 1)


class SqueezeNet(nn.Module):
    """SqueezeNet v1.1 layout (Fire modules) + BatchNorm so it trains from scratch with a normal LR; CIFAR stem."""
    feat_ch = 256

    def __init__(self, num_classes=100):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, 64, 3, 1, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(3, 2, 1))   # 16x16
        self.s1 = nn.Sequential(Fire(64, 16, 64, 64), Fire(128, 16, 64, 64), nn.MaxPool2d(3, 2, 1))                                        # 8x8
        self.s2 = nn.Sequential(Fire(128, 32, 128, 128), Fire(256, 32, 128, 128))                                                          # 8x8 <- feature tap
        self.s3 = nn.Sequential(nn.MaxPool2d(3, 2, 1), Fire(256, 48, 192, 192), Fire(384, 48, 192, 192), Fire(384, 64, 256, 256), Fire(512, 64, 256, 256))  # 4x4
        self.cls = nn.Sequential(nn.Dropout(0.3), nn.Conv2d(512, num_classes, 1))

    def forward(self, x, return_feat=False):
        x = self.s1(self.stem(x)); f = self.s2(x)
        out = self.cls(self.s3(f)).mean((2, 3))
        return (out, f) if return_feat else out


STUDENTS = {"mobilenetv2": mobilenet_v2, "squeezenet": SqueezeNet, "efficientnet_lite0": efficientnet_lite0}


def build_student(name, num_classes=100):
    m = STUDENTS[name](num_classes)
    if not hasattr(m, "feat_ch"):
        m.feat_ch = 256
    return m


if __name__ == "__main__":
    from common import n_params
    x = torch.randn(2, 3, 32, 32)
    t = Teacher(); o, f = t(x, True); print("teacher", n_params(t) / 1e6, "M", o.shape, f.shape)
    for n in STUDENTS:
        m = build_student(n); o, f = m(x, True); print(n, round(n_params(m) / 1e6, 2), "M", o.shape, f.shape, m.feat_ch)
