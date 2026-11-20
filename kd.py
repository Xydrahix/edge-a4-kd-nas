"""Knowledge-distillation losses: soft-target KL (Hinton), combined hard+soft loss, and feature (hint) distillation."""
import torch
import torch.nn as nn
import torch.nn.functional as F


def soft_kd_loss(s_logits, t_logits, T):
    """KL(softmax(t/T) || softmax(s/T)) scaled by T^2 so gradients keep the same magnitude for any temperature."""
    s = F.log_softmax(s_logits.float() / T, 1)
    t = F.softmax(t_logits.float() / T, 1)
    return F.kl_div(s, t, reduction="batchmean") * T * T


class FeatureHint(nn.Module):
    """1x1 conv + BN regressor mapping the student's 8x8 feature to the teacher's channel width; loss = MSE to the (BN-free,
    L2-normalised-per-channel) teacher feature. Regressor is only used for training and is NOT exported."""
    def __init__(self, s_ch, t_ch):
        super().__init__()
        self.proj = nn.Sequential(nn.Conv2d(s_ch, t_ch, 1, bias=False), nn.BatchNorm2d(t_ch))

    def forward(self, fs, ft):
        fs = self.proj(fs.float())
        return F.mse_loss(F.normalize(fs, dim=1), F.normalize(ft.float(), dim=1)) * fs.shape[1] / 10.0


def kd_loss_fn(teacher, T=4.0, alpha=0.7, hint=None, beta=1.0):
    """Returns loss_fn(model, x, y) for common.fit.
    L = alpha * T^2 * KL(soft) + (1-alpha) * CE(hard) [+ beta * feature MSE when `hint` is given]."""
    teacher.eval()

    def fn(model, x, y):
        with torch.no_grad():
            t_out, t_feat = teacher(x, return_feat=True)
        if hint is not None:
            s_out, s_feat = model(x, return_feat=True)
        else:
            s_out, s_feat = model(x), None
        loss = alpha * soft_kd_loss(s_out, t_out, T) + (1 - alpha) * F.cross_entropy(s_out.float(), y)
        if hint is not None:
            loss = loss + beta * hint(s_feat, t_feat)
        return loss
    return fn
