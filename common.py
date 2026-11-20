"""Shared helpers: data (GPU-resident, GPU augmentation), training loop, evaluation, CPU latency."""
import json, os, random, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.abspath(__file__))
CKPT, RES, DATA, ONNX = (os.path.join(ROOT, d) for d in ("checkpoints", "results", "data", "onnx"))
for _d in (CKPT, RES, DATA, ONNX, os.path.join(RES, "logs")):
    os.makedirs(_d, exist_ok=True)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
STATS = {"cifar10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
         "cifar100": ((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))}
NCLS = {"cifar10": 10, "cifar100": 100}


def seed_all(s=0):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def load_np(name):
    """uint8 NHWC arrays (xtr, ytr, xte, yte); downloads through torchvision if the cache is missing."""
    p = os.path.join(DATA, f"{name}_u8.npz")
    if not os.path.exists(p):
        import torchvision
        C = torchvision.datasets.CIFAR100 if name == "cifar100" else torchvision.datasets.CIFAR10
        tr, te = C(DATA, train=True, download=True), C(DATA, train=False, download=True)
        np.savez(p, xtr=tr.data, ytr=np.array(tr.targets), xte=te.data, yte=np.array(te.targets))
    d = np.load(p)
    return d["xtr"], d["ytr"], d["xte"], d["yte"]


class Data:
    """Whole dataset lives on the GPU as uint8; augmentation (reflect-pad crop + flip) is done per batch on the GPU."""
    def __init__(self, name, device=DEV, val_split=0):
        xtr, ytr, xte, yte = load_np(name)
        self.name, self.n_classes, self.device = name, NCLS[name], device
        m, s = STATS[name]
        self.mean = torch.tensor(m, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(s, device=device).view(1, 3, 1, 1)
        t = lambda a: torch.from_numpy(a).to(device)
        self.xtr, self.ytr, self.xte, self.yte = t(xtr), t(ytr), t(xte), t(yte)
        if val_split:  # fixed held-out validation split taken from the training set (used by the NAS)
            g = torch.Generator().manual_seed(123)
            perm = torch.randperm(len(self.xtr), generator=g).to(device)
            v, tr = perm[:val_split], perm[val_split:]
            self.xva, self.yva = self.xtr[v], self.ytr[v]
            self.xtr, self.ytr = self.xtr[tr], self.ytr[tr]

    def prep(self, xb, res=None):
        x = xb.permute(0, 3, 1, 2).float().div_(255)
        x = (x - self.mean) / self.std
        if res and res != x.shape[-1]:
            x = F.interpolate(x, size=(res, res), mode="bilinear", align_corners=False, antialias=res < x.shape[-1])
        return x

    @staticmethod
    def augment(x):
        n, c, h, w = x.shape
        xp = F.pad(x, (4, 4, 4, 4), mode="reflect")
        i = torch.randint(0, 9, (n,), device=x.device); j = torch.randint(0, 9, (n,), device=x.device)
        ar = torch.arange(h, device=x.device)
        rows, cols = (i[:, None] + ar)[:, None, :, None], (j[:, None] + ar)[:, None, None, :]
        x = xp[torch.arange(n, device=x.device)[:, None, None, None], torch.arange(c, device=x.device)[None, :, None, None], rows, cols]
        flip = torch.rand(n, device=x.device) < 0.5
        return torch.where(flip[:, None, None, None], x.flip(3), x)

    def train_batches(self, bs, res=None, limit=None):
        n = len(self.xtr) if limit is None else min(limit, len(self.xtr))
        perm = torch.randperm(len(self.xtr), device=self.device)[:n]
        for k in range(0, n - bs + 1, bs):
            idx = perm[k:k + bs]
            yield self.augment(self.prep(self.xtr[idx], res)), self.ytr[idx]

    def eval_batches(self, split="test", bs=500, res=None):
        X, Y = (self.xte, self.yte) if split == "test" else (self.xva, self.yva)
        for k in range(0, len(X), bs):
            yield self.prep(X[k:k + bs], res), Y[k:k + bs]


@torch.no_grad()
def evaluate(model, data, split="test", res=None, bs=500):
    model.eval(); ok = n = 0
    for x, y in data.eval_batches(split, bs, res):
        with torch.autocast(DEV, dtype=torch.bfloat16, enabled=DEV == "cuda"):
            out = model(x.contiguous(memory_format=torch.channels_last))
        ok += (out.argmax(1) == y).sum().item(); n += len(y)
    return 100.0 * ok / n


def make_opt(model, lr, wd, epochs, steps_per_epoch, extra_params=()):
    params = list(model.parameters()) + list(extra_params)
    opt = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=wd, nesterov=True)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, epochs=epochs, steps_per_epoch=steps_per_epoch, pct_start=0.2)
    return opt, sched


def fit(model, data, epochs, bs=128, lr=0.1, wd=5e-4, loss_fn=None, extra_params=(), res=None, tag="", log=None,
        eval_split="test", limit=None, verbose=True):
    """Generic training loop. loss_fn(model, x, y) -> scalar loss (default: cross-entropy). Returns per-epoch history."""
    model.to(DEV).to(memory_format=torch.channels_last)
    spe = (len(data.xtr) if limit is None else min(limit, len(data.xtr))) // bs
    opt, sched = make_opt(model, lr, wd, epochs, spe, extra_params)
    loss_fn = loss_fn or (lambda m, x, y: F.cross_entropy(m(x), y))
    hist, t0 = [], time.time()
    for ep in range(1, epochs + 1):
        model.train(); tl = 0.0; nb = 0
        for x, y in data.train_batches(bs, res, limit):
            x = x.contiguous(memory_format=torch.channels_last)
            with torch.autocast(DEV, dtype=torch.bfloat16, enabled=DEV == "cuda"):
                loss = loss_fn(model, x, y)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
            tl += loss.item(); nb += 1
        acc = evaluate(model, data, eval_split, res)
        hist.append({"epoch": ep, "loss": tl / nb, "acc": acc, "time_s": time.time() - t0})
        if verbose:
            print(f"[{tag}] ep {ep}/{epochs} loss {tl / nb:.4f} {eval_split}-acc {acc:.2f} ({time.time() - t0:.0f}s)", flush=True)
        if log:
            json.dump(hist, open(log, "w"))
    return hist


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def cpu_latency_ms(model_fn, x, warmup=10, iters=50):
    """Median wall-clock latency (ms) of model_fn(x); x is a CPU tensor / numpy array. Single-threaded caller sets threads."""
    for _ in range(warmup):
        model_fn(x)
    ts = []
    for _ in range(iters):
        t = time.perf_counter(); model_fn(x); ts.append((time.perf_counter() - t) * 1e3)
    return float(np.median(ts)), float(np.percentile(ts, 90))
