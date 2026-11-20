"""Unified inference script (ONNX Runtime, CPU): same CLI for the teacher, every student and the NAS model.
  python inference.py --model mobilenetv2_kd --n 8                 # CIFAR-100 test images
  python inference.py --model nas_final --dataset cifar10 --image my.png
"""
import argparse, json, os, numpy as np, onnxruntime as ort
from PIL import Image
ROOT = os.path.dirname(os.path.abspath(__file__))
STATS = {"cifar10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)), "cifar100": ((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))}
C10 = "airplane automobile bird cat deer dog frog horse ship truck".split()


def class_names(ds):
    if ds == "cifar10":
        return C10
    p = os.path.join(ROOT, "data", "cifar100_names.json")
    return json.load(open(p)) if os.path.exists(p) else [str(i) for i in range(100)]


def preprocess(img_u8, ds, res):
    x = np.asarray(Image.fromarray(img_u8).resize((res, res), Image.BILINEAR), dtype=np.float32) / 255.0
    m, s = (np.array(v, dtype=np.float32) for v in STATS[ds])
    return ((x - m) / s).transpose(2, 0, 1)[None]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="name of onnx/<name>.onnx")
    ap.add_argument("--dataset", choices=["cifar10", "cifar100"], default=None)
    ap.add_argument("--image", help="path of an image file (any size; resized to the model's input)")
    ap.add_argument("--n", type=int, default=5, help="number of test-set images to classify when --image is not given")
    ap.add_argument("--threads", type=int, default=1)
    a = ap.parse_args()
    ds = a.dataset or ("cifar10" if a.model.startswith("nas") else "cifar100")
    so = ort.SessionOptions(); so.intra_op_num_threads = a.threads
    sess = ort.InferenceSession(os.path.join(ROOT, "onnx", a.model + ".onnx"), so, providers=["CPUExecutionProvider"])
    res = sess.get_inputs()[0].shape[2]; names = class_names(ds)
    if a.image:
        items = [(np.array(Image.open(a.image).convert("RGB")), None)]
    else:
        d = np.load(os.path.join(ROOT, "data", f"{ds}_u8.npz")); items = [(d["xte"][i], int(d["yte"][i])) for i in range(a.n)]
    for img, y in items:
        lg = sess.run(None, {"input": preprocess(img, ds, res)})[0][0]
        p = np.exp(lg - lg.max()); p /= p.sum(); top = np.argsort(-p)[:5]
        print(("true=" + names[y] + "  " if y is not None else "") + "top5: " + ", ".join(f"{names[i]} {p[i]:.2f}" for i in top))


if __name__ == "__main__":
    main()
