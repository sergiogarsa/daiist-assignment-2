"""
Assignment 2 training + evaluation pipeline (run via `uv run python main.py train`).

Decision supported: should a winter-maintenance duty officer dispatch a spot
treatment (grit/plough) to a road segment, based on its roadside-camera image?

Stages, all run end to end with no manual steps:
  1. Download UDOT WINTER ROAD CONDITIONS v1 (Roboflow Universe, classification, folder format).
  2. Audit the dataset BEFORE training: split/class counts, image sizes, image sources,
     duplicate and near-duplicate images across splits.
  3. Build a downscaled working copy (same files, same split/class folders; only pixels resized)
     so CPU training is practical. Labels and official splits are untouched.
  4. Fine-tune yolo11n-cls (ImageNet-pretrained) on train; best epoch chosen on valid top-1.
  5. Score valid and test with the fine-tuned model and with two zero-shot baselines
     (pretrained ImageNet yolo11n-cls with a stated snow-class mapping; COCO yolo11n detector).
  6. Multiclass metrics, case-level metrics, cost-vs-threshold sweep and the
     pass / review / fail threshold pair, all SELECTED ON VALID and REPORTED ON TEST.
  7. Sensitivity analysis, bootstrap uncertainty, failure examples, plots.

Outputs: weights/best.pt (committed) and artifacts/ (predictions, results.json, plots),
which app.py reads without retraining.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors  # noqa: E402
import matplotlib.lines  # noqa: E402
import matplotlib.patches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

from roadcost import policy as P  # noqa: E402
from roadcost.config import (  # noqa: E402
    ART_DIR, ASSUMPTION_NOTES, CLASSES, DATA_DIR, ECON, NEGATIVE_CLASSES, POSITIVE_CLASSES,
    RF_FORMAT, RF_LICENSE, RF_PROJECT, RF_URL, RF_VERSION, RF_WORKSPACE, ROOT, SMALL_DIR, SPLITS,
    WEIGHTS_DIR, WEIGHTS_PATH, TrainConfig,
)

CFG = TrainConfig()
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
# Of 256 dHash bits. Calibrated by eye on test/train pairs: up to ~24 bits the two images
# are visibly the same scene from the same drive (seconds apart); beyond ~35 they differ.
NEAR_DUP_BITS = 24
NEAR_DUP_CURVE = (0, 10, 16, 24, 30, 40)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass


def device_str() -> str:
    import torch

    return "0" if torch.cuda.is_available() else "cpu"


# ======================================================================================
# 1. Download
# ======================================================================================
def dataset_present(root: Path) -> bool:
    return all((root / s).is_dir() and any((root / s).rglob("*.jpg")) for s in SPLITS)


def download() -> None:
    if dataset_present(DATA_DIR):
        log(f"Dataset already present at {DATA_DIR.relative_to(ROOT)}; skipping download.")
        return
    from roboflow import Roboflow

    from api_key import load_api_key

    log(f"Downloading {RF_WORKSPACE}/{RF_PROJECT} v{RF_VERSION} ({RF_FORMAT}) from Roboflow Universe ...")
    rf = Roboflow(api_key=load_api_key())
    version = rf.workspace(RF_WORKSPACE).project(RF_PROJECT).version(RF_VERSION)
    version.download(RF_FORMAT, location=str(DATA_DIR), overwrite=False)
    if not dataset_present(DATA_DIR):
        raise RuntimeError("Download finished but the expected train/valid/test folders are missing.")


def list_images(root: Path) -> pd.DataFrame:
    rows = []
    for split in SPLITS:
        for cls_dir in sorted((root / split).iterdir()):
            if not cls_dir.is_dir():
                continue
            for f in sorted(cls_dir.iterdir()):
                if f.suffix.lower() in IMG_EXT:
                    rows.append({"split": split, "cls": cls_dir.name, "file": f.name,
                                 "rel": f"{split}/{cls_dir.name}/{f.name}"})
    df = pd.DataFrame(rows)
    unknown = set(df["cls"]) - set(CLASSES)
    if unknown:
        raise RuntimeError(f"Unexpected class folders: {unknown}")
    return df


def source_family(name: str) -> str:
    if name.startswith("screenshot"):
        return "screenshot"
    if name.startswith("frame"):
        return "video_frame"
    if name.startswith("Test_Image"):
        return "Test_Image"
    return "other"


# ======================================================================================
# 2-3. Working copy + audit helpers
# ======================================================================================
def _shrink_one(args):
    src, dst, max_side = args
    dst = Path(dst)
    with Image.open(src) as im:
        w, h = im.size
        scale = max_side / max(w, h)
        if im.format == "JPEG" and scale < 0.5:
            im.draft("RGB", (int(w * scale * 1.01) + 1, int(h * scale * 1.01) + 1))  # fast JPEG decode
        im = im.convert("RGB")
        if max(im.size) > max_side:
            im.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        if not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            tmp = dst.with_suffix(".tmp.jpg")
            im.save(tmp, quality=92)
            os.replace(tmp, dst)
        # 16x16 difference hash (256 bits) for near-duplicate detection
        g = im.convert("L").resize((17, 16), Image.Resampling.LANCZOS)
        a = np.asarray(g, dtype=np.int16)
        bits = (a[:, 1:] > a[:, :-1]).flatten()
    return (w, h, np.packbits(bits).tobytes().hex())


def build_small_copy_and_hashes(df: pd.DataFrame) -> pd.DataFrame:
    index = SMALL_DIR / "_index.csv"
    if index.exists():
        cached = pd.read_csv(index)
        if len(cached) == len(df) and set(cached["rel"]) == set(df["rel"]) and all(
                (SMALL_DIR / r).exists() for r in df["rel"]):
            log("Downscaled working copy and hashes already built; reusing.")
            return df.merge(cached[["rel", "width", "height", "dhash"]], on="rel", how="left")
    log(f"Building downscaled working copy (longest side {CFG.small_max_side}px) and image hashes ...")
    jobs = [(str(DATA_DIR / r), str(SMALL_DIR / r), CFG.small_max_side) for r in df["rel"]]
    t = time.time()
    with ProcessPoolExecutor(max_workers=max(1, min(4, os.cpu_count() or 1))) as ex:
        res = list(ex.map(_shrink_one, jobs, chunksize=32))
    log(f"  done in {time.time() - t:.0f}s")
    df = df.copy()
    df["width"] = [r[0] for r in res]
    df["height"] = [r[1] for r in res]
    df["dhash"] = [r[2] for r in res]
    df[["rel", "width", "height", "dhash"]].to_csv(index, index=False)
    return df


def file_md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


_POP = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def hash_matrix(hexes) -> np.ndarray:
    return np.stack([np.frombuffer(bytes.fromhex(h), dtype=np.uint8) for h in hexes])


def nearest_hamming(a: np.ndarray, b: np.ndarray, chunk: int = 64):
    """For every row of a, the min Hamming distance to rows of b and its index."""
    dmin = np.empty(len(a), dtype=np.int32)
    imin = np.empty(len(a), dtype=np.int64)
    for i in range(0, len(a), chunk):
        x = np.bitwise_xor(a[i:i + chunk, None, :], b[None, :, :])
        d = _POP[x].sum(axis=2, dtype=np.int32)
        imin[i:i + chunk] = d.argmin(axis=1)
        dmin[i:i + chunk] = d.min(axis=1)
    return dmin, imin


def audit(df: pd.DataFrame) -> dict:
    log("Auditing dataset ...")
    out: dict = {}
    counts = df.pivot_table(index="cls", columns="split", values="file", aggfunc="count").reindex(
        index=list(CLASSES), columns=list(SPLITS)).fillna(0).astype(int)
    out["counts"] = counts.to_dict()
    out["split_sizes"] = df["split"].value_counts().reindex(list(SPLITS)).to_dict()
    out["positive_share_by_split"] = {
        s: float(df[df.split == s]["cls"].isin(POSITIVE_CLASSES).mean()) for s in SPLITS}
    df["source"] = df["file"].map(source_family)
    out["source_by_split_class"] = (df.groupby(["split", "cls", "source"]).size()
                                    .rename("n").reset_index().to_dict(orient="records"))
    out["source_by_class_all"] = pd.crosstab(df["cls"], df["source"]).to_dict()
    sizes = df.groupby(["width", "height"]).size().sort_values(ascending=False)
    out["image_sizes"] = [{"width": int(w), "height": int(h), "n": int(n)} for (w, h), n in sizes.items()]
    out["image_size_by_source"] = (df.groupby(["source", "width", "height"]).size().rename("n")
                                   .reset_index().to_dict(orient="records"))

    # exact duplicates (byte-identical files)
    log("  hashing files for exact duplicates ...")
    df["md5"] = [file_md5(DATA_DIR / r) for r in df["rel"]]
    g = df.groupby("md5")
    exact = [grp for _, grp in g if len(grp) > 1]
    out["exact_duplicate_groups"] = len(exact)
    out["exact_duplicate_groups_cross_split"] = sum(grp["split"].nunique() > 1 for grp in exact)
    out["exact_duplicate_groups_label_conflict"] = sum(grp["cls"].nunique() > 1 for grp in exact)

    # same original filename used more than once (Roboflow adds a random .rf.<hash> suffix)
    df["orig_name"] = df["file"].str.replace(r"\.rf\.[0-9a-f]+\.\w+$", "", regex=True)
    on = df.groupby("orig_name")
    multi = [grp for _, grp in on if len(grp) > 1]
    out["orig_name_reused_groups"] = len(multi)
    out["orig_name_reused_cross_split"] = sum(grp["split"].nunique() > 1 for grp in multi)
    out["orig_name_reused_label_conflict"] = sum(grp["cls"].nunique() > 1 for grp in multi)

    # near duplicates by 256-bit dHash
    log("  near-duplicate search (dHash) across splits ...")
    H = {s: hash_matrix(df[df.split == s]["dhash"]) for s in SPLITS}
    idx = {s: df.index[df.split == s].to_numpy() for s in SPLITS}
    df["nn_train_dist"] = np.nan
    df["nn_train_same_label"] = np.nan
    for s in ("valid", "test"):
        d, j = nearest_hamming(H[s], H["train"])
        df.loc[idx[s], "nn_train_dist"] = d
        df.loc[idx[s], "nn_train_same_label"] = (
            df.loc[idx["train"][j], "cls"].to_numpy() == df.loc[idx[s], "cls"].to_numpy()).astype(float)
    d_tv, _ = nearest_hamming(H["test"], H["valid"])
    df.loc[idx["test"], "nn_valid_dist"] = d_tv
    nd = {}
    for s in ("valid", "test"):
        sub = df[df.split == s]
        near = sub["nn_train_dist"] <= NEAR_DUP_BITS
        nd[s] = {
            "n": int(len(sub)),
            "near_dup_in_train": int(near.sum()),
            "share": float(near.mean()),
            "label_agrees_with_near_dup": float(sub.loc[near, "nn_train_same_label"].mean()) if near.any() else None,
            "exact_or_identical_hash(dist=0)": int((sub["nn_train_dist"] == 0).sum()),
            "dist_quantiles": {q: float(sub["nn_train_dist"].quantile(q)) for q in (0.05, 0.25, 0.5, 0.75)},
        }
    nd["curve"] = []
    for t in NEAR_DUP_CURVE:
        row = {"bits": t}
        for s in ("valid", "test"):
            sub = df[df.split == s]
            near = sub["nn_train_dist"] <= t
            row[f"{s}_share"] = float(near.mean())
            row[f"{s}_label_agree"] = float(sub.loc[near, "nn_train_same_label"].mean()) if near.any() else None
        nd["curve"].append(row)
    nd["test_vs_valid_near_dup"] = int((df.loc[idx["test"], "nn_valid_dist"] <= NEAR_DUP_BITS).sum())
    nd["threshold_bits"] = NEAR_DUP_BITS
    out["near_duplicates"] = nd
    out["bounding_boxes"] = ("Not applicable: this is an image-classification dataset (one label per "
                             "image, no boxes). Object-size statistics do not exist for it.")

    # plot class counts
    fig, ax = plt.subplots(figsize=(7, 3.6))
    counts.plot(kind="bar", ax=ax, color=["#4c72b0", "#dd8452", "#55a868"])
    ax.set_title("Images per class and split (UDOT Winter Road Conditions v1)")
    ax.set_ylabel("images")
    ax.set_xlabel("")
    plt.xticks(rotation=0)
    fig.tight_layout()
    fig.savefig(ART_DIR / "class_counts.png", dpi=120)
    plt.close(fig)
    return out


# ======================================================================================
# 4. Fine-tune
# ======================================================================================
CKPT_DIR = ROOT / "runs" / "cls_finetune"


class _FolderDataset:
    """(image, label) pairs from a split folder; labels follow CLASSES (alphabetical)."""

    def __init__(self, split: str, transform):
        self.items = [(p, CLASSES.index(p.parent.name))
                      for c in CLASSES for p in sorted((SMALL_DIR / split / c).iterdir())
                      if p.suffix.lower() in IMG_EXT]
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p, y = self.items[i]
        with Image.open(p) as im:
            return self.transform(im.convert("RGB")), y


def _worker_init(worker_id):
    import torch

    s = torch.initial_seed() % 2**32
    np.random.seed(s)
    random.seed(s)


def _save_ultralytics_weights(model, path: Path, extra: dict) -> None:
    """Save in the Ultralytics checkpoint layout so YOLO(path) loads it like any trained model."""
    import copy
    from datetime import datetime

    import torch
    import ultralytics

    m = copy.deepcopy(model).cpu().float().eval()
    m.names = {i: c for i, c in enumerate(CLASSES)}
    ckpt = {"date": datetime.now().isoformat(), "version": ultralytics.__version__, "license": "AGPL-3.0",
            "docs": "https://docs.ultralytics.com", "epoch": -1, "best_fitness": extra.get("valid_top1"),
            "model": m.half(), "ema": None, "updates": None, "optimizer": None,
            "train_args": {"task": "classify", "mode": "train", "model": CFG.model, "imgsz": CFG.imgsz,
                           "epochs": CFG.epochs, "batch": CFG.batch, "seed": CFG.seed,
                           "data": f"{RF_WORKSPACE}/{RF_PROJECT}/{RF_VERSION}"},
            "train_metrics": extra}
    tmp = path.with_suffix(".tmp")
    torch.save(ckpt, tmp)
    os.replace(tmp, path)


def _atomic_torch_save(obj, path: Path) -> None:
    import torch

    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def train_model() -> dict:
    """Fine-tune the ImageNet-pretrained yolo11n-cls on the train split.

    A plain PyTorch loop around the Ultralytics model, using Ultralytics' own
    classification transforms. It checkpoints every ~45 s and resumes automatically if the
    process is interrupted (the run used for the submission was trained on a CPU sandbox
    whose processes are capped at a few minutes). Uninterrupted, it simply runs start to finish.
    Model selection: the epoch with the highest valid top-1 accuracy (ties -> lower valid loss).
    """
    import torch
    from torch.utils.data import DataLoader, Subset
    from ultralytics import YOLO
    from ultralytics.data.augment import classify_augmentations, classify_transforms
    from ultralytics.nn.tasks import ClassificationModel

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    state_path, best_path = CKPT_DIR / "state.pt", CKPT_DIR / "best_ultralytics.pt"
    cfg_sig = json.dumps(CFG.__dict__, sort_keys=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = YOLO(CFG.model).model  # ClassificationModel, 1000 ImageNet classes
    ClassificationModel.reshape_outputs(model, len(CLASSES))
    model.names = {i: c for i, c in enumerate(CLASSES)}
    for p in model.parameters():
        p.requires_grad = True
    model.to(dev)

    train_tf = classify_augmentations(size=CFG.imgsz, scale=(0.35, 1.0), hflip=0.5, hsv_h=0.015, hsv_s=0.4,
                                      hsv_v=0.3, auto_augment=None, erasing=0.0)
    eval_tf = classify_transforms(CFG.imgsz)
    ds_train, ds_val = _FolderDataset("train", train_tf), _FolderDataset("valid", eval_tf)
    n_batches = int(np.ceil(len(ds_train) / CFG.batch))
    total_steps = CFG.epochs * n_batches
    opt = torch.optim.AdamW(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)
    warm = n_batches // 2

    def lr_at(step):
        if step < warm:
            return CFG.lr * (step + 1) / warm
        prog = (step - warm) / max(1, total_steps - warm)
        return CFG.lr * (0.02 + 0.98 * 0.5 * (1 + np.cos(np.pi * prog)))

    st = {"epoch": 0, "batch": 0, "history": [], "best": None, "train_seconds": 0.0, "cfg": cfg_sig}
    if state_path.exists():
        saved = torch.load(state_path, map_location="cpu", weights_only=False)
        if saved.get("cfg") == cfg_sig:
            model.load_state_dict(saved["model"])
            opt.load_state_dict(saved["opt"])
            st = saved["st"]
            log(f"Resuming fine-tuning at epoch {st['epoch'] + 1}, batch {st['batch']}/{n_batches}")
        else:
            log("Found a checkpoint from a different configuration; starting fresh.")
    if st["epoch"] >= CFG.epochs and best_path.exists():
        log("Fine-tuning already complete for this configuration.")
    else:
        log(f"Fine-tuning {CFG.model} for {CFG.epochs} epochs on {len(ds_train)} images (device={dev}) ...")
    loss_fn = torch.nn.CrossEntropyLoss()
    last_save = time.time()
    t_seg = time.time()

    def save_state():
        _atomic_torch_save({"model": model.state_dict(), "opt": opt.state_dict(), "cfg": cfg_sig,
                            "st": {**st, "train_seconds": st["train_seconds"] + time.time() - t_seg}}, state_path)

    while st["epoch"] < CFG.epochs:
        ep = st["epoch"]
        if st["batch"] < n_batches:
            g = torch.Generator().manual_seed(CFG.seed * 1000 + ep)
            perm = torch.randperm(len(ds_train), generator=g).tolist()
            start = st["batch"] * CFG.batch
            torch.manual_seed(CFG.seed * 100003 + ep * 1009 + st["batch"])  # augmentation stream
            loader = DataLoader(Subset(ds_train, perm[start:]), batch_size=CFG.batch, shuffle=False,
                                num_workers=CFG.workers, worker_init_fn=_worker_init, drop_last=False)
            model.train()
            for x, y in loader:
                step = ep * n_batches + st["batch"]
                for gp in opt.param_groups:
                    gp["lr"] = lr_at(step)
                x, y = x.to(dev), y.to(dev)
                out = model(x)
                out = out[0] if isinstance(out, (list, tuple)) else out
                loss = loss_fn(out, y)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                st["loss_sum"] = st.get("loss_sum", 0.0) + float(loss) * len(y)
                st["seen"] = st.get("seen", 0) + len(y)
                st["batch"] += 1
                if time.time() - last_save > 45:
                    save_state()
                    last_save = time.time()
                    log(f"  epoch {ep + 1}/{CFG.epochs} batch {st['batch']}/{n_batches} "
                        f"loss {st['loss_sum'] / st['seen']:.4f}")
            st.setdefault("epoch_train_loss", {})[str(ep)] = st.get("loss_sum", 0.0) / max(st.get("seen", 0), 1)
            st["loss_sum"], st["seen"] = 0.0, 0
            save_state()
        # validation for model selection
        model.eval()
        vl, vc, vn = 0.0, 0, 0
        with torch.no_grad():
            for x, y in DataLoader(ds_val, batch_size=64, shuffle=False, num_workers=CFG.workers):
                out = model(x.to(dev))
                logits = out[1] if isinstance(out, (list, tuple)) else out
                vl += float(torch.nn.functional.cross_entropy(logits, y.to(dev), reduction="sum"))
                vc += int((logits.argmax(1).cpu() == y).sum())
                vn += len(y)
        row = {"epoch": ep + 1, "train_loss": st["epoch_train_loss"][str(ep)],
               "valid_loss": vl / vn, "valid_top1": vc / vn, "lr_end": lr_at((ep + 1) * n_batches - 1)}
        st["history"].append(row)
        log(f"  epoch {ep + 1}/{CFG.epochs}: valid top-1 {row['valid_top1']:.4f}, valid loss {row['valid_loss']:.4f}")
        b = st["best"]
        if b is None or row["valid_top1"] > b["valid_top1"] or (
                row["valid_top1"] == b["valid_top1"] and row["valid_loss"] < b["valid_loss"]):
            st["best"] = row
            _save_ultralytics_weights(model, best_path, {"best_epoch": ep + 1, **row})
        st["epoch"] += 1
        st["batch"] = 0
        save_state()
        last_save = time.time()

    WEIGHTS_DIR.mkdir(exist_ok=True)
    shutil.copy2(best_path, WEIGHTS_PATH)
    hist = pd.DataFrame(st["history"])
    hist.to_csv(ART_DIR / "training_history.csv", index=False)
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    ax[0].plot(hist["epoch"], hist["train_loss"], marker="o", label="train loss")
    ax[0].plot(hist["epoch"], hist["valid_loss"], marker="o", label="valid loss")
    ax[0].legend()
    ax[0].set_xlabel("epoch")
    ax[1].plot(hist["epoch"], hist["valid_top1"], marker="o")
    ax[1].axvline(st["best"]["epoch"], color="red", lw=0.8, ls="--")
    ax[1].set_title("valid top-1 accuracy (red = selected epoch)")
    ax[1].set_xlabel("epoch")
    fig.tight_layout()
    fig.savefig(ART_DIR / "training_curves.png", dpi=120)
    plt.close(fig)
    _ = YOLO(str(WEIGHTS_PATH))  # sanity check: the saved weights load through Ultralytics
    log(f"  best epoch {st['best']['epoch']} (valid top-1 {st['best']['valid_top1']:.4f}); "
        f"weights -> {WEIGHTS_PATH.relative_to(ROOT)}")
    return {"train_seconds_cumulative": st["train_seconds"], "best_epoch": st["best"]["epoch"],
            "best_valid_top1": st["best"]["valid_top1"], "history": st["history"],
            "loop": "PyTorch loop around Ultralytics yolo11n-cls with Ultralytics transforms; AdamW, "
                    "linear warm-up then cosine decay; checkpoint + auto-resume"}


# ======================================================================================
# 5. Scoring
# ======================================================================================
CACHE_DIR = ROOT / "runs" / "cache"


def predict_probs(weights: str, paths: list[Path], batch: int = 32) -> tuple[np.ndarray, dict]:
    """Softmax probabilities for every image (all classes kept, nothing discarded). Cached per weights file."""
    from ultralytics import YOLO

    wp = Path(weights) if Path(weights).exists() else ROOT / weights
    key = hashlib.md5((file_md5(wp) if wp.exists() else weights).encode()
                      + "|".join(str(p.relative_to(ROOT)) for p in paths).encode()).hexdigest()[:16]
    cache = CACHE_DIR / f"probs_{key}.npz"
    m = YOLO(weights)
    if cache.exists():
        log(f"  reusing cached scores {cache.name}")
        return np.load(cache)["p"], m.names
    dev = device_str()
    out = []
    for i in range(0, len(paths), batch):
        res = m.predict([str(p) for p in paths[i:i + batch]], imgsz=CFG.imgsz, device=dev, verbose=False)
        out.extend(r.probs.data.cpu().numpy() for r in res)
    probs = np.stack(out)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(cache, p=probs)
    return probs, m.names


# ImageNet-1k classes whose presence implies a snowy scene. Chosen a priori from the class
# list (not tuned on any split). Animals/clothing that merely contain "snow"/"ski"/"ice"
# in their name (snow_leopard, ski_mask, ice_cream, ...) are deliberately excluded.
IMAGENET_SNOW_CLASSES = ("snowplow", "snowmobile", "ski", "dogsled", "bobsled", "alp")


def imagenet_snow_indices(names: dict) -> dict:
    """A-priori semantic mapping of ImageNet classes to 'snowy scene'."""
    found = {i: n for i, n in names.items() if n in IMAGENET_SNOW_CLASSES}
    if len(found) != len(IMAGENET_SNOW_CLASSES):
        raise RuntimeError(f"ImageNet mapping incomplete: {found}")
    return found


COCO_WINTER_CLASSES = ("skis", "snowboard")  # the only COCO classes tied to snow at all
COCO_CONF = 0.25  # Ultralytics default; fixed a priori, not tuned


def coco_zero_shot(paths: list[Path], true_cls: list[str]) -> dict:
    """Run the pretrained COCO detector zero-shot and document what it sees.

    COCO has no road-surface-condition class. The closest thing to a mapping is
    'a skis or snowboard detection (conf >= 0.25) means snow' -- stated here, not tuned.
    Per-image max confidence for those classes is kept down to conf 0.01 so the
    behaviour can be inspected at other thresholds too.
    """
    from ultralytics import YOLO

    cache = CACHE_DIR / "coco_zero_shot_v2.json"
    if cache.exists():
        log("  reusing cached COCO zero-shot results")
        return json.loads(cache.read_text())
    m = YOLO("yolo11n.pt")
    dev = device_str()
    det_counts = {c: {} for c in CLASSES}
    any_det = {c: 0 for c in CLASSES}
    winter_score = []
    for i in range(0, len(paths), 16):
        res = m.predict([str(p) for p in paths[i:i + 16]], imgsz=640, conf=0.01, device=dev, verbose=False)
        for r, c in zip(res, true_cls[i:i + 16]):
            cls_ids = [int(k) for k in r.boxes.cls.tolist()]
            confs = r.boxes.conf.tolist()
            names = [m.names[k] for k, cf in zip(cls_ids, confs) if cf >= COCO_CONF]
            if names:
                any_det[c] += 1
            for n in names:
                det_counts[c][n] = det_counts[c].get(n, 0) + 1
            winter_score.append(max([cf for k, cf in zip(cls_ids, confs) if m.names[k] in COCO_WINTER_CLASSES],
                                    default=0.0))
    n_by = pd.Series(true_cls).value_counts().to_dict()
    top = {c: dict(sorted(v.items(), key=lambda kv: -kv[1])[:6]) for c, v in det_counts.items()}
    out = {
        "n_images": len(paths),
        "conf_for_summary_and_mapping": COCO_CONF,
        "share_images_with_any_detection": {c: any_det[c] / n_by[c] for c in CLASSES},
        "top_detected_classes_by_true_class": top,
        "coco_snow_related_classes": list(COCO_WINTER_CLASSES),
        "has_road_surface_condition_class": False,
        "mapping_used": "Stated a priori: FAIL if a 'skis' or 'snowboard' box with conf >= 0.25 is detected, "
                        "else PASS. These classes describe winter sports, not road surfaces, so the mapping is "
                        "weak by construction; it is reported only to show the detector's operational behaviour.",
        "winter_score": winter_score,
    }
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(out))
    return out


# ======================================================================================
# 6. Metrics
# ======================================================================================
def multiclass_metrics(y_true: list[str], y_pred: list[str]) -> dict:
    cm = pd.crosstab(pd.Categorical(y_true, categories=CLASSES), pd.Categorical(y_pred, categories=CLASSES),
                     rownames=["true"], colnames=["pred"], dropna=False)
    per = {}
    for c in CLASSES:
        tp = int(cm.loc[c, c])
        fp = int(cm[c].sum() - tp)
        fn = int(cm.loc[c].sum() - tp)
        prec = tp / (tp + fp) if tp + fp else float("nan")
        rec = tp / (tp + fn) if tp + fn else float("nan")
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else float("nan")
        per[c] = {"precision": prec, "recall": rec, "f1": f1, "support": int(cm.loc[c].sum()),
                  "recall_ci95": P.wilson(tp, tp + fn)}
    acc = float(np.trace(cm.to_numpy()) / cm.to_numpy().sum())
    macro_f1 = float(np.nanmean([v["f1"] for v in per.values()]))
    return {"accuracy": acc, "macro_f1": macro_f1, "per_class": per,
            "confusion": cm.to_numpy().tolist(), "labels": list(CLASSES)}


def auc(scores, y) -> float:
    s, y = np.asarray(scores), np.asarray(y)
    pos, neg = s[y == 1], s[y == 0]
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order))
    allv = np.concatenate([pos, neg])[order]
    # average ranks for ties
    i = 0
    r = np.empty(len(allv))
    while i < len(allv):
        j = i
        while j + 1 < len(allv) and allv[j + 1] == allv[i]:
            j += 1
        r[i:j + 1] = (i + j) / 2 + 1
        i = j + 1
    ranks[order] = r
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def plot_confusion(cm, labels, title, path, normalize=True):
    cm = np.asarray(cm, dtype=float)
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    shown = cm / cm.sum(axis=1, keepdims=True) if normalize else cm
    ax.imshow(shown, cmap="Blues", vmin=0, vmax=1 if normalize else None)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            txt = f"{int(cm[i, j])}" + (f"\n{shown[i, j]:.0%}" if normalize else "")
            ax.text(j, i, txt, ha="center", va="center",
                    color="white" if shown[i, j] > (0.5 if normalize else shown.max() / 2) else "black", fontsize=9)
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def case_confusion(r: dict) -> dict:
    c = r["counts"]
    return {"TP(FAIL|pos)": c["fail_pos"], "FN(PASS|pos)": c["pass_pos"], "REVIEW|pos": c["review_pos"],
            "FP(FAIL|neg)": c["fail_neg"], "TN(PASS|neg)": c["pass_neg"], "REVIEW|neg": c["review_neg"]}


# ======================================================================================
# 7. Economics: selection on valid, evaluation on test, sensitivity
# ======================================================================================
def select_policies(s_val, y_val, e) -> dict:
    grid = P.candidate_thresholds(s_val)
    sw = P.sweep_single(s_val, y_val, grid, e)
    t_star, c_star = P.best_single(sw)
    pairs = P.sweep_pairs(s_val, y_val, grid, e)
    unconstrained = P.best_pair(pairs, e)
    constrained = P.best_pair(pairs, e, capacity_hours=e.review_capacity_hours_per_day)
    return {"grid": grid, "sweep": sw, "t_star": t_star, "cost_star_val": c_star, "pairs": pairs,
            "pair_unconstrained": unconstrained, "pair_constrained": constrained}


def evaluate_policies(s, y, t_star, pair, e) -> list[dict]:
    rows = [
        P.policy_summary("Do nothing (never dispatch on camera evidence)", P.constant_rates("PASS"), e),
        P.policy_summary("Treat every segment (always dispatch)", P.constant_rates("FAIL"), e),
        P.policy_summary("Human reviews every case", P.constant_rates("REVIEW"), e),
        P.policy_summary(f"Model only, t={t_star:.3f}", P.three_way_rates(s, y, t_star, t_star), e),
        P.policy_summary(f"Model + review, [{pair['t_low']:.3f}, {pair['t_high']:.3f})",
                         P.three_way_rates(s, y, pair["t_low"], pair["t_high"]), e),
    ]
    return rows


def sensitivity(s_val, y_val, s_test, y_test, sel, e) -> dict:
    grid = sel["grid"]
    out = {}
    # 1D: C_FN at baseline prevalence; thresholds re-optimised on VALID for every value
    cfn_vals = np.unique(np.round(np.geomspace(100, 50000, 60)))
    rows = []
    for cfn in cfn_vals:
        ee = P.with_overrides(e, c_fn=float(cfn))
        sw = P.sweep_single(s_val, y_val, grid, ee)
        t, _ = P.best_single(sw)
        pr = P.best_pair(P.sweep_pairs(s_val, y_val, grid, ee), ee, capacity_hours=ee.review_capacity_hours_per_day)
        rows.append({
            "c_fn": float(cfn), "t_star": t, "t_low": pr["t_low"], "t_high": pr["t_high"],
            "do_nothing": P.expected_cost(P.constant_rates("PASS"), ee),
            "treat_all": P.expected_cost(P.constant_rates("FAIL"), ee),
            "review_all": P.expected_cost(P.constant_rates("REVIEW"), ee),
            "model_only_reopt_test": P.expected_cost(P.three_way_rates(s_test, y_test, t, t), ee),
            "model_review_reopt_test": P.expected_cost(P.three_way_rates(s_test, y_test, pr["t_low"], pr["t_high"]), ee),
            "model_only_frozen_test": P.expected_cost(P.three_way_rates(s_test, y_test, sel["t_star"], sel["t_star"]), ee),
        })
    out["c_fn"] = rows
    # 1D: prevalence
    rows = []
    for pi in np.round(np.concatenate([np.linspace(0.01, 0.1, 10), np.linspace(0.12, 0.8, 35)]), 4):
        ee = P.with_overrides(e, prevalence=float(pi))
        sw = P.sweep_single(s_val, y_val, grid, ee)
        t, _ = P.best_single(sw)
        pr = P.best_pair(P.sweep_pairs(s_val, y_val, grid, ee), ee, capacity_hours=ee.review_capacity_hours_per_day)
        rows.append({
            "prevalence": float(pi), "t_star": t, "t_low": pr["t_low"], "t_high": pr["t_high"],
            "do_nothing": P.expected_cost(P.constant_rates("PASS"), ee),
            "treat_all": P.expected_cost(P.constant_rates("FAIL"), ee),
            "review_all": P.expected_cost(P.constant_rates("REVIEW"), ee),
            "model_only_reopt_test": P.expected_cost(P.three_way_rates(s_test, y_test, t, t), ee),
            "model_review_reopt_test": P.expected_cost(P.three_way_rates(s_test, y_test, pr["t_low"], pr["t_high"]), ee),
            "model_only_frozen_test": P.expected_cost(P.three_way_rates(s_test, y_test, sel["t_star"], sel["t_star"]), ee),
        })
    out["prevalence"] = rows
    # 2D policy map on valid (cheapest policy among do nothing / treat all / review all / model / model+review)
    cfn_axis = np.geomspace(200, 30000, 25)
    pi_axis = np.linspace(0.02, 0.6, 25)
    names = ["do nothing", "treat all", "review all", "model only", "model + review"]
    best = np.zeros((len(pi_axis), len(cfn_axis)), int)
    best_no_review = np.zeros((len(pi_axis), len(cfn_axis)), int)
    for i, pi in enumerate(pi_axis):
        for j, cfn in enumerate(cfn_axis):
            ee = P.with_overrides(e, prevalence=float(pi), c_fn=float(cfn))
            sw = P.sweep_single(s_val, y_val, grid, ee)
            _, cm = P.best_single(sw)
            pr = P.best_pair(P.sweep_pairs(s_val, y_val, grid, ee), ee, capacity_hours=ee.review_capacity_hours_per_day)
            costs = [P.expected_cost(P.constant_rates("PASS"), ee), P.expected_cost(P.constant_rates("FAIL"), ee),
                     P.expected_cost(P.constant_rates("REVIEW"), ee), cm, pr["cost_per_case"]]
            best[i, j] = int(np.argmin(costs))
            best_no_review[i, j] = [0, 1, 3][int(np.argmin([costs[0], costs[1], costs[3]]))]
    out["policy_map"] = {"c_fn": cfn_axis.tolist(), "prevalence": pi_axis.tolist(), "best": best.tolist(),
                         "best_no_review": best_no_review.tolist(), "names": names}
    # reviewer-error scenarios (thresholds re-optimised on valid for each)
    rows = []
    for label, m, f in (("perfect reviewer (optimistic)", 0.0, 0.0), ("base 5%/5%", 0.05, 0.05),
                        ("pessimistic 10%/10%", 0.10, 0.10), ("hard cases 20%/20%", 0.20, 0.20)):
        ee = P.with_overrides(e, reviewer_miss_rate=m, reviewer_false_alarm_rate=f)
        pr = P.best_pair(P.sweep_pairs(s_val, y_val, grid, ee), ee, capacity_hours=ee.review_capacity_hours_per_day)
        r_test = P.three_way_rates(s_test, y_test, pr["t_low"], pr["t_high"])
        frozen = P.three_way_rates(s_test, y_test, sel["pair_constrained"]["t_low"], sel["pair_constrained"]["t_high"])
        rows.append({"scenario": label, "miss": m, "false_alarm": f, "t_low": pr["t_low"], "t_high": pr["t_high"],
                     "test_cost_per_1000_reopt": 1000 * P.expected_cost(r_test, ee),
                     "test_cost_per_1000_frozen_pair": 1000 * P.expected_cost(frozen, ee),
                     "review_all_per_1000": 1000 * P.expected_cost(P.constant_rates("REVIEW"), ee),
                     "test_review_share_reopt": P.review_share(r_test, ee)})
    out["reviewer_error"] = rows
    return out


def break_even(e, s_val, y_val, s_test, y_test, sel) -> dict:
    """Closed-form-ish break-even points using the frozen model-only rates measured on TEST."""
    r = P.three_way_rates(s_test, y_test, sel["t_star"], sel["t_star"])
    tpr, fpr = r["p_fail_pos"], r["p_fail_neg"]
    pi = e.prevalence
    out = {}
    # model-only beats do nothing when pi*TPR*C_FN > (1-pi)*FPR*C_FP (C_TP=C_TN=0)
    out["c_fn_model_beats_do_nothing"] = (1 - pi) * fpr * e.c_fp / (pi * tpr) if tpr > 0 else None
    # treat-all beats model-only when (1-pi)*C_FP*(1-FPR) < pi*C_FN*(1-TPR)
    out["c_fn_treat_all_beats_model"] = ((1 - pi) * e.c_fp * (1 - fpr) / (pi * (1 - tpr))) if tpr < 1 else None
    # prevalence at which treat-all beats model-only (at baseline costs)
    a = e.c_fp * (1 - fpr)
    b = e.c_fn * (1 - tpr)
    out["prevalence_treat_all_beats_model"] = a / (a + b) if (a + b) > 0 else None
    a2 = e.c_fp * fpr
    b2 = e.c_fn * tpr
    out["prevalence_model_beats_do_nothing"] = a2 / (a2 + b2) if (a2 + b2) > 0 else None
    # theoretical Bayes threshold IF scores were calibrated probabilities (for comparison only)
    # Where the threshold WOULD be if s were a calibrated probability at the training prior:
    # FAIL iff production posterior > C_FP/(C_FP+C_FN); production odds = data odds * prior-odds ratio.
    p_d = float(np.mean(y_val))  # valid split positive share (same curation as train)
    k = e.c_fp / (e.c_fp + e.c_fn)
    ratio = (e.prevalence / (1 - e.prevalence)) / (p_d / (1 - p_d))
    data_odds_needed = (k / (1 - k)) / ratio
    out["bayes_cutoff_on_production_posterior"] = k
    out["split_positive_share_used"] = p_d
    out["prior_shift_adjusted_threshold_if_calibrated"] = data_odds_needed / (1 + data_odds_needed)
    return out


# ======================================================================================
# Plots for the economics
# ======================================================================================
def plot_cost_curve(sel, s_test, y_test, e):
    sw = sel["sweep"]
    swt = P.sweep_single(s_test, y_test, sel["grid"], e)
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.plot(sw["threshold"], 1000 * sw["cost_per_case"], label="valid (used for selection)")
    ax.plot(swt["threshold"], 1000 * swt["cost_per_case"], "--", alpha=0.7, label="test (report only)")
    ax.axvline(sel["t_star"], color="red", lw=1)
    ax.scatter([sel["t_star"]], [1000 * sel["cost_star_val"]], color="red", zorder=5,
               label=f"chosen t* = {sel['t_star']:.3f}")
    ax.axhline(1000 * P.expected_cost(P.constant_rates("PASS"), e), color="grey", ls=":", label="do nothing")
    ax.axhline(1000 * P.expected_cost(P.constant_rates("FAIL"), e), color="black", ls=":", label="treat all")
    ax.set_xlabel("threshold on s = P(Snow)+P(Slush)   (FAIL if s >= t)")
    ax.set_ylabel("expected cost per 1,000 cases (EUR)")
    ax.set_title(f"Model-only policy cost (prevalence {e.prevalence:.0%}, C_FP {e.c_fp:.0f}, C_FN {e.c_fn:.0f})",
                 fontsize=10)
    ax.set_xlim(-0.01, 1.01)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(ART_DIR / "cost_vs_threshold.png", dpi=120)
    plt.close(fig)


def plot_sensitivity(sens, e):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    d = pd.DataFrame(sens["c_fn"])
    for col, lab in (("do_nothing", "do nothing"), ("treat_all", "treat all"), ("review_all", "review all"),
                     ("model_only_reopt_test", "model only (re-optimised)"),
                     ("model_review_reopt_test", "model + review (re-optimised)")):
        ax[0].plot(d["c_fn"], 1000 * d[col], label=lab)
    ax[0].axvline(e.c_fn, color="red", lw=0.8, ls="--")
    ax[0].set_xscale("log")
    ax[0].set_xlabel("C_FN (EUR)")
    ax[0].set_ylabel("EUR per 1,000 cases (test rates)")
    ax[0].set_yscale("log")
    ax[0].legend(fontsize=7)
    ax[0].set_title(f"Sensitivity to C_FN (prevalence {e.prevalence:.0%})", fontsize=10)
    d = pd.DataFrame(sens["prevalence"])
    for col, lab in (("do_nothing", "do nothing"), ("treat_all", "treat all"), ("review_all", "review all"),
                     ("model_only_reopt_test", "model only (re-optimised)"),
                     ("model_review_reopt_test", "model + review (re-optimised)")):
        ax[1].plot(d["prevalence"], 1000 * d[col], label=lab)
    ax[1].axvline(e.prevalence, color="red", lw=0.8, ls="--")
    ax[1].set_xlabel("production prevalence of snow/slush")
    ax[1].set_title("Sensitivity to prevalence (baseline costs)", fontsize=10)
    ax[1].set_yscale("log")
    ax[1].set_ylabel("EUR per 1,000 cases (test rates, log)")
    fig.tight_layout()
    fig.savefig(ART_DIR / "sensitivity.png", dpi=120)
    plt.close(fig)

    pm = sens["policy_map"]
    cmap = matplotlib.colors.ListedColormap(["#bbbbbb", "#333333", "#e6a23c", "#4c72b0", "#55a868"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, key, title in ((axes[0], "best", "Cheapest policy, reviewer available"),
                           (axes[1], "best_no_review", "Cheapest policy if NO reviewer is available")):
        ax.pcolormesh(pm["c_fn"], pm["prevalence"], np.asarray(pm[key]), cmap=cmap, vmin=-0.5, vmax=4.5,
                      shading="nearest")
        ax.set_xscale("log")
        ax.scatter([e.c_fn], [e.prevalence], color="red", marker="x", s=80)
        ax.set_xlabel("C_FN (EUR), C_FP fixed at %.0f" % e.c_fp)
        ax.set_title(title + " (thresholds re-selected on valid)", fontsize=9)
    axes[0].set_ylabel("production prevalence")
    handles = [matplotlib.patches.Patch(color=cmap(i), label=n) for i, n in enumerate(pm["names"])]
    handles.append(matplotlib.lines.Line2D([], [], color="red", marker="x", ls="", label="baseline"))
    axes[1].legend(handles=handles, fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(ART_DIR / "policy_map.png", dpi=120)
    plt.close(fig)


def plot_pair_heatmap(sel):
    pairs = sel["pairs"]
    g = pairs["grid"]
    cost = np.where(np.isfinite(pairs["cost"]), 1000 * pairs["cost"], np.nan)
    # subsample for readability on a 0-1 axis
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.pcolormesh(g, g, cost.T, shading="nearest", cmap="viridis",
                       norm=matplotlib.colors.LogNorm(vmin=np.nanmin(cost), vmax=np.nanmax(cost)))
    fig.colorbar(im, ax=ax, label="EUR per 1,000 cases (valid)")
    pc, pu = sel["pair_constrained"], sel["pair_unconstrained"]
    ax.scatter([pu["t_low"]], [pu["t_high"]], color="white", marker="o", label="unconstrained optimum")
    ax.scatter([pc["t_low"]], [pc["t_high"]], color="red", marker="x", s=80, label="chosen (within capacity)")
    ax.set_xlabel("t_low")
    ax.set_ylabel("t_high")
    ax.legend(fontsize=8)
    ax.set_title("Two-threshold policy cost on valid", fontsize=10)
    fig.tight_layout()
    fig.savefig(ART_DIR / "review_pair_heatmap.png", dpi=120)
    plt.close(fig)


def save_error_examples(test_df, t_star, k=6):
    err_dir = ART_DIR / "errors"
    if err_dir.exists():
        shutil.rmtree(err_dir)
    err_dir.mkdir(parents=True)
    fn = test_df[(test_df.y == 1) & (test_df.score < t_star)].sort_values("score").head(k)
    fp = test_df[(test_df.y == 0) & (test_df.score >= t_star)].sort_values("score", ascending=False).head(k)
    rows = []
    for kind, sub in (("FN", fn), ("FP", fp)):
        for _, r in sub.iterrows():
            src = SMALL_DIR / r["rel"]
            dst = err_dir / f"{kind}_{r['cls']}_{Path(r['file']).stem[:40]}.jpg"
            with Image.open(src) as im:
                im = im.convert("RGB")
                im.thumbnail((360, 360))
                im.save(dst, quality=85)
            rows.append({"kind": kind, "true": r["cls"], "pred": r["pred"], "score": float(r["score"]),
                         "source": r["source"], "rel": r["rel"], "thumb": dst.relative_to(ROOT).as_posix()})
    if rows:
        n = len(rows)
        cols = min(6, n)
        nrow = int(np.ceil(n / cols))
        fig, axes = plt.subplots(nrow, cols, figsize=(2.6 * cols, 2.2 * nrow))
        axes = np.atleast_1d(axes).ravel()
        for a in axes:
            a.axis("off")
        for a, r in zip(axes, rows):
            a.imshow(Image.open(ROOT / r["thumb"]))
            a.set_title(f"{r['kind']}: true {r['true']}, pred {r['pred']}\ns={r['score']:.3f}", fontsize=7)
        fig.tight_layout()
        fig.savefig(ART_DIR / "error_examples.png", dpi=110)
        plt.close(fig)
    return rows


# ======================================================================================
# main
# ======================================================================================
def to_jsonable(o):
    if isinstance(o, dict):
        return {str(k): to_jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [to_jsonable(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, float) and not np.isfinite(o):
        return None
    if isinstance(o, np.ndarray):
        return to_jsonable(o.tolist())
    return o


def main() -> None:
    t0 = time.time()
    seed_everything(CFG.seed)
    ART_DIR.mkdir(exist_ok=True)

    # 1-3 data
    download()
    df = list_images(DATA_DIR)
    df = build_small_copy_and_hashes(df)
    aud = audit(df)
    (ART_DIR / "dataset_audit.json").write_text(json.dumps(to_jsonable(aud), indent=2))
    log(f"  split sizes {aud['split_sizes']}; near-dups test->train "
        f"{aud['near_duplicates']['test']['near_dup_in_train']}/{aud['near_duplicates']['test']['n']}")

    # 4 train
    import torch
    import ultralytics

    env = {"python": sys.version.split()[0], "platform": platform.platform(), "processor": platform.processor(),
           "cpu_count": os.cpu_count(), "torch": torch.__version__, "ultralytics": ultralytics.__version__,
           "cuda": torch.cuda.is_available(), "device": device_str()}
    tr = train_model()

    # 5 score valid + test
    ev = df[df.split.isin(["valid", "test"])].copy().reset_index(drop=True)
    paths = [SMALL_DIR / r for r in ev["rel"]]
    log("Scoring valid + test with the fine-tuned model ...")
    probs, names = predict_probs(str(WEIGHTS_PATH), paths)
    order = [names[i] for i in range(len(names))]
    if tuple(order) != CLASSES:
        raise RuntimeError(f"Model class order {order} != expected {CLASSES}")
    for i, c in enumerate(CLASSES):
        ev[f"p_{c}"] = probs[:, i]
    ev["pred"] = [CLASSES[i] for i in probs.argmax(1)]
    ev["score"] = ev[[f"p_{c}" for c in POSITIVE_CLASSES]].sum(axis=1)
    ev["y"] = P.is_positive(ev["cls"])

    log("Scoring valid + test with the pretrained ImageNet yolo11n-cls (zero-shot) ...")
    zprobs, znames = predict_probs(CFG.model, paths)
    snow_idx = imagenet_snow_indices(znames)
    ev["zs_score"] = zprobs[:, list(snow_idx)].sum(axis=1)
    ev["zs_top1"] = [znames[i] for i in zprobs.argmax(1)]

    test = ev[ev.split == "test"].reset_index(drop=True)
    val = ev[ev.split == "valid"].reset_index(drop=True)
    log("Running the COCO detector zero-shot on test ...")
    coco = coco_zero_shot([SMALL_DIR / r for r in test["rel"]], test["cls"].tolist())
    test["coco_winter_score"] = coco.pop("winter_score")

    keep = ["split", "cls", "rel", "file", "source", "width", "height", "nn_train_dist", "nn_train_same_label",
            "pred", "score", "y", "zs_score", "zs_top1"] + [f"p_{c}" for c in CLASSES]
    val[keep].to_csv(ART_DIR / "predictions_valid.csv", index=False)
    test[keep].to_csv(ART_DIR / "predictions_test.csv", index=False)

    # 6 multiclass metrics
    mc = {s: multiclass_metrics(d["cls"].tolist(), d["pred"].tolist()) for s, d in (("valid", val), ("test", test))}
    plot_confusion(mc["test"]["confusion"], CLASSES, "Fine-tuned yolo11n-cls, TEST (row-normalised)",
                   ART_DIR / "confusion_multiclass_test.png")
    majority = df[df.split == "train"]["cls"].value_counts().idxmax()
    base_major = multiclass_metrics(test["cls"].tolist(), [majority] * len(test))
    zs_multiclass_note = ("ImageNet has no Clear/Wet/Slush/Snow classes; multiclass zero-shot accuracy is "
                          "undefined. It is scored only at case level via the stated snow-class mapping.")

    # leakage-stratified accuracy
    near = test["nn_train_dist"] <= NEAR_DUP_BITS
    leak = {"test_with_near_dup_in_train": {"n": int(near.sum()),
                                            "accuracy": float((test.loc[near, "pred"] == test.loc[near, "cls"]).mean()) if near.any() else None},
            "test_without_near_dup": {"n": int((~near).sum()),
                                      "accuracy": float((test.loc[~near, "pred"] == test.loc[~near, "cls"]).mean()) if (~near).any() else None}}
    by_source = {s: {"n": int(len(g)), "accuracy": float((g["pred"] == g["cls"]).mean())}
                 for s, g in test.groupby("source")}

    # 7 economics: select on valid, freeze, evaluate on test
    e = ECON
    sel = select_policies(val["score"].to_numpy(), val["y"].to_numpy(), e)
    pc = sel["pair_constrained"]
    log(f"Selected on valid: t*={sel['t_star']:.4f}; review pair [{pc['t_low']:.4f}, {pc['t_high']:.4f}) "
        f"(unconstrained [{sel['pair_unconstrained']['t_low']:.4f}, {sel['pair_unconstrained']['t_high']:.4f}))")
    s_t, y_t = test["score"].to_numpy(), test["y"].to_numpy()
    s_v, y_v = val["score"].to_numpy(), val["y"].to_numpy()
    pol_test = evaluate_policies(s_t, y_t, sel["t_star"], pc, e)
    pol_val = evaluate_policies(s_v, y_v, sel["t_star"], pc, e)
    pol_unconstrained_test = P.policy_summary(
        "Model + review, unconstrained capacity",
        P.three_way_rates(s_t, y_t, sel["pair_unconstrained"]["t_low"], sel["pair_unconstrained"]["t_high"]), e)

    # zero-shot ImageNet: threshold selected on VALID with the same cost function, frozen on test
    zsel = P.best_single(P.sweep_single(val["zs_score"], y_v, P.candidate_thresholds(val["zs_score"]), e))
    zs_test = P.policy_summary(f"Zero-shot ImageNet (snow-class mapping), t={zsel[0]:.4f}",
                               P.three_way_rates(test["zs_score"], y_t, zsel[0], zsel[0]), e)
    case_metrics = {
        "fine_tuned": {
            "auc_valid": auc(s_v, y_v), "auc_test": auc(s_t, y_t),
            "test_case_confusion_model_only": case_confusion(P.three_way_rates(s_t, y_t, sel["t_star"], sel["t_star"])),
            "test_case_confusion_review": case_confusion(P.three_way_rates(s_t, y_t, pc["t_low"], pc["t_high"])),
            "valid_case_confusion_model_only": case_confusion(P.three_way_rates(s_v, y_v, sel["t_star"], sel["t_star"])),
        },
        "zero_shot_imagenet": {"auc_test": auc(test["zs_score"], y_t), "auc_valid": auc(val["zs_score"], y_v),
                               "threshold_from_valid": zsel[0], "mapped_classes": list(snow_idx.values()),
                               "test_policy": zs_test,
                               "test_case_confusion": case_confusion(P.three_way_rates(test["zs_score"], y_t, zsel[0], zsel[0])),
                               "top1_imagenet_labels_on_test": test["zs_top1"].value_counts().head(8).to_dict()},
    }
    coco_r = P.three_way_rates(test["coco_winter_score"], y_t, COCO_CONF, COCO_CONF)
    coco["test_policy"] = P.policy_summary("Zero-shot COCO yolo11n (skis/snowboard => FAIL, conf 0.25)", coco_r, e)
    coco["test_case_confusion"] = case_confusion(coco_r)
    coco["auc_test_winter_score"] = auc(test["coco_winter_score"], y_t)
    r_ms = P.three_way_rates(s_t, y_t, sel["t_star"], sel["t_star"])
    tp, fn_ = r_ms["counts"]["fail_pos"], r_ms["counts"]["pass_pos"]
    fp, tn = r_ms["counts"]["fail_neg"], r_ms["counts"]["pass_neg"]
    unc = {
        "model_only_test_tpr_ci95": P.wilson(tp, tp + fn_), "model_only_test_fpr_ci95": P.wilson(fp, fp + tn),
        "model_only_test_cost_per_1000_bootstrap": {k: 1000 * v for k, v in P.bootstrap_cost(
            s_t, y_t, sel["t_star"], sel["t_star"], e).items()},
        "review_test_cost_per_1000_bootstrap": {k: 1000 * v for k, v in P.bootstrap_cost(
            s_t, y_t, pc["t_low"], pc["t_high"], e).items()},
        "model_only_valid_cost_per_1000_bootstrap": {k: 1000 * v for k, v in P.bootstrap_cost(
            s_v, y_v, sel["t_star"], sel["t_star"], e).items()},
    }
    log("Sensitivity analysis ...")
    sens = sensitivity(s_v, y_v, s_t, y_t, sel, e)
    be = break_even(e, s_v, y_v, s_t, y_t, sel)

    plot_cost_curve(sel, s_t, y_t, e)
    plot_pair_heatmap(sel)
    plot_sensitivity(sens, e)
    test["pred_case"] = np.where(test["score"] >= sel["t_star"], "FAIL", "PASS")
    errors = save_error_examples(test, sel["t_star"])
    pd.DataFrame({"threshold": sel["sweep"]["threshold"], "tpr": sel["sweep"]["tpr"], "fpr": sel["sweep"]["fpr"],
                  "cost_per_case": sel["sweep"]["cost_per_case"]}).to_csv(ART_DIR / "cost_sweep_valid.csv", index=False)

    results = {
        "dataset": {"workspace": RF_WORKSPACE, "project": RF_PROJECT, "version": RF_VERSION, "format": RF_FORMAT,
                    "task": "classification", "url": RF_URL, "license": RF_LICENSE},
        "definitions": {"positive_classes": POSITIVE_CLASSES, "negative_classes": NEGATIVE_CLASSES,
                        "score": "s = P(Snow) + P(Slush) (softmax)", "near_dup_bits": NEAR_DUP_BITS,
                        "rule": "PASS if s < t_low; REVIEW if t_low <= s < t_high; FAIL if s >= t_high"},
        "train_config": CFG.__dict__, "environment": env, "training": tr,
        "economics": e.to_dict(), "assumption_notes": ASSUMPTION_NOTES,
        "multiclass": mc, "majority_baseline": {"class": majority, **base_major},
        "zero_shot_multiclass_note": zs_multiclass_note, "coco_zero_shot": coco,
        "leakage_stratified_test_accuracy": leak, "test_accuracy_by_source": by_source,
        "case_metrics": case_metrics,
        "selection": {"t_star": sel["t_star"], "cost_per_1000_valid_at_t_star": 1000 * sel["cost_star_val"],
                      "pair_constrained": pc, "pair_unconstrained": sel["pair_unconstrained"]},
        "policies_test": pol_test, "policies_valid": pol_val, "policy_unconstrained_review_test": pol_unconstrained_test,
        "uncertainty": unc, "sensitivity": sens, "break_even": be, "error_examples": errors,
        "pipeline_seconds": time.time() - t0,
    }
    (ART_DIR / "results.json").write_text(json.dumps(to_jsonable(results), indent=2))
    log(f"Done in {(time.time() - t0) / 60:.1f} min. Weights: {WEIGHTS_PATH.relative_to(ROOT)}; "
        f"outputs: {ART_DIR.relative_to(ROOT)}/")
    for row in pol_test:
        log(f"  TEST {row['policy']:<55s} EUR/1000 {row['cost_per_1000']:9.1f}  review {row['review_share']:.1%}")


if __name__ == "__main__":
    main()
