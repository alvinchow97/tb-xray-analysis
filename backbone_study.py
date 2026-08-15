"""Standalone backbone study runner — faithful extraction of analysis.ipynb cell 61.

Usage:
    python backbone_study.py <backbone> [condition]
        backbone:  mobilenetv2 | densenet121 | efficientnetb0
        condition: raw | masked   (default: both)

Resumable: results accumulate in backbone_results.json keyed "<backbone>|<cond>|<fold>";
completed keys are skipped, so re-running after an interruption is safe. Run each
invocation in a fresh process (the run_backbone_study.sh wrapper does this) to avoid
the tensorflow-metal memory leak observed when switching backbones inside one session.

Reproducibility: replicates the notebook's exact data splits (unsorted glob order +
seeded train_test_split) and verifies them against the data/masked/kaggle_* cache
before training. Aborts if the split does not match the notebook's.
"""
import gc
import glob
import json
import os
import pathlib
import sys

import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

# ── Configuration (cell 4) ───────────────────────────────────────────────────
IMG_SIZE = 224
RANDOM_SEED = 42
EPOCHS_STAGE_1 = 15
EPOCHS_STAGE_2 = 15
LEARNING_RATE_1 = 1e-3
LEARNING_RATE_2 = 1e-5

ROOT = pathlib.Path("/Users/alvinchow/analysis")
DATA_DIR = ROOT / "data"
KAGGLE_DATA_DIR = DATA_DIR / "kaggle"
MONTGOMERY_DIR = DATA_DIR / "montgomery"
SHENZHEN_DIR = DATA_DIR / "shenzhen"
MASK_DIR = DATA_DIR / "masked"
RESULTS_PATH = ROOT / "backbone_results.json"

STUDY_BATCH = 16

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


# ── Data splits (cell 10 — glob order deliberately unsorted, as in the notebook) ──
def prepare_data_splits():
    base_dir = KAGGLE_DATA_DIR / "TB_Chest_Radiography_Database"
    normal_paths = glob.glob(str(base_dir / "Normal" / "*.*"))
    tb_paths = glob.glob(str(base_dir / "Tuberculosis" / "*.*"))
    all_paths = normal_paths + tb_paths
    all_labels = [0] * len(normal_paths) + [1] * len(tb_paths)
    X_train, X_temp, y_train, y_temp = train_test_split(
        all_paths, all_labels, test_size=0.2, stratify=all_labels, random_state=RANDOM_SEED)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=RANDOM_SEED)
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def verify_split_against_mask_cache(X_train, X_val, X_test):
    for tag, split in [("kaggle_train", X_train), ("kaggle_val", X_val), ("kaggle_test", X_test)]:
        cached = {p.name for p in (MASK_DIR / tag).glob("*.png")}
        mine = {pathlib.Path(p).name for p in split}
        if cached != mine:
            sys.exit(f"ABORT: reproduced {tag} split does not match data/masked/{tag} "
                     f"({len(mine)} vs {len(cached)} files). Results would not be "
                     "comparable with cached folds — investigate before running.")
    print("Split verification against mask cache: OK")


# ── External cohorts (cells 31–32) ───────────────────────────────────────────
def load_nlm_dataset(image_dir, prefix):
    image_dir = pathlib.Path(image_dir)
    paths, labels = [], []
    for img_file in sorted(image_dir.glob("*.png")):
        if img_file.name.startswith("._"):
            continue
        parts = img_file.stem.rsplit("_", 1)
        if len(parts) != 2 or parts[1] not in ("0", "1"):
            continue
        paths.append(str(img_file))
        labels.append(int(parts[1]))
    if not paths:
        raise FileNotFoundError(f"[{prefix}] no labelled PNGs in {image_dir}")
    print(f"[{prefix}] {len(paths)} images — TB: {sum(labels)}  Normal: {len(labels) - sum(labels)}")
    return paths, labels


def find_image_dir(base, pattern="CXR_png"):
    base = pathlib.Path(base)
    candidates = [p for p in base.rglob(pattern) if p.is_dir() and "__MACOSX" not in p.parts]
    for p in candidates:
        if any(f for f in p.glob("*.png") if not f.name.startswith("._")):
            return p
    return candidates[0] if candidates else base


# ── Augmentation (cell 12) ───────────────────────────────────────────────────
def image_augmentation(img, label):
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_brightness(img, max_delta=0.1)
    img = tf.image.random_contrast(img, lower=0.9, upper=1.1)
    return img, label


# ── Loss, callbacks, two-stage training (cells 15–17) ────────────────────────
def focal_loss(alpha=0.25, gamma=2.0):
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        bce = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        alpha_t = y_true * alpha + (1 - y_true) * (1 - alpha)
        return tf.reduce_mean(alpha_t * tf.pow(1.0 - p_t, gamma) * bce)
    return loss


def make_callbacks():
    ES, RLR = tf.keras.callbacks.EarlyStopping, tf.keras.callbacks.ReduceLROnPlateau
    stage1 = [ES(monitor="val_auc", patience=5, restore_best_weights=True, verbose=1),
              RLR(monitor="val_loss", factor=0.3, patience=3, min_lr=1e-6, verbose=1)]
    stage2 = [ES(monitor="val_auc", patience=7, restore_best_weights=True, verbose=1),
              RLR(monitor="val_loss", factor=0.2, patience=4, min_lr=1e-6, verbose=1)]
    return stage1, stage2


def train_model(model, backbone, train_ds, val_ds):
    callbacks_stage1, callbacks_stage2 = make_callbacks()
    print("\nStage 1: Training head (backbone frozen) ...")
    backbone.trainable = False
    model.compile(optimizer=tf.keras.optimizers.Adam(LEARNING_RATE_1, clipnorm=1.0),
                  loss=focal_loss(alpha=0.25, gamma=2.0),
                  metrics=["accuracy", tf.keras.metrics.AUC(name="auc")])
    h1 = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_STAGE_1, callbacks=callbacks_stage1)

    print("\nStage 2: Fine-tuning (top 50 backbone layers) ...")
    backbone.trainable = True
    for layer in backbone.layers[:-50]:
        layer.trainable = False
    model.compile(optimizer=tf.keras.optimizers.Adam(LEARNING_RATE_2, clipnorm=1.0),
                  loss=focal_loss(alpha=0.25, gamma=2.0),
                  metrics=["accuracy", tf.keras.metrics.AUC(name="auc")])
    h2 = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_STAGE_2, callbacks=callbacks_stage2)
    return model, h1, h2


# ── Backbone registry + builders (cell 61) ───────────────────────────────────
BACKBONE_REG = {
    "mobilenetv2": (tf.keras.applications.MobileNetV2, tf.keras.applications.mobilenet_v2.preprocess_input),
    "densenet121": (tf.keras.applications.DenseNet121, tf.keras.applications.densenet.preprocess_input),
    "efficientnetb0": (tf.keras.applications.EfficientNetB0, tf.keras.applications.efficientnet.preprocess_input),
}


def build_backbone_model(name, dropout=0.2):
    ctor, _ = BACKBONE_REG[name]
    bb = ctor(input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights="imagenet")
    bb.trainable = False
    inp = tf.keras.Input((IMG_SIZE, IMG_SIZE, 3))
    x = bb(inp, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    out = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    return tf.keras.Model(inp, out, name=f"TB_{name}"), bb


def build_ds_bb(paths, labels, preprocess_fn, train=False):
    def _load(p, l):
        im = tf.io.read_file(p)
        im = tf.image.decode_image(im, channels=3, expand_animations=False)
        im = tf.image.resize(im, (IMG_SIZE, IMG_SIZE))
        im = tf.cast(im, tf.float32)
        return preprocess_fn(im), l
    ds = tf.data.Dataset.from_tensor_slices((paths, labels)).map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    if train:
        ds = ds.map(image_augmentation, num_parallel_calls=tf.data.AUTOTUNE).shuffle(1000)
    return ds.batch(STUDY_BATCH).prefetch(tf.data.AUTOTUNE)


def collect_predictions(model, dataset):
    y_true, y_prob = [], []
    for x, y in dataset:
        preds = model.predict(x, verbose=0).ravel()
        y_prob.extend(preds.tolist())
        y_true.extend(y.numpy().tolist())
    return np.array(y_true), np.array(y_prob)


def _auc(m, ds):
    yt, yp = collect_predictions(m, ds)
    return roc_auc_score(yt, yp)


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in BACKBONE_REG:
        sys.exit(f"usage: python backbone_study.py <{'|'.join(BACKBONE_REG)}> [raw|masked]")
    bb_name = sys.argv[1]
    conditions = [sys.argv[2]] if len(sys.argv) > 2 else ["raw", "masked"]

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = prepare_data_splits()
    verify_split_against_mask_cache(X_train, X_val, X_test)
    X_mont, y_mont = load_nlm_dataset(find_image_dir(MONTGOMERY_DIR), "MCUCXR")
    X_shenz, y_shenz = load_nlm_dataset(find_image_dir(SHENZHEN_DIR), "CHNCXR")

    # masked-path resolver (cell 61)
    _mask_index = {}
    for _sub in ["kaggle_train", "kaggle_val", "kaggle_test", "montgomery", "shenzhen"]:
        _d = MASK_DIR / _sub
        if _d.exists():
            for _f in _d.glob("*.png"):
                _mask_index[_f.name] = str(_f)

    def _to_masked(paths):
        out = []
        for p in paths:
            nm = pathlib.Path(p).name
            if nm not in _mask_index:
                raise FileNotFoundError(f"No cached mask for {nm}; run notebook Section 14 first.")
            out.append(_mask_index[nm])
        return out

    results = json.load(open(RESULTS_PATH)) if os.path.exists(RESULTS_PATH) else {}

    cv_paths = np.array(X_train + X_val + X_test)
    cv_labels = np.array(y_train + y_val + y_test)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    folds = list(skf.split(cv_paths, cv_labels))

    _, prep = BACKBONE_REG[bb_name]
    for cond in conditions:
        for fold, (tr_idx, te_idx) in enumerate(folds):
            key = f"{bb_name}|{cond}|{fold}"
            if key in results:
                print(f"skip {key} (cached)")
                continue
            print(f"\n==== {key} ====", flush=True)
            Xtr, Xte = list(cv_paths[tr_idx]), list(cv_paths[te_idx])
            ytr, yte = list(cv_labels[tr_idx]), list(cv_labels[te_idx])
            Xtr, Xv, ytr, yv = train_test_split(Xtr, ytr, test_size=0.15, stratify=ytr,
                                                random_state=RANDOM_SEED)
            ext_m, ext_s = list(X_mont), list(X_shenz)
            if cond == "masked":
                Xtr, Xv, Xte = _to_masked(Xtr), _to_masked(Xv), _to_masked(Xte)
                ext_m, ext_s = _to_masked(ext_m), _to_masked(ext_s)

            tf.keras.utils.set_random_seed(RANDOM_SEED + fold)
            tf.keras.backend.clear_session()
            tr_ds = build_ds_bb(Xtr, ytr, prep, train=True)
            va_ds = build_ds_bb(Xv, yv, prep, train=False)
            model_f, bb_f = build_backbone_model(bb_name)
            model_f, _, _ = train_model(model_f, bb_f, tr_ds, va_ds)

            res = {
                "in": _auc(model_f, build_ds_bb(Xte, yte, prep)),
                "mont": _auc(model_f, build_ds_bb(ext_m, y_mont, prep)),
                "shen": _auc(model_f, build_ds_bb(ext_s, y_shenz, prep)),
            }
            # re-read so parallel/prior invocations are never clobbered
            results = json.load(open(RESULTS_PATH)) if os.path.exists(RESULTS_PATH) else {}
            results[key] = res
            json.dump(results, open(RESULTS_PATH, "w"), indent=2)
            print(f"  {key}: in {res['in']:.3f} | Mont {res['mont']:.3f} | Shen {res['shen']:.3f}", flush=True)
            del model_f, bb_f
            tf.keras.backend.clear_session()
            gc.collect()

    done = sum(1 for k in results if k.startswith(bb_name))
    print(f"\n{bb_name} complete: {done} fold-conditions cached in {RESULTS_PATH.name}.")


if __name__ == "__main__":
    main()
