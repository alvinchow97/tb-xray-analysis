"""Probe: why does DenseNet121 collapse to AUC~0.5 on Shenzhen?

Extracts frozen ImageNet GAP features from MobileNetV2 and DenseNet121 for
Kaggle (train/test split, seed 42 as in the notebook), Montgomery and Shenzhen,
fits a logistic-regression head on Kaggle train features, and reports AUC and
score distributions per cohort. Also reports the AUC of a trivial
mean-pixel-intensity classifier as a low-level-shortcut check.
"""
import json
import pathlib
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

SEED = 42
IMG_SIZE = 224
BATCH = 32
DATA = pathlib.Path("/Users/alvinchow/analysis/data")
OUT = pathlib.Path("/Users/alvinchow/analysis/probe_results.json")

kaggle = DATA / "kaggle/TB_Chest_Radiography_Database"
normal = sorted(str(p) for p in (kaggle / "Normal").glob("*.png"))
tb = sorted(str(p) for p in (kaggle / "Tuberculosis").glob("*.png"))
X = normal + tb
y = [0] * len(normal) + [1] * len(tb)
X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
X_val, X_te, y_val, y_te = train_test_split(X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=SEED)

def load_nlm(d):
    files = sorted(p for p in d.glob("*.png") if not p.name.startswith("."))
    return [str(p) for p in files], [int(p.stem[-1]) for p in files]

X_mont, y_mont = load_nlm(DATA / "montgomery/MontgomerySet/CXR_png")
X_shen, y_shen = load_nlm(DATA / "shenzhen/ChinaSet_AllFiles/CXR_png")

BACKBONES = {
    "mobilenetv2": (tf.keras.applications.MobileNetV2, tf.keras.applications.mobilenet_v2.preprocess_input),
    "densenet121": (tf.keras.applications.DenseNet121, tf.keras.applications.densenet.preprocess_input),
}

def make_ds(paths, prep):
    def _load(p):
        im = tf.io.read_file(p)
        im = tf.image.decode_image(im, channels=3, expand_animations=False)
        im = tf.image.resize(im, (IMG_SIZE, IMG_SIZE))
        return prep(tf.cast(im, tf.float32))
    return (tf.data.Dataset.from_tensor_slices(paths)
            .map(_load, num_parallel_calls=tf.data.AUTOTUNE)
            .batch(BATCH).prefetch(tf.data.AUTOTUNE))

results = {}

# Low-level shortcut check: mean pixel intensity as the only feature
def mean_intensity(paths):
    ds = (tf.data.Dataset.from_tensor_slices(paths)
          .map(lambda p: tf.reduce_mean(tf.cast(tf.image.decode_image(
              tf.io.read_file(p), channels=3, expand_animations=False), tf.float32)),
               num_parallel_calls=tf.data.AUTOTUNE)
          .batch(64))
    return np.concatenate([b.numpy() for b in ds])

print("== mean-intensity classifier ==", flush=True)
mi = {}
for name, paths, labels in [("kaggle_test", X_te, y_te), ("montgomery", X_mont, y_mont), ("shenzhen", X_shen, y_shen)]:
    m = mean_intensity(paths)
    mi[name] = roc_auc_score(labels, m)
    print(f"  {name}: AUC(mean intensity) = {mi[name]:.3f}", flush=True)
results["mean_intensity_auc"] = mi

for bb_name, (ctor, prep) in BACKBONES.items():
    print(f"\n== {bb_name}: extracting frozen GAP features ==", flush=True)
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(SEED)
    bb = ctor(input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights="imagenet", pooling="avg")
    feats = {}
    for name, paths in [("train", X_tr), ("test", X_te), ("montgomery", X_mont), ("shenzhen", X_shen)]:
        feats[name] = bb.predict(make_ds(paths, prep), verbose=0)
        print(f"  {name}: {feats[name].shape}", flush=True)

    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(feats["train"], y_tr)
    res = {}
    for name, labels in [("test", y_te), ("montgomery", y_mont), ("shenzhen", y_shen)]:
        p = clf.predict_proba(feats[name])[:, 1]
        res[name] = {
            "auc": float(roc_auc_score(labels, p)),
            "score_pcts": [float(np.percentile(p, q)) for q in (1, 25, 50, 75, 99)],
        }
        print(f"  {name}: AUC={res[name]['auc']:.4f}  score pcts(1/25/50/75/99)="
              f"{['%.3f' % v for v in res[name]['score_pcts']]}", flush=True)
    results[bb_name] = res
    np.savez_compressed(f"/Users/alvinchow/analysis/probe_feats_{bb_name}.npz",
                        shen=feats["shenzhen"], mont=feats["montgomery"], y_shen=y_shen, y_mont=y_mont)

OUT.write_text(json.dumps(results, indent=2))
print("\nSaved:", OUT, flush=True)
