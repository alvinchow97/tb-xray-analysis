"""Generate the five publication figures for the paper into figures/.

Runs CPU-only so it can execute alongside the GPU backbone-study training.
Fig 5 uses whatever folds exist in backbone_results.json — re-run after the
backbone study completes to include EfficientNet-B0.
"""
import json
import pathlib

import numpy as np
import tensorflow as tf

tf.config.set_visible_devices([], "GPU")  # leave the GPU to the training run

import cv2
import glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import auc, roc_curve
from sklearn.model_selection import train_test_split

ROOT = pathlib.Path("/Users/alvinchow/analysis")
DATA = ROOT / "data"
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)
IMG_SIZE = 224
SEED = 42

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "legend.fontsize": 8, "figure.dpi": 100, "savefig.dpi": 300,
    "axes.spines.top": False, "axes.spines.right": False,
})
# Okabe–Ito colourblind-safe palette
C_IN, C_MONT, C_SHEN = "#0072B2", "#D55E00", "#009E73"


def save(fig, name):
    fig.savefig(FIG / f"{name}.png", bbox_inches="tight")
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"saved figures/{name}.png (+pdf)", flush=True)


# ── Data splits (identical to notebook cell 10) ──────────────────────────────
base_dir = DATA / "kaggle/TB_Chest_Radiography_Database"
normal_paths = glob.glob(str(base_dir / "Normal" / "*.*"))
tb_paths = glob.glob(str(base_dir / "Tuberculosis" / "*.*"))
all_paths = normal_paths + tb_paths
all_labels = [0] * len(normal_paths) + [1] * len(tb_paths)
X_train, X_temp, y_train, y_temp = train_test_split(
    all_paths, all_labels, test_size=0.2, stratify=all_labels, random_state=SEED)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=SEED)


def load_nlm(d):
    files = sorted(p for p in pathlib.Path(d).glob("*.png") if not p.name.startswith("._"))
    files = [p for p in files if p.stem.rsplit("_", 1)[-1] in ("0", "1")]
    return [str(p) for p in files], [int(p.stem.rsplit("_", 1)[-1]) for p in files]

X_mont, y_mont = load_nlm(DATA / "montgomery/MontgomerySet/CXR_png")
X_shen, y_shen = load_nlm(DATA / "shenzhen/ChinaSet_AllFiles/CXR_png")


def build_ds(paths, labels):
    def _load(p, l):
        im = tf.io.read_file(p)
        im = tf.image.decode_image(im, channels=3, expand_animations=False)
        im = tf.image.resize(im, (IMG_SIZE, IMG_SIZE))
        im = tf.keras.applications.mobilenet_v2.preprocess_input(tf.cast(im, tf.float32))
        return im, l
    return (tf.data.Dataset.from_tensor_slices((paths, labels))
            .map(_load, num_parallel_calls=tf.data.AUTOTUNE).batch(32).prefetch(tf.data.AUTOTUNE))


print("Loading baseline model ...", flush=True)
model = tf.keras.models.load_model(ROOT / "tb_detector.keras", compile=False)


def predict_probs(paths, labels):
    y_true, y_prob = [], []
    for x, y in build_ds(paths, labels):
        y_prob.extend(model.predict(x, verbose=0).ravel().tolist())
        y_true.extend(y.numpy().tolist())
    return np.array(y_true), np.array(y_prob)


# ── Figure 2: ROC curves, in-domain vs external ──────────────────────────────
print("Fig 2: predictions on test / Montgomery / Shenzhen ...", flush=True)
yt_in, yp_in = predict_probs(X_test, y_test)
yt_m, yp_m = predict_probs(X_mont, y_mont)
yt_s, yp_s = predict_probs(X_shen, y_shen)

fig, ax = plt.subplots(figsize=(4.2, 4.2))
for yt, yp, label, color in [
    (yt_in, yp_in, "In-domain test", C_IN),
    (yt_s, yp_s, "Shenzhen", C_SHEN),
    (yt_m, yp_m, "Montgomery", C_MONT),
]:
    fpr, tpr, _ = roc_curve(yt, yp)
    ax.plot(fpr, tpr, color=color, lw=1.8, label=f"{label} (AUC {auc(fpr, tpr):.3f})")
ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5)
ax.set(xlabel="False positive rate", ylabel="True positive rate (sensitivity)",
       xlim=(0, 1), ylim=(0, 1.02))
ax.legend(loc="lower right")
ax.set_aspect("equal")
save(fig, "fig2_roc")

# ── Figure 3: reliability diagrams before/after temperature scaling ──────────
print("Fig 3: calibration ...", flush=True)
feature_model = tf.keras.Model(model.input, model.layers[-2].output)
_W, _b = model.layers[-1].get_weights()

def collect_logits(paths, labels):
    y_true, logits = [], []
    for x, y in build_ds(paths, labels):
        feats = feature_model(x, training=False).numpy()
        logits.extend((feats @ _W + _b).ravel().tolist())
        y_true.extend(y.numpy().tolist())
    return np.array(y_true), np.array(logits)

yv_true, yv_logit = collect_logits(X_val, y_val)
yt_true, yt_logit = collect_logits(X_test, y_test)


def expected_calibration_error(y_true, y_prob, n_bins=15):
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_prob, bins) - 1
    ece = 0.0
    for i in range(n_bins):
        mask = bin_ids == i
        if mask.any():
            acc = np.mean((y_prob[mask] >= 0.5) == y_true[mask])
            conf = np.mean(y_prob[mask])
            ece += np.abs(acc - conf) * mask.mean()
    return ece


def fit_temperature(logits, labels, epochs=300, lr=0.05):
    logits_t = tf.constant(logits, dtype=tf.float32)
    labels_t = tf.cast(labels, tf.float32)
    log_T = tf.Variable(0.0, dtype=tf.float32)
    opt = tf.keras.optimizers.Adam(lr)
    for _ in range(epochs):
        with tf.GradientTape() as tape:
            scaled = logits_t / tf.exp(log_T)
            loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(labels=labels_t, logits=scaled))
        opt.apply_gradients(zip(tape.gradient(loss, [log_T]), [log_T]))
    return float(tf.exp(log_T).numpy())

T = fit_temperature(yv_logit, yv_true)
p_before = tf.sigmoid(yt_logit).numpy()
p_after = tf.sigmoid(yt_logit / T).numpy()
ece_b = expected_calibration_error(yt_true, p_before)
ece_a = expected_calibration_error(yt_true, p_after)
print(f"  T={T:.3f}  ECE before={ece_b:.3f} after={ece_a:.3f}", flush=True)


def plot_reliability(ax, y_true, y_prob, title, n_bins=15):
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    centers = (bins[:-1] + bins[1:]) / 2
    accs, confs = [], []
    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
        accs.append(np.mean((y_prob[mask] >= 0.5) == y_true[mask]) if mask.any() else 0)
        confs.append(np.mean(y_prob[mask]) if mask.any() else centers[i])
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5, label="Perfect calibration")
    ax.bar(centers, accs, width=0.06, alpha=0.65, color=C_IN, label="Accuracy per bin")
    ax.plot(centers, confs, "o-", color=C_MONT, ms=3, lw=1.2, label="Mean confidence")
    ax.set(xlim=(0, 1), ylim=(0, 1.02), xlabel="Predicted probability", title=title)

fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.7), sharey=True)
plot_reliability(axes[0], yt_true, p_before, f"Before scaling (ECE {ece_b:.3f})")
plot_reliability(axes[1], yt_true, p_after, f"After scaling, T={T:.2f} (ECE {ece_a:.3f})")
axes[0].set_ylabel("Accuracy")
axes[0].legend(loc="upper left")
save(fig, "fig3_reliability")

# ── Figure 1: Grad-CAM, in-domain vs external ────────────────────────────────
print("Fig 1: Grad-CAM ...", flush=True)
from tensorflow.keras.layers import Conv2D, DepthwiseConv2D


def make_gradcam_heatmap(img_tensor, model):
    backbone = model.get_layer("mobilenetv2_1.00_224")
    last_conv_layer = next(l for l in reversed(backbone.layers)
                           if isinstance(l, (Conv2D, DepthwiseConv2D)))
    backbone_probe = tf.keras.Model(backbone.input, [last_conv_layer.output, backbone.output])
    gap = model.get_layer("global_average_pooling2d")
    dropout = model.get_layer("dropout")
    dense = model.get_layer("dense")
    with tf.GradientTape() as tape:
        conv_maps, backbone_out = backbone_probe(img_tensor, training=False)
        tape.watch(conv_maps)
        x = gap(backbone_out)
        x = dropout(x, training=False)
        class_score = dense(x)[:, 0]
    grads = tape.gradient(class_score, conv_maps)
    pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
    hm = tf.squeeze(conv_maps[0] @ pooled[..., tf.newaxis])
    hm = tf.maximum(hm, 0)
    return (hm / (tf.reduce_max(hm) + 1e-8)).numpy()


def gradcam_row(img_path):
    img = Image.open(img_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    img_rgb = np.array(img)
    t = tf.keras.applications.mobilenet_v2.preprocess_input(tf.cast(img_rgb, tf.float32))
    t = tf.expand_dims(t, 0)
    score = float(model.predict(t, verbose=0)[0][0])
    hm = cv2.resize(make_gradcam_heatmap(t, model), (IMG_SIZE, IMG_SIZE))
    hm_col = cv2.applyColorMap(np.uint8(255 * hm), cv2.COLORMAP_JET)
    hm_col = cv2.cvtColor(hm_col, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(img_rgb, 0.6, hm_col, 0.4, 0)
    return img_rgb, hm, overlay, score

# deterministic samples: first in-domain test TB, first Shenzhen TB
in_tb = X_test[next(i for i, l in enumerate(y_test) if l == 1)]
ext_tb = X_shen[next(i for i, l in enumerate(y_shen) if l == 1)]

fig, axes = plt.subplots(2, 3, figsize=(7.2, 5.0))
for row, (path, tag) in enumerate([(in_tb, "In-domain TB (Kaggle test)"),
                                   (ext_tb, "External TB (Shenzhen)")]):
    img_rgb, hm, overlay, score = gradcam_row(path)
    axes[row, 0].imshow(img_rgb)
    axes[row, 0].set_title(f"{tag}", loc="left")
    axes[row, 1].imshow(hm, cmap="jet")
    axes[row, 1].set_title("Grad-CAM")
    axes[row, 2].imshow(overlay)
    axes[row, 2].set_title(f"Overlay (score {score:.2f})")
    for ax in axes[row]:
        ax.axis("off")
save(fig, "fig1_gradcam")

# ── Figure 4: segmentation example ───────────────────────────────────────────
print("Fig 4: segmentation example ...", flush=True)
segmenter = tf.keras.models.load_model(ROOT / "tb_lung_unet.keras", compile=False)
ex_path = X_mont[0]
g = cv2.resize(cv2.imread(ex_path, cv2.IMREAD_GRAYSCALE), (IMG_SIZE, IMG_SIZE))
mask = segmenter.predict((g / 255.0)[None, ..., None].astype("float32"), verbose=0)[0, ..., 0] > 0.5
masked = (g * mask).astype("uint8")

fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7))
for ax, im, title in zip(axes, [g, mask.astype(float), masked],
                         ["Chest radiograph", "Predicted lung mask", "Masked input"]):
    ax.imshow(im, cmap="gray")
    ax.set_title(title)
    ax.axis("off")
save(fig, "fig4_segmentation")

# ── Figure 5: backbone × cohort, fine-tuned + frozen probe ───────────────────
print("Fig 5: backbone study ...", flush=True)
bb_res = json.load(open(ROOT / "backbone_results.json"))
probe = json.load(open(ROOT / "probe_results.json"))

BB_LABEL = {"mobilenetv2": "MobileNetV2", "densenet121": "DenseNet-121",
            "efficientnetb0": "EfficientNet-B0"}
BB_COLOR = {"mobilenetv2": C_IN, "densenet121": C_MONT, "efficientnetb0": C_SHEN}
COHORTS = [("in", "test", "In-domain test"), ("mont", "montgomery", "Montgomery"),
           ("shen", "shenzhen", "Shenzhen")]
backbones = [b for b in BB_LABEL
             if sum(1 for k in bb_res if k.startswith(f"{b}|raw|")) == 5]

fig, ax = plt.subplots(figsize=(7.0, 3.6))
n_bar = 2 * len(backbones)          # fine-tuned + frozen per backbone
width = 0.8 / n_bar
for gi, (bb_key, probe_key, coh_label) in enumerate(COHORTS):
    for bi, bb in enumerate(backbones):
        vals = np.array([bb_res[f"{bb}|raw|{f}"][bb_key] for f in range(5)])
        x_ft = gi - 0.4 + (2 * bi + 0.5) * width
        x_fz = gi - 0.4 + (2 * bi + 1.5) * width
        ax.bar(x_ft, vals.mean(), width * 0.92, yerr=vals.std(ddof=1), capsize=2,
               color=BB_COLOR[bb], label=f"{BB_LABEL[bb]} (fine-tuned)" if gi == 0 else None)
        if bb in probe:
            ax.bar(x_fz, probe[bb][probe_key]["auc"], width * 0.92, color=BB_COLOR[bb],
                   alpha=0.45, hatch="///", edgecolor="white",
                   label=f"{BB_LABEL[bb]} (frozen probe)" if gi == 0 else None)
ax.axhline(0.5, color="k", ls=":", lw=0.9)
ax.text(2.42, 0.505, "chance", fontsize=7.5, va="bottom", ha="right")
ax.set_xticks(range(len(COHORTS)))
ax.set_xticklabels([c[2] for c in COHORTS])
ax.set_ylabel("ROC-AUC")
ax.set_ylim(0.4, 1.02)
ax.legend(ncol=2, loc="lower left", framealpha=0.9)
note = "" if "efficientnetb0" in backbones else "  (EfficientNet-B0 pending)"
ax.set_title(f"External generalisation by backbone{note}", fontsize=9)
save(fig, "fig5_backbones")

print("\nAll figures written to figures/", flush=True)
