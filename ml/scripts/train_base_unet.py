#!/usr/bin/env python3
"""
train_base_unet.py — Pre-train a base DDPM U-Net on synthetic B&W 32×32 images.

Architecture mirrors the browser model (src/core/trainer.ts) exactly:
  Filters 16 / 32 / 64, cosine beta schedule T=400, Adam(lr=0.001).

The pretrained weights are loaded by the browser as a starting point, so
fine-tuning the user's own drawn smileys only needs ~50–100 steps instead
of training from scratch.

Training data (no external downloads):
  35 % face-like shapes   ← most relevant for smiley task
  20 % circles / ellipses
  20 % rectangles / squares
  15 % lines / arcs
  10 % random blobs / dot clusters

Usage:
  python ml/scripts/train_base_unet.py              # full run (~5 min CPU)
  python ml/scripts/train_base_unet.py --steps 500  # quick smoke-test
  python ml/scripts/train_base_unet.py --no-save    # skip saving
"""

import argparse
import os
import time

import numpy as np
import json
import shutil

import tensorflow as tf
from tensorflow import keras


# ── Keras 3 → TF.js model.json compatibility patch ──────────────────────────
def _fix_model_json_for_tfjs(path: str) -> None:
    """
    Keras 3.x saves model.json in a format that tf.loadLayersModel() cannot
    parse.  This function patches the three incompatibilities in-place:

      1. dtype: DTypePolicy objects  →  plain "float32" strings
      2. InputLayer config: batch_shape  →  batch_input_shape  (+ drops unknown keys)
      3. inbound_nodes: Keras-3 args/kwargs dicts  →  Keras-2 [[name,ni,ti,{}]] lists
      4. output_layers: flat ["name",0,0]  →  nested [["name",0,0]]
      5. Initializer dicts: strip Keras-3-only fields (module, registered_name)
    """
    import json, shutil

    def _dtype(v):
        if isinstance(v, dict) and v.get("class_name") == "DTypePolicy":
            return v["config"]["name"]
        if isinstance(v, dict):
            return {k: _dtype(vv) for k, vv in v.items()}
        if isinstance(v, list):
            return [_dtype(i) for i in v]
        return v

    def _init(d):
        if not isinstance(d, dict):
            return d
        out = {}
        if "class_name" in d:
            out["class_name"] = d["class_name"]
        if "config" in d:
            out["config"] = _init(d["config"])
        return out

    def _tensor_history(t):
        if isinstance(t, dict) and t.get("class_name") == "__keras_tensor__":
            h = t["config"]["keras_history"]
            return h[0], h[1], h[2]
        return None

    def _inbound(nodes):
        out = []
        for node in nodes:
            if not isinstance(node, dict) or "args" not in node:
                out.append(node); continue
            first = node["args"][0] if node["args"] else None
            if first is None:
                continue
            if isinstance(first, list):          # Concatenate / multi-input
                inner = []
                for t in first:
                    h = _tensor_history(t)
                    if h:
                        inner.append([h[0], h[1], h[2], {}])
                out.append(inner)
            else:                                # single-input layer
                h = _tensor_history(first)
                if h:
                    out.append([[h[0], h[1], h[2], {}]])
        return out

    INIT_KEYS = ["kernel_initializer", "bias_initializer", "kernel_regularizer",
                 "bias_regularizer", "activity_regularizer",
                 "kernel_constraint",  "bias_constraint"]

    def _layer(l):
        l = dict(l)
        if "inbound_nodes" in l:
            l["inbound_nodes"] = _inbound(l["inbound_nodes"])
        if "config" in l:
            cfg = dict(l["config"])
            cfg = {k: _dtype(v) if k == "dtype" else v for k, v in cfg.items()}
            if l.get("class_name") == "InputLayer" and "batch_shape" in cfg:
                cfg["batch_input_shape"] = cfg.pop("batch_shape")
            for k in ("optional", "ragged"):
                cfg.pop(k, None)
            for k in INIT_KEYS:
                if k in cfg and cfg[k] is not None:
                    cfg[k] = _init(cfg[k])
            l["config"] = cfg
        return l

    shutil.copy(path, path + ".keras3_bak")
    with open(path) as f:
        data = json.load(f)

    mc_cfg = data["modelTopology"]["model_config"]["config"]
    mc_cfg["layers"] = [_layer(l) for l in mc_cfg["layers"]]
    ol = mc_cfg.get("output_layers", [])
    if ol and isinstance(ol[0], str):       # flat ["name",0,0] → [["name",0,0]]
        mc_cfg["output_layers"] = [ol]

    with open(path, "w") as f:
        json.dump(data, f, separators=(",", ":"))
    print(f"✅  model.json patched for TF.js (Keras 3 → Keras 2 format)  → {path}")


# ── Config (must stay in sync with src/core/config.ts) ──────────────────────
IMAGE_SIZE    = 32
T             = 400         # diffusion timesteps
BETA_START    = 0.0001
BETA_END      = 0.02
LEARNING_RATE = 0.001
BATCH_SIZE    = 64          # larger batch for faster pretraining
TOTAL_STEPS   = 5000        # ≈ 5 min CPU / 30 s GPU
LOG_EVERY     = 200
SAVE_DIR      = "ml/models"
TFJS_OUT_DIR  = "public/model"
MODEL_NAME    = "base_unet"

# ── Cosine beta schedule (matches diffusion.ts) ──────────────────────────────
def _make_schedule():
    t          = np.arange(T) / max(T - 1, 1)
    cosine     = (1.0 - np.cos(t * np.pi)) / 2.0
    betas      = (BETA_START + (BETA_END - BETA_START) * cosine).astype(np.float32)
    alphas     = (1.0 - betas).astype(np.float32)
    alpha_bars = np.cumprod(alphas).astype(np.float32)
    return alpha_bars

ALPHA_BARS         = _make_schedule()          # [T]
SQRT_AB            = np.sqrt(ALPHA_BARS)        # [T]
SQRT_ONE_MINUS_AB  = np.sqrt(1.0 - ALPHA_BARS) # [T]

# ── Synthetic training data ───────────────────────────────────────────────────
S = IMAGE_SIZE

def _blank():
    return np.full((S, S), -0.9, dtype=np.float32)

def _circle(filled=False):
    img = _blank()
    cx  = np.random.uniform(7, S - 7)
    cy  = np.random.uniform(7, S - 7)
    r   = np.random.uniform(5, 12)
    Y, X = np.mgrid[0:S, 0:S].astype(np.float32)
    d   = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    img[d < r if filled else np.abs(d - r) < 1.5] = 0.9
    return img

def _ellipse(filled=False):
    img = _blank()
    cx  = np.random.uniform(7, S - 7)
    cy  = np.random.uniform(7, S - 7)
    rx  = np.random.uniform(4, 12)
    ry  = np.random.uniform(3, 10)
    Y, X = np.mgrid[0:S, 0:S].astype(np.float32)
    d   = ((X - cx) / rx) ** 2 + ((Y - cy) / ry) ** 2
    img[(d < 1.0) if filled else (np.abs(d - 1.0) < 0.22)] = 0.9
    return img

def _rect(filled=False):
    img = _blank()
    x1, y1 = int(np.random.uniform(3, S // 2)), int(np.random.uniform(3, S // 2))
    x2, y2 = int(np.random.uniform(S // 2, S - 3)), int(np.random.uniform(S // 2, S - 3))
    if filled:
        img[y1:y2, x1:x2] = 0.9
    else:
        img[y1:y2, x1]  = 0.9
        img[y1:y2, x2]  = 0.9
        img[y1, x1:x2]  = 0.9
        img[y2, x1:x2]  = 0.9
    return img

def _line():
    img   = _blank()
    angle = np.random.uniform(0, np.pi)
    cx    = S / 2 + np.random.uniform(-4, 4)
    cy    = S / 2 + np.random.uniform(-4, 4)
    L     = np.random.uniform(10, 22)
    for s in np.linspace(-L / 2, L / 2, 100):
        x, y = int(round(cx + s * np.cos(angle))), int(round(cy + s * np.sin(angle)))
        if 0 <= x < S and 0 <= y < S:
            img[y, x] = 0.9
    return img

def _arc():
    img     = _blank()
    cx      = np.random.uniform(7, S - 7)
    cy      = np.random.uniform(7, S - 7)
    r       = np.random.uniform(5, 12)
    a_start = np.random.uniform(0, np.pi)
    a_end   = a_start + np.random.uniform(np.pi * 0.4, np.pi * 1.8)
    for a in np.linspace(a_start, a_end, 150):
        x, y = int(round(cx + r * np.cos(a))), int(round(cy + r * np.sin(a)))
        if 0 <= x < S and 0 <= y < S:
            img[y, x] = 0.9
    return img

def _dots():
    img = _blank()
    n   = np.random.randint(5, 20)
    for _ in range(n):
        x, y = np.random.randint(2, S - 2), np.random.randint(2, S - 2)
        r    = np.random.uniform(0.8, 2.5)
        Y, X = np.mgrid[0:S, 0:S].astype(np.float32)
        img[np.sqrt((X - x) ** 2 + (Y - y) ** 2) < r] = 0.9
    return img

def _face():
    """Face-like shape: outline circle + eyes + mouth (smile / flat / frown)."""
    img = _blank()
    cx  = S / 2 + np.random.uniform(-3, 3)
    cy  = S / 2 + np.random.uniform(-3, 3)
    r   = np.random.uniform(9, 13)
    Y, X = np.mgrid[0:S, 0:S].astype(np.float32)

    # Face outline
    d = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    img[np.abs(d - r) < 1.5] = 0.9

    # Eyes
    eye_r = np.random.uniform(1.0, 2.2)
    dx    = r * np.random.uniform(0.28, 0.40)
    dy    = r * np.random.uniform(0.15, 0.28)
    for ex in [cx - dx, cx + dx]:
        ey  = cy - dy
        ed  = np.sqrt((X - ex) ** 2 + (Y - ey) ** 2)
        img[ed < eye_r] = 0.9

    # Mouth
    expr   = np.random.choice(["smile", "flat", "frown"], p=[0.5, 0.25, 0.25])
    mx_r   = r * np.random.uniform(0.35, 0.50)
    my_off = r * np.random.uniform(0.28, 0.45)

    if expr == "smile":
        for a in np.linspace(0.15 * np.pi, 0.85 * np.pi, 50):
            px = cx + mx_r * np.cos(a)
            py = cy + my_off + mx_r * 0.30 * np.sin(a) - mx_r * 0.12
            xi, yi = int(round(px)), int(round(py))
            if 0 <= xi < S and 0 <= yi < S:
                img[yi, xi] = 0.9
    elif expr == "frown":
        for a in np.linspace(0.15 * np.pi, 0.85 * np.pi, 50):
            px = cx + mx_r * np.cos(a)
            py = cy + my_off - mx_r * 0.30 * np.sin(a) + mx_r * 0.12
            xi, yi = int(round(px)), int(round(py))
            if 0 <= xi < S and 0 <= yi < S:
                img[yi, xi] = 0.9
    else:  # flat line
        y0 = int(round(cy + my_off))
        if 0 <= y0 < S:
            img[y0, max(0, int(cx - mx_r)):min(S, int(cx + mx_r))] = 0.9

    return img

# Shape distribution: (generator_fn, weight)
_SHAPES = [
    (_face,                    35),  # face-like → most relevant for the smiley task
    (lambda: _circle(False),   10),
    (lambda: _circle(True),    10),
    (lambda: _ellipse(False),   8),
    (lambda: _ellipse(True),    7),
    (lambda: _rect(False),      8),
    (lambda: _rect(True),       7),
    (_line,                     8),
    (_arc,                      4),
    (_dots,                     3),
]
_WEIGHTS = np.array([w for _, w in _SHAPES], dtype=np.float64)
_WEIGHTS /= _WEIGHTS.sum()


def generate_batch(batch_size: int) -> np.ndarray:
    """Return (batch_size, S, S, 1) float32 array in [-1, 1]."""
    imgs = []
    for _ in range(batch_size):
        idx = int(np.random.choice(len(_SHAPES), p=_WEIGHTS))
        imgs.append(_SHAPES[idx][0]()[..., np.newaxis])
    return np.stack(imgs, axis=0)


# ── Sinusoidal time embedding (must mirror trainer.ts sinEmb) ─────────────────
TIME_DIM = 16   # 8 sin + 8 cos, must equal CONFIG.timeDim in config.ts

def sin_emb_batch(t_idx: np.ndarray, T: int) -> np.ndarray:
    """
    t_idx : int array [B], 1-based timesteps
    Returns float32 [B, TIME_DIM]
    """
    t_norm = t_idx.astype(np.float32) / T          # [B], in (0, 1]
    scaled = t_norm * 1000.0                        # stretch [0,1] → [0,1000]
    emb    = np.zeros((len(t_idx), TIME_DIM), dtype=np.float32)
    for i in range(TIME_DIM // 2):
        freq = 1.0 / (10000.0 ** (2 * i / TIME_DIM))
        emb[:, 2 * i]     = np.sin(scaled * freq)
        emb[:, 2 * i + 1] = np.cos(scaled * freq)
    return emb                                      # [B, TIME_DIM]


# ── Model (must mirror src/core/trainer.ts createModel) ──────────────────────
def build_model(S: int = IMAGE_SIZE) -> keras.Model:
    """
    U-Net with sinusoidal time-embedding input and two-point time conditioning:
      1. INPUT     : Dense(TD→S²) → Reshape(S,S,1) → concat with image
      2. BOTTLENECK: Dense(TD→64,relu) → Reshape(1,1,64) → add to bottleneck
    Must produce identical layer *names and shapes* to the TF.js build so
    converted weights are compatible with tf.loadLayersModel() in the browser.
    """
    input_image = keras.Input(shape=(S, S, 1),    name="input_image")
    input_time  = keras.Input(shape=(TIME_DIM,),  name="input_time")

    # 1. Input-level time conditioning (no relu — sin/cos spans [-1,1])
    te  = keras.layers.Dense(S * S, name="time_dense")(input_time)
    tm  = keras.layers.Reshape((S, S, 1), name="time_reshape")(te)
    x   = keras.layers.Concatenate(name="cat_input")([input_image, tm])

    # Encoder
    e1  = keras.layers.Conv2D(16, 3, padding="same", activation="relu",
                               kernel_initializer="he_normal", name="enc1a")(x)
    e1b = keras.layers.Conv2D(16, 3, padding="same", activation="relu",
                               kernel_initializer="he_normal", name="enc1b")(e1)
    p1  = keras.layers.MaxPooling2D(2, name="pool1")(e1b)

    e2  = keras.layers.Conv2D(32, 3, padding="same", activation="relu",
                               kernel_initializer="he_normal", name="enc2a")(p1)
    e2b = keras.layers.Conv2D(32, 3, padding="same", activation="relu",
                               kernel_initializer="he_normal", name="enc2b")(e2)
    p2  = keras.layers.MaxPooling2D(2, name="pool2")(e2b)

    # Bottleneck
    b   = keras.layers.Conv2D(64, 3, padding="same", activation="relu",
                               kernel_initializer="he_normal", name="bot_a")(p2)
    bb  = keras.layers.Conv2D(64, 3, padding="same", activation="relu",
                               kernel_initializer="he_normal", name="bot_b")(b)

    # 2. Bottleneck-level time conditioning (additive bias, broadcast over spatial)
    tbot = keras.layers.Dense(64, activation="relu", name="time_bot")(input_time)
    tbot = keras.layers.Reshape((1, 1, 64), name="time_bot_reshape")(tbot)
    bb   = keras.layers.Add(name="bot_time_add")([bb, tbot])

    # Decoder
    u2   = keras.layers.UpSampling2D(2, name="up2")(bb)
    cat2 = keras.layers.Concatenate(name="cat2")([u2, e2b])
    d2   = keras.layers.Conv2D(32, 3, padding="same", activation="relu",
                                kernel_initializer="he_normal", name="dec2a")(cat2)
    d2b  = keras.layers.Conv2D(32, 3, padding="same", activation="relu",
                                kernel_initializer="he_normal", name="dec2b")(d2)

    u1   = keras.layers.UpSampling2D(2, name="up1")(d2b)
    cat1 = keras.layers.Concatenate(name="cat1")([u1, e1b])
    d1   = keras.layers.Conv2D(16, 3, padding="same", activation="relu",
                                kernel_initializer="he_normal", name="dec1")(cat1)

    # Output: zero-initialised so starting predictions ≈ 0 (neutral start)
    out  = keras.layers.Conv2D(1, 1, padding="same",
                                kernel_initializer="zeros", name="output")(d1)

    return keras.Model(inputs=[input_image, input_time], outputs=out,
                       name="base_unet")


# ── Custom training step (faster than model.fit per-call) ────────────────────
_SQRT_AB_TF   = tf.constant(SQRT_AB)
_SQRT_OMAB_TF = tf.constant(SQRT_ONE_MINUS_AB)

@tf.function
def train_step(model, optimizer, x0: tf.Tensor, t_emb: tf.Tensor,
               t_idx: tf.Tensor) -> tf.Tensor:
    """
    x0:    [B, S, S, 1]    clean images
    t_emb: [B, TIME_DIM]   sinusoidal time embeddings
    t_idx: [B]             integer timesteps (1-based, for noise schedule lookup)
    """
    eps       = tf.random.normal(tf.shape(x0))
    idx0      = t_idx - 1
    sqrt_ab   = tf.reshape(tf.gather(_SQRT_AB_TF,   idx0), [-1, 1, 1, 1])
    sqrt_omab = tf.reshape(tf.gather(_SQRT_OMAB_TF, idx0), [-1, 1, 1, 1])
    xt        = sqrt_ab * x0 + sqrt_omab * eps

    with tf.GradientTape() as tape:
        eps_pred = model([xt, t_emb], training=True)
        loss     = tf.reduce_mean(tf.square(eps_pred - eps))

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Pre-train base DDPM U-Net")
    parser.add_argument("--steps",   type=int,   default=TOTAL_STEPS,
                        help="Number of training steps")
    parser.add_argument("--batch",   type=int,   default=BATCH_SIZE,
                        help="Batch size")
    parser.add_argument("--lr",      type=float, default=LEARNING_RATE,
                        help="Learning rate")
    parser.add_argument("--no-save", action="store_true",
                        help="Skip saving — useful for quick tests")
    args = parser.parse_args()

    gpus = tf.config.list_physical_devices("GPU")
    print(f"TensorFlow {tf.__version__}  |  GPU: {bool(gpus)}")
    print(f"steps={args.steps}  batch={args.batch}  lr={args.lr}\n")

    model     = build_model()
    optimizer = keras.optimizers.Adam(learning_rate=args.lr)
    model.summary(line_length=80)

    t0          = time.time()
    loss_history = []

    for step in range(1, args.steps + 1):
        x0    = generate_batch(args.batch)
        t_idx = np.random.randint(1, T + 1, size=(args.batch,)).astype(np.int32)
        t_emb = sin_emb_batch(t_idx, T)   # [B, TIME_DIM] sinusoidal features

        loss  = train_step(
            model,
            optimizer,
            tf.constant(x0,    dtype=tf.float32),
            tf.constant(t_emb, dtype=tf.float32),
            tf.constant(t_idx, dtype=tf.int32),
        )
        loss_val = float(loss)
        loss_history.append(loss_val)

        if step % LOG_EVERY == 0 or step == 1:
            avg  = float(np.mean(loss_history[-LOG_EVERY:]))
            sps  = step / (time.time() - t0)
            eta  = (args.steps - step) / max(sps, 1e-6)
            print(f"step {step:>5}/{args.steps}  loss={loss_val:.5f}  "
                  f"avg={avg:.5f}  {sps:.1f} steps/s  ETA {eta:.0f}s")

    elapsed = time.time() - t0
    print(f"\nTraining complete in {elapsed:.1f}s  "
          f"({elapsed / args.steps * 1000:.1f} ms/step)")

    if args.no_save:
        print("--no-save set, skipping model export.")
        return

    # ── Save ────────────────────────────────────────────────────────────────
    os.makedirs(SAVE_DIR, exist_ok=True)

    # 1. Try direct TF.js export (cleanest, no CLI needed)
    try:
        import tensorflowjs as tfjs
        os.makedirs(TFJS_OUT_DIR, exist_ok=True)
        tfjs.converters.save_keras_model(model, TFJS_OUT_DIR)
        print(f"\n✅  TF.js model saved → {TFJS_OUT_DIR}/  (ready for browser)")
        # Keras 3.x produces a model.json format that TF.js loadLayersModel()
        # cannot parse (DTypePolicy objects, flat output_layers, new inbound_nodes
        # format).  Patch it in-place so the browser can load it directly.
        _fix_model_json_for_tfjs(os.path.join(TFJS_OUT_DIR, "model.json"))
    except ImportError:
        print("\n⚠️  tensorflowjs not installed — saving Keras format instead.")
        print("    Install: pip install tensorflowjs")
        print("    Then run: bash ml/convert_to_tfjs.sh\n")

    # 2. Always save .keras format (for inspection / re-training)
    keras_path = os.path.join(SAVE_DIR, MODEL_NAME + ".keras")
    model.save(keras_path)
    print(f"✅  Keras model saved → {keras_path}")

    # 3. SavedModel format (most universally compatible for conversion)
    saved_path = os.path.join(SAVE_DIR, MODEL_NAME + "_savedmodel")
    model.export(saved_path)
    print(f"✅  SavedModel saved  → {saved_path}/")

    print("\nNext step (if not already done):")
    print(f"  bash ml/convert_to_tfjs.sh")


if __name__ == "__main__":
    main()
