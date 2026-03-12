#!/usr/bin/env python3
"""
ml/scripts/fix_model_json.py

Post-processes a Keras 3.x model.json → Keras 2.x format so that
tf.loadLayersModel() can parse it in the browser (TF.js 4.x).

Two incompatibilities are fixed:
  1. dtype  : DTypePolicy objects  → plain "float32" strings
  2. inbound_nodes: Keras 3 args/kwargs style
                 → Keras 2 [[layer_name, node_idx, tensor_idx, kwargs]] style
  3. initializers : strip unknown Keras 3 fields (module, registered_name)

Usage:
    python ml/scripts/fix_model_json.py                     # default: public/model/model.json
    python ml/scripts/fix_model_json.py public/model/model.json
"""

import json
import shutil
import sys
from pathlib import Path


# ── helpers ────────────────────────────────────────────────────────────────────

def _tensor_history(t):
    """Extract (layer_name, node_idx, tensor_idx) from a __keras_tensor__ dict."""
    if isinstance(t, dict) and t.get("class_name") == "__keras_tensor__":
        h = t["config"]["keras_history"]
        return (h[0], h[1], h[2])
    return None


def _fix_dtype(val):
    """Recursively replace DTypePolicy objects with plain dtype strings."""
    if isinstance(val, dict):
        if val.get("class_name") == "DTypePolicy":
            return val["config"]["name"]           # e.g. "float32"
        return {k: _fix_dtype(v) for k, v in val.items()}
    if isinstance(val, list):
        return [_fix_dtype(v) for v in val]
    return val


def _fix_initializer(init):
    """Strip Keras-3-only fields (module, registered_name) from initializer dicts."""
    if not isinstance(init, dict):
        return init
    out = {}
    if "class_name" in init:
        out["class_name"] = init["class_name"]
    if "config" in init:
        out["config"] = _fix_initializer(init["config"])
    return out


def _convert_inbound_nodes(nodes_v3):
    """
    Keras 3.x inbound_nodes  →  Keras 2.x inbound_nodes.

    Keras 3 format (one node):
      {"args": [single_tensor_ref | [list_of_tensor_refs]], "kwargs": {}}

    Keras 2 format (one node):
      [[layer_name, node_idx, tensor_idx, {}], ...]
      (one sub-list per input tensor, the whole thing is wrapped in another list
       so that inbound_nodes = [node0, node1, ...] where each nodeN is a list)
    """
    if not nodes_v3:
        return []

    out = []
    for node in nodes_v3:
        if not isinstance(node, dict) or "args" not in node:
            # Already old-style → keep as-is
            out.append(node)
            continue

        args = node["args"]
        if not args:
            continue

        first = args[0]

        if isinstance(first, list):
            # Multi-input layer (Concatenate, Add, …)
            # Keras 3: args = [[tensor_ref, tensor_ref, …]]
            inner = []
            for t in first:
                h = _tensor_history(t)
                if h:
                    inner.append([h[0], h[1], h[2], {}])
            out.append(inner)
        else:
            # Single-input layer
            # Keras 3: args = [tensor_ref]
            h = _tensor_history(first)
            if h:
                out.append([[h[0], h[1], h[2], {}]])

    return out


INITIALIZER_KEYS = [
    "kernel_initializer", "bias_initializer", "embeddings_initializer",
    "kernel_regularizer", "bias_regularizer", "activity_regularizer",
    "kernel_constraint", "bias_constraint",
]


def _convert_layer(layer):
    """Convert a single layer dict from Keras 3.x to Keras 2.x."""
    layer = dict(layer)

    # Fix inbound_nodes
    if "inbound_nodes" in layer:
        layer["inbound_nodes"] = _convert_inbound_nodes(layer["inbound_nodes"])

    if "config" in layer:
        cfg = dict(layer["config"])

        # Fix dtype
        if "dtype" in cfg:
            cfg["dtype"] = _fix_dtype(cfg["dtype"])

        # Fix InputLayer: Keras 3 uses `batch_shape`, TF.js expects `batch_input_shape`
        if layer.get("class_name") == "InputLayer" and "batch_shape" in cfg:
            cfg["batch_input_shape"] = cfg.pop("batch_shape")
        # Also remove Keras-3-only fields TF.js doesn't understand
        for k in ("optional", "ragged"):
            cfg.pop(k, None)

        # Fix initializers / regularizers
        for key in INITIALIZER_KEYS:
            if key in cfg and cfg[key] is not None:
                cfg[key] = _fix_initializer(cfg[key])

        layer["config"] = cfg

    return layer


def _fix_output_layers(cfg):
    """
    Keras 3.x serialises a single output as a flat list: ["output", 0, 0]
    TF.js (and Keras 2.x) expects a list of tuples:     [["output", 0, 0]]
    If the first element is a string we're in the flat (Keras 3) form.
    """
    ol = cfg.get("output_layers")
    if ol and isinstance(ol[0], str):
        cfg["output_layers"] = [ol]
    return cfg


def convert_model_json(data):
    """Top-level converter: walk model_config.layers and fix each one."""
    data = dict(data)
    topo = dict(data.get("modelTopology", {}))

    mc = dict(topo.get("model_config", {}))
    cfg = dict(mc.get("config", {}))

    if "layers" in cfg:
        cfg["layers"] = [_convert_layer(l) for l in cfg["layers"]]

    cfg = _fix_output_layers(cfg)

    mc["config"] = cfg
    topo["model_config"] = mc
    data["modelTopology"] = topo
    return data


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("public/model/model.json")

    if not path.exists():
        print(f"❌  File not found: {path}")
        sys.exit(1)

    bak = path.with_suffix(".json.bak")
    shutil.copy(path, bak)
    print(f"✅  Backed up original → {bak}")

    with open(path) as f:
        data = json.load(f)

    converted = convert_model_json(data)

    with open(path, "w") as f:
        json.dump(converted, f, separators=(",", ":"))

    n_layers = len(converted["modelTopology"]["model_config"]["config"]["layers"])
    print(f"✅  Patched {n_layers} layers  →  {path}")
    print("    DTypePolicy objects    → plain dtype strings")
    print("    Keras 3 inbound_nodes  → Keras 2 inbound_nodes")
    print("    Initializer dicts      → TF.js-compatible format")


if __name__ == "__main__":
    main()
