#!/usr/bin/env bash
# ml/convert_to_tfjs.sh
#
# Converts the saved Keras model to TF.js Layers-Model format
# (model.json + *.bin) so it can be loaded by the browser via:
#   tf.loadLayersModel('/model/model.json')
#
# Prerequisites:
#   pip install tensorflowjs
#
# Usage:
#   bash ml/convert_to_tfjs.sh
#   bash ml/convert_to_tfjs.sh --model ml/models/conditional_unet.keras

set -euo pipefail

MODEL="${1:-ml/models/base_unet.keras}"
OUT="public/model"

echo "Converting $MODEL → $OUT/"
mkdir -p "$OUT"

if [ ! -f "$MODEL" ] && [ ! -d "$MODEL" ]; then
  echo "❌  Model not found at $MODEL"
  echo "    Run python ml/scripts/train_base_unet.py first."
  exit 1
fi

# Try .keras / .h5 format first
if [[ "$MODEL" == *.keras || "$MODEL" == *.h5 ]]; then
  tensorflowjs_converter \
    --input_format=keras \
    --output_format=tfjs_layers_model \
    --weight_shard_size_bytes=4194304 \
    "$MODEL" "$OUT"

# Fallback: SavedModel directory
elif [ -d "$MODEL" ]; then
  tensorflowjs_converter \
    --input_format=tf_saved_model \
    --output_format=tfjs_layers_model \
    --weight_shard_size_bytes=4194304 \
    "$MODEL" "$OUT"
else
  echo "❌  Unknown model format. Provide a .keras file or SavedModel directory."
  exit 1
fi

echo ""
echo "✅  TF.js model written to $OUT/"
echo "    Files: $(ls "$OUT" | tr '\n' ' ')"

# Patch the model.json for TF.js browser compatibility.
# Keras 3.x uses a newer serialisation format; this converts it to Keras 2.x.
echo ""
echo "Patching $OUT/model.json for TF.js browser compatibility..."
python3 ml/scripts/fix_model_json.py "$OUT/model.json"

echo ""
echo "The browser will automatically load this model from /model/model.json"
echo "next time you train — no code changes needed."
