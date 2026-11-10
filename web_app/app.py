#!/usr/bin/env python
"""Flask web UI for SmartBin trash classification.

Expected folder layout (relative to this file):
  ../model/smartbin_head.h5          – TensorFlow/Keras model (MobileNetV2 head)
  ../model/label_map.json           – mapping from class indices to names

Run with:
  python -m web_app.app   # from the project root (smart bin folder)
"""
import os
import json
import numpy as np
from flask import Flask, request, render_template, jsonify
from PIL import Image
import tensorflow as tf

# ------------------------------------------------------------------
# CONFIGURATION – adjust paths if you move files around
# ------------------------------------------------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_DIR = os.path.join(BASE_DIR, "model")
MODEL_CANDIDATES = [
    os.path.join(MODEL_DIR, "smartbin2_best.keras"),
    os.path.join(MODEL_DIR, "smartbin_head.h5"),
]
LABEL_MAP_CANDIDATES = [
    os.path.join(MODEL_DIR, "class_labels.json"),
    os.path.join(MODEL_DIR, "label_map.json"),
]

MODEL_PATH = next((path for path in MODEL_CANDIDATES if os.path.isfile(path)), None)
LABEL_MAP_PATH = next((path for path in LABEL_MAP_CANDIDATES if os.path.isfile(path)), None)
if MODEL_PATH is None:
    raise FileNotFoundError(
        "No trained model found. Expected model/smartbin2_best.keras or model/smartbin_head.h5."
    )
if LABEL_MAP_PATH is None:
    raise FileNotFoundError(
        "No class labels found. Expected model/class_labels.json or model/label_map.json."
    )

print(f"Loading model from {MODEL_PATH} …")
model = tf.keras.models.load_model(MODEL_PATH)

print(f"Loading label map from {LABEL_MAP_PATH} …")
with open(LABEL_MAP_PATH, "r", encoding="utf-8") as f:
    raw_labels = json.load(f)
if isinstance(raw_labels, list):
    label_map = {str(index): name for index, name in enumerate(raw_labels)}
else:
    label_map = raw_labels

model_input_shape = model.input_shape
IMAGE_SIZE = tuple(int(value) for value in model_input_shape[1:3])

# ------------------------------------------------------------------
# Helper: preprocess an image to the shape MobileNetV2 expects (224×224)
# ------------------------------------------------------------------
def prepare_image(pil_img):
    img = pil_img.convert("RGB").resize(IMAGE_SIZE)
    arr = np.array(img, dtype=np.float32)
    arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)
    return np.expand_dims(arr, axis=0)  # (1, height, width, 3)

app = Flask(__name__, static_folder="static", template_folder="templates")

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400
    f = request.files["file"]
    if f.filename == "":
        return jsonify({"error": "No selected file"}), 400
    try:
        img = Image.open(f.stream)
        x = prepare_image(img)
        preds = model.predict(x)
        class_idx = int(np.argmax(preds, axis=1)[0])
        class_name = label_map.get(str(class_idx), f"Unknown({class_idx})")
        return jsonify({
            "class_index": class_idx,
            "class_name": class_name,
            "probabilities": preds.tolist()[0]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # accessible at http://127.0.0.1:5000
    app.run(host="127.0.0.1", port=5000, debug=True)

