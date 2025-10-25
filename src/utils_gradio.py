import numpy as np
from PIL import Image
from .utils_model import build_multitask_efficientnet
from .inference import predict_multitask
import json
from pathlib import Path
import tempfile
import os

# ------------------------- Load maps -------------------------
cat_map_path = Path.cwd() / "annotations/cat_map_filtered.json"
attr_map_path = Path.cwd() / "annotations/attr_map_filtered.json"

with open(cat_map_path, "r") as f:
    cat_map = json.load(f)
cat_map = {int(v): k for k, v in cat_map.items()}

with open(attr_map_path, "r") as f:
    attr_map = json.load(f)
attr_map = {int(v): k for k, v in attr_map.items()}

# ------------------------- Load model -------------------------
model_path = Path.cwd() / "model/model.keras"
num_classes = len(cat_map)
num_attributes = len(attr_map)

model = build_multitask_efficientnet(
    num_classes=num_classes,
    num_attributes=num_attributes,
    resize_shape=(224, 224),
    num_channels=3
)
model.load_weights(model_path)

# ------------------------- Prediction function -------------------------
def gradio_predict(img, attr_threshold=0.5):
    """
    Run multitask prediction on an uploaded image and return human-readable results.

    Works on Windows, Linux, and Colab.
    """
    # 1. Convert input to PIL.Image
    if not isinstance(img, Image.Image):
        img = Image.open(img)
    img = img.convert("RGB")
    img = img.resize((224, 224))

    # 2. Save temp file in cross-platform way
    tmp_dir = Path(tempfile.gettempdir())  # System temp folder
    tmp_dir.mkdir(exist_ok=True)
    img_path = tmp_dir / "gradio_input.jpg"
    img.save(img_path)  # Safe on all platforms

    # 3. Run prediction
    pred = predict_multitask(
        model=model,
        img_path=img_path,
        attr_threshold=attr_threshold,
        resize_shape=(224, 224),
        cat_map=cat_map,
        attr_map=attr_map
    )

    # 4. Map category
    category_name = cat_map[pred["predicted_category"]]

    # 5. Map attributes
    attribute_list = [
        attr_map[i] for i, val in enumerate(pred["predicted_attributes"]) if val == 1
    ]
    attributes_text = ", ".join(attribute_list) if attribute_list else "None"

    return f"Category: {category_name}\nAttributes: {attributes_text}"
