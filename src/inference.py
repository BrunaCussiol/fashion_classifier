import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing import image


def predict_multitask(
    model,
    img_path,
    cat_map=None,
    attr_map=None,
    attr_threshold=0.5,
    resize_shape=(224, 224),
):
    """
    Run multitask inference on a single image with enforced 3 channels (RGB).
    """
    # 1. Load image and force RGB and resize
    img = Image.open(img_path)
    img = img.convert("RGB")
    img = img.resize(resize_shape)

    # 2. Convert to array
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # batch dimension

    # 3. Predict
    category_probs, attributes_probs = model.predict(img_array, verbose=0)

    # 4. Post-process predictions
    predicted_category = int(np.argmax(category_probs[0]))
    predicted_category_name = cat_map[predicted_category] if cat_map else None

    predicted_attributes = (attributes_probs[0] > attr_threshold).astype(int).tolist()
    predicted_attributes_names = (
        [attr_map[i] for i, val in enumerate(predicted_attributes) if val == 1]
        if attr_map
        else None
    )

    return {
        "category_probs": category_probs[0],
        "predicted_category": predicted_category,
        "predicted_category_name": predicted_category_name,
        "attributes_probs": attributes_probs[0],
        "predicted_attributes": predicted_attributes,
        "predicted_attributes_names": predicted_attributes_names,
    }