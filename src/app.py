from pathlib import Path

import gradio as gr

from src.utils_gradio import gradio_predict

# ------------------------- Define Gradio interface -------------------------
description = """
<h2 style="color: #4A90E2;">Multitask Fashion Classifier</h2>
<p>Upload an image of a clothing item. The model predicts:</p>
<ul>
<li><b>Category</b> (e.g., t-shirt, dress)</li>
<li><b>Attributes</b> (e.g., floral, sleeveless, cotton)</li>
</ul>
"""

# Input and output components
inputs = gr.Image(type="pil", label="Upload Image")
outputs = [gr.Textbox(label="Predicted Category & Attributes", lines=5)]

# Optional: add some example images (paths local to Colab or URLs)
examples = [
    [
        "https://eu.manduka.com/cdn/shop/files/7516113_W_Dhara_Legging_PhantomHeather_02.jpg?v=1750410341&width=871"
    ],
    [
        "https://encrypted-tbn0.gstatic.com/shopping?q=tbn:ANd9GcSTiJkqHLvNj8r3juJZ5-_5vXmY4T26I47SXKgeiwF7xcpuukZUYjc1G28V1kYYf2jT5aAJxWzf9ieBSBA5czDjAUWz-7RQusUGbTkZtdlr0QrpC9O6-BQw"
    ],
]

iface = gr.Interface(
    fn=gradio_predict,
    inputs=inputs,
    outputs=outputs,
    title="Fashion Classifier",
    description=description,
    examples=examples,
    allow_flagging="never",
    theme="default",
)

# ------------------------- Launch -------------------------
if __name__ == "__main__":
    iface.launch()
