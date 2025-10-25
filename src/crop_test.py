import os
from PIL import Image

# Annotation files
coords_file = "C:\\Users\\bruna\\OneDrive\\Desktop\\AI bootcamp\\Final Project\\annotations\\val_bbox.txt"
images_file = "C:\\Users\\bruna\\OneDrive\\Desktop\\AI bootcamp\\Final Project\\annotations\\val.txt"
output_dir = "img_cropped/val"

# Read coordinates
with open(coords_file, "r") as f:
    coords = [list(map(int, line.strip().split())) for line in f if line.strip()]

# Read image routes
with open(images_file, "r") as f:
    images = [line.strip() for line in f if line.strip()]

# Validation
if len(coords) != len(images):
    print(f"⚠️ Ojo: {len(coords)} coordenadas pero {len(images)} imágenes")
else:
    print(f"Procesando {len(images)} imágenes...")

# Processing
for i, (img_path, (x1, y1, x2, y2)) in enumerate(zip(images, coords)):
    try:
        # Open image
        img = Image.open(img_path)
        cropped = img.crop((x1, y1, x2, y2))  # (left, upper, right, lower)

        # Build new route keeping same structure
        rel_path = os.path.relpath(img_path, "img")  # removes the prefix "img/"
        save_dir = os.path.join(output_dir, os.path.dirname(rel_path))
        os.makedirs(save_dir, exist_ok=True)

        # Save with modified name
        base_name, ext = os.path.splitext(os.path.basename(img_path))
        save_path = os.path.join(save_dir, f"{base_name}_crop{ext}")

        cropped.save(save_path)
        print(f"[{i+1}] Saved: {save_path}")
    except Exception as e:
        print(f"❌ Error in {img_path}: {e}")
