import pandas as pd
import os
import shutil

# Rutas de los CSV
csv_train = r"C:\Users\bruna\OneDrive\Desktop\AI bootcamp\Final Project\img_cropped\ann_train.csv"
csv_val   = r"C:\Users\bruna\OneDrive\Desktop\AI bootcamp\Final Project\img_cropped\ann_val.csv"
csv_test  = r"C:\Users\bruna\OneDrive\Desktop\AI bootcamp\Final Project\img_cropped\ann_test.csv"

# Carpetas donde están las imágenes originales
img_train_folder = r"C:\Users\bruna\OneDrive\Desktop\AI bootcamp\Final Project\img_cropped\train"
img_val_folder  = r"C:\Users\bruna\OneDrive\Desktop\AI bootcamp\Final Project\img_cropped\val"
img_test_folder  = r"C:\Users\bruna\OneDrive\Desktop\AI bootcamp\Final Project\img_cropped\test"

# Carpeta de salida donde vamos a copiar las imágenes filtradas
output_folder = r"C:\Users\bruna\OneDrive\Desktop\AI bootcamp\Final Project\img_filtered"
os.makedirs(output_folder, exist_ok=True)

# Función para filtrar imágenes según un CSV
def filter_images(csv_file, src_folder, subset_name):
    df = pd.read_csv(csv_file)
    image_names = df["image_name"].tolist()
    
    subset_folder = os.path.join(output_folder, subset_name)
    os.makedirs(subset_folder, exist_ok=True)
    
    for img_name in image_names:
        # Construir ruta original
        src_path = os.path.join(src_folder, img_name)
        if os.path.exists(src_path):
            # Crear subcarpetas si es necesario
            dest_path = os.path.join(subset_folder, img_name)
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            shutil.copy2(src_path, dest_path)
        else:
            print(f"⚠️ No se encontró: {src_path}")

# Filtrar cada subset
filter_images(csv_train, img_train_folder, "train")
filter_images(csv_val,   img_val_folder, "val")    # usualmente val también está en train
filter_images(csv_test,  img_test_folder,  "test")

print("✅ Filtrado completado")
