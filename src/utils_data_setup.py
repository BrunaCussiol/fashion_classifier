import tensorflow as tf

def load_image(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [224,224]) / 255.0
    return img

def format_labels(cat, attrs):
    return {"category": tf.cast(cat, tf.int32),
            "attributes": tf.cast(attrs, tf.float32)}

def augment(image, labels):
    if tf.random.uniform([]) < 0.5:
        image = tf.image.flip_left_right(image)
    if tf.random.uniform([]) < 0.5:
        image = tf.image.random_brightness(image, max_delta=0.1)
    if tf.random.uniform([]) < 0.5:
        image = tf.image.random_contrast(image, 0.9, 1.1)
    return image, labels

def make_dataset(df, attribute_cols, base_path, batch_size=32, augment_data=False):
    df = df.copy()
    df["image_path_full"] = base_path + df["image_name"]
    categories = df["category_encoded"].values.astype('int32')
    attributes = df[attribute_cols].values.astype('float32')
    path_ds = tf.data.Dataset.from_tensor_slices(df["image_path_full"].values)
    image_ds = path_ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    label_ds = tf.data.Dataset.from_tensor_slices((categories, attributes))
    ds = tf.data.Dataset.zip((image_ds, label_ds))
    ds = ds.map(lambda img, lbl: (img, format_labels(lbl[0], lbl[1])), num_parallel_calls=tf.data.AUTOTUNE)
    if augment_data:
        ds = ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.shuffle(1000)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds
