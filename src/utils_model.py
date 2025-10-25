import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.metrics import (
    SparseCategoricalAccuracy,
    BinaryAccuracy,
    AUC,
    Precision,
    Recall
)

class BalancedAccuracy(tf.keras.metrics.Metric):
    def __init__(self, num_classes, name="balanced_accuracy", **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.acc_per_class = [tf.keras.metrics.Mean() for _ in range(num_classes)]

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred_class = tf.argmax(y_pred, axis=-1)
        for i in range(self.num_classes):
            mask = tf.equal(y_true, i)
            correct = tf.reduce_sum(tf.cast(tf.logical_and(mask, tf.equal(y_pred_class, i)), tf.float32))
            total = tf.reduce_sum(tf.cast(mask, tf.float32))
            acc = tf.cond(total > 0, lambda: correct / total, lambda: 0.0)
            self.acc_per_class[i].update_state(acc)

    def result(self):
        return tf.reduce_mean([m.result() for m in self.acc_per_class])

    def reset_state(self):
        for m in self.acc_per_class:
            m.reset_state()




def build_multitask_efficientnet(
    num_classes: int,
    num_attributes: int,
    resize_shape=(224, 224),
    num_channels: int = 3,
):
    """
    Build an EfficientNetB0 multitask model (without compile) for inference/prediction.

    Args:
        num_classes (int): Number of category classes.
        num_attributes (int): Number of binary attributes to predict.
        resize_shape (tuple): Input image size (height, width).
        num_channels (int): Number of input channels (default=3 for RGB).

    Returns:
        tf.keras.Model: A Keras multitask model ready for inference.
    """

    # Input layer
    inputs = layers.Input(shape=(resize_shape[0], resize_shape[1], num_channels))

    # Backbone EfficientNetB0 pretrained on ImageNet
    backbone = EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_tensor=inputs,
        pooling="avg"
    )

    # Shared backbone output
    x = backbone.output

    # Task 1: Category classification
    category_output = layers.Dense(
        num_classes, activation="softmax", name="category"
    )(x)

    # Task 2: Attribute prediction
    attribute_output = layers.Dense(
        num_attributes, activation="sigmoid", name="attributes"
    )(x)

    # Build the model (no compile)
    model = models.Model(inputs=inputs, outputs=[category_output, attribute_output])

    return model


def compile_multitask_model(model, num_classes, learning_rate=0.0001):
    """
    Compile a multi-task Keras model with category (softmax) and attributes (sigmoid) outputs.

    Args:
        model (tf.keras.Model): The Keras model to compile.
        num_classes (int): Number of classes for the category classification.
        learning_rate (float, optional): Learning rate for the Adam optimizer. Default is 0.0001.

    Returns:
        tf.keras.Model: The compiled Keras model.
    """

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss={
            "category": tf.keras.losses.SparseCategoricalCrossentropy(),
            "attributes": tf.keras.losses.BinaryCrossentropy(),
        },
        metrics={
            "category": [
                BalancedAccuracy(num_classes=num_classes),  # Custom metric
                SparseCategoricalAccuracy(name="accuracy"),
            ],
            "attributes": [
                BinaryAccuracy(name="binary_accuracy"),
                AUC(name="auc"),
                Precision(name="precision"),
                Recall(name="recall"),
            ],
        },
    )

    return model

