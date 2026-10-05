import tensorflow as tf
from tensorflow.keras import layers, models

IMAGE_SIZE = 128
BATCH_SIZE = 32
CHANNELS = 3


def build_waste_classifier():
    model = models.Sequential([
        layers.Conv2D(
            32,
            (3, 3),
            activation="relu",
            input_shape=(IMAGE_SIZE, IMAGE_SIZE, CHANNELS)
        ),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(128, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),

        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(128, activation="relu"),

        # 2 classes: Organic and Recyclable
        layers.Dense(2, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


if __name__ == "__main__":
    print("--- Waste Classification CNN Model Summary ---")

    model = build_waste_classifier()
    model.summary()