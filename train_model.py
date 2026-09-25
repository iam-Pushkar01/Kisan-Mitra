import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
import json

# ==========================================
# DATASET LOCATION
# ==========================================

DATASET_PATH = "data/plantvillage dataset/color"

# ==========================================
# IMAGE SETTINGS
# ==========================================

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# ==========================================
# TRAINING DATA PREPARATION
# ==========================================

datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input,
    validation_split=0.2
)

train_data = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training",
    shuffle=True
)

validation_data = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    shuffle=False
)

# ==========================================
# NUMBER OF CLASSES
# ==========================================

num_classes = len(train_data.class_indices)

print("Number of classes:", num_classes)
print("Classes:", train_data.class_indices)

# ==========================================
# SAVE CLASS NAMES
# ==========================================

class_names = [None] * num_classes

for class_name, class_index in train_data.class_indices.items():
    class_names[class_index] = class_name

with open("model/class_names.json", "w") as f:
    json.dump(class_names, f, indent=4)

print("Class names saved successfully.")

# ==========================================
# MOBILENETV2 BASE MODEL
# ==========================================

base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

# Freeze pretrained layers
base_model.trainable = False

# ==========================================
# BUILD MODEL
# ==========================================

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.2),
    layers.Dense(num_classes, activation="softmax")
])

# ==========================================
# COMPILE MODEL
# ==========================================

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# ==========================================
# MODEL SUMMARY
# ==========================================

model.summary()

# ==========================================
# TRAIN MODEL
# ==========================================

history = model.fit(
    train_data,
    validation_data=validation_data,
    epochs=3
)

# ==========================================
# SAVE TRAINED MODEL
# ==========================================

model.save("model/kisan_mitra_model.keras")

print("===================================")
print("Model training complete!")
print("Model saved successfully.")
print("Class names saved successfully.")
print("===================================")