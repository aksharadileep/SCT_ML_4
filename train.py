import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, callbacks, optimizers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from PIL import Image
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

# ─── Configuration ────────────────────────────────────────────────────────────
DATA_PATH  = r"A:\SCT_ML_4\data\leapGestRecog"
MODEL_DIR  = r"A:\SCT_ML_4\models"
IMG_SIZE   = (128, 128)
BATCH_SIZE = 64
NUM_EPOCHS_PHASE1 = 20
NUM_EPOCHS_PHASE2 = 15

GESTURES = [
    "01_palm", "02_I", "03_fist", "04_fist_moved", "05_thumb",
    "06_index", "07_ok", "08_palm_moved", "09_c", "10_down"
]

DISPLAY_NAMES = {
    "01_palm":       "Palm",
    "02_I":          "I Shape",
    "03_fist":       "Fist",
    "04_fist_moved": "Fist Moved",
    "05_thumb":      "Thumb",
    "06_index":      "Index",
    "07_ok":         "OK",
    "08_palm_moved": "Palm Moved",
    "09_c":          "C Shape",
    "10_down":       "Down",
}

# ─── Data loading ─────────────────────────────────────────────────────────────
def load_all_data():
    """
    Load every image as RGB (H×W×3).
    No grayscale conversion — MobileNetV2 expects colour input and
    preprocess_input handles normalisation internally at model time.
    """
    images, labels = [], []
    subfolders = sorted(
        d for d in os.listdir(DATA_PATH)
        if os.path.isdir(os.path.join(DATA_PATH, d))
    )
    print(f"Scanning {len(subfolders)} subfolders…")

    for subfolder in subfolders:
        subfolder_path = os.path.join(DATA_PATH, subfolder)
        for gesture in GESTURES:
            folder = os.path.join(subfolder_path, gesture)
            if not os.path.exists(folder):
                continue
            for fname in os.listdir(folder):
                if not fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue
                try:
                    img = (
                        Image.open(os.path.join(folder, fname))
                             .convert('RGB')                  # always 3-ch RGB
                             .resize(IMG_SIZE, Image.Resampling.LANCZOS)
                    )
                    images.append(np.array(img, dtype=np.uint8))
                    labels.append(gesture)
                except Exception:
                    continue

    return np.array(images), np.array(labels)


# ─── Model ────────────────────────────────────────────────────────────────────
def create_model(num_classes: int):
    """
    MobileNetV2 transfer-learning model.
    Input  : uint8 or float32 RGB image, range [0, 255] or [0, 1] — both work
             because MobileNetV2.preprocess_input rescales to [-1, 1].
    Output : softmax probabilities over num_classes.
    """
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(128, 128, 3),
        include_top=False,
        weights='imagenet',
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=(128, 128, 3))

    # Cast to float32 and run MobileNetV2 preprocessing (handles any input range)
    x = tf.cast(inputs, tf.float32)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)

    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs)
    return model, base_model


# ─── Main ─────────────────────────────────────────────────────────────────────
print("=" * 60)
print("HAND GESTURE RECOGNITION — OPTIMISED TRAINING")
print("=" * 60)

print("\n📂 Loading dataset…")
X, y = load_all_data()
print(f"✅ Total images : {len(X)}")
print(f"📐 Image shape  : {X[0].shape}")

if len(X) == 0:
    raise RuntimeError("No images found — check DATA_PATH.")

# Encode labels
label_encoder = LabelEncoder()
y_enc = label_encoder.fit_transform(y)
num_classes = len(label_encoder.classes_)
print(f"🏷️  Classes : {num_classes}")

# Train / val / test split
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y_enc, test_size=0.3, random_state=42, stratify=y_enc
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

print(f"\n📊 Split — train: {len(X_train)}  val: {len(X_val)}  test: {len(X_test)}")

# Class weights
class_weights = compute_class_weight(
    'balanced', classes=np.unique(y_enc), y=y_enc
)
class_weight_dict = dict(enumerate(class_weights))

# Build & compile (Phase 1)
print("\n🏗️  Building model…")
model, base_model = create_model(num_classes)
model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-3),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'],
)

os.makedirs(MODEL_DIR, exist_ok=True)
best_ckpt = os.path.join(MODEL_DIR, 'best_model.keras')

cb = [
    callbacks.EarlyStopping(
        monitor='val_accuracy', patience=8,
        restore_best_weights=True, verbose=1,
    ),
    callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5,
        patience=3, min_lr=1e-7, verbose=1,
    ),
    callbacks.ModelCheckpoint(
        best_ckpt, monitor='val_accuracy',
        save_best_only=True, verbose=1,
    ),
]

print("\n🔒 PHASE 1: Training classifier head…")
history1 = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=NUM_EPOCHS_PHASE1,
    batch_size=BATCH_SIZE,
    class_weight=class_weight_dict,
    callbacks=cb,
    verbose=1,
)

# Phase 2: fine-tune top layers
print("\n🔓 PHASE 2: Fine-tuning top 50 layers…")
base_model.trainable = True
for layer in base_model.layers[:-50]:
    layer.trainable = False

model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-5),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'],
)

history2 = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=NUM_EPOCHS_PHASE2,
    batch_size=BATCH_SIZE,
    class_weight=class_weight_dict,
    callbacks=cb,
    verbose=1,
)

# ─── Evaluation ───────────────────────────────────────────────────────────────
print("\n📊 Final evaluation on test set:")
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"   Accuracy : {test_acc*100:.2f}%")
print(f"   Loss     : {test_loss:.4f}")

# Save artefacts
model.save(os.path.join(MODEL_DIR, 'gesture_model.keras'))
with open(os.path.join(MODEL_DIR, 'label_encoder.pkl'), 'wb') as f:
    pickle.dump(label_encoder, f)
with open(os.path.join(MODEL_DIR, 'display_names.pkl'), 'wb') as f:
    pickle.dump(DISPLAY_NAMES, f)

print(f"\n💾 Saved to {MODEL_DIR}")

# Per-class accuracy
y_pred_classes = np.argmax(model.predict(X_test, verbose=0), axis=1)
print("\n📋 Per-class accuracy:")
for i, gesture in enumerate(label_encoder.classes_):
    mask = y_test == i
    if mask.any():
        acc = (y_pred_classes[mask] == i).mean()
        icon = "✅" if acc > 0.9 else "⚠️" if acc > 0.7 else "❌"
        print(f"   {icon} {DISPLAY_NAMES.get(gesture, gesture):15s}: {acc*100:.2f}%")

print("\n📊 Classification Report:")
print(classification_report(
    y_test, y_pred_classes,
    target_names=[DISPLAY_NAMES.get(g, g) for g in label_encoder.classes_],
))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred_classes)
plt.figure(figsize=(12, 10))
sns.heatmap(
    cm, annot=True, fmt='d', cmap='Blues',
    xticklabels=[DISPLAY_NAMES.get(g, g) for g in label_encoder.classes_],
    yticklabels=[DISPLAY_NAMES.get(g, g) for g in label_encoder.classes_],
)
plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
plt.ylabel('True Label'); plt.xlabel('Predicted Label')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, 'confusion_matrix.png'), dpi=150)
plt.show()

# Training curves
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
acc  = history1.history['accuracy']     + history2.history['accuracy']
vacc = history1.history['val_accuracy'] + history2.history['val_accuracy']
loss = history1.history['loss']         + history2.history['loss']
vloss= history1.history['val_loss']     + history2.history['val_loss']
ep   = range(1, len(acc) + 1)
split= len(history1.history['accuracy'])

for ax, (tr, val, title) in zip(axes, [
    (acc, vacc, 'Accuracy'), (loss, vloss, 'Loss')
]):
    ax.plot(ep, tr,  'b-', label=f'Training {title}',   linewidth=2)
    ax.plot(ep, val, 'r-', label=f'Validation {title}', linewidth=2)
    ax.axvline(x=split, color='g', linestyle='--', label='Fine-tune start', alpha=0.7)
    ax.set_title(f'Model {title}', fontsize=14, fontweight='bold')
    ax.set_xlabel('Epoch'); ax.set_ylabel(title)
    ax.legend(); ax.grid(True, alpha=0.3)

plt.suptitle('Training History', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, 'training_history.png'), dpi=150)
plt.show()

print("\n" + "=" * 60)
print(f"🎉 DONE!  Test accuracy: {test_acc*100:.2f}%")
print("=" * 60)