"""
Brain MRI Transfer Learning — RETRAIN SCRIPT
Fixes:
  1) epochs=2 → 15 frozen + 5 fine-tune
  2) rescale=1./255 → preprocess_input (MobileNetV2 expects 0-255 → -1..+1)
  3) Added callbacks: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
  4) Fine-tuning phase: unfreeze top 30 layers of MobileNetV2
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# ── Paths ─────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR  = os.path.join(BASE_DIR, 'media', 'MRI Stroke', 'test')
SAVE_DIR  = os.path.join(BASE_DIR, 'training_results', 'brain')
os.makedirs(SAVE_DIR, exist_ok=True)

IMG_SIZE   = 224
BATCH_SIZE = 16
EPOCHS_1   = 15   # Phase 1: frozen base
EPOCHS_2   = 5    # Phase 2: fine-tune top layers

# ── Data Generators ───────────────────────────────────────────────────
# CRITICAL: Use preprocess_input (NOT rescale=1/255) for MobileNetV2
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,   # ← FIXED
    validation_split=0.2,
    rotation_range=20,
    width_shift_range=0.15,
    height_shift_range=0.15,
    zoom_range=0.15,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,   # ← FIXED (same as inference)
    validation_split=0.2
)

train_gen = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training'
)

val_gen = val_datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation',
    shuffle=False
)

print(f"Classes: {train_gen.class_indices}")
print(f"Train samples: {train_gen.samples}  |  Val samples: {val_gen.samples}")

# ── Build Model ───────────────────────────────────────────────────────
base = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
base.trainable = False   # Phase 1: freeze everything

x = base.output
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.4)(x)
x = Dense(64, activation='relu')(x)
x = Dropout(0.3)(x)
out = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base.input, outputs=out)
model.compile(optimizer=Adam(learning_rate=1e-3),
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.summary()

# ── Callbacks ─────────────────────────────────────────────────────────
ckpt_path = os.path.join(SAVE_DIR, 'brain_tl_best.h5')
callbacks_phase1 = [
    ModelCheckpoint(ckpt_path, monitor='val_accuracy', save_best_only=True, verbose=1),
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=3, min_lr=1e-6, verbose=1)
]

# ── Phase 1: Train top layers ─────────────────────────────────────────
print("\n" + "="*50)
print("PHASE 1: Training classifier head (base frozen)")
print("="*50)
history1 = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS_1,
    callbacks=callbacks_phase1
)

# ── Phase 2: Fine-tune top 30 layers ──────────────────────────────────
print("\n" + "="*50)
print("PHASE 2: Fine-tuning top 30 MobileNetV2 layers")
print("="*50)
base.trainable = True
for layer in base.layers[:-30]:
    layer.trainable = False

model.compile(optimizer=Adam(learning_rate=1e-5),   # much smaller LR for fine-tune
              loss='binary_crossentropy',
              metrics=['accuracy'])

callbacks_phase2 = [
    ModelCheckpoint(ckpt_path, monitor='val_accuracy', save_best_only=True, verbose=1),
    EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=2, min_lr=1e-8, verbose=1)
]

history2 = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS_2,
    callbacks=callbacks_phase2
)

# ── Save Final Model ──────────────────────────────────────────────────
final_path = os.path.join(SAVE_DIR, 'brain_tl_model.h5')
model.save(final_path)
print(f"\nModel saved: {final_path}")

# ── Merge history for plotting ─────────────────────────────────────────
def merge_histories(h1, h2, key):
    return h1.history.get(key, []) + h2.history.get(key, [])

acc     = merge_histories(history1, history2, 'accuracy')
val_acc = merge_histories(history1, history2, 'val_accuracy')
loss    = merge_histories(history1, history2, 'loss')
val_loss= merge_histories(history1, history2, 'val_loss')
epochs  = range(1, len(acc) + 1)

# ── Accuracy / Loss Plot ───────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(epochs, acc, 'b-o', label='Train Accuracy')
axes[0].plot(epochs, val_acc, 'g-o', label='Val Accuracy')
axes[0].axvline(x=len(history1.history['accuracy']), color='orange', linestyle='--', label='Fine-tuning start')
axes[0].set_title('Brain MRI — Accuracy over Epochs', fontsize=13)
axes[0].legend(); axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Accuracy')

axes[1].plot(epochs, loss, 'r-o', label='Train Loss')
axes[1].plot(epochs, val_loss, 'm-o', label='Val Loss')
axes[1].axvline(x=len(history1.history['loss']), color='orange', linestyle='--', label='Fine-tuning start')
axes[1].set_title('Brain MRI — Loss over Epochs', fontsize=13)
axes[1].legend(); axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Loss')

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'accuracy_loss.png'), dpi=120, bbox_inches='tight')
plt.close()

# ── Evaluate ───────────────────────────────────────────────────────────
val_gen.reset()
preds = model.predict(val_gen, verbose=1)
y_pred = (preds > 0.5).astype(int).flatten()
y_true = val_gen.classes

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=val_gen.class_indices.keys(),
            yticklabels=val_gen.class_indices.keys())
plt.title('Brain MRI — Confusion Matrix')
plt.ylabel('True Label'); plt.xlabel('Predicted Label')
plt.savefig(os.path.join(SAVE_DIR, 'confusion_matrix.png'), dpi=120, bbox_inches='tight')
plt.close()

# ROC Curve
fpr, tpr, _ = roc_curve(y_true, preds.flatten())
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], 'navy', lw=2, linestyle='--')
plt.xlim([0, 1]); plt.ylim([0, 1.05])
plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
plt.title('Brain MRI — ROC Curve')
plt.legend(loc='lower right')
plt.savefig(os.path.join(SAVE_DIR, 'roc_curve.png'), dpi=120, bbox_inches='tight')
plt.close()

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=list(val_gen.class_indices.keys())))
print(f"\nFinal Val Accuracy: {val_acc[-1]*100:.2f}%")
print(f"AUC: {roc_auc:.4f}")
print(f"\nBrain MRI TL Training Complete. Artifacts saved in: {SAVE_DIR}")
