"""
Mammography Transfer Learning — RETRAIN SCRIPT
Fixes:
  1) epochs=2 → 15 frozen + 5 fine-tune
  2) X /= 255 during training → preprocess_input (must match inference!)
  3) Added callbacks: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
  4) Fine-tuning phase: unfreeze top 30 layers of MobileNetV2
"""
import os
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input    # ← FIXED
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# ── Paths ─────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR  = os.path.join(BASE_DIR, 'media', 'Mamography Samples')
SAVE_DIR  = os.path.join(BASE_DIR, 'training_results', 'mammography')
os.makedirs(SAVE_DIR, exist_ok=True)

IMG_SIZE   = 224
BATCH_SIZE = 16
EPOCHS_1   = 15
EPOCHS_2   = 5

classes    = {"benign": 0, "InSitu": 1, "Invasive": 2, "Normal": 3}
class_names= ["Benign", "InSitu", "Invasive", "Normal"]

# ── Load and Crop Dataset ──────────────────────────────────────────────
def getCropImgs(img_rgb, needRotations=False):
    z = np.asarray(img_rgb, dtype=np.float32)
    crops = []
    h, w = z.shape[:2]
    rows = min(3, h // 512)
    cols = min(4, w // 512)
    if rows == 0 or cols == 0:
        return [cv2.resize(z, (IMG_SIZE, IMG_SIZE))]
    for i in range(rows):
        for j in range(cols):
            crop = z[512*i:512*(i+1), 512*j:512*(j+1), :]
            crop_resized = cv2.resize(crop, (IMG_SIZE, IMG_SIZE))
            crops.append(crop_resized)
            if needRotations:
                crops.append(np.rot90(np.rot90(crop_resized)).copy())
    return crops

print("Loading Mammography dataset...")
X_raw = []
y_raw = []

for file in os.listdir(DATA_DIR):
    if not (file.endswith('.jpg') or file.endswith('.png')):
        continue
    label = -1
    for key in classes:
        if key.lower() in file.lower():
            label = classes[key]
            break
    if label == -1:
        continue
    img = cv2.imread(os.path.join(DATA_DIR, file))
    if img is None:
        continue
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    crops = getCropImgs(img_rgb, needRotations=False)
    for crop in crops:
        X_raw.append(crop)   # still 0-255 float32
        y_raw.append(label)

print(f"Total patches: {len(X_raw)}")
if len(X_raw) == 0:
    print("No data found — check DATA_DIR:", DATA_DIR)
    exit(1)

X_raw = np.array(X_raw, dtype=np.float32)   # shape (N, 224, 224, 3)  values 0-255
y_raw = np.array(y_raw)
y_cat = to_categorical(y_raw, num_classes=len(classes))

X_train_raw, X_val_raw, y_train, y_val = train_test_split(
    X_raw, y_cat, test_size=0.2, random_state=42, stratify=y_raw
)
y_true_val = np.argmax(y_val, axis=1)

# Apply preprocess_input CORRECTLY: input must be 0-255, output is -1..+1
X_train = preprocess_input(X_train_raw.copy())    # ← FIXED
X_val   = preprocess_input(X_val_raw.copy())      # ← FIXED

print(f"Train: {len(X_train)}  Val: {len(X_val)}")

# ── Build Model ───────────────────────────────────────────────────────
base = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
base.trainable = False

x = base.output
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.4)(x)
x = Dense(64, activation='relu')(x)
x = Dropout(0.3)(x)
out = Dense(len(classes), activation='softmax')(x)

model = Model(inputs=base.input, outputs=out)
model.compile(optimizer=Adam(learning_rate=1e-3),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

# ── Callbacks ─────────────────────────────────────────────────────────
ckpt_path = os.path.join(SAVE_DIR, 'mammography_tl_best.h5')
cb_p1 = [
    ModelCheckpoint(ckpt_path, monitor='val_accuracy', save_best_only=True, verbose=1),
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=3, min_lr=1e-7, verbose=1)
]

# ── Phase 1 ───────────────────────────────────────────────────────────
print("\n" + "="*50)
print("PHASE 1: Training classifier head (base frozen)")
print("="*50)
history1 = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS_1,
    batch_size=BATCH_SIZE,
    callbacks=cb_p1
)

# ── Phase 2: Fine-tune top 30 layers ──────────────────────────────────
print("\n" + "="*50)
print("PHASE 2: Fine-tuning top 30 MobileNetV2 layers")
print("="*50)
base.trainable = True
for layer in base.layers[:-30]:
    layer.trainable = False

model.compile(optimizer=Adam(learning_rate=1e-5),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

cb_p2 = [
    ModelCheckpoint(ckpt_path, monitor='val_accuracy', save_best_only=True, verbose=1),
    EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=2, min_lr=1e-8, verbose=1)
]

history2 = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS_2,
    batch_size=BATCH_SIZE,
    callbacks=cb_p2
)

# ── Save ──────────────────────────────────────────────────────────────
final_path = os.path.join(SAVE_DIR, 'mammography_tl_model.h5')
model.save(final_path)
print(f"\nModel saved: {final_path}")

# ── Plots ─────────────────────────────────────────────────────────────
def concat(h1, h2, key): return h1.history.get(key,[]) + h2.history.get(key,[])
acc      = concat(history1,history2,'accuracy')
val_acc  = concat(history1,history2,'val_accuracy')
loss_v   = concat(history1,history2,'loss')
val_loss = concat(history1,history2,'val_loss')
ep       = range(1, len(acc)+1)
split    = len(history1.history['accuracy'])

fig,axes = plt.subplots(1,2,figsize=(14,5))
axes[0].plot(ep,acc,'b-o',label='Train Acc'); axes[0].plot(ep,val_acc,'g-o',label='Val Acc')
axes[0].axvline(x=split,color='orange',linestyle='--',label='Fine-tune start')
axes[0].set_title('Mammography — Accuracy'); axes[0].legend()
axes[1].plot(ep,loss_v,'r-o',label='Train Loss'); axes[1].plot(ep,val_loss,'m-o',label='Val Loss')
axes[1].axvline(x=split,color='orange',linestyle='--'); axes[1].set_title('Mammography — Loss'); axes[1].legend()
plt.tight_layout(); plt.savefig(os.path.join(SAVE_DIR,'accuracy_loss.png'),dpi=120,bbox_inches='tight'); plt.close()

preds = model.predict(X_val, verbose=1)
y_pred = np.argmax(preds,axis=1)

cm = confusion_matrix(y_true_val, y_pred)
plt.figure(figsize=(6,5)); sns.heatmap(cm,annot=True,fmt='d',cmap='Blues',xticklabels=class_names,yticklabels=class_names)
plt.title('Mammography — Confusion Matrix'); plt.savefig(os.path.join(SAVE_DIR,'confusion_matrix.png'),dpi=120,bbox_inches='tight'); plt.close()

plt.figure()
for i in range(len(classes)):
    fpr,tpr,_ = roc_curve(y_val[:,i],preds[:,i])
    plt.plot(fpr,tpr,lw=2,label=f'{class_names[i]} (AUC={auc(fpr,tpr):.2f})')
plt.plot([0,1],[0,1],'navy',lw=2,linestyle='--')
plt.title('Mammography — ROC Curves'); plt.legend(loc='lower right')
plt.savefig(os.path.join(SAVE_DIR,'roc_curve.png'),dpi=120,bbox_inches='tight'); plt.close()

print("\nClassification Report:")
print(classification_report(y_true_val, y_pred, target_names=class_names))
print(f"\nFinal Val Accuracy: {val_acc[-1]*100:.2f}%")
print(f"\nMammography TL Training Complete. Artifacts saved in: {SAVE_DIR}")
