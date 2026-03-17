"""
Chest X-Ray Transfer Learning — RETRAIN SCRIPT
Fixes:
  1) epochs=2 → 15 frozen + 5 fine-tune
  2) rescale=1./255 → preprocessing_function=preprocess_input (matches inference!)
  3) Added callbacks: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
  4) Fine-tuning phase: unfreeze top 30 layers of MobileNetV2
"""
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input    # ← FIXED
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# ── Paths ─────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGE_DIR  = os.path.join(BASE_DIR, 'media', 'chest X-ray', 'images')
TRAIN_CSV  = os.path.join(BASE_DIR, 'media', 'nih', 'train-small.csv')
VALID_CSV  = os.path.join(BASE_DIR, 'media', 'nih', 'valid-small.csv')
SAVE_DIR   = os.path.join(BASE_DIR, 'training_results', 'chest')
os.makedirs(SAVE_DIR, exist_ok=True)

IMG_SIZE   = 224
BATCH_SIZE = 32
EPOCHS_1   = 15
EPOCHS_2   = 5

labels = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion',
          'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass', 'Nodule',
          'Pleural_Thickening', 'Pneumonia', 'Pneumothorax']

# ── Load CSVs ─────────────────────────────────────────────────────────
print("Loading CSV files...")
train_df = pd.read_csv(TRAIN_CSV)
valid_df = pd.read_csv(VALID_CSV)

filename_col = 'Image' if 'Image' in train_df.columns else 'id'

def check_exists(img_id):
    return os.path.exists(os.path.join(IMAGE_DIR, img_id))

train_df = train_df[train_df[filename_col].apply(check_exists)]
valid_df = valid_df[valid_df[filename_col].apply(check_exists)]

print(f"Train images: {len(train_df)}  |  Val images: {len(valid_df)}")
if len(train_df) == 0:
    print("No images found in:", IMAGE_DIR)
    exit(1)

# ── Data Generators ───────────────────────────────────────────────────
# CRITICAL: Use preprocessing_function=preprocess_input (NOT rescale=1/255)
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,   # ← FIXED
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input    # ← FIXED (same as inference)
)

train_gen = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    directory=IMAGE_DIR,
    x_col=filename_col,
    y_col=labels,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='raw'
)

val_gen = val_datagen.flow_from_dataframe(
    dataframe=valid_df,
    directory=IMAGE_DIR,
    x_col=filename_col,
    y_col=labels,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='raw',
    shuffle=False
)

# ── Build Model ───────────────────────────────────────────────────────
base = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
base.trainable = False

x = base.output
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.4)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
out = Dense(len(labels), activation='sigmoid')(x)   # multi-label

model = Model(inputs=base.input, outputs=out)
model.compile(optimizer=Adam(learning_rate=1e-3),
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.summary()

# ── Callbacks ─────────────────────────────────────────────────────────
ckpt_path = os.path.join(BASE_DIR, 'models', 'chest_tl_best.h5')
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
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS_1,
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
              loss='binary_crossentropy',
              metrics=['accuracy'])

cb_p2 = [
    ModelCheckpoint(ckpt_path, monitor='val_accuracy', save_best_only=True, verbose=1),
    EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=2, min_lr=1e-8, verbose=1)
]

history2 = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS_2,
    callbacks=cb_p2
)

# ── Save ──────────────────────────────────────────────────────────────
final_path = os.path.join(BASE_DIR, 'models', 'chest_tl_model.h5')
model.save(final_path)
print(f"\nModel saved: {final_path}")

# ── Plots ─────────────────────────────────────────────────────────────
def concat(h1,h2,k): return h1.history.get(k,[])+h2.history.get(k,[])
acc      = concat(history1,history2,'accuracy')
val_acc  = concat(history1,history2,'val_accuracy')
loss_v   = concat(history1,history2,'loss')
val_loss = concat(history1,history2,'val_loss')
ep       = range(1,len(acc)+1)
split    = len(history1.history['accuracy'])

fig,axes = plt.subplots(1,2,figsize=(14,5))
axes[0].plot(ep,acc,'b-o',label='Train Acc'); axes[0].plot(ep,val_acc,'g-o',label='Val Acc')
axes[0].axvline(x=split,color='orange',linestyle='--',label='Fine-tune start')
axes[0].set_title('Chest X-Ray — Accuracy'); axes[0].legend()
axes[1].plot(ep,loss_v,'r-o',label='Train Loss'); axes[1].plot(ep,val_loss,'m-o',label='Val Loss')
axes[1].axvline(x=split,color='orange',linestyle='--'); axes[1].set_title('Chest X-Ray — Loss'); axes[1].legend()
plt.tight_layout(); plt.savefig(os.path.join(SAVE_DIR,'accuracy_loss.png'),dpi=120,bbox_inches='tight'); plt.close()

# Evaluate ROC Curves
val_gen.reset()
preds = model.predict(val_gen, verbose=1)
y_true = valid_df[labels].values

plt.figure(figsize=(12,8))
for i in range(len(labels)):
    try:
        fpr, tpr, _ = roc_curve(y_true[:,i], preds[:,i])
        plt.plot(fpr,tpr,lw=2,label=f'{labels[i]} (AUC={auc(fpr,tpr):.2f})')
    except Exception:
        pass
plt.plot([0,1],[0,1],'navy',lw=2,linestyle='--')
plt.title('Chest X-Ray — ROC Curves'); plt.legend(loc='lower right',prop={'size':8})
plt.savefig(os.path.join(SAVE_DIR,'roc_curve.png'),dpi=120,bbox_inches='tight'); plt.close()

y_pred_bin = (preds > 0.5).astype(int)
print("\nClassification Report:")
print(classification_report(y_true, y_pred_bin, target_names=labels))
print(f"\nFinal Val Accuracy: {val_acc[-1]*100:.2f}%")
print(f"\nChest TL Training Complete. Artifacts saved in: {SAVE_DIR}")
