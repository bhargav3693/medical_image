import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'

from users.utility.generative_ai import get_clinical_advice

import cv2
import numpy as np
from django.conf import settings
from tf_keras.utils import load_img
from tf_keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess


def generate_heatmap_with_bbox(image_path, prefix="mammo"):
    """
    Activation-based heatmap with bounding box for mammography images.
    """
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        return None

    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (31, 31), 0)

    # Invert for dark-background mammography images (tissue appears bright)
    # Keep if tissue is bright
    if np.mean(gray) < 100:
        blurred = cv2.bitwise_not(blurred)

    norm = cv2.normalize(blurred, None, 0, 255, cv2.NORM_MINMAX)
    heatmap = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img_bgr, 0.55, heatmap, 0.45, 0)

    _, thresh = cv2.threshold(norm, 175, 255, cv2.THRESH_BINARY)
    kernel = np.ones((7, 7), np.uint8)
    thresh = cv2.dilate(thresh, kernel, iterations=4)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        c = max(contours, key=cv2.contourArea)
        bx, by, bw, bh = cv2.boundingRect(c)
        pad_x = max(8, int(bw * 0.06))
        pad_y = max(8, int(bh * 0.06))
        bx = max(0, bx - pad_x)
        by = max(0, by - pad_y)
        bw = min(w - bx, bw + 2 * pad_x)
        bh = min(h - by, bh + 2 * pad_y)
        # Bounding box in magenta/red for mammography
        cv2.rectangle(overlay, (bx, by), (bx + bw, by + bh), (80, 0, 230), 3)
        cl = 22
        for px, py, dx, dy in [
            (bx, by, 1, 1), (bx+bw, by, -1, 1),
            (bx, by+bh, 1, -1), (bx+bw, by+bh, -1, -1)
        ]:
            cv2.line(overlay, (px, py), (px + dx*cl, py), (0, 80, 255), 5)
            cv2.line(overlay, (px, py), (px, py + dy*cl), (0, 80, 255), 5)

    heatmap_name = f"heatmap_{prefix}_" + os.path.basename(image_path)
    heatmap_path = os.path.join(settings.MEDIA_ROOT, heatmap_name)
    cv2.imwrite(heatmap_path, overlay)
    return heatmap_name


def start_process(imagepath):

    img_path = os.path.join(settings.MEDIA_ROOT, imagepath)
    model_path = os.path.join(settings.BASE_DIR, 'models', "mammography_model3.h5")
    tl_model_path = os.path.join(settings.BASE_DIR, 'models', 'mammography_tl_model.h5')

    model = load_model(model_path, compile=False)
    try:
        tl_model = load_model(tl_model_path, compile=False)
    except Exception as e:
        print(f"[WARN] Could not load TL model: {e}")
        tl_model = None

    classes = ["Benign", "InSitu", "Invasive", "Normal"]

    def getCropImgs(img, needRotations=False):
        z = np.asarray(img, dtype=np.float32)
        crops = []
        for i in range(3):
            for j in range(4):
                crop = z[512*i:512*(i+1), 512*j:512*(j+1), :]
                crops.append(crop)
                if needRotations:
                    crops.append(np.rot90(np.rot90(crop)))
        return crops

    def softmaxToProbs(soft):
        z_exp = np.exp(soft[0] - np.max(soft[0]))  # numerically stable softmax
        return (z_exp / np.sum(z_exp)) * 100

    def predict_crop(crop_img_0_255):
        """
        crop_img_0_255: a 512x512x3 float32 image in 0–255 range
        """
        target_size = (model.input_shape[1], model.input_shape[2])
        crop_resized_cnn = cv2.resize(crop_img_0_255, target_size)
        
        # --- CNN: normalise to 0-1 as trained ---
        cnn_in = np.expand_dims(crop_resized_cnn / 255.0, axis=0)
        cnn_out = model.predict(cnn_in, verbose=0)
        cnn_probs = softmaxToProbs(cnn_out)
        cnn_idx = int(np.argmax(cnn_probs))

        if tl_model is not None:
            # --- TL: resize to 224 and apply MobileNetV2 preprocess_input ---
            crop_resized = cv2.resize(crop_img_0_255, (224, 224))
            tl_in = mobilenet_preprocess(crop_resized.copy())  # -1..+1  ← THE FIX
            tl_in = np.expand_dims(tl_in, axis=0)
            tl_out = tl_model.predict(tl_in, verbose=0)
            # Handle both softmax and sigmoid output
            if tl_out.shape[-1] == len(classes):
                tl_probs = tl_out[0] * 100
            else:
                # Wrong number of outputs, fall back to cnn
                tl_probs = np.zeros(len(classes))
                tl_probs[cnn_idx] = cnn_probs[cnn_idx]
        else:
            tl_probs = np.zeros(len(classes))
            tl_probs[cnn_idx] = cnn_probs[cnn_idx]

        return cnn_idx, cnn_probs, tl_probs

    def predictImage():
        # Load using cv2 for explicit control as requested
        img_bgr = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_arr = img_rgb.astype(np.float32)

        crops = getCropImgs(img_arr, needRotations=False)

        sum_cnn = np.zeros(len(classes))
        sum_tl  = np.zeros(len(classes))

        for i, crop in enumerate(crops):
            _, cp, tp = predict_crop(crop)
            sum_cnn += cp
            sum_tl  += tp

        n = len(crops)
        cnn_idx       = int(np.argmax(sum_cnn))
        tl_idx        = int(np.argmax(sum_tl))
        prediction    = classes[cnn_idx]
        prediction_tl = classes[tl_idx]
        cnn_confidence = float(sum_cnn[cnn_idx] / n)
        tl_confidence  = float(sum_tl[tl_idx] / n)

        print(f"CNN: {prediction} @ {cnn_confidence:.2f}%")
        print(f"TL:  {prediction_tl} @ {tl_confidence:.2f}%")

        heatmap_name = generate_heatmap_with_bbox(img_path, prefix="mammo")

        # --- AI Clinical Suggestion ---
        dominant = prediction_tl if tl_confidence > cnn_confidence else prediction
        dynamic_suggestion = get_clinical_advice(dominant, 'mammography')

        return prediction, prediction_tl, cnn_confidence, tl_confidence, heatmap_name, dynamic_suggestion

    return predictImage()