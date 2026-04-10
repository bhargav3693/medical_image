import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_USE_LEGACY_KERAS'] = '1'

import cv2
import numpy as np
import imutils
from django.conf import settings


def start_process(imagepath):
    import tensorflow as tf
    from tf_keras.models import load_model
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess

    # Limit CPU threads to prevent OOM on Render free (512MB)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)

    img_path = os.path.join(settings.MEDIA_ROOT, imagepath)
    model_path = os.path.join(settings.BASE_DIR, 'models', 'brain_tumor_detector.h5')
    tl_model_path = os.path.join(settings.BASE_DIR, 'models', 'brain_tl_model.h5')

    model = load_model(model_path, compile=False)
    try:
        tl_model = load_model(tl_model_path, compile=False)
    except Exception as e:
        print(f"[WARN] Could not load Brain TL model: {e}")
        tl_model = None

    image_orig = cv2.imread(img_path)
    print('*' * 50)
    print("CNN Model:", model)
    print("TL Model:", tl_model)

    gray = cv2.cvtColor(image_orig, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)

    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)

    extLeft  = tuple(c[c[:, :, 0].argmin()][0])
    extRight = tuple(c[c[:, :, 0].argmax()][0])
    extTop   = tuple(c[c[:, :, 1].argmin()][0])
    extBot   = tuple(c[c[:, :, 1].argmax()][0])

    new_image = image_orig[extTop[1]:extBot[1], extLeft[0]:extRight[0]]

    # --- CNN Inference (240x240, 0-1) ---
    cnn_img = cv2.resize(new_image, (240, 240), interpolation=cv2.INTER_CUBIC)
    cnn_arr = cnn_img.astype(np.float32) / 255.0
    cnn_arr = cnn_arr.reshape((1, 240, 240, 3))
    res = float(model.predict(cnn_arr, verbose=0)[0][0])

    # --- TL Inference: MobileNetV2 preprocess_input (0-255 → -1..+1) ---
    if tl_model is not None:
        try:
            # Ensure RGB conversion (MRI are grayscale, but BGR→RGB ensures 3 channels)
            tl_rgb = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
            tl_img = cv2.resize(tl_rgb, (224, 224), interpolation=cv2.INTER_CUBIC)
            tl_arr = tl_img.astype(np.float32)   # keep 0-255 for preprocess_input
            tl_arr = mobilenet_preprocess(tl_arr) # scale to -1..+1
            tl_arr = tl_arr.reshape((1, 224, 224, 3))
            res_tl = float(tl_model.predict(tl_arr, verbose=0)[0][0])
        except Exception as e:
            print(f"[WARN] Brain TL inference error: {e}")
            res_tl = res  # fallback
    else:
        res_tl = res

    # --- Generate Heatmap with Bounding Box for Brain MRI ---
    heatmap_name = _generate_brain_heatmap(img_path, extLeft, extRight, extTop, extBot)

    # MAGIC FIX: Clear massive memory graph to stop OOM / SIGKILL!
    import gc
    tf.keras.backend.clear_session()
    del model
    if tl_model: del tl_model
    gc.collect()

    return res, res_tl, heatmap_name


def _generate_brain_heatmap(image_path, extLeft, extRight, extTop, extBot):
    """Generate a heatmap overlay with a bounding box marking the brain region."""
    import os, cv2, numpy as np
    from django.conf import settings

    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        return None

    h, w = img_bgr.shape[:2]

    # Convert to grayscale for activation map
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (25, 25), 0)
    norm = cv2.normalize(blurred, None, 0, 255, cv2.NORM_MINMAX)

    heatmap = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img_bgr, 0.55, heatmap, 0.45, 0)

    # Draw bounding box around the brain region found during preprocessing
    bx = max(0, extLeft[0])
    by = max(0, extTop[1])
    bw = min(w, extRight[0]) - bx
    bh = min(h, extBot[1]) - by

    if bw > 10 and bh > 10:
        cv2.rectangle(overlay, (bx, by), (bx + bw, by + bh), (0, 200, 255), 3)
        cl = 22
        for px, py, dx, dy in [
            (bx, by, 1, 1), (bx+bw, by, -1, 1),
            (bx, by+bh, 1, -1), (bx+bw, by+bh, -1, -1)
        ]:
            cv2.line(overlay, (px, py), (px + dx*cl, py), (0, 255, 255), 5)
            cv2.line(overlay, (px, py), (px, py + dy*cl), (0, 255, 255), 5)

    heatmap_name = "heatmap_brain_" + os.path.basename(image_path)
    heatmap_path = os.path.join(settings.MEDIA_ROOT, heatmap_name)
    cv2.imwrite(heatmap_path, overlay)
    return heatmap_name