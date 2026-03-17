import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Let's write a generic Grad-CAM function
def get_img_array(img_path, size):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=size)
    array = tf.keras.preprocessing.image.img_to_array(img)
    array = np.expand_dims(array, axis=0) / 255.0
    return array

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        model.inputs, [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def save_and_display_gradcam(img_path, heatmap, cam_path="cam.jpg", alpha=0.4):
    img = cv2.imread(img_path)
    # Ensure heatmap is nan safe
    heatmap = np.nan_to_num(heatmap)

    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    superimposed_img = heatmap_colored * alpha + img
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)
    
    # Bounding box derived from heatmap
    _, thresh = cv2.threshold(heatmap, 150, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(superimposed_img, (x, y), (x+w, y+h), (0, 0, 255), 2)
        cv2.putText(superimposed_img, "Disease Region", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imwrite(cam_path, superimposed_img)

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # We will pick one image for each
    brain_img = os.path.join(base_dir, 'media', 'MRI Stroke', 'test', 'yes', 'Y168.jpg')
    mammo_img = os.path.join(base_dir, 'media', 'Mamography Samples', 'InSitu1.jpg')
    chest_img = os.path.join(base_dir, 'media', 'chest X-ray', 'images', '00000001_000.png')

    reports_dir = os.path.join(base_dir, 'training_results', 'evaluation')
    os.makedirs(reports_dir, exist_ok=True)

    # Brain
    if os.path.exists(brain_img):
        try:
            brain_model = load_model(os.path.join(base_dir, 'models', 'brain_tl_model.h5'))
            img_arr = get_img_array(brain_img, size=(224, 224))
            # MobileNetV2 typically has an output layer 'out_relu'
            # Let's find the last conv layer in MobileNetV2
            last_conv_layer = "out_relu"
            hm = make_gradcam_heatmap(img_arr, brain_model, last_conv_layer)
            save_and_display_gradcam(brain_img, hm, os.path.join(reports_dir, 'brain_heatmap.jpg'))
            print("Brain heatmap generated.")
        except Exception as e:
            print("Brain heatmap failed:", e)

    # Mammography
    if os.path.exists(mammo_img):
        try:
            mammo_model = load_model(os.path.join(base_dir, 'models', 'mammography_tl_model.h5'))
            img_arr = get_img_array(mammo_img, size=(224, 224))
            last_conv_layer = "out_relu"
            hm = make_gradcam_heatmap(img_arr, mammo_model, last_conv_layer)
            save_and_display_gradcam(mammo_img, hm, os.path.join(reports_dir, 'mammography_heatmap.jpg'))
            print("Mammography heatmap generated.")
        except Exception as e:
            print("Mammography heatmap failed:", e)

    # Chest
    if os.path.exists(chest_img):
        try:
            chest_model = load_model(os.path.join(base_dir, 'models', 'chest_tl_model.h5'))
            img_arr = get_img_array(chest_img, size=(224, 224))
            last_conv_layer = "out_relu"
            hm = make_gradcam_heatmap(img_arr, chest_model, last_conv_layer)
            save_and_display_gradcam(chest_img, hm, os.path.join(reports_dir, 'chest_heatmap.jpg'))
            print("Chest heatmap generated.")
        except Exception as e:
            print("Chest heatmap failed:", e)
    
    print("Heatmap generation complete.")
