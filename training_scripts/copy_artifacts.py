import os
import shutil

src_base = r"c:\Users\botta\Desktop\project\medicals\training_results"
dst_base = r"c:\Users\botta\.gemini\antigravity\brain\dd00534e-7de1-426e-8374-fc9479c07168"

for model in ["brain", "mammography", "chest"]:
    src_dir = os.path.join(src_base, model)
    if os.path.exists(src_dir):
        for f in ["accuracy_loss.png", "confusion_matrix.png", "roc_curve.png"]:
            src_f = os.path.join(src_dir, f)
            if os.path.exists(src_f):
                dst_f = os.path.join(dst_base, f"{model}_{f}")
                shutil.copy(src_f, dst_f)
                
# evaluation heatmaps already copied successfully as brain_heatmap.jpg etc.
print("Copied successfully.")
