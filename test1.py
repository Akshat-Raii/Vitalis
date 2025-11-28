"""
eval_and_plots.py

Creates:
 - files/confusion_matrix.png
 - files/roc_curve.png
 - files/train_val_loss.png
 - files/train_val_metric.png

Ensure: train.py's load_dataset, unet.py, metrics.py are importable and files/model.h5 exists.
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras.utils import CustomObjectScope
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc, classification_report
)
from metrics import dice_loss, dice_coef
from train import load_dataset  # assumes your train.py is in same folder and exposes load_dataset

# -------------------- settings --------------------
H = 256
W = 256
RESULT_DIR = "files"
os.makedirs(RESULT_DIR, exist_ok=True)
SAVE_CONFUSION = os.path.join(RESULT_DIR, "confusion_matrix.png")
SAVE_ROC = os.path.join(RESULT_DIR, "roc_curve.png")
SAVE_LOSS = os.path.join(RESULT_DIR, "train_val_loss.png")
SAVE_METRIC = os.path.join(RESULT_DIR, "train_val_metric.png")
CSV_LOG = os.path.join("files", "log.csv")  # CSVLogger output during training
MODEL_PATH = os.path.join("files", "model.h5")
BATCH_SIZE = 1  # we will predict image-by-image (keeps memory low)
THRESHOLD = 0.5  # for confusion matrix and binary metrics
DPI = 300
# --------------------------------------------------

def read_image_as_array(path, H=H, W=W):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (W, H))
    arr = img.astype(np.float32) / 255.0
    return arr, img  # normalized for model, and original 0-255 for saving if needed

def read_mask_as_array(path, H=H, W=W):
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    m = cv2.resize(m, (W, H))
    # keep values 0..255; caller will normalize/threshold as required
    return m

def save_prediction_visual(image_rgb, mask_gray, pred_binary, save_path):
    # image_rgb: original resized BGR (0..255)
    # mask_gray: grayscale 0..255
    # pred_binary: binary (0/1) HxW
    # produce side-by-side image: input | mask | pred
    mask_rgb = np.stack([mask_gray, mask_gray, mask_gray], axis=-1)
    pred_vis = (pred_binary.astype(np.uint8) * 255)
    pred_rgb = np.stack([pred_vis, pred_vis, pred_vis], axis=-1)

    sep = np.ones((H, 10, 3), dtype=np.uint8) * 255
    concat = np.concatenate([image_rgb, sep, mask_rgb, sep, pred_rgb], axis=1)
    cv2.imwrite(save_path, concat)

def main():
    # load model
    with CustomObjectScope({"dice_loss": dice_loss, "dice_coef": dice_coef}):
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    print("Model loaded:", MODEL_PATH)

    # load dataset splits (adapt path if necessary)
    dataset_path = "D:/U_Net/Brain_Tumor/archive"  # as in your script
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_dataset(dataset_path)
    print(f"Test images: {len(test_x)}")

    # containers for pixel-wise evaluation
    all_gt_pixels = []
    all_pred_probs_pixels = []
    # optionally save small visual results in results/ (already done in your script), but we save a few examples here
    vis_dir = os.path.join("results", "eval_vis")
    os.makedirs(vis_dir, exist_ok=True)

    # iterate test set
    for idx, (x_path, y_path) in enumerate(tqdm(list(zip(test_x, test_y)), total=len(test_x))):
        # read inputs
        x_norm, x_orig = read_image_as_array(x_path)  # normalized and original (BGR)
        mask_gray = read_mask_as_array(y_path)

        # predict probability map
        x_in = np.expand_dims(x_norm, axis=0)  # shape (1,H,W,3)
        pred_prob = model.predict(x_in, verbose=0)[0]  # shape (H,W,1)
        pred_prob = np.squeeze(pred_prob, axis=-1)  # (H,W)
        pred_binary = (pred_prob >= THRESHOLD).astype(np.uint8)

        # collect flattened pixels (gt normalized to 0/1)
        gt_binary = (mask_gray / 255.0) >= 0.5
        all_gt_pixels.append(gt_binary.reshape(-1).astype(np.uint8))
        all_pred_probs_pixels.append(pred_prob.reshape(-1).astype(np.float32))

        # optionally save a handful of visuals (first 20)
        if idx < 20:
            name = os.path.basename(x_path)
            save_path = os.path.join(vis_dir, name)
            save_prediction_visual(x_orig, mask_gray, pred_binary, save_path)

    all_gt_pixels = np.concatenate(all_gt_pixels, axis=0)
    all_pred_probs_pixels = np.concatenate(all_pred_probs_pixels, axis=0)

    print("Total pixels:", all_gt_pixels.shape[0])

    # ---------------- ROC curve (pixel-wise) ----------------
    fpr, tpr, roc_thresh = roc_curve(all_gt_pixels, all_pred_probs_pixels)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6,6), dpi=DPI)
    plt.plot(fpr, tpr, linewidth=2, label=f"Pixel-wise ROC (AUC = {roc_auc:.4f})")
    plt.plot([0,1],[0,1], linestyle="--", linewidth=1, color="gray")
    plt.xlabel("False Positive Rate", fontsize=14)
    plt.ylabel("True Positive Rate", fontsize=14)
    plt.title("Pixel-wise ROC Curve", fontsize=16)
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(alpha=0.3, linestyle="--")
    plt.tight_layout()
    plt.savefig(SAVE_ROC, dpi=DPI, bbox_inches="tight")
    plt.close()
    print("Saved ROC curve to:", SAVE_ROC)

    # ---------------- Confusion matrix (pixel-wise) ----------------
    # Binarize predicted probabilities at THRESHOLD for confusion matrix
    pred_bin_pixels = (all_pred_probs_pixels >= THRESHOLD).astype(np.uint8)
    cm = confusion_matrix(all_gt_pixels, pred_bin_pixels)
    # if binary, make sure shape (2,2)
    if cm.shape != (2,2):
        # handle rare corner where only one class exists
        # Expand to 2x2 with zeros
        tmp = np.zeros((2,2), dtype=int)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                tmp[i,j] = cm[i,j]
        cm = tmp

    # normalize for display (row-wise)
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-12)

    plt.figure(figsize=(6,6), dpi=DPI)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Pred 0", "Pred 1"], yticklabels=["True 0", "True 1"])
    plt.ylabel("True label", fontsize=14)
    plt.xlabel("Predicted label", fontsize=14)
    plt.title(f"Pixel-wise Confusion Matrix (th={THRESHOLD})", fontsize=16)
    plt.tight_layout()
    plt.savefig(SAVE_CONFUSION, dpi=DPI, bbox_inches="tight")
    plt.close()
    print("Saved Confusion matrix to:", SAVE_CONFUSION)
    print("Confusion matrix (counts):\n", cm)
    print("\nClassification report (pixel-wise):\n")
    try:
        # small helper to print classification_report
        from sklearn.metrics import classification_report as cr
        print(cr(all_gt_pixels, pred_bin_pixels, digits=4))
    except Exception:
        pass

    # ---------------- Train/Validation curves from CSVLogger ----------------
    if os.path.exists(CSV_LOG):
        df = pd.read_csv(CSV_LOG)
        # find loss columns
        loss_col = "loss" if "loss" in df.columns else None
        val_loss_col = None
        # typical val loss column naming: "val_loss"
        for c in df.columns:
            if c.startswith("val_") and "loss" in c:
                val_loss_col = c
                break
        if loss_col and val_loss_col:
            plt.figure(figsize=(6,5), dpi=DPI)
            plt.plot(df[loss_col], label="train loss", linewidth=2)
            plt.plot(df[val_loss_col], label="val loss", linewidth=2)
            plt.xlabel("Epoch", fontsize=14)
            plt.ylabel("Loss", fontsize=14)
            plt.title("Train / Validation Loss", fontsize=16)
            plt.legend(fontsize=12)
            plt.grid(alpha=0.3, linestyle="--")
            plt.tight_layout()
            plt.savefig(SAVE_LOSS, dpi=DPI, bbox_inches="tight")
            plt.close()
            print("Saved train/val loss to:", SAVE_LOSS)
        else:
            print("Could not find loss/val_loss columns in", CSV_LOG)

        # plot metric (if present). Commonly 'dice_coef' and 'val_dice_coef'
        # try to find a metric column (non-loss, non-val_loss)
        metric_col = None
        val_metric_col = None
        # pick first pair of metric/val_metric that is not 'loss' or 'val_loss'
        for c in df.columns:
            if c in ("epoch", "loss", "val_loss"):
                continue
            if not c.startswith("val_"):
                # check if val counterpart exists
                if "val_" + c in df.columns:
                    metric_col = c
                    val_metric_col = "val_" + c
                    break
        if metric_col and val_metric_col:
            plt.figure(figsize=(6,5), dpi=DPI)
            plt.plot(df[metric_col], label=f"train {metric_col}", linewidth=2)
            plt.plot(df[val_metric_col], label=f"val {metric_col}", linewidth=2)
            plt.xlabel("Epoch", fontsize=14)
            plt.ylabel(metric_col, fontsize=14)
            plt.title(f"Train / Validation {metric_col}", fontsize=16)
            plt.legend(fontsize=12)
            plt.grid(alpha=0.3, linestyle="--")
            plt.tight_layout()
            plt.savefig(SAVE_METRIC, dpi=DPI, bbox_inches="tight")
            plt.close()
            print("Saved train/val metric to:", SAVE_METRIC)
        else:
            print("Could not find a metric + val_metric pair in", CSV_LOG)
    else:
        print("CSV log not found at:", CSV_LOG)
        print("If you ran training in a different location, set CSV_LOG path accordingly.")

if __name__ == "__main__":
    main()
