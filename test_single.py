import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.utils import CustomObjectScope
from metrics import dice_coef, dice_loss

H, W = 256, 256
model_path = "files/model.h5"
output_mask_dir = "generated_masks"

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def generate_mask(image_path):
    with CustomObjectScope({"dice_coef": dice_coef, "dice_loss": dice_loss}):
        model = tf.keras.models.load_model(model_path)

    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image = cv2.resize(image, (W, H))
    input_img = image / 255.0
    input_img = np.expand_dims(input_img, axis=0)

    pred = model.predict(input_img, verbose=0)[0]
    pred = np.squeeze(pred, axis=-1)
    mask = (pred >= 0.5).astype(np.uint8) * 255

    create_dir(output_mask_dir)
    filename = os.path.basename(image_path)
    mask_path = os.path.join(output_mask_dir, f"mask_{filename}")
    cv2.imwrite(mask_path, mask)
    print(f"Mask saved at: {mask_path}")

if __name__ == "__main__":
    image_path = "D:/U_Net/Brain_Tumor/mri1.png"
    generate_mask(image_path)
