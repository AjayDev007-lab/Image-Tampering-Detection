import os
import cv2
import shutil
import random

CASIA_DIR = r"C:\Users\anto5\Desktop\tamper_project\dataset\CASIA2"
TP_DIR = os.path.join(CASIA_DIR, "Tp")
GT_DIR = os.path.join(CASIA_DIR, "Gt")

OUT = r"C:\Users\anto5\Desktop\tamper_project\dataset"

train_ratio = 0.8

img_train = os.path.join(OUT, "images/train")
img_val = os.path.join(OUT, "images/val")
lbl_train = os.path.join(OUT, "labels/train")
lbl_val = os.path.join(OUT, "labels/val")

os.makedirs(img_train, exist_ok=True)
os.makedirs(img_val, exist_ok=True)
os.makedirs(lbl_train, exist_ok=True)
os.makedirs(lbl_val, exist_ok=True)

images = [f for f in os.listdir(TP_DIR) if f.endswith(".jpg")]
random.shuffle(images)

split = int(len(images) * train_ratio)
train_imgs = images[:split]
val_imgs = images[split:]


def create_bbox(mask_path):
    mask = cv2.imread(mask_path, 0)
    if mask is None:
        return None

    _, thresh = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    cnt = max(contours, key=cv2.contourArea)

    x, y, w, h = cv2.boundingRect(cnt)

    H, W = mask.shape

    x_c = (x + w/2) / W
    y_c = (y + h/2) / H
    w_n = w / W
    h_n = h / H

    return x_c, y_c, w_n, h_n


def process(images, img_out, lbl_out):
    for img_name in images:
        img_path = os.path.join(TP_DIR, img_name)

        mask_name = img_name.replace(".jpg", "_gt.png")
        mask_path = os.path.join(GT_DIR, mask_name)

        bbox = create_bbox(mask_path)
        if bbox is None:
            continue

        shutil.copy(img_path, os.path.join(img_out, img_name))

        label_path = os.path.join(lbl_out, img_name.replace(".jpg", ".txt"))

        with open(label_path, "w") as f:
            f.write(f"0 {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}\n")


process(train_imgs, img_train, lbl_train)
process(val_imgs, img_val, lbl_val)

print("✅ REAL bounding-box dataset created!")