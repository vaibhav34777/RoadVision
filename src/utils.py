import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sampling import inference

# Mapping for CamVid dataset
def rgb_to_label(mask, colormap, default_label=0):
    label_mask = np.full(mask.shape[:2], default_label, dtype=np.uint8)
    for rgb, label in colormap.items():
        matches = np.all(mask == np.array(rgb, dtype=np.uint8), axis=-1)
        label_mask[matches] = label
    return label_mask

# Reduce number of classes in the label mask
def remap_labels(label_mask, mapping, ignore_value=255):
    remapped = np.full(label_mask.shape, ignore_value, dtype=np.uint8)
    for orig_label, new_label in mapping.items():
        remapped[label_mask == orig_label] = new_label
    return remapped

# Custom colormap for output
custom_colormap = {(64, 128, 64): 0,(192, 0, 128): 1,(0, 128, 192): 2,(0, 128, 64): 3,(200, 200, 0): 4,
 (0, 0, 128): 5,(64, 0, 192): 6,(192, 128, 64): 7,(192, 192, 128): 8,(64, 64, 128): 9,(128, 0, 192): 10,
 (192, 0, 64): 11,(128, 128, 64): 12,(192, 0, 192): 13,(128, 64, 64): 14,(64, 192, 128): 15,(64, 64, 0): 16,
 (64, 192, 30): 17,(128, 128, 192): 18,(192, 0, 0): 19,(192, 128, 128): 20,(135, 206, 235): 21,
 (64, 128, 192): 22,(0, 0, 64): 23,(0, 64, 64): 24,(192, 64, 128): 25,(128, 150, 0): 26,(192, 128, 192): 27,
 (64, 0, 64): 28, (192, 192, 0): 29,(106, 90, 205): 30,(64, 128, 128): 31}


# Calculating pixel accuracy
def pixel_accuracy(preds, labels):
    correct = (preds == labels).sum().item()
    total = preds.numel()
    return correct, total
    
# Convert label mask to RGB using the custom colormap
def label_to_rgb(label_mask, custom_colormap=custom_colormap):
    h, w = label_mask.shape
    rgb_mask = np.zeros((h, w, 3), dtype=np.uint8)
    inv_colormap = {v: k for k, v in custom_colormap.items()}
    for label_val, rgb in inv_colormap.items():
        rgb_mask[label_mask == label_val] = rgb
    return rgb_mask

# For visualizing the output of segmentation model
def visualize_prediction(img_path, true_mask_path):
    original = Image.open(img_path).convert("RGB")
    pred_rgb = inference(img_path)  
    true_label = Image.open(true_mask_path).convert('RGB')

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(original)
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(pred_rgb)
    plt.title("Model Output")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(true_label)
    plt.title("Ground Truth Label")
    plt.axis('off')

    plt.tight_layout()
    plt.show()
