import cv2
import numpy as np
import torch 
import albumentations as A
import pandas as pd
import os
from torch.utils.data import Dataset, DataLoader

# Set seeds for reproducibility
torch.manual_seed(43)
if torch.cuda.is_available():
    torch.cuda.manual_seed(43)  

geo_transform = A.Compose([
    A.Resize(608, 608, interpolation=cv2.INTER_LINEAR),
    A.HorizontalFlip(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=15, p=0.5),
    A.ToTensorV2()
])

# Colormap for CamVid dataset
df = pd.read_csv("/kaggle/input/camvid/CamVid/class_dict.csv")
df['rgb'] = list(zip(df['r'], df['g'], df['b']))
colormap = {}
for i, val in enumerate(df['rgb']):
    colormap[val] = i

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

# Reduced Maping for CamVid dataset
reduce_mapping = {4: 4,5: 5,9: 9,11: 11,13: 13,14: 13,16: 16,17: 17,18: 17,19: 19,20: 20,21: 21,
    24: 24,22: 5,26: 26,27: 5,30: 30,31: 31,0: 255,10: 255,1: 255,2: 255,3: 255,6: 255,28: 255,
    29: 255,23: 255,25: 255,15: 255,12: 255,7: 255, 8: 255
}

# Augmentation function
def augment_fn(image, mask, n_aug, transform):
    imgs = []
    masks = []
    for _ in range(n_aug):
        augmented = transform(image=image, mask=mask)
        img = augmented['image']           
        mask_out = augmented['mask']         
        mask_out = mask_out.long()
        imgs.append(img)
        masks.append(mask_out)
    return imgs, masks

# Custom Dataset class
class ImageDataset(Dataset):
    def __init__(self, image_folder, mask_folder, colormap, mapping):
        self.image_folder = image_folder
        self.mask_folder = mask_folder
        self.image_paths = sorted(os.listdir(self.image_folder))
        self.mask_paths = sorted(os.listdir(self.mask_folder))
        self.colormap = colormap
        self.mapping = mapping

    def __len__(self):
        return len(self.image_paths)
        
    def __getitem__(self, idx):
        image_path = os.path.join(self.image_folder, self.image_paths[idx])
        mask_path = os.path.join(self.mask_folder, self.mask_paths[idx])
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        label_mask = rgb_to_label(mask, self.colormap)  # 2D array
        remapped_mask = remap_labels(label_mask, self.mapping)  
        return image, remapped_mask

# Custom collate function for DataLoader
def custom_collate(batch, n_aug=8, transform=None):
    final_images = []
    final_masks = []
    for image, mask in batch:
        aug_img, aug_mask = augment_fn(image, mask, n_aug, transform)
        final_images.extend(aug_img)
        final_masks.extend(aug_mask)
    return torch.stack(final_images), torch.stack(final_masks)


train_folder = "/kaggle/working/final_dir"  # Path to your training images
train_mask_folder = "/kaggle/working/final_label_dir"  # Path to your training masks
val_folder = "/kaggle/working/test_dir"               # Path to your validation images
val_mask_folder = "/kaggle/working/test_label_dir"     # Path to your validation masks

# Create the dataset and dataloader
def get_train_loader():
    train_dataset = ImageDataset(train_folder, train_mask_folder, colormap, reduce_mapping)
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=4,
                            collate_fn=lambda batch: custom_collate(batch, n_aug=2, 
                                                                    transform=geo_transform))
    return train_loader

def get_val_loader():
    val_dataset = ImageDataset(val_folder, val_mask_folder, colormap, reduce_mapping)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=True, num_workers=4,
                            collate_fn=lambda batch: custom_collate(batch, n_aug=2,
                                                                     transform=geo_transform))
    return val_loader