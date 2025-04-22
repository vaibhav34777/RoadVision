import os 
from PIL import Image
import cv2
import numpy as np
import torch 
from torch import nn
import streamlit as st
import torch.nn.functional as F
from ultralytics import YOLO
import albumentations as A
from albumentations.pytorch import ToTensorV2
import tempfile

# MODEL DEFINITION
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, max_pooling=True, dropout=False, dropout_prob=0.3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=2)
        self.dropout = nn.Dropout2d(p=dropout_prob)
        self.max_pooling = max_pooling
        self.drop = dropout

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        if self.drop:
            x = self.dropout(x)
        skip = x
        next = self.max_pool(x) if self.max_pooling else x
        return next, skip

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.trans_conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.relu = nn.ReLU()

    def forward(self, expansive, contractive):
        up = self.trans_conv(expansive)
        diffY = contractive.size(2) - up.size(2)
        diffX = contractive.size(3) - up.size(3)
        up = F.pad(up, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([up, contractive], dim=1)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        return x

class UNet(nn.Module):
    def __init__(self, in_channels, base_filters, out_classes):
        super().__init__()
        self.conv1 = ConvBlock(in_channels, base_filters)
        self.conv2 = ConvBlock(base_filters, 2 * base_filters)
        self.conv3 = ConvBlock(2 * base_filters, 4 * base_filters)
        self.conv4 = ConvBlock(4 * base_filters, 8 * base_filters, dropout=True)
        self.conv5 = ConvBlock(8 * base_filters, 16 * base_filters, max_pooling=False, dropout=True)
        self.up6 = UpBlock(16 * base_filters, 8 * base_filters)
        self.up7 = UpBlock(8 * base_filters, 4 * base_filters)
        self.up8 = UpBlock(4 * base_filters, 2 * base_filters)
        self.up9 = UpBlock(2 * base_filters, base_filters)
        self.conv10_1 = nn.Conv2d(base_filters, base_filters, kernel_size=3, padding=1)
        self.conv10_2 = nn.Conv2d(base_filters, base_filters, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.final_conv = nn.Conv2d(base_filters, out_classes, kernel_size=1)

    def forward(self, x):
        n1, s1 = self.conv1(x)
        n2, s2 = self.conv2(n1)
        n3, s3 = self.conv3(n2)
        n4, s4 = self.conv4(n3)
        n5, _ = self.conv5(n4)
        d = self.up6(n5, s4)
        d = self.up7(d, s3)
        d = self.up8(d, s2)
        d = self.up9(d, s1)
        d = self.relu(self.conv10_1(d))
        d = self.relu(self.conv10_2(d))
        d = self.final_conv(d)
        d = F.interpolate(d, size=(608, 608), mode='bilinear', align_corners=False)
        return d
    
@st.cache_resource
def load_models():
    # segmentation
    base_filters, out_classes, in_channels = 32, 32, 3
    seg_model = UNet(in_channels, base_filters, out_classes)
    ckpt = torch.load("model/model.pth", map_location="cpu")
    seg_model.load_state_dict(ckpt)
    seg_model.eval()
    yolo_model = YOLO("yolov8n.pt")
    return seg_model, yolo_model

seg_model, yolo_model = load_models()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seg_model.to(device)

custom_colormap = {(64, 128, 64): 0,
 (192, 0, 128): 1,
 (0, 128, 192): 2,
 (0, 128, 64): 3,
 (200, 200, 0): 4,
 (0, 0, 128): 5,
 (64, 0, 192): 6,
 (192, 128, 64): 7,
 (192, 192, 128): 8,
 (64, 64, 128): 9,
 (128, 0, 192): 10,
 (192, 0, 64): 11,
 (128, 128, 64): 12,
 (192, 0, 192): 13,
 (128, 64, 64): 14,
 (64, 192, 128): 15,
 (64, 64, 0): 16,
 (64, 192, 30): 17,
 (128, 128, 192): 18,
 (192, 0, 0): 19,
 (192, 128, 128): 20,
 (135, 206, 235): 21,
 (64, 128, 192): 22,
 (0, 0, 64): 23,
 (0, 64, 64): 24,
 (192, 64, 128): 25,
 (128, 150, 0): 26,
 (192, 128, 192): 27,
 (64, 0, 64): 28,
 (192, 192, 0): 29,
 (106, 90, 205): 30,
 (64, 128, 128): 31}

transform = A.Compose([
    A.Resize(608, 608, interpolation=cv2.INTER_LINEAR),
    ToTensorV2(),
])

def label_to_rgb(label_mask, colormap):
    h, w = label_mask.shape
    rgb_mask = np.zeros((h, w, 3), dtype=np.uint8)
    inv_colormap = {v: k for k, v in colormap.items()}
    for label_val, rgb in inv_colormap.items():
        rgb_mask[label_mask == label_val] = rgb
    return rgb_mask

def inference(img_path):
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = transform(image=image)['image']
    image = image.float()/255.0
    image = image.unsqueeze(0)
    image = image.to(device)
    model.eval()
    logits = model(image)
    probs = F.softmax(logits,dim=1)
    preds = torch.argmax(probs,dim=1)
    preds = preds.squeeze(0).cpu().numpy()
    output = label_to_rgb(preds, custom_colormap)
    return output


def final_output(img_path):
    CLASS_COLORS = {'person':(0,255,255), 'bicycle':(255,0,255), 'bus':(0,128,255)}
    orig_bgr = cv2.imread(img_path)
    orig_rgb = cv2.cvtColor(orig_bgr,cv2.COLOR_BGR2RGB)
    image = transform(image=orig_rgb)['image']
    image = image.float()/255.0
    image = image.unsqueeze(0)
    yolo_results = yolo_model(image)
    seg_output = inference(img_path)
    for r in yolo_results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            #label = r.names[int(box.cls[0])]
            cls = r.names[int(box.cls[0])]
            conf = box.conf[0]
            color = CLASS_COLORS.get(cls, (255,255,255))
            cv2.rectangle(seg_output,(x1,y1),(x2,y2),color,2)
            label=f"{cls} {conf:.2f}"
            tw,th = cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX,0.5,1)[0]
            cv2.rectangle(seg_output,(x1,y1-th-4),(x1+tw+2,y1),color,-1)
            cv2.putText(seg_output,label,(x1,y1-4),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),1)
    return seg_output

st.title("Segmentation + YOLO Viewer")
option = st.radio("Choose input method:", ["Upload an image", "Use webcam"])

def show_output(image_path):
    output_img = final_output(image_path)  
    st.write("## Segmentation + Detection Output")
    st.image(output_img, use_column_width=True)

# Upload image from file
if option == "Upload an image":
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            tmp.write(uploaded_file.read())
            temp_path = tmp.name
        st.image(temp_path, caption="Input Image", use_column_width=True)
        show_output(temp_path)

# Use webcam input
elif option == "Use webcam":
    camera_img = st.camera_input("Take a picture")
    if camera_img:
        img = Image.open(camera_img).convert("RGB")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            img.save(tmp.name)
            temp_path = tmp.name
        st.image(temp_path, caption="Input Image", use_column_width=True)
        show_output(temp_path)
   