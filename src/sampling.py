from model import UNet,get_yolo_model
import albumentations as A
import torch 
import cv2
import os
from utils import label_to_rgb
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image

# Getting the models for segmentation and detection
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load Segmentation mdoel
model = UNet(in_channels=3, base_filters=32, out_classes=32).to(device)
# Load the model weights
model.load_state_dict(torch.load("model.pth", map_location = device,weights_only=True))

# Load YOLOv8 model for detection
yolo_model = get_yolo_model()


transform = A.Compose([
    A.Resize(608, 608, interpolation=cv2.INTER_LINEAR),
    A.ToTensorV2(),
])

# Function to perform inference on the image using the segmentation model
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
    output = label_to_rgb(preds)
    return output


# Function to get the final output with YOLOv8 detection and segmentation
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
            cls = r.names[int(box.cls[0])]
            conf = box.conf[0]
            color = CLASS_COLORS.get(cls, (255,255,255))
            cv2.rectangle(seg_output,(x1,y1),(x2,y2),color,2)
            label=f"{cls} {conf:.2f}"
            tw,th = cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX,0.5,1)[0]
            cv2.rectangle(seg_output,(x1,y1-th-4),(x1+tw+2,y1),color,-1)
            cv2.putText(seg_output,label,(x1,y1-4),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),1)
    return seg_output
   

# Example usage
img_path = "/kaggle/input/camvid/CamVid/val/0001TP_009870.png"  # Path to your image
img = Image.open(img_path).convert('RGB')
f_output = final_output(img_path)
plt.figure(figsize=(12,8))
plt.subplot(1,2,1)
plt.imshow(img)
plt.axis('off')
plt.title('Input')
plt.subplot(1,2,2)
plt.imshow(f_output)

plt.axis('off')
plt.title('Final Output')
plt.savefig('yolo.png')
plt.show()

# Creating a video from the output frames
output_path = 'demo_output.mp4'
fps = 4
frame_size = (608, 608) 
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec
video_writer = cv2.VideoWriter(output_path, fourcc, fps, frame_size)

input_folder = "/kaggle/input/camvid/CamVid/val"   # Path to your input folder with images
image_paths = sorted([os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith(".png")])

for img_path in image_paths:
    overlay_frame = final_output(img_path) 
    video_frame = cv2.cvtColor(overlay_frame, cv2.COLOR_RGB2BGR)
    video_writer.write(video_frame)

video_writer.release()
print(f"Video saved at {output_path}")
