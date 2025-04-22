# RoadVision 🚗🛣️  
**Semantic Segmentation + Object Detection for Intelligent Road Scene Understanding**

RoadVision is a deep learning web application that fuses **semantic segmentation** (U‑Net) with **object detection** (YOLO) to analyze road scenes. It outputs an RGB segmentation mask with color‑coded classes and overlays YOLO bounding boxes—ideal for autonomous driving demos, smart‑city prototypes, and research.

---

##  Features

- 🔍 **Object Detection**: YOLOv8 pretrained on COCO  
- 🌈 **Semantic Segmentation**: Custom U‑Net (32 classes)  
- 🎯 **Combined Loss**: Focal Loss + Dice Loss for robust class learning  
- ⚙️ **Training Regimen**: Adam optimizer + StepLR scheduler, 30 epochs  
- 📈 **Metrics**:  
  - Final training loss: **1.2**  
  - Final validation loss: **2.1**  
  - Pixel accuracy: **85 %**  
- 🖼️ **Inference Pipeline**:  
  1. Pass input through U‑Net → segmentation mask  
  2. Pass input through YOLO → bounding boxes  
  3. Color‑code each mask class (e.g., road=green, sidewalk=red, building=yellow, car=blue)  
  4. Overlay YOLO boxes on the colored mask  
- 📱 **Streamlit UI**: File upload or webcam input  
- ☁️ **Deployable** on Hugging Face Spaces (CPU only)

---

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/vaibhav34777/RoadVision.git
   cd roadvision
2. ** Install Dependencies **
   ```bash
   pip install -r requirements.txt
3. ** Load the model **
   load weights in the unet model from outputs/README.md and load the pretrained YOLO model from ultralytics.
   
