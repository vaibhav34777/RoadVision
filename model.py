import torch 
from torch import nn
import torch.nn.functional as F
from ultralytics import YOLO

# Segmentation Model Definition
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
    
#  Pretrained YOLO Model
def get_yolo_model():
    return YOLO("yolov8n.pt")
