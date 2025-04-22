import cv2
import torch 
from torch import nn
import torch.nn.functional as F
from dataloader import get_train_loader, get_val_loader
from model import UNet

# Set seeds for reproducibility
torch.manual_seed(43)
if torch.cuda.is_available():
    torch.cuda.manual_seed(43)  

# getting the data loadeers
train_loader = get_train_loader()
val_loader = get_val_loader()
    
# Model Initialization
base_filters = 32
out_classes = 32
in_channels = 3
model = UNet(in_channels,base_filters,out_classes)
device='cpu'
if torch.cuda.is_available():
    device='cuda'
model.to(device)

# Loss Fucntion

# Calculate class weights based on frequency of each class in the training set
from collections import Counter
counter = Counter()
for _, mask in train_loader:
    counter.update(mask.flatten().tolist())
freqs = torch.tensor([counter[i] for i in range(32)], dtype=torch.float)
weights = 1.0 / (torch.log(1.02 + freqs / freqs.sum()))

# Focal Loss
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None, ignore_index=255):
        super().__init__()
        self.gamma = gamma
        self.ce  = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)

    def forward(self, logits, targets):
        # Standard cross‑entropy
        logp = F.log_softmax(logits, dim=1)
        ce   = F.nll_loss(logp, targets, weight=self.ce.weight,
                          ignore_index=self.ce.ignore_index, reduction='none')

        p    = torch.exp(-ce)         
        focal = (1 - p)**self.gamma    
        loss  = focal * ce
        # average over non‑ignored pixels
        valid = (targets != self.ce.ignore_index)
        return loss[valid].mean()

# Dice Loss
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5, ignore_index=255):
        super().__init__()
        self.smooth = smooth
        self.ignore = ignore_index

    def forward(self, logits, targets):
        # one‑hot encode predictions & targets
        num_classes = logits.shape[1]
        probs = F.softmax(logits, dim=1)
        mask = (targets != self.ignore).unsqueeze(1)
        probs = probs * mask
        t_onehot = F.one_hot(targets.clamp(0, num_classes-1), num_classes) \
                      .permute(0,3,1,2).float() * mask

        intersect = (probs * t_onehot).sum(dim=(0,2,3))
        union     = probs.sum(dim=(0,2,3)) + t_onehot.sum(dim=(0,2,3))
        dice = (2*intersect + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()

focal = FocalLoss(gamma=2.0, weight=weights.to(device), ignore_index=255)
dice  = DiceLoss(ignore_index=255)

def combined_loss(logits, masks):
    return 0.4 * focal(logits, masks) + 0.6 * dice(logits, masks)

# Optimizer
num_epochs = 30
optimizer = torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=10,gamma=0.1)

# Calculating Pixel Accuracy
def pixel_accuracy(preds, labels):
    correct = (preds == labels).sum().item()
    total = preds.numel()
    return correct, total

train_losses = []
val_losses = []
min_pixel_acc = 0.0

# Training Loop
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    loop = tqdm(train_loader, desc=f"Epoch {epoch + 1} / {num_epochs}")
    for img, mask in loop:
        img, mask = img.to(device), mask.to(device)
        img = img.float()/255.0
        mask = mask.long()
        optimizer.zero_grad()
        logits = model(img)  
        loss = combined_loss(logits, mask)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        train_losses.append(loss.item())
        loop.set_postfix(loss=loss.item())

    avg_train_loss = epoch_loss / len(train_loader)
    print(f"Epoch {epoch + 1} training loss: {avg_train_loss:.4f}")
    
    model.eval()
    val_loss = 0.0
    total_correct_pixels = 0
    total_pixels = 0
    with torch.no_grad():
        for img, mask in val_loader:
            img, mask = img.to(device), mask.to(device)
            img = img.float()/255.0
            mask = mask.long()
            logits = model(img)
            loss = combined_loss(logits, mask)
            val_loss += loss.item()
            preds = torch.argmax(logits, dim=1)  # shape: [batch*n_aug, H, W]
            correct, total = pixel_accuracy(preds, mask)
            total_correct_pixels += correct
            total_pixels += total
    avg_val_loss = val_loss / len(val_loader)
    avg_pixel_acc = total_correct_pixels / total_pixels
    val_losses.append(avg_val_loss)
    if avg_pixel_acc > min_pixel_acc:
        min_pixel_acc = avg_pixel_acc
        torch.save(model.state_dict(), "model.pth") # save the model state dict
    
    print(f"Validation loss: {avg_val_loss:.4f} | Pixel Accuracy: {avg_pixel_acc:.4f}")
    scheduler.step()
