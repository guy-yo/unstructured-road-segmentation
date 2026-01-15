import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm  # הוספנו פס התקדמות

# =========================
# Dataset
# =========================

class RoadSegmentationDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        # שימוש ב-sorted כדי להבטיח סדר זהה
        self.image_files = sorted(list(Path(images_dir).glob("*.jpg")))
        
        # אנחנו מניחים שהשמות זהים (רק הסיומת שונה - png למסיכות)
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        # מציאת המסיכה המתאימה לפי השם
        mask_path = os.path.join(self.masks_dir, img_path.stem + ".png")

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L") # גווני אפור

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask

from pathlib import Path

def get_dataloaders(data_root, batch_size=8):
    # --- תיקון קריטי: רזולוציה תואמת למודל הראשי ---
    # PyTorch Resize מקבל (Height, Width)
    # המודל שלנו עובד על 512 רוחב ו-256 גובה
    transform = transforms.Compose([
        transforms.Resize((256, 512)), 
        transforms.ToTensor()
    ])

    train_images = os.path.join(data_root, "images", "train")
    train_masks = os.path.join(data_root, "masks", "train")

    val_images = os.path.join(data_root, "images", "val")
    val_masks = os.path.join(data_root, "masks", "val")

    train_dataset = RoadSegmentationDataset(train_images, train_masks, transform=transform)
    val_dataset = RoadSegmentationDataset(val_images, val_masks, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader

# =========================
# U-Net model (Simple Baseline)
# =========================

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels), # הוספנו BatchNorm ליציבות
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        self.enc1 = DoubleConv(in_channels, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256) # הוספנו עומק כדי שיהיה הוגן יותר מול ResNet
        self.pool = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(256, 512)

        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(512, 256)
        
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(256, 128)
        
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(128, 64)

        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        
        b = self.bottleneck(self.pool(e3))
        
        d3 = self.up3(b)
        # תיקון גדלים אם צריך (לרוב ב-Resize של חזקות 2 זה בסדר)
        d3 = torch.cat((d3, e3), dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat((d2, e2), dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat((d1, e1), dim=1)
        d1 = self.dec1(d1)

        return self.out_conv(d1)

# =========================
# Metrics
# =========================

def compute_iou(preds, targets, threshold=0.5):
    preds = (preds > threshold).float()
    targets = (targets > threshold).float()
    intersection = (preds * targets).sum()
    union = preds.sum() + targets.sum() - intersection
    return (intersection / (union + 1e-6)).item()

# =========================
# Training
# =========================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Starting Baseline Training on {device} ---")

    data_root = "DATA"
    train_loader, val_loader = get_dataloaders(data_root, batch_size=8)

    model = UNet().to(device)
    
    # שימוש ב-BCE (Binary Cross Entropy) כי זה רק כביש/לא כביש
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    epochs = 10 
    best_iou = 0.0

    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        # --- Training Loop ---
        model.train()
        train_loss = 0
        loop = tqdm(train_loader, desc="Training", leave=False)
        
        for images, masks in loop:
            images = images.to(device)
            masks = masks.to(device).float()

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_train_loss = train_loss / len(train_loader)

        # --- Validation Loop ---
        model.eval()
        iou_scores = []
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device).float()
                outputs = model(images) # Raw logits
                preds = torch.sigmoid(outputs)
                iou_scores.append(compute_iou(preds, masks))

        avg_val_iou = sum(iou_scores) / len(iou_scores)
        
        print(f"Train Loss: {avg_train_loss:.4f} | Val IoU: {avg_val_iou:.4f}")

        # --- Saving Best Model ---
        if avg_val_iou > best_iou:
            best_iou = avg_val_iou
            torch.save(model.state_dict(), "baseline_unet.pth")
            print(">>> New best baseline model saved!")

    print("\nDone. Baseline model saved as 'baseline_unet.pth'")

if __name__ == "__main__":
    main()