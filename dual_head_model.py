import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class DualHeadResNetUNet(nn.Module):
    def __init__(self, n_classes=1):
        super().__init__()
        
        # --- 1. Encoder (Pre-trained ResNet18) ---
        # טוענים את ResNet18 שכבר למדה לראות תמונות ב-ImageNet
        self.base_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.base_layers = list(self.base_model.children())
        
        # פירוק ה-Encoder לשכבות כדי שנוכל לחבר Skip Connections
        self.layer0 = nn.Sequential(*self.base_layers[:3]) # גודל מקורי / 2
        self.layer1 = nn.Sequential(*self.base_layers[3:5]) # גודל / 4
        self.layer2 = self.base_layers[5]  # גודל / 8
        self.layer3 = self.base_layers[6]  # גודל / 16
        self.layer4 = self.base_layers[7]  # גודל / 32
        
        # --- 2. Decoder (Upsampling path) ---
        # שכבות שמעלות את הרזולוציה חזרה
        self.up4 = self._up_block(512, 256)
        self.up3 = self._up_block(256 + 256, 128) # +256 בגלל החיבור לשכבה המקבילה מה-Encoder
        self.up2 = self._up_block(128 + 128, 64)
        self.up1 = self._up_block(64 + 64, 64)
        self.up0 = self._up_block(64 + 64, 32)
        
        # --- 3. Heads (הפיצול) ---
        
        # ראש א': Potential Map (1 Channel) -> Sigmoid (0 to 1)
        self.potential_head = nn.Sequential(
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid() 
        )
        
        # ראש ב': Flow Field (2 Channels: X, Y) -> Tanh (-1 to 1)
        self.flow_head = nn.Sequential(
            nn.Conv2d(32, 2, kernel_size=1),
            nn.Tanh()
        )

    def _up_block(self, in_channels, out_channels):
        """בלוק עזר לעלייה ברזולוציה"""
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # --- שלב ה-Encoder (ירידה למטה) ---
        x0 = self.layer0(x) # (64, H/2, W/2)
        x1 = self.layer1(x0) # (64, H/4, W/4)
        x2 = self.layer2(x1) # (128, H/8, W/8)
        x3 = self.layer3(x2) # (256, H/16, W/16)
        x4 = self.layer4(x3) # (512, H/32, W/32) - התחתית של ה-U
        
        # --- שלב ה-Decoder (עלייה למעלה) ---
        
        # עולים מ-x4 ומחברים את x3
        up4 = self.up4(x4)
        # אם הגדלים לא תואמים בול בגלל עיגול אי-זוגי, נחתוך או נמתח (בדרך כלל עובד בסדר בחזקות של 2)
        # לצורך הפשטות נניח שהגדלים תואמים (512x256 מתחלק יפה ב-32)
        merge3 = torch.cat([up4, x3], dim=1) 
        
        up3 = self.up3(merge3)
        merge2 = torch.cat([up3, x2], dim=1)
        
        up2 = self.up2(merge2)
        merge1 = torch.cat([up2, x1], dim=1)
        
        up1 = self.up1(merge1)
        merge0 = torch.cat([up1, x0], dim=1)
        
        features = self.up0(merge0) # חזרה לגודל המקורי
        
        # --- שלב הפיצול (Heads) ---
        potential = self.potential_head(features)
        flow = self.flow_head(features)
        
        return potential, flow

# --- בדיקה שהמודל עובד ---
if __name__ == "__main__":
    # יצירת דוגמה של המודל
    model = DualHeadResNetUNet()
    
    # יצירת קלט דמה (Batch=1, Channels=3, Height=256, Width=512)
    dummy_input = torch.randn(1, 3, 256, 512)
    
    # הרצה
    pred_potential, pred_flow = model(dummy_input)
    
    print("\n--- Model Check ---")
    print(f"Input Shape:     {dummy_input.shape}")
    print(f"Potential Shape: {pred_potential.shape} (Expected: 1, 1, 256, 512)")
    print(f"Flow Shape:      {pred_flow.shape}      (Expected: 1, 2, 256, 512)")
    print("Test passed successfully!")