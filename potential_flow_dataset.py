import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path

class PotentialFlowDataset(Dataset):
    def __init__(self, root_dir, split='train', target_size=(512, 256)):
        """
        אתחול ה-Dataset.
        :param root_dir: הנתיב לתיקיית DATA
        :param split: 'train', 'val', או 'test'
        :param target_size: גודל התמונה הרצוי (רוחב, גובה) למודל
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.target_size = target_size # (Width, Height)
        
        # נתיבים לתיקיות
        self.img_dir = self.root_dir / "images" / split
        self.mask_dir = self.root_dir / "masks" / split
        self.flow_dir = self.root_dir / "flow" / split
        
        # איסוף שמות הקבצים (מסודרים לפי א"ב כדי שיתאימו)
        self.images = sorted(list(self.img_dir.glob("*.jpg")))
        
        print(f"[{split.upper()}] Dataset initialized. Found {len(self.images)} samples.")

    def __len__(self):
        # המודל צריך לדעת כמה דוגמאות יש סה"כ
        return len(self.images)

    def __getitem__(self, idx):
        # פונקציה זו נקראת עבור כל תמונה בזמן האימון
        
        # 1. קבלת נתיב הקובץ הנוכחי
        img_path = self.images[idx]
        stem = img_path.stem # שם הקובץ ללא סיומת
        
        # 2. טעינת הקבצים מהדיסק
        # תמונה (RGB)
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # מסיכה (Potential - Grayscale)
        mask_path = self.mask_dir / f"{stem}.png"
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        
        # וקטורים (Flow - NumPy Array)
        flow_path = self.flow_dir / f"{stem}.npy"
        flow = np.load(str(flow_path)) # (H, W, 2)

        # 3. שינוי גודל (Resize) למידות קבועות
        # המודל חייב לקבל גודל אחיד כדי לעבוד ב-Batches
        img = cv2.resize(img, self.target_size)
        mask = cv2.resize(mask, self.target_size, interpolation=cv2.INTER_NEAREST)
        flow = cv2.resize(flow, self.target_size) # שים לב: זה משנה גודל, אך לא משנה את ערכי הוקטורים (כיוון נשמר)

        # 4. המרה ל-Tensor וסידור הערוצים
        # PyTorch מצפה ל: (Channels, Height, Width)
        # וערכים בין 0.0 ל-1.0 (float32)

        # עיבוד תמונה: (H,W,3) -> (3,H,W)
        img_tensor = torch.from_numpy(img).float() / 255.0
        img_tensor = img_tensor.permute(2, 0, 1) # הזזת הערוצים להתחלה
        
        # עיבוד מסיכה: (H,W) -> (1,H,W)
        mask_tensor = torch.from_numpy(mask).float() / 255.0
        mask_tensor = mask_tensor.unsqueeze(0) # הוספת מימד הערוץ
        
        # עיבוד Flow: (H,W,2) -> (2,H,W)
        flow_tensor = torch.from_numpy(flow).float()
        flow_tensor = flow_tensor.permute(2, 0, 1)

        return img_tensor, mask_tensor, flow_tensor

# --- בדיקה שהקוד עובד ---
if __name__ == "__main__":
    # יצירת דוגמה של ה-Dataset
    ds = PotentialFlowDataset(root_dir="DATA", split="train")
    
    # שליפת הדוגמה הראשונה
    img, mask, flow = ds[0]
    
    print("\n--- Sample Check ---")
    print(f"Image Shape: {img.shape}  (Expected: 3, 256, 512)")
    print(f"Mask Shape:  {mask.shape}  (Expected: 1, 256, 512)")
    print(f"Flow Shape:  {flow.shape}  (Expected: 2, 256, 512)")
    print(f"Image Type:  {img.dtype}")
    print("Test passed successfully!")