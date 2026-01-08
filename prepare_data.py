import os
import cv2
import numpy as np
import shutil
from pathlib import Path

# --- הגדרות ---
SOURCE_ROOT = "idd20k_lite"      # מאיפה לוקחים את הנתונים המקוריים
DEST_ROOT = "DATA"               # לאן שומרים את הדאטה המעובד
DRIVABLE_IDS = [0, 1, 2]         # מזהים של כביש

def create_dir_structure():
    """יוצר את עץ התיקיות החדש (כולל תיקיית flow לווקטורים)"""
    paths = {
        "train_img": Path(DEST_ROOT) / "images" / "train",
        "train_mask": Path(DEST_ROOT) / "masks" / "train",
        "train_flow": Path(DEST_ROOT) / "flow" / "train",
        
        "val_img": Path(DEST_ROOT) / "images" / "val",
        "val_mask": Path(DEST_ROOT) / "masks" / "val",
        "val_flow": Path(DEST_ROOT) / "flow" / "val",
        
        "test_img": Path(DEST_ROOT) / "images" / "test"
    }
    
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
        
    return paths

def find_matching_mask(img_path, img_dir_root, mask_dir_root):
    """מוצא את המסיכה המתאימה בתיקיית המקור"""
    try:
        relative_path = img_path.relative_to(img_dir_root)
        options = [
            relative_path.name.replace("_leftImg8bit.png", "_gtFine_label.png").replace(".jpg", "_gtFine_label.png"),
            img_path.stem + "_label.png",
            img_path.stem.replace("_image", "") + "_label.png"
        ]
        for opt in options:
            p = mask_dir_root / relative_path.parent / opt
            if p.exists(): return p
    except Exception:
        pass
    return None

def compute_targets(mask_path):
    """
    מחשב גם את מפת המרחקים (Potential) וגם את שדה הווקטורים (Flow).
    """
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None: return None, None
    
    # 1. יצירת מסיכה בינארית (1=כביש)
    binary_road = np.isin(mask, DRIVABLE_IDS).astype(np.uint8)
    
    # 2. חישוב מפת מרחקים
    dist_map = cv2.distanceTransform(binary_road, cv2.DIST_L2, 5)
    
    # 3. חישוב ה-Flow (נגזרות)
    dy, dx = np.gradient(dist_map)
    
    magnitude = np.sqrt(dx**2 + dy**2)
    magnitude[magnitude == 0] = 1 # מניעת חלוקה באפס
    
    dx_norm = dx / magnitude
    dy_norm = dy / magnitude
    
    # שמירת ה-Flow כ-Float32 (ערוץ 0=X, ערוץ 1=Y)
    flow_field = np.dstack((dx_norm, dy_norm)).astype(np.float32)

    # נרמול מפת המרחקים לתצוגה ושמירה כתמונה (0-255)
    max_val = dist_map.max()
    if max_val > 0:
        dist_map_visual = (dist_map / max_val) * 255
    else:
        dist_map_visual = dist_map
        
    return dist_map_visual.astype(np.uint8), flow_field

def process_dataset():
    print(f"--- Starting Data Preparation ---")
    print(f"Reading from: {SOURCE_ROOT}")
    print(f"Saving to:    {DEST_ROOT}")
    
    dirs = create_dir_structure()
    
    for split in ['train', 'val']:
        root = Path(SOURCE_ROOT).resolve()
        img_dir_src = root / "leftImg8bit" / split
        mask_dir_src = root / "gtFine" / split
        
        if not img_dir_src.exists(): continue
            
        print(f"Processing '{split}' folder...")
        image_paths = sorted(list(img_dir_src.rglob("*.png")) + list(img_dir_src.rglob("*.jpg")))
        
        count_processed = 0
        
        for i, img_path in enumerate(image_paths):
            mask_path = find_matching_mask(img_path, img_dir_src, mask_dir_src)
            filename_stem = img_path.stem
            
            if mask_path:
                processed_mask, flow_field = compute_targets(mask_path)
                
                if processed_mask is not None:
                    img = cv2.imread(str(img_path))
                    if img is not None:
                        # שמירת התמונה המקורית
                        cv2.imwrite(str(dirs[f"{split}_img"] / f"{filename_stem}.jpg"), img)
                        # שמירת מפת המרחקים (Target 1)
                        cv2.imwrite(str(dirs[f"{split}_mask"] / f"{filename_stem}.png"), processed_mask)
                        # שמירת שדה הווקטורים (Target 2)
                        np.save(str(dirs[f"{split}_flow"] / f"{filename_stem}.npy"), flow_field)
                        
                        count_processed += 1
            else:
                # תמונות ללא מסיכה עוברות ל-Test
                img = cv2.imread(str(img_path))
                if img is not None:
                    cv2.imwrite(str(dirs["test_img"] / f"{filename_stem}.jpg"), img)
            
            if (i + 1) % 100 == 0:
                print(f"  Processed {i + 1} images...")

        print(f"Finished '{split}': {count_processed} pairs generated.")

    print("-" * 30)
    print("Done! Data is ready in 'DATA/' folder.")

if __name__ == "__main__":
    process_dataset()