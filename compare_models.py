import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import random
import sys

# --- ייבוא הארכיטקטורות ---
from dual_head_model import DualHeadResNetUNet
try:
    from train_baseline import UNet 
except ImportError:
    print("Error: Could not import UNet from 'train_baseline.py'. Make sure the file exists.")
    sys.exit(1)

# --- הגדרות ---
OUR_MODEL_PATH = "best_model_improved.pth"
BASELINE_MODEL_PATH = "baseline_unet.pth"

# שינוי חשוב: משתמשים ב-VAL כי שם יש לנו Ground Truth להשוואה
TEST_DIR = Path("DATA") / "images" / "val" 

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_models():
    print(f"Loading models on {DEVICE}...")
    
    # 1. טעינת המודל שלך (Dual Head)
    model_ours = DualHeadResNetUNet()
    if Path(OUR_MODEL_PATH).exists():
        # weights_only=True הוא פרמטר אבטחה חדש, נשתמש בברירת המחדל למניעת אזהרות אם הגרסה תומכת
        try:
            model_ours.load_state_dict(torch.load(OUR_MODEL_PATH, map_location=DEVICE, weights_only=True))
        except TypeError: # גרסאות ישנות של טורץ'
            model_ours.load_state_dict(torch.load(OUR_MODEL_PATH, map_location=DEVICE))
    else:
        print(f"Warning: {OUR_MODEL_PATH} not found!")
        return None, None
    model_ours.to(DEVICE).eval()
    
    # 2. טעינת מודל הבסיס (Baseline UNet)
    model_base = UNet()
    if Path(BASELINE_MODEL_PATH).exists():
        try:
            model_base.load_state_dict(torch.load(BASELINE_MODEL_PATH, map_location=DEVICE, weights_only=True))
        except TypeError:
            model_base.load_state_dict(torch.load(BASELINE_MODEL_PATH, map_location=DEVICE))
    else:
        print(f"Warning: {BASELINE_MODEL_PATH} not found!")
        return None, None
    model_base.to(DEVICE).eval()
    
    return model_ours, model_base

def calculate_iou(pred, gt):
    """חישוב מדד חפיפה (Intersection over Union)"""
    pred_bin = (pred > 0.5).astype(np.uint8).flatten()
    gt_bin = (gt > 0.5).astype(np.uint8).flatten()
    
    intersection = np.logical_and(pred_bin, gt_bin).sum()
    union = np.logical_or(pred_bin, gt_bin).sum()
    
    if union == 0: return 1.0 
    return intersection / union

def calculate_path_rmse(path_x, path_y, gt_x, gt_y):
    """חישוב שגיאת סטייה ממוצעת (RMSE)"""
    if len(path_x) < 5 or len(gt_x) < 5:
        return 999.0 
        
    min_y = max(min(path_y), min(gt_y))
    max_y = min(max(path_y), max(gt_y))
    
    if min_y >= max_y: return 999.0
    
    common_y = np.arange(min_y, max_y, 1)
    
    # מיון ואינטרפולציה
    path_y_sorted, path_x_sorted = zip(*sorted(zip(path_y, path_x)))
    gt_y_sorted, gt_x_sorted = zip(*sorted(zip(gt_y, gt_x)))
    
    # הסרת כפילויות ל-interp
    _, unique_indices = np.unique(path_y_sorted, return_index=True)
    path_y_unique = np.array(path_y_sorted)[unique_indices]
    path_x_unique = np.array(path_x_sorted)[unique_indices]

    _, unique_indices_gt = np.unique(gt_y_sorted, return_index=True)
    gt_y_unique = np.array(gt_y_sorted)[unique_indices_gt]
    gt_x_unique = np.array(gt_x_sorted)[unique_indices_gt]

    interp_x = np.interp(common_y, path_y_unique, path_x_unique)
    interp_gt = np.interp(common_y, gt_y_unique, gt_x_unique)
    
    rmse = np.sqrt(np.mean((interp_x - interp_gt)**2))
    return rmse

def calculate_center_path(mask):
    """האלגוריתם ה'טיפש' עבור הבייסליין"""
    h, w = mask.shape
    path_x, path_y = [], []
    start_y = h - 10
    curr_x = w // 2
    
    for y in range(start_y, 0, -5):
        row = mask[y, :]
        road_pixels = np.where(row > 0.5)[0]
        
        if len(road_pixels) == 0: break
            
        center_x = np.mean(road_pixels)
        if abs(center_x - curr_x) > 50 and len(path_x) > 0: break
            
        curr_x = center_x
        path_x.append(curr_x)
        path_y.append(y)
        
    return path_x, path_y

def simulate_flow_driving(flow, potential, start_pos):
    """האלגוריתם ה'חכם' (ווקטורים)"""
    curr_x, curr_y = float(start_pos[0]), float(start_pos[1])
    path_x, path_y = [curr_x], [curr_y]
    
    h, w = potential.shape
    forward_speed = 3.0
    steering_gain = 4.0
    momentum = 0.8
    current_steering = 0.0
    
    for _ in range(300):
        ix, iy = int(curr_x), int(curr_y)
        if ix < 0 or ix >= w or iy < 0 or iy >= h: break
        if potential[iy, ix] < 0.1: break 
            
        raw_steering = flow[iy, ix, 0]
        current_steering = (current_steering * momentum) + (raw_steering * (1 - momentum))
        curr_y -= forward_speed
        curr_x += current_steering * steering_gain
        
        path_x.append(curr_x)
        path_y.append(curr_y)
        
    return path_x, path_y

def compare_visualizations(model_ours, model_base, num_samples=5):
    all_images = list(TEST_DIR.glob("*.jpg"))
    
    if not all_images:
        print(f"No images found in {TEST_DIR}")
        return

    # תיקיות ה-Ground Truth
    current_split = TEST_DIR.name # 'val'
    mask_root = Path("DATA") / "masks" / current_split
    flow_root = Path("DATA") / "flow" / current_split

    samples = random.sample(all_images, min(len(all_images), num_samples))
    
    for img_path in samples:
        # טעינה
        original_img = cv2.imread(str(img_path))
        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        input_img = cv2.resize(original_img, (512, 256))
        img_tensor = torch.from_numpy(input_img).float() / 255.0
        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0).to(DEVICE)
        
        # טעינת GT
        gt_mask_path = mask_root / f"{img_path.stem}.png"
        gt_flow_path = flow_root / f"{img_path.stem}.npy"
        
        if not gt_mask_path.exists() or not gt_flow_path.exists():
            print(f"Skipping {img_path.stem} - missing GT files")
            continue
            
        mask_gt = cv2.imread(str(gt_mask_path), cv2.IMREAD_GRAYSCALE)
        mask_gt = cv2.resize(mask_gt, (512, 256), interpolation=cv2.INTER_NEAREST)
        mask_gt = mask_gt.astype(float) / 255.0
        
        flow_gt = np.load(str(gt_flow_path))
        flow_gt = cv2.resize(flow_gt, (512, 256)) 

        # --- הרצה ---
        with torch.no_grad():
            out_base = model_base(img_tensor)
            mask_base = torch.sigmoid(out_base).squeeze().cpu().numpy()
        
        with torch.no_grad():
            out_pot, out_flow = model_ours(img_tensor)
            mask_ours = out_pot.squeeze().cpu().numpy()
            flow_ours = out_flow.squeeze().cpu().numpy()
            flow_ours = np.moveaxis(flow_ours, 0, -1)

        # --- מסלולים ---
        h, w = mask_gt.shape
        start_pos = (w // 2, h - 10)
        
        path_base_x, path_base_y = calculate_center_path(mask_base)
        path_ours_x, path_ours_y = simulate_flow_driving(flow_ours, mask_ours, start_pos)
        path_gt_x, path_gt_y = simulate_flow_driving(flow_gt, mask_gt, start_pos)

        # --- ציונים ---
        iou_base = calculate_iou(mask_base, mask_gt)
        iou_ours = calculate_iou(mask_ours, mask_gt)
        
        rmse_base = calculate_path_rmse(path_base_x, path_base_y, path_gt_x, path_gt_y)
        rmse_ours = calculate_path_rmse(path_ours_x, path_ours_y, path_gt_x, path_gt_y)

        # --- גרף ---
        plt.figure(figsize=(18, 6))
        
        # Baseline
        plt.subplot(1, 3, 1)
        plt.imshow(input_img)
        plt.plot(path_base_x, path_base_y, color='red', linewidth=3)
        plt.title(f"Baseline (U-Net)\nIoU: {iou_base:.2%}\nPath Error: {rmse_base:.1f} px", color='red')
        plt.axis('off')
        
        # Ours
        plt.subplot(1, 3, 2)
        plt.imshow(input_img)
        plt.plot(path_ours_x, path_ours_y, color='lime', linewidth=3)
        plt.title(f"Ours (Dual-Head)\nIoU: {iou_ours:.2%}\nPath Error: {rmse_ours:.1f} px", color='green')
        plt.axis('off')
        
        # Overlay
        plt.subplot(1, 3, 3)
        plt.imshow(input_img)
        plt.plot(path_gt_x, path_gt_y, color='blue', linewidth=2, linestyle='--', label='Ground Truth')
        plt.plot(path_base_x, path_base_y, color='red', linewidth=2, label='Baseline')
        plt.plot(path_ours_x, path_ours_y, color='lime', linewidth=2, label='Ours')
        plt.title("Comparison Overlay")
        plt.legend()
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    model_ours, model_base = load_models()
    if model_ours and model_base:
        compare_visualizations(model_ours, model_base, num_samples=5)