import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# --- הגדרות ---
DATA_ROOT = "DATA"  # התיקייה שבה נמצא הדאטה המוכן

def visualize_random_samples(num_samples=5):
    """
    לוקח דוגמאות אקראיות מתיקיית ה-DATA ומציג:
    1. תמונה מקורית
    2. מפת מרחקים (Potential)
    3. סימולציית נהיגה על בסיס ה-Flow (חצים + קו ירוק)
    """
    print(f"--- Visualizing {num_samples} Random Samples from '{DATA_ROOT}' ---")
    
    img_dir = Path(DATA_ROOT) / "images" / "train"
    mask_dir = Path(DATA_ROOT) / "masks" / "train"
    flow_dir = Path(DATA_ROOT) / "flow" / "train"
    
    if not img_dir.exists():
        print(f"Error: Directory {img_dir} does not exist. Run prepare_data.py first.")
        return

    # שליפת כל הקבצים
    all_files = list(img_dir.glob("*.jpg"))
    if not all_files:
        print("No images found.")
        return
        
    # בחירה אקראית
    samples = np.random.choice(all_files, min(len(all_files), num_samples), replace=False)
    
    for i, img_path in enumerate(samples):
        stem = img_path.stem
        mask_path = mask_dir / f"{stem}.png"
        flow_path = flow_dir / f"{stem}.npy"
        
        # טעינה
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # המרה לתצוגה נכונה ב-Matplotlib
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        
        if not flow_path.exists():
            print(f"Flow file missing for {stem}")
            continue
            
        flow = np.load(str(flow_path)) # (H, W, 2)
        
        # יצירת הגרף
        plt.figure(figsize=(18, 6))
        
        # --- תמונה 1: מקור ---
        plt.subplot(1, 3, 1)
        plt.imshow(img)
        plt.title(f"Sample #{i+1}: Original")
        plt.axis('off')
        
        # --- תמונה 2: מפת מרחקים (המטרה של ראש א') ---
        plt.subplot(1, 3, 2)
        plt.imshow(mask, cmap='inferno')
        plt.title("Target: Potential Map")
        plt.axis('off')
        
        # --- תמונה 3: סימולציית נהיגה (המטרה של ראש ב') ---
        plt.subplot(1, 3, 3)
        plt.imshow(img) # רקע: התמונה המקורית
        plt.title("Flow Field + Path Simulation")
        
        h, w = mask.shape
        
        # א. ציור החצים (Flow Arrows)
        step = 20 # דילול (מציג חץ כל 20 פיקסלים למניעת עומס)
        y, x = np.mgrid[0:h:step, 0:w:step]
        fx = flow[::step, ::step, 0]
        fy = flow[::step, ::step, 1]
        
        # מציגים חצים רק איפה שיש כביש (מפת המרחקים > 20)
        mask_arrow = mask[::step, ::step] > 20 
        
        plt.quiver(x[mask_arrow], y[mask_arrow], 
                   fx[mask_arrow], fy[mask_arrow], 
                   color='cyan', scale=25, headwidth=3, alpha=0.5, label='Flow Vectors')
        
        # ב. ציור הנתיב (Green Path Simulation)
        # נקודת התחלה: אמצע למטה
        start_x, start_y = w // 2, h - 10
        path_x, path_y = [start_x], [start_y]
        curr_x, curr_y = float(start_x), float(start_y)
        
        # פרמטרים לסימולציה חלקה
        forward_speed = 3.0
        steering_gain = 4.0
        momentum = 0.8
        current_steering_x = 0.0
        
        for _ in range(300): # מקסימום 300 צעדים
            ix, iy = int(curr_x), int(curr_y)
            
            # בדיקת גבולות ויציאה מאזור בטוח
            if ix < 0 or ix >= w or iy < 0 or iy >= h: break
            if mask[iy, ix] < 10: break # פגיעה במכשול
            
            # קריאת ההיגוי מה-Flow (ציר X)
            raw_steering = flow[iy, ix, 0]
            
            # החלקה (Smoothing)
            current_steering_x = (current_steering_x * momentum) + (raw_steering * (1 - momentum))
            
            # עדכון מיקום
            curr_y -= forward_speed
            curr_x += current_steering_x * steering_gain
            
            path_x.append(curr_x)
            path_y.append(curr_y)
            
        plt.plot(path_x, path_y, color='lime', linewidth=4, label='Predicted Path')
        plt.legend(loc='upper right')
        plt.axis('off')
        
        plt.show()

if __name__ == "__main__":
    visualize_random_samples(5)