import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import random

# ייבוא המודל שלנו
from dual_head_model import DualHeadResNetUNet

# --- הגדרות ---
MODEL_PATH = "best_model.pth"
# מגדירים את שני המקורות האפשריים לבדיקה
TEST_DIR = Path("DATA") / "images" / "test" # תמונות שהמודל מעולם לא ראה (עדיפות עליונה)
VAL_DIR = Path("DATA") / "images" / "val"   # תמונות ששימשו לבדיקה בזמן האימון (גיבוי)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_trained_model():
    print(f"Loading model from {MODEL_PATH}...")
    model = DualHeadResNetUNet()
    # טעינת המשקולות (Weights) שנלמדו
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval() # מעבר למצב בדיקה (חשוב!)
    return model

def predict_and_visualize(model, num_samples=5):
    # מנסים קודם את תיקיית ה-TEST (הכי נכון מתודולוגית)
    data_source = TEST_DIR
    all_images = list(data_source.glob("*.jpg"))
    
    # אם אין תמונות ב-TEST (אולי לכולן היו לייבלים), עוברים ל-VAL
    if not all_images:
        print(f"Note: No images found in '{TEST_DIR}'.")
        print(f"Switching to validation set '{VAL_DIR}' for visualization...")
        data_source = VAL_DIR
        all_images = list(data_source.glob("*.jpg"))
        
    if not all_images:
        print("Error: No images found in Test or Validation folders.")
        return
        
    samples = random.sample(all_images, min(len(all_images), num_samples))
    
    print(f"Running inference on {len(samples)} images from: {data_source}")
    
    for img_path in samples:
        # 1. הכנת התמונה למודל
        # קריאה
        original_img = cv2.imread(str(img_path))
        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        
        # שינוי גודל למה שהמודל מכיר (512x256)
        input_img = cv2.resize(original_img, (512, 256))
        
        # המרה ל-Tensor
        img_tensor = torch.from_numpy(input_img).float() / 255.0
        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0) # (1, 3, 256, 512)
        img_tensor = img_tensor.to(DEVICE)
        
        # 2. הרצת המודל (Inference)
        with torch.no_grad():
            pred_potential, pred_flow = model(img_tensor)
            
        # 3. עיבוד התוצאות חזרה לתמונה/numpy
        # פוטנציאל: (1, 1, 256, 512) -> (256, 512)
        potential_map = pred_potential.squeeze().cpu().numpy()
        
        # פלואו: (1, 2, 256, 512) -> (256, 512, 2)
        flow_map = pred_flow.squeeze().cpu().numpy()
        flow_map = np.moveaxis(flow_map, 0, -1) # העברת הערוצים לסוף
        
        # 4. ויזואליזציה (ציור)
        plot_results(input_img, potential_map, flow_map)

def plot_results(img, potential, flow):
    plt.figure(figsize=(18, 5))
    
    # --- א. התמונה המקורית ---
    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.title("Input Image")
    plt.axis('off')
    
    # --- ב. מפת הבטיחות החזויה (Predicted Safety) ---
    plt.subplot(1, 3, 2)
    plt.imshow(potential, cmap='inferno')
    plt.title("Model Prediction: Safety Map")
    plt.axis('off')
    
    # --- ג. נתיב נסיעה חזוי (Predicted Path) ---
    plt.subplot(1, 3, 3)
    plt.imshow(img) # רקע
    plt.title("Model Prediction: Driving Path")
    
    h, w = potential.shape
    
    # ציור חצים (מדולל)
    step = 20
    y, x = np.mgrid[0:h:step, 0:w:step]
    fx = flow[::step, ::step, 0]
    fy = flow[::step, ::step, 1]
    
    # מציגים חצים רק איפה שהמודל חושב שיש כביש בטוח
    mask = potential[::step, ::step] > 0.1 # סף ביטחון
    
    plt.quiver(x[mask], y[mask], fx[mask], fy[mask], 
               color='cyan', scale=25, headwidth=3, alpha=0.5)
    
    # סימולציית נהיגה (הקו הירוק)
    path_x, path_y = simulate_driving(flow, potential, start_pos=(w//2, h-10))
    
    plt.plot(path_x, path_y, color='lime', linewidth=4, label='Model Path')
    plt.legend()
    plt.axis('off')
    
    plt.show()

def simulate_driving(flow, potential, start_pos):
    """אותה לוגיקה של נהיגה, אבל הפעם על התחזית של המודל"""
    curr_x, curr_y = float(start_pos[0]), float(start_pos[1])
    path_x, path_y = [curr_x], [curr_y]
    
    h, w = potential.shape
    
    forward_speed = 3.0
    steering_gain = 4.0
    momentum = 0.8
    current_steering = 0.0
    
    for _ in range(300):
        ix, iy = int(curr_x), int(curr_y)
        
        # גבולות
        if ix < 0 or ix >= w or iy < 0 or iy >= h: break
        
        # אם המודל חושב שזה לא כביש (ערך נמוך בפוטנציאל) - עצור
        if potential[iy, ix] < 0.1: 
            break
            
        # היגוי מהמודל
        raw_steering = flow[iy, ix, 0]
        
        # החלקה
        current_steering = (current_steering * momentum) + (raw_steering * (1 - momentum))
        
        curr_y -= forward_speed
        curr_x += current_steering * steering_gain
        
        path_x.append(curr_x)
        path_y.append(curr_y)
        
    return path_x, path_y

if __name__ == "__main__":
    model = load_trained_model()
    predict_and_visualize(model, num_samples=5)