import json
import cv2
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm

# --- הגדרות ---
ROOT_DIR = "idd20Kfull" 

# מיפוי: כביש מקבל מזהה, כל השאר מקבלים ברירת מחדל של 255
# שים לב: אנחנו לא מגדירים פה את 'car' או 'person', הם יטופלו אוטומטית כ-255
LABEL_MAPPING = {
    'road': 0,
    'parking': 1,
    'drivable fallback': 2
}

def convert_json_to_label(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    h = data['imgHeight']
    w = data['imgWidth']
    
    # 1. מתחילים עם קנבס של "מכשול" (255)
    mask = np.full((h, w), 255, dtype=np.uint8)
    
    objects = data.get('objects', [])
    
    # 2. עוברים על האובייקטים לפי הסדר בקובץ (חשוב! הסדר קובע מי דורס את מי)
    for obj in objects:
        label_name = obj['label']
        
        # בדיקה האם זה כביש (0-2) או מכשול (255)
        # שיטת .get מחזירה 255 אם התווית לא נמצאת במילון (למשל car, person)
        val = LABEL_MAPPING.get(label_name, 255)
        
        # המרת קואורדינטות ל-int32 (התיקון שעשינו קודם)
        poly_points = np.array(obj['polygon'], dtype=np.int32)
        
        if poly_points.size > 0:
            # התיקון הקריטי: אנחנו מציירים גם אם זה 255!
            # זה מבטיח שמכונית תצויר כ"מכשול" ותמחק את הכביש שמתחתיה
            cv2.fillPoly(mask, [poly_points], val)
                
    return mask

def main():
    print("--- Starting JSON to PNG Conversion (Corrected for Obstacles) ---")
    root = Path(ROOT_DIR)
    
    json_files = list(root.rglob("*_polygons.json"))
    
    if not json_files:
        print("Error: No JSON files found!")
        return

    print(f"Found {len(json_files)} JSON files. Converting...")
    
    count = 0
    errors = 0
    
    for json_path in tqdm(json_files):
        try:
            mask = convert_json_to_label(json_path)
            
            new_name = json_path.name.replace("_polygons.json", "_label.png")
            save_path = json_path.parent / new_name
            
            cv2.imwrite(str(save_path), mask)
            count += 1
            
        except Exception as e:
            errors += 1

    print("-" * 30)
    print(f"Summary:")
    print(f"  Successfully converted: {count}")
    print(f"  Errors: {errors}")
    print("\nIMPORTANT: Now run 'prepare_data.py' again to update the distance maps.")

if __name__ == "__main__":
    main()