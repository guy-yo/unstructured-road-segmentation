import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm # פס התקדמות יפה
import os

# ייבוא המחלקות שיצרנו בקבצים הקודמים
from potential_flow_dataset import PotentialFlowDataset
from dual_head_model import DualHeadResNetUNet

# --- הגדרות (Hyperparameters) ---
BATCH_SIZE = 8          # כמה תמונות המודל רואה במכה (תלוי בזיכרון ה-GPU שלך)
LEARNING_RATE = 0.0001  # קצב הלמידה (צעדים קטנים כדי לא לפספס את המינימום)
NUM_EPOCHS = 10         # כמה פעמים עוברים על כל הדאטה
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def train_one_epoch(model, loader, optimizer, criterion_potential, criterion_flow):
    model.train() # מעביר למצב אימון (חשוב ל-Dropout/BatchNorm)
    total_loss = 0
    
    # שימוש ב-tqdm לפס התקדמות
    loop = tqdm(loader, desc="Training", leave=False)
    
    for images, masks, flows in loop:
        images = images.to(DEVICE)
        masks = masks.to(DEVICE)
        flows = flows.to(DEVICE)
        
        # 1. הרצה קדימה (Forward Pass)
        pred_potential, pred_flow = model(images)
        
        # 2. חישוב השגיאה (Loss)
        loss_pot = criterion_potential(pred_potential, masks)
        loss_flow = criterion_flow(pred_flow, flows)
        
        # שגיאה משולבת (אפשר לתת משקלים שונים, כרגע 1:1)
        loss = loss_pot + loss_flow
        
        # 3. עדכון משקולות (Backward Pass)
        optimizer.zero_grad() # איפוס נגזרות קודמות
        loss.backward()       # חישוב נגזרות חדשות
        optimizer.step()      # עדכון המשקולות
        
        total_loss += loss.item()
        
        # עדכון פס ההתקדמות
        loop.set_postfix(loss=loss.item())
        
    return total_loss / len(loader)

def evaluate(model, loader, criterion_potential, criterion_flow):
    model.eval() # מעביר למצב בדיקה (מנטרל Dropout)
    total_loss = 0
    
    with torch.no_grad(): # לא מחשבים נגזרות (חוסך זיכרון)
        for images, masks, flows in loader:
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)
            flows = flows.to(DEVICE)
            
            pred_potential, pred_flow = model(images)
            
            loss_pot = criterion_potential(pred_potential, masks)
            loss_flow = criterion_flow(pred_flow, flows)
            loss = loss_pot + loss_flow
            
            total_loss += loss.item()
            
    return total_loss / len(loader)

def main():
    print(f"--- Starting Training on {DEVICE} ---")
    
    # 1. הכנת הדאטה
    train_ds = PotentialFlowDataset(root_dir="DATA", split="train")
    val_ds = PotentialFlowDataset(root_dir="DATA", split="val")
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # 2. הכנת המודל
    model = DualHeadResNetUNet().to(DEVICE)
    
    # 3. הגדרת פונקציות שגיאה (Loss) ואופטימייזר
    # MSE (Mean Squared Error) מתאים כי הערכים הם רציפים (0.0 עד 1.0 או וקטורים)
    criterion = nn.MSELoss() 
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # משתנה לשמירת המודל הטוב ביותר
    best_val_loss = float("inf")
    
    # 4. לולאת האימון הראשית
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        
        # אימון
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, criterion)
        
        # בדיקה (ולידציה)
        val_loss = evaluate(model, val_loader, criterion, criterion)
        
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        # שמירת המודל אם הוא השיג תוצאה טובה יותר בולידציה
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pth")
            print(">>> New best model saved!")

    print("\nDone! Training finished.")
    print("Best model is saved as 'best_model.pth'")

if __name__ == "__main__":
    main()