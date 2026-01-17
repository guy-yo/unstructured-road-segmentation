import torch
import torch.nn as nn
from train_baseline import UNet, get_dataloaders, compute_iou

def run_experiment(optimizer_name, lr=1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UNet().to(device)
    criterion = nn.BCEWithLogitsLoss()

    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    elif optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    else:
        raise ValueError("Unknown optimizer")

    print(f"Prepared experiment with optimizer: {optimizer_name}")
    return model, optimizer, criterion


if __name__ == "__main__":
    optimizers = ["adam", "sgd", "adamw"]

    for opt in optimizers:
        run_experiment(opt)
