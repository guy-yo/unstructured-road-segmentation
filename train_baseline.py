"""
Baseline training script
U-Net for road segmentation only (no flow, no path)
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
