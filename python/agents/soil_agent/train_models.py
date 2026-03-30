"""
Run this once to train and save the Random Forest models.
Usage: python train_models.py
"""
from fertilizer_model import train_models

if __name__ == "__main__":
    print("Training fertilizer recommendation models...")
    train_models()
    print("Done. Models saved in models/ folder.")