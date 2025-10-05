#!/usr/bin/env python3
"""
Test script for SVHN dataset implementation
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset.SVHN import SVHN
import torchvision.transforms as transforms
from utils.aggregate_block.dataset_and_transform_generate import get_num_classes, get_input_shape, get_dataset_normalization

def test_svhn():
    print("Testing SVHN dataset implementation...")
    
    # Test dataset parameters
    dataset_name = "svhn"
    
    # Test utility functions
    print(f"Number of classes: {get_num_classes(dataset_name)}")
    print(f"Input shape: {get_input_shape(dataset_name)}")
    
    # Test normalization
    norm = get_dataset_normalization(dataset_name)
    print(f"Normalization: {norm}")
    
    # Create transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        norm
    ])
    
    # Test dataset loading
    print("\nLoading SVHN dataset...")
    try:
        train_dataset = SVHN("data/svhn", train=True, transform=transform, download=True)
        test_dataset = SVHN("data/svhn", train=False, transform=transform, download=True)
        
        print(f"Training samples: {len(train_dataset)}")
        print(f"Test samples: {len(test_dataset)}")
        
        # Test getting a sample
        image, label = train_dataset[0]
        print(f"First sample - Image shape: {image.shape}, Label: {label}")
        
        print("\nSVHN dataset implementation test completed successfully!")
        return True
        
    except Exception as e:
        print(f"Error testing SVHN dataset: {e}")
        return False

if __name__ == "__main__":
    test_svhn()
