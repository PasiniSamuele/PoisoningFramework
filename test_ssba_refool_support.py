#!/usr/bin/env python3

"""
Test script to verify that both refool and ssba attacks are supported in bd_attack_generate.py
"""

import sys
import os
import argparse
import tempfile
import numpy as np
sys.path.append('.')

# Test imports
try:
    from utils.aggregate_block.bd_attack_generate import bd_attack_img_trans_generate
    print("✓ bd_attack_generate import successful")
except ImportError as e:
    print(f"✗ bd_attack_generate import failed: {e}")
    exit(1)

# Create a mock args object for testing
class MockArgs:
    def __init__(self, attack_type):
        self.attack = attack_type
        self.attack_type = attack_type
        self.img_size = (32, 32, 3)
        
        # For SSBA
        self.attack_train_replace_imgs_path = "./resource/ssba/cifar10_ssba_train_b1.npy"
        self.attack_test_replace_imgs_path = "./resource/ssba/cifar10_ssba_test_b1.npy"
        
        # For refool
        self.r_adv_img_folder_path = "./resource/refool/Refool-SelectedReflectionImages/selected_out-images"
        self.ghost_rate = 0.49
        self.alpha_t = 0.4
        self.offset = (0, 0)
        self.sigma = -1
        self.ghost_alpha = -1

def test_attack_support(attack_name):
    """Test if an attack type is supported in bd_attack_generate"""
    print(f"\n--- Testing {attack_name} support ---")
    
    # Check resource files first
    if attack_name.lower() in ['ssba']:
        train_path = "./resource/ssba/cifar10_ssba_train_b1.npy"
        test_path = "./resource/ssba/cifar10_ssba_test_b1.npy"
        if os.path.exists(train_path) and os.path.exists(test_path):
            print(f"✓ SSBA resource files found")
        else:
            print(f"✗ SSBA resource files missing: {train_path}, {test_path}")
            return False
    
    elif attack_name.lower() == 'refool':
        refool_path = "./resource/refool/Refool-SelectedReflectionImages/selected_out-images"
        if os.path.exists(refool_path):
            img_count = len([f for f in os.listdir(refool_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            print(f"✓ Refool reflection images found: {img_count} images")
        else:
            print(f"✗ Refool reflection images not found at: {refool_path}")
            return False
    
    # Test transform generation
    try:
        args = MockArgs(attack_name)
        train_transform, test_transform = bd_attack_img_trans_generate(args)
        print(f"✓ {attack_name} transform generation successful")
        print(f"  - Train transform: {type(train_transform)}")
        print(f"  - Test transform: {type(test_transform)}")
        return True
    except Exception as e:
        print(f"✗ {attack_name} transform generation failed: {e}")
        return False

def main():
    print("Testing attack support in bd_attack_generate.py")
    
    # Test both attacks
    attacks_to_test = ['SSBA', 'refool']
    results = {}
    
    for attack in attacks_to_test:
        results[attack] = test_attack_support(attack)
    
    # Summary
    print(f"\n--- Test Results Summary ---")
    for attack, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{attack}: {status}")
    
    all_passed = all(results.values())
    print(f"\nOverall: {'✓ ALL TESTS PASSED' if all_passed else '✗ SOME TESTS FAILED'}")
    return 0 if all_passed else 1

if __name__ == "__main__":
    exit(main())
