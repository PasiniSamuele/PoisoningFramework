#!/usr/bin/env python3

"""
Test script to verify that refool attack is supported in multi-step framework
"""

import sys
import os
sys.path.append('.')

# Test imports
try:
    from attack.multi_step_attack import MultiStepBackdoorAttack
    print("✓ Multi-step attack import successful")
except ImportError as e:
    print(f"✗ Multi-step attack import failed: {e}")
    exit(1)

try:
    from utils.aggregate_block.bd_attack_generate import bd_attack_img_trans_generate
    print("✓ bd_attack_generate import successful")
except ImportError as e:
    print(f"✗ bd_attack_generate import failed: {e}")
    exit(1)

# Test attack class mapping
try:
    attack_class = MultiStepBackdoorAttack._get_attack_class_static('refool')
    print(f"✓ Attack class mapping successful: {attack_class}")
except Exception as e:
    print(f"✗ Attack class mapping failed: {e}")

# Test that refool resources exist
refool_path = "./resource/refool/Refool-SelectedReflectionImages/selected_out-images"
if os.path.exists(refool_path):
    img_count = len([f for f in os.listdir(refool_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    print(f"✓ Refool reflection images found: {img_count} images")
else:
    print(f"✗ Refool reflection images not found at: {refool_path}")

print("\nRefool support test completed!")
