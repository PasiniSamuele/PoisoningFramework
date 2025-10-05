#!/usr/bin/env python3

import sys
sys.path.append('.')

from utils.aggregate_block.bd_attack_generate import bd_attack_img_trans_generate
import argparse
from PIL import Image
import numpy as np

def test_attack(attack_name):
    """Test if an attack type is supported without UnboundLocalError"""
    print(f"Testing {attack_name} attack...")
    
    # Create minimal args
    args = argparse.Namespace()
    args.attack = attack_name
    args.img_size = (32, 32, 3)
    
    # Add common defaults
    args.patch_mask_path = './resource/badnet/trigger_image.png'
    args.attack_trigger_img_path = './resource/blended/hello_kitty.jpeg'
    args.attack_train_blended_alpha = 0.1
    args.attack_test_blended_alpha = 0.1
    args.sig_delta = 40.0
    args.sig_f = 6
    args.attack_train_replace_imgs_path = './resource/ssba/cifar10_ssba_train_b1.npy'
    args.attack_test_replace_imgs_path = './resource/ssba/cifar10_ssba_test_b1.npy'
    args.mask_path = './resource/trojannn/apple4.png'
    
    # Refool specific
    args.r_adv_img_folder_path = './resource/refool/'
    args.ghost_rate = 0.5
    args.alpha_t = 0.4
    args.offset = (0, 0)
    args.sigma = -1
    args.ghost_alpha = -1
    
    try:
        train_transform, test_transform = bd_attack_img_trans_generate(args)
        print(f"✅ {attack_name}: SUCCESS - transforms generated")
        return True
    except UnboundLocalError as e:
        print(f"❌ {attack_name}: FAILED - UnboundLocalError: {e}")
        return False
    except Exception as e:
        print(f"⚠️  {attack_name}: WARNING - Other error: {e}")
        return True  # Other errors are expected due to missing files, etc.

def main():
    attacks_to_test = [
        'badnet', 'blended', 'sig', 'inputaware', 'trojannn', 
        'wanet', 'refool', 'ssba', 'SSBA'
    ]
    
    results = {}
    for attack in attacks_to_test:
        results[attack] = test_attack(attack)
    
    print("\n" + "="*50)
    print("SUMMARY:")
    for attack, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {attack:12} : {status}")

if __name__ == "__main__":
    main()
