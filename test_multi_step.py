#!/usr/bin/env python3
"""
Test script for the multi-step attack clean pairs training
"""

import os
import sys
import argparse
import logging

# Add current directory to path
sys.path = ["./"] + sys.path

from attack.multi_step_attack import MultiStepAttack

def test_clean_pairs_training():
    """Test the clean pairs training functionality."""
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)-8s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Create test arguments
    parser = argparse.ArgumentParser()
    
    # Initialize the multi-step attack and get default arguments
    attack = MultiStepAttack()
    parser = attack.set_args(parser)
    parser = attack.set_bd_args(parser)
    
    # Set test parameters
    test_args = [
        '--dataset', 'cifar10',
        '--epochs', '2',  # Very few epochs for testing
        '--clean_pairs', '2',  # Only 2 pairs for testing
        '--train_set_clean_percentage', '0.8',
        '--attack_type', 'badnet',
        '--save_path', './test_multi_step_output',
        '--seed', '42'
    ]
    
    args = parser.parse_args(test_args)
    
    logging.info("Starting multi-step attack test")
    logging.info(f"Dataset: {args.dataset}")
    logging.info(f"Epochs: {args.epochs}")
    logging.info(f"Clean pairs: {args.clean_pairs}")
    logging.info(f"Train set clean percentage: {args.train_set_clean_percentage}")
    
    try:
        # Execute the test
        attack.add_bd_yaml_to_args(args)
        attack.add_yaml_to_args(args)
        args = attack.process_args(args)
        
        logging.info("Preparing attack")
        attack.prepare(args)
        
        logging.info("Stage 1: Dataset preparation")
        attack.stage1_non_training_data_prepare()
        
        logging.info("Stage 2: Clean pairs training")
        attack.stage2_training()
        
        logging.info("Test completed successfully!")
        
        # Print summary of what was created
        if os.path.exists(args.save_path):
            logging.info(f"Created output in: {args.save_path}")
            for item in os.listdir(args.save_path):
                item_path = os.path.join(args.save_path, item)
                if os.path.isdir(item_path):
                    logging.info(f"  Folder: {item}/")
                    for subitem in os.listdir(item_path):
                        logging.info(f"    {subitem}")
        
    except Exception as e:
        logging.error(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == '__main__':
    success = test_clean_pairs_training()
    if success:
        print("\n✅ Test completed successfully!")
    else:
        print("\n❌ Test failed!")
        sys.exit(1)
