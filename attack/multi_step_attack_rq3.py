#!/usr/bin/env python3
"""
General Multi-Step Attack Framework

This module provides a flexible framework for executing multi-step attacks where:
1. Multiple clean training steps are performed first
2. A final poisoned attack step is executed using any specified attack method

Usage examples:
    # Using BadNet attack
    python multi_step_attack.py --attack_type badnet --clean_steps 2

    # Using Blended attack  
    python multi_step_attack.py --attack_type blended --clean_steps 3

    # Using SIG attack
    python multi_step_attack.py --attack_type sig --clean_steps 1
"""

import os
import sys
from numpy.core.arrayprint import format_float_scientific
import yaml
import importlib
import argparse
import logging

sys.path = ["./"] + sys.path

import numpy as np
import torch

from utils.backdoor_generate_poison_index import generate_poison_index_from_label_transform
from utils.aggregate_block.bd_attack_generate import bd_attack_img_trans_generate, bd_attack_label_trans_generate
from copy import deepcopy
from utils.aggregate_block.model_trainer_generate import generate_cls_model
from utils.aggregate_block.train_settings_generate import argparser_opt_scheduler, argparser_criterion
from utils.save_load_attack import save_attack_result
from attack.prototype import NormalCase
from utils.trainer_cls import BackdoorModelTrainer
from utils.bd_dataset_v2 import prepro_cls_DatasetBD_v2, dataset_wrapper_with_transform


class MultiStepAttack(NormalCase):
    """
    A general multi-step attack wrapper that can execute any attack class in multiple steps.
    
    This class allows for executing clean training steps followed by a poisoned attack step.
    The attack class to be used in the final step is configurable.
    """

    def __init__(self, attack_class=None):
        """
        Initialize MultiStepAttack.
        
        Args:
            attack_class: The attack class to use for the final poisoned step.
                         If None, will be determined from command line arguments.
        """
        super(MultiStepAttack, self).__init__()
        self.attack_class = attack_class

    def set_bd_args(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser.add_argument("--clean_pairs", type=int, default=1,
                           help="Number of clean training pairs before the attack step")
        parser.add_argument("--skip_clean_base_steps", type=bool, default=False,
                           help="Whether to skip clean base steps before the attack step")
        parser.add_argument('--train_set_clean_percentage', type=float, default=0.8,
                           help="Percentage of the set to use to train the initial clean model")
        parser.add_argument("--attack_type", type=str, default="badnet",
                           help="Type of attack to use in the final step (e.g., 'badnet', 'blended', 'sig', etc.)")
        
        # Add common attack arguments that most attacks share
        try:
            from attack.badnet import add_common_attack_args
            parser = add_common_attack_args(parser)
        except ImportError:
            # Fallback: add common arguments manually
            parser.add_argument('--attack', type=str, help='name of attack')
            parser.add_argument('--attack_target', type=int, help='target class in all2one attack')
            parser.add_argument('--attack_label_trans', type=str, help='which type of label modification in backdoor attack')
            parser.add_argument('--pratio', type=float, help='the poison rate')
        
        # Add common backdoor arguments (most attacks use these)
        parser.add_argument("--patch_mask_path", type=str, help="path for patch mask")
        parser.add_argument('--bd_yaml_path', type=str, default='./config/attack/badnet/default.yaml',
                           help='path for yaml file provide additional default attributes')
            
        return parser
    
    @classmethod
    def add_attack_specific_args(cls, parser: argparse.ArgumentParser, attack_type: str) -> argparse.ArgumentParser:
        """
        Add attack-specific arguments based on the attack type.
        This method can be called after we know the attack type.
        """
        try:
            attack_class = cls._get_attack_class_static(attack_type)
            if hasattr(attack_class, 'set_bd_args'):
                # Get attack-specific arguments by calling the attack class's set_bd_args
                temp_parser = argparse.ArgumentParser()
                attack_parser = attack_class.set_bd_args(attack_class, temp_parser)
                
                # Extract attack-specific arguments (excluding common ones)
                common_args = {'attack', 'attack_target', 'attack_label_trans', 'pratio', 
                              'patch_mask_path', 'bd_yaml_path', 'clean_steps', 'attack_type'}
                
                for action in attack_parser._actions:
                    if action.dest not in common_args and action.dest != 'help':
                        # Copy the action to our main parser
                        parser.add_argument(*action.option_strings, **{
                            'type': action.type,
                            'default': action.default,
                            'help': action.help,
                            'choices': action.choices,
                            'required': action.required,
                            'nargs': action.nargs,
                            'const': action.const,
                            'metavar': action.metavar
                        })
        except Exception as e:
            logging.debug(f"Could not add attack-specific args for {attack_type}: {e}")
        
        return parser
    
    @staticmethod
    def _get_attack_class_static(attack_type):
        """Static version of _get_attack_class for use in class methods."""
        attack_mapping = {
            'badnet': ('attack.badnet', 'BadNet'),
            'blended': ('attack.blended', 'Blended'),
            'sig': ('attack.sig', 'SIG'),
            'wanet': ('attack.wanet', 'Wanet'),
            'ssba': ('attack.ssba', 'SSBA'),
            'inputaware': ('attack.inputaware', 'InputAware'),
            'ctrl': ('attack.ctrl', 'CTRL'),
            'lf': ('attack.lf', 'LF'),
            'lc': ('attack.lc', 'LC'),
            'trojannn': ('attack.trojannn', 'TrojanNN'),
            'refool': ('attack.refool', 'Refool'),
        }
        
        if attack_type.lower() not in attack_mapping:
            raise ValueError(f"Unknown attack type: {attack_type}")
        
        module_name, class_name = attack_mapping[attack_type.lower()]
        module = importlib.import_module(module_name)
        return getattr(module, class_name)
    
    def _get_attack_class(self, attack_type):
        """
        Dynamically import and return the attack class based on attack_type.
        
        Args:
            attack_type (str): Name of the attack type (e.g., 'badnet', 'blended')
            
        Returns:
            Attack class
        """
        try:
            # Map attack types to their module names and class names
            attack_mapping = {
                'badnet': ('attack.badnet', 'BadNet'),
                'blended': ('attack.blended', 'Blended'),
                'sig': ('attack.sig', 'SIG'),
                'wanet': ('attack.wanet', 'Wanet'),
                'ssba': ('attack.ssba', 'SSBA'),
                'inputaware': ('attack.inputaware', 'InputAware'),
                'ctrl': ('attack.ctrl', 'CTRL'),
                'lf': ('attack.lf', 'LF'),
                'lc': ('attack.lc', 'LC'),
                'trojannn': ('attack.trojannn', 'TrojanNN'),
                'refool': ('attack.refool', 'Refool'),
                # Add more mappings as needed
            }
            
            if attack_type.lower() not in attack_mapping:
                raise ValueError(f"Unknown attack type: {attack_type}")
            
            module_name, class_name = attack_mapping[attack_type.lower()]
            module = importlib.import_module(module_name)
            attack_class = getattr(module, class_name)
            
            return attack_class
            
        except (ImportError, AttributeError) as e:
            logging.error(f"Failed to import attack class for {attack_type}: {e}")
            # Fallback to BadNet
            from attack.badnet import BadNet
            return BadNet
    
    def prepare(self, args):
        """
        Prepare the multi-step attack with new simplified logic.
        Now we just store the args and determine the attack class.
        """
        super().prepare(args)
        
        # Determine attack class for the final step
        if self.attack_class is None:
            # Use 'attack' from config if available, otherwise fall back to 'attack_type' parameter
            attack_type = getattr(args, 'attack', None) or getattr(args, 'attack_type', 'badnet')
            self.attack_class = self._get_attack_class(attack_type)
            
        logging.info(f"Multi-step attack prepared with attack type: {getattr(args, 'attack', None) or getattr(args, 'attack_type', 'badnet')}")
        logging.info(f"Clean percentage for dataset split: {args.train_set_clean_percentage}")

    def add_bd_yaml_to_args(self, args):
        """Add backdoor YAML configuration to args if the attack step supports it."""
        if hasattr(args, 'bd_yaml_path') and args.bd_yaml_path:
            try:
                with open(args.bd_yaml_path, 'r') as f:
                    mix_defaults = yaml.safe_load(f)
                mix_defaults.update({k: v for k, v in args.__dict__.items() if v is not None})
                args.__dict__ = mix_defaults
            except FileNotFoundError:
                logging.warning(f"BD YAML file not found: {args.bd_yaml_path}")

    def stage1_non_training_data_prepare(self):
        """
        New multi-step stage1 logic:
        
        1. Prepare the original datasets using benign_prepare
        2. Split both training and test datasets into two parts using stratified sampling
           based on train_set_clean_percentage parameter
        3. The first part (clean percentage) will be used for clean model training
        4. The second part (remaining percentage) will be used for the attack step
        """
        logging.info("Multi-step stage1 start - New algorithm")
        
        # Get the original datasets using benign_prepare
        train_dataset_without_transform, \
        train_img_transform, \
        train_label_transform, \
        test_dataset_without_transform, \
        test_img_transform, \
        test_label_transform, \
        clean_train_dataset_with_transform, \
        clean_train_dataset_targets, \
        clean_test_dataset_with_transform, \
        clean_test_dataset_targets = self.benign_prepare()
        
        # Set random seed if provided for reproducible splits
        if hasattr(self.args, 'seed') and self.args.seed is not None:
            np.random.seed(self.args.seed)
            torch.manual_seed(self.args.seed)
            logging.info(f"Set random seed to {self.args.seed} for dataset splitting")
        
        # Split datasets into two parts using stratified sampling
        clean_percentage = self.args.train_set_clean_percentage
        logging.info(f"Splitting training dataset with clean percentage: {clean_percentage}")
        
        # Split training dataset only
        train_clean_part, train_attack_part = self._split_dataset_by_percentage(
            train_dataset_without_transform, clean_percentage, "training"
        )
        
        # Keep test dataset unique (no splitting)
        logging.info("Test dataset will remain unique and not be split")
        
        # Store the split datasets (original clean and attack parts)
        self.train_clean_part = train_clean_part
        self.test_dataset = test_dataset_without_transform  # Unique test set
        
        # Split the attack part into two equal stratified parts
        logging.info("Splitting attack part into two equal stratified parts for clean and poisoned training")
        train_attack_clean_part, train_attack_poisoned_part = self._split_dataset_by_percentage(
            train_attack_part, 0.5, "attack_part"
        )
        
        # Store the two attack parts separately
        self.train_attack_clean_copy = train_attack_clean_part
        self.train_attack_poisoned_part = train_attack_poisoned_part
        # Test set remains the same for all steps
        self.test_attack_clean_copy = test_dataset_without_transform
        
        # Create properly poisoned copies of attack parts using the actual attack method
        logging.info("Creating poisoned attack dataset using attack transforms")
        self._create_poisoned_attack_dataset(self.train_attack_poisoned_part, test_dataset_without_transform)
        
        # Now split the clean parts into multiple pairs based on clean_pairs parameter
        clean_pairs = getattr(self.args, 'clean_pairs', 5)
        logging.info(f"Creating {clean_pairs} clean pairs from the clean parts")
        
        # Split clean training part into pairs
        self.train_clean_pairs = []
        for pair_idx in range(clean_pairs):
            train_clean_1, train_clean_2 = self._split_dataset_by_percentage(
                train_clean_part, clean_percentage, f"training_clean_pair_{pair_idx+1}"
            )
            self.train_clean_pairs.append((train_clean_1, train_clean_2))
        
        # Test dataset remains the same for all clean pairs (no splitting)
        # Each clean pair will use the same unique test dataset
        
        # Store transforms for later use
        self.train_img_transform = train_img_transform
        self.train_label_transform = train_label_transform
        self.test_img_transform = test_img_transform
        self.test_label_transform = test_label_transform
        
        logging.info(f"Dataset splitting completed successfully: {clean_pairs} clean pairs created with unique test set")

    def _split_dataset_by_percentage(self, dataset, clean_percentage, dataset_type=""):
        """
        Split a dataset into two parts using stratified sampling based on percentage.
        
        Args:
            dataset: The dataset to split (torch Dataset or similar)
            clean_percentage (float): Percentage (0.0 to 1.0) for the first part (clean training)
            dataset_type (str): Description for logging ("training" or "test")
            
        Returns:
            tuple: (first_part, second_part) where first_part contains clean_percentage
                   of samples and second_part contains the remaining samples
        """
        # Import necessary modules for dataset handling
        from utils.bd_dataset_v2 import get_labels
        from torch.utils.data import Subset
        import copy
        
        # Get labels from the dataset
        if hasattr(dataset, 'targets'):
            # For datasets with targets attribute (e.g., CIFAR, MNIST)
            labels = np.array(dataset.targets)
        elif hasattr(dataset, 'wrapped_dataset') and hasattr(dataset.wrapped_dataset, 'targets'):
            # For wrapped datasets
            labels = np.array(dataset.wrapped_dataset.targets)
        else:
            # For other dataset types, iterate through to get labels
            labels = np.array(get_labels(dataset))
        
        # Get unique classes and their counts
        unique_classes = np.unique(labels)
        total_samples = len(labels)
        
        logging.info(f"Splitting {dataset_type} dataset with {total_samples} samples across "
                    f"{len(unique_classes)} classes using {clean_percentage:.1%} for clean part")
        
        # Create indices for each part
        clean_indices = []
        attack_indices = []
        
        # For each class, split its samples based on the percentage
        for class_label in unique_classes:
            class_indices = np.where(labels == class_label)[0]
            np.random.shuffle(class_indices)  # Shuffle to ensure randomness
            
            # Calculate how many samples for clean part for this class
            num_class_samples = len(class_indices)
            num_clean_samples = int(num_class_samples * clean_percentage)
            
            # Split the class indices
            clean_class_indices = class_indices[:num_clean_samples]
            attack_class_indices = class_indices[num_clean_samples:]
            
            clean_indices.extend(clean_class_indices)
            attack_indices.extend(attack_class_indices)
            
            logging.debug(f"Class {class_label}: {num_class_samples} total, "
                         f"{len(clean_class_indices)} clean, {len(attack_class_indices)} attack")
        
        # Shuffle indices within each part to mix classes
        np.random.shuffle(clean_indices)
        np.random.shuffle(attack_indices)
        
        # Create dataset parts using Subset
        clean_part = Subset(dataset, clean_indices)
        attack_part = Subset(dataset, attack_indices)
        
        # Log split information
        logging.info(f"{dataset_type.capitalize()} dataset split: "
                    f"{len(clean_indices)} samples for clean part ({len(clean_indices)/total_samples:.1%}), "
                    f"{len(attack_indices)} samples for attack part ({len(attack_indices)/total_samples:.1%})")
        
        return clean_part, attack_part

    def set_data(self, train_dataset, test_dataset, clean_train_dataset, clean_test_dataset):
        """
        Set the data for this training step.
        
        Args:
            train_dataset: Training dataset without transform
            test_dataset: Test dataset without transform  
            clean_train_dataset: Clean training dataset with transform
            clean_test_dataset: Clean test dataset with transform
        """
        self.train_dataset_without_transform = train_dataset
        self.test_dataset_without_transform = test_dataset
        self.clean_train_dataset_with_transform = clean_train_dataset
        self.clean_test_dataset_with_transform = clean_test_dataset

    def _create_set_data_method(self, step):
        """
        Create a set_data method for a training step that doesn't have one.
        
        Args:
            step: The training step instance
            
        Returns:
            A set_data method bound to the step
        """
        def set_data(train_dataset, test_dataset, clean_train_dataset, clean_test_dataset):
            """Set the datasets for this training step."""
            step.train_dataset_without_transform = train_dataset
            step.test_dataset_without_transform = test_dataset  
            step.clean_train_dataset_with_transform = clean_train_dataset
            step.clean_test_dataset_with_transform = clean_test_dataset
            
            # Update stage1_results if it exists
            if hasattr(step, 'stage1_results'):
                step.stage1_results = (clean_train_dataset, clean_test_dataset, None, None)
        
        return set_data

    def _create_benign_prepare_override(self, training_step, step_index):
        """Create an override for the benign_prepare method to return split datasets."""
        
        def benign_prepare_override():
            # Get the split datasets assigned to this training step
            split_train_dataset = training_step._split_train_dataset
            split_test_dataset = training_step._split_test_dataset
            split_clean_train_dataset = training_step._split_clean_train_dataset
            split_clean_test_dataset = training_step._split_clean_test_dataset
            
            # Get transforms from the original method
            # We need the original transforms that were set up during stage1
            if hasattr(training_step, 'stage1_results'):
                original_clean_train, original_clean_test, _, _ = training_step.stage1_results
                train_img_transform = original_clean_train.wrap_img_transform
                train_label_transform = original_clean_train.wrap_label_transform
                test_img_transform = original_clean_test.wrap_img_transform
                test_label_transform = original_clean_test.wrap_label_transform
            else:
                # Fallback - call original method to get transforms
                _, train_img_transform, train_label_transform, _, test_img_transform, test_label_transform, _, _, _, _ = training_step._original_benign_prepare()
            
            # Replace datasets with our split versions
            split_train_dataset = training_step._split_train_dataset
            split_test_dataset = training_step._split_test_dataset
            split_clean_train_dataset = training_step._split_clean_train_dataset
            split_clean_test_dataset = training_step._split_clean_test_dataset
            
            # Get targets from split datasets
            from utils.bd_dataset_v2 import get_labels
            split_clean_train_targets = get_labels(split_train_dataset) 
            split_clean_test_targets = get_labels(split_test_dataset)
            
            # Ensure split datasets with transforms are properly wrapped
            from utils.bd_dataset_v2 import dataset_wrapper_with_transform
            
            # If the split clean datasets don't have proper transforms, recreate them
            if not hasattr(split_clean_train_dataset, 'wrap_img_transform'):
                split_clean_train_dataset = dataset_wrapper_with_transform(
                    split_train_dataset,
                    train_img_transform,
                    train_label_transform
                )
            
            if not hasattr(split_clean_test_dataset, 'wrap_img_transform'):
                split_clean_test_dataset = dataset_wrapper_with_transform(
                    split_test_dataset,
                    test_img_transform,
                    test_label_transform
                )
            
            return split_train_dataset, \
                   train_img_transform, \
                   train_label_transform, \
                   split_test_dataset, \
                   test_img_transform, \
                   test_label_transform, \
                   split_clean_train_dataset, \
                   split_clean_train_targets, \
                   split_clean_test_dataset, \
                   split_clean_test_targets
        
        return benign_prepare_override

    def _create_benign_prepare_override(self, training_step, step_index):
        """
        Create an override for benign_prepare that returns the split datasets.
        
        Args:
            training_step: The training step instance
            step_index: Index of this step in the training sequence
            
        Returns:
            A benign_prepare method that returns split datasets
        """
        def benign_prepare_override():
            """
            Override of benign_prepare that returns the pre-split datasets for this step.
            """
            # Get the original transforms by calling the original benign_prepare
            # but we'll replace the datasets with our split versions
            original_result = training_step._original_benign_prepare()
            
            train_dataset_without_transform, \
            train_img_transform, \
            train_label_transform, \
            test_dataset_without_transform, \
            test_img_transform, \
            test_label_transform, \
            clean_train_dataset_with_transform, \
            clean_train_dataset_targets, \
            clean_test_dataset_with_transform, \
            clean_test_dataset_targets = original_result
            
            # Replace datasets with our split versions
            split_train_dataset = training_step._split_train_dataset
            split_test_dataset = training_step._split_test_dataset
            split_clean_train_dataset = training_step._split_clean_train_dataset
            split_clean_test_dataset = training_step._split_clean_test_dataset
            
            # Get targets from split datasets
            from utils.bd_dataset_v2 import get_labels
            split_clean_train_targets = get_labels(split_train_dataset) 
            split_clean_test_targets = get_labels(split_test_dataset)
            
            # Ensure split datasets with transforms are properly wrapped
            from utils.bd_dataset_v2 import dataset_wrapper_with_transform
            
            # If the split clean datasets don't have proper transforms, recreate them
            if not hasattr(split_clean_train_dataset, 'wrap_img_transform'):
                split_clean_train_dataset = dataset_wrapper_with_transform(
                    split_train_dataset,
                    train_img_transform,
                    train_label_transform
                )
            
            if not hasattr(split_clean_test_dataset, 'wrap_img_transform'):
                split_clean_test_dataset = dataset_wrapper_with_transform(
                    split_test_dataset,
                    test_img_transform,
                    test_label_transform
                )
            
            return split_train_dataset, \
                   train_img_transform, \
                   train_label_transform, \
                   split_test_dataset, \
                   test_img_transform, \
                   test_label_transform, \
                   split_clean_train_dataset, \
                   split_clean_train_targets, \
                   split_clean_test_dataset, \
                   split_clean_test_targets
        
        return benign_prepare_override

    def _restore_original_methods(self):
        """
        Restore original benign_prepare methods for all training steps.
        This can be called after training is complete to clean up overrides.
        """
        for training_step in self.training_steps:
            if hasattr(training_step, '_original_benign_prepare'):
                training_step.benign_prepare = training_step._original_benign_prepare
                delattr(training_step, '_original_benign_prepare')
            
            # Clean up split dataset attributes
            for attr in ['_split_train_dataset', '_split_test_dataset', 
                        '_split_clean_train_dataset', '_split_clean_test_dataset',
                        '_split_bd_train_dataset', '_split_bd_test_dataset']:
                if hasattr(training_step, attr):
                    delattr(training_step, attr)

    def _create_poisoned_attack_dataset(self, train_attack_part, test_dataset):
        """
        Create a properly poisoned dataset using the attack's transforms.
        This creates the actual backdoored data for the attack_poisoned step.
        """
        logging.info("Generating poisoned attack dataset with backdoor transforms")
        
        # Import necessary components for backdoor attack
        from utils.bd_dataset_v2 import get_labels
        from copy import deepcopy
        import torch
        
        # Get the attack part labels
        attack_part_labels = get_labels(train_attack_part)
        
        # Generate backdoor transforms based on the attack type
        train_bd_img_transform, test_bd_img_transform = bd_attack_img_trans_generate(self.args)
        bd_label_transform = bd_attack_label_trans_generate(self.args)
        
        # For attack part, we want to poison a portion (not all) to create a realistic backdoor attack
        # Use the specified poison ratio (pratio)
        pratio = getattr(self.args, 'pratio', 0.1)
        
        logging.info(f"Attack part dataset size: {len(train_attack_part)}")
        logging.info(f"Using poison ratio: {pratio}")
        logging.info(f"Expected poison samples: {round(pratio * len(train_attack_part))}")
        
        # Generate poison indices for training attack part
        train_poison_index = generate_poison_index_from_label_transform(
            attack_part_labels,
            label_transform=bd_label_transform,
            train=True,  # This is training data
            pratio=pratio,  # Use specified poison ratio
        )
        
        logging.info(f"Poisoning {len(train_poison_index)} samples from attack part ({pratio*100}% poison rate)")
        
        # Create the poisoned training dataset for attack part
        self.train_attack_poisoned_copy = prepro_cls_DatasetBD_v2(
            deepcopy(train_attack_part),
            poison_indicator=train_poison_index,
            bd_image_pre_transform=train_bd_img_transform,
            bd_label_pre_transform=bd_label_transform,
            save_folder_path=None  # We don't need to save this
        )
        
        # For test dataset, create a poisoned version for evaluation
        test_labels = get_labels(test_dataset)
        
        # Generate poison indices for test set using the same poison ratio as specified
        test_poison_index = generate_poison_index_from_label_transform(
            test_labels,
            label_transform=bd_label_transform,
            train=False,  # This is test data
            pratio=pratio,  # Use specified poison ratio
        )
        
        logging.info(f"Creating poisoned test dataset with {len(test_poison_index)} poisoned samples")
        
        # Create poisoned test dataset
        self.test_attack_poisoned_copy = prepro_cls_DatasetBD_v2(
            deepcopy(test_dataset),
            poison_indicator=test_poison_index,
            bd_image_pre_transform=test_bd_img_transform,
            bd_label_pre_transform=bd_label_transform,
            save_folder_path=None  # We don't need to save this
        )
        
        logging.info("Poisoned attack datasets created successfully")

    def stage2_training(self):
        """
        Execute the new multi-step training process:
        1. Train clean pairs (each pair: train -> save -> load -> fine-tune -> save)
        2. Execute final attack step
        """
        logging.info('Multi-step stage2 training start - New algorithm')
        
        # # Step 1: Train all clean pairs
        # if not self.args.skip_clean_base_steps:
        #     self._train_clean_pairs()

        # # Step 2: Train attack base model
        # if not self.args.skip_clean_base_steps:
        # self._train_attack_base()

        # Step 3: Train attack clean and poisoned models
        self._train_attack_variants()
        
        logging.info('Multi-step stage2 training completed')

    def _train_clean_pairs(self):
        """
        Train all clean pairs following the procedure:
        For each pair: train with first element -> save -> load -> fine-tune with second element -> save
        """
        import os
        import copy
        
        clean_pairs = getattr(self.args, 'clean_pairs', 5)
        logging.info(f'Starting clean pairs training for {clean_pairs} pairs')
        
        for pair_idx in range(clean_pairs):
            logging.info(f'Training clean pair {pair_idx + 1}/{clean_pairs}')
            
            # Get the pair datasets
            train_clean_1, train_clean_2 = self.train_clean_pairs[pair_idx]
            
            # Create folder for this pair (train_0, train_1, ...)
            pair_folder = f"train_{pair_idx}"
            pair_path = os.path.join(self.args.save_path, pair_folder)
            os.makedirs(pair_path, exist_ok=True)
            
            # Step 1: Train with first element of the pair (from scratch)
            step_0_path = os.path.join(pair_path, "step_0")
            self._train_clean_step(
                train_dataset=train_clean_1,
                test_dataset=self.test_dataset,  # Use unique test set
                step_name=f"pair_{pair_idx}_step_0",
                save_path=step_0_path,
                weights_path=None  # Start from scratch
            )
            
            # Step 2: Fine-tune with second element of the pair (load checkpoint)
            step_1_path = os.path.join(pair_path, "step_1")
            checkpoint_path = os.path.join(step_0_path, "clean_model.pt")
            self._train_clean_step(
                train_dataset=train_clean_2,
                test_dataset=self.test_dataset,  # Use unique test set
                step_name=f"pair_{pair_idx}_step_1",
                save_path=step_1_path,
                weights_path=checkpoint_path  # Load from step 0
            )
            
            logging.info(f'Completed clean pair {pair_idx + 1}/{clean_pairs}')
        
        logging.info('All clean pairs training completed')

    def _train_clean_step(self, train_dataset, test_dataset, step_name, save_path, weights_path=None):
        """
        Train a single clean step using NormalCase (clean training) logic.
        
        Args:
            train_dataset: Training dataset (Subset)
            test_dataset: Test dataset (Subset)
            step_name: Name for logging
            save_path: Path to save results
            weights_path: Path to load weights from (None for fresh start)
        """
        from utils.bd_dataset_v2 import dataset_wrapper_with_transform
        import copy
        import os
        
        logging.info(f'Starting clean training step: {step_name}')
        
        # Create NormalCase instance for clean training
        clean_trainer = NormalCase()
        
        # Create modified args for this step
        step_args = copy.deepcopy(self.args)
        # Set the save_folder_name to include the subdirectory structure
        relative_path = os.path.relpath(save_path, './record')
        step_args.save_folder_name = relative_path
        step_args.weights_path = weights_path
        
        # Create the directory manually to avoid the conflict
        os.makedirs(save_path, exist_ok=True)
        
        # Prepare the trainer
        clean_trainer.prepare(step_args)
        
        # Create datasets with transforms
        train_dataset_with_transform = dataset_wrapper_with_transform(
            train_dataset,
            self.train_img_transform,
            self.train_label_transform
        )
        
        test_dataset_with_transform = dataset_wrapper_with_transform(
            test_dataset,
            self.test_img_transform,
            self.test_label_transform
        )
        
        # Override the benign_prepare method to return our custom datasets
        def custom_benign_prepare():
            from utils.bd_dataset_v2 import get_labels
            
            # Get targets from datasets
            train_targets = get_labels(train_dataset)
            test_targets = get_labels(test_dataset)
            
            return (
                train_dataset,  # train_dataset_without_transform
                self.train_img_transform,  # train_img_transform
                self.train_label_transform,  # train_label_transform
                test_dataset,  # test_dataset_without_transform
                self.test_img_transform,  # test_img_transform
                self.test_label_transform,  # test_label_transform
                train_dataset_with_transform,  # clean_train_dataset_with_transform
                train_targets,  # clean_train_dataset_targets
                test_dataset_with_transform,  # clean_test_dataset_with_transform
                test_targets  # clean_test_dataset_targets
            )
        
        # Store original method and override
        original_benign_prepare = clean_trainer.benign_prepare
        clean_trainer.benign_prepare = custom_benign_prepare
        
        try:
            # Execute training
            clean_trainer.stage1_non_training_data_prepare()
            clean_trainer.stage2_training()
            
            logging.info(f'Completed clean training step: {step_name}')
            
        except Exception as e:
            logging.error(f'Error in clean training step {step_name}: {e}')
            raise
        finally:
            # Restore original method
            clean_trainer.benign_prepare = original_benign_prepare

    def _train_attack_base(self):
        """
        Train the attack base model using the entire train_clean dataset.
        This model serves as the foundation for attack_base_2.
        """
        logging.info('Starting attack base model training')
        
        # Create attack_base directory
        attack_base_path = os.path.join(self.args.save_path, 'attack_base')
        
        # Use the entire train_clean_part dataset for attack base training
        train_dataset = self.train_clean_part
        test_dataset = self.test_dataset
        
        # Step 1: Train attack_base from scratch
        # self._train_attack_step(
        #     step_name='attack_base',
        #     save_path=attack_base_path,
        #     train_dataset=train_dataset,
        #     test_dataset=test_dataset,
        #     weights_path=None  # Train from scratch
        # )
        
        logging.info('Attack base model training completed')
        
        # Step 2: Train attack_base_2 by fine-tuning attack_base with additional data
        logging.info('Starting attack_base_2 model training (fine-tuning attack_base)')
        
        attack_base_2_path = os.path.join(self.args.save_path, 'attack_base_2')
        attack_base_model_path = os.path.abspath(os.path.join(attack_base_path, 'clean_model.pt'))
        
        # Verify the attack_base weights exist before proceeding
        if not os.path.exists(attack_base_model_path):
            raise FileNotFoundError(f'Attack base model not found at: {attack_base_model_path}. '
                                   f'Make sure attack_base training completed successfully.')
        
        logging.info(f'Will load weights from: {attack_base_model_path}')
        
        # Combine train_clean_part + train_attack_clean_part for attack_base_2
        from torch.utils.data import ConcatDataset
        combined_dataset = ConcatDataset([self.train_clean_part, self.train_attack_clean_copy])
        
        self._train_attack_step(
            step_name='attack_base_2',
            save_path=attack_base_2_path,
            train_dataset=combined_dataset,
            test_dataset=test_dataset,
            weights_path=attack_base_model_path  # Load from attack_base
        )
        
        logging.info('Attack_base_2 model training completed')

    def _train_attack_variants(self):
        """
        Train attack_clean and attack_poisoned variants by fine-tuning the attack_base_2 model.
        """
        logging.info('Starting attack variants training')
        
        attack_base_2_model_path = os.path.abspath(os.path.join(self.args.save_path, 'attack_base_2', 'clean_model.pt'))
        
        # Verify the attack_base_2 weights exist before proceeding
        if not os.path.exists(attack_base_2_model_path):
            raise FileNotFoundError(f'Attack base 2 model not found at: {attack_base_2_model_path}. '
                                   f'Make sure attack_base_2 training completed successfully.')
        
        logging.info(f'Will load weights for attack variants from: {attack_base_2_model_path}')
        
        # if not self.args.skip_clean_base_steps:

        # 1. Train attack_clean: train_clean + train_attack_clean_part + train_attack_poisoned_part (clean)
        # self._train_attack_clean(attack_base_2_model_path)
  
        # 2. Train attack_poisoned: train_clean + train_attack_clean_part + train_attack_poisoned_part (poisoned)
        self._train_attack_poisoned(attack_base_2_model_path)

        logging.info('Attack variants training completed')

    def _train_attack_clean(self, base_model_path):
        """
        Train attack_clean model: fine-tune attack_base_2 using the full dataset (clean, no poisoning).
        This uses train_clean_part + train_attack_clean_copy + train_attack_poisoned_part (as clean data).
        """
        logging.info('Training attack_clean model')
        
        # Create attack_clean directory
        attack_clean_path = os.path.join(self.args.save_path, 'attack_clean')
        
        # Combine all parts: train_clean_part + train_attack_clean_copy + train_attack_poisoned_part
        # Note: train_attack_poisoned_part is used here as clean data (without poisoning)
        from torch.utils.data import ConcatDataset
        combined_dataset = ConcatDataset([
            self.train_clean_part, 
            self.train_attack_clean_copy, 
            self.train_attack_poisoned_part  # Use as clean data
        ])
        
        self._train_attack_step(
            step_name='attack_clean',
            save_path=attack_clean_path,
            train_dataset=combined_dataset,
            test_dataset=self.test_dataset,
            weights_path=base_model_path
        )
        
        logging.info('Attack_clean model training completed')

    def _train_attack_poisoned(self, base_model_path):
        """
        Train attack_poisoned model: fine-tune attack_base_2 using the full dataset with poisoning.
        This uses train_clean_part + train_attack_clean_copy + train_attack_poisoned_copy (poisoned version).
        The poisoned copy contains backdoored samples in train_attack_poisoned_part.
        """
        logging.info('Training attack_poisoned model')
        
        # Create attack_poisoned directory
        attack_poisoned_path = os.path.join(self.args.save_path, 'attack_poisoned')
        
        # Combine all parts: train_clean_part + train_attack_clean_copy + train_attack_poisoned_copy
        # Note: train_attack_poisoned_copy contains the backdoored samples
        from torch.utils.data import ConcatDataset
        combined_dataset = ConcatDataset([
            self.train_clean_part, 
            self.train_attack_clean_copy, 
            self.train_attack_poisoned_copy  # Contains poisoned samples
        ])
        
        self._train_attack_step(
            step_name='attack_poisoned',
            save_path=attack_poisoned_path,
            train_dataset=combined_dataset,
            test_dataset=self.test_dataset,
            weights_path=base_model_path
        )
        
        logging.info('Attack_poisoned model training completed')

    def _train_attack_step(self, step_name, save_path, train_dataset, test_dataset, weights_path=None):
        """
        Generic method for training attack steps (base, clean, poisoned).
        
        Args:
            step_name: Name of the training step
            save_path: Directory to save the model
            train_dataset: Training dataset to use
            test_dataset: Test dataset to use
            weights_path: Path to pre-trained weights (None for training from scratch)
        """
        logging.info(f'Starting attack training step: {step_name}')
        
        # Import copy for this method
        import copy
        
        # Use the correct attack class for attack_poisoned, otherwise NormalCase
        if step_name == 'attack_poisoned':
            attack_trainer = self.attack_class()
        else:
            attack_trainer = NormalCase()
        
        # Create modified args for this step
        step_args = copy.deepcopy(self.args)
        
        # For attack_poisoned step, merge BD YAML config to get proper pratio and other attack settings
        if step_name == 'attack_poisoned':
            self.add_bd_yaml_to_args(step_args)
        
        # Set the save_folder_name to include the subdirectory structure
        relative_path = os.path.relpath(save_path, './record')
        step_args.save_folder_name = relative_path
        step_args.weights_path = weights_path
        
        # Log the weights path for debugging
        if weights_path is not None:
            logging.info(f'Setting weights_path to: {weights_path}')
            logging.info(f'Weights file exists: {os.path.exists(weights_path)}')
        else:
            logging.info('No weights_path provided, training from scratch')
        
        # Create the directory manually
        os.makedirs(save_path, exist_ok=True)
        
        # Store original method
        original_benign_prepare = attack_trainer.benign_prepare
        
        # Override benign_prepare to use our custom datasets
        def custom_benign_prepare():
            return train_dataset, test_dataset
        attack_trainer.benign_prepare = custom_benign_prepare
        
        # Prepare the trainer
        attack_trainer.prepare(step_args)
        
        # Create datasets with transforms - these will be captured by the closure
        train_dataset_with_transform = dataset_wrapper_with_transform(
            train_dataset,
            self.train_img_transform,
            self.train_label_transform
        )
        
        test_dataset_with_transform = dataset_wrapper_with_transform(
            test_dataset,
            self.test_img_transform,
            self.test_label_transform
        )
        
        # Override benign_prepare to use our custom datasets and return all required values
        def custom_benign_prepare():
            # We need to return 10 values just like the original benign_prepare method
            # Get labels for the datasets
            from utils.bd_dataset_v2 import get_labels
            
            # For attack_poisoned step, use the poisoned dataset
            if step_name == 'attack_poisoned' and hasattr(self, 'train_attack_poisoned_copy'):
                # Use the poisoned dataset
                train_targets = get_labels(self.train_attack_poisoned_copy)
                test_targets = get_labels(self.test_attack_poisoned_copy)
                
                # Create wrapper with transform for poisoned datasets
                poisoned_train_dataset_with_transform = dataset_wrapper_with_transform(
                    self.train_attack_poisoned_copy,
                    self.train_img_transform,
                    self.train_label_transform
                )
                poisoned_test_dataset_with_transform = dataset_wrapper_with_transform(
                    self.test_attack_poisoned_copy,
                    self.test_img_transform,
                    self.test_label_transform
                )
                
                return (self.train_attack_poisoned_copy,  # train_dataset_without_transform
                       self.train_img_transform,  # train_img_transform
                       self.train_label_transform,  # train_label_transform
                       self.test_attack_poisoned_copy,  # test_dataset_without_transform
                       self.test_img_transform,  # test_img_transform
                       self.test_label_transform,  # test_label_transform
                       poisoned_train_dataset_with_transform,  # clean_train_dataset_with_transform
                       train_targets,  # clean_train_dataset_targets
                       poisoned_test_dataset_with_transform,  # clean_test_dataset_with_transform
                       test_targets)  # clean_test_dataset_targets
            else:
                # For non-poisoned steps, use regular datasets
                train_targets = get_labels(train_dataset)
                test_targets = get_labels(test_dataset)
                
                # Use the train_dataset_with_transform and test_dataset_with_transform defined above
                return (train_dataset,  # train_dataset_without_transform
                       self.train_img_transform,  # train_img_transform
                       self.train_label_transform,  # train_label_transform
                       test_dataset,  # test_dataset_without_transform
                       self.test_img_transform,  # test_img_transform
                       self.test_label_transform,  # test_label_transform
                       train_dataset_with_transform,  # clean_train_dataset_with_transform
                       train_targets,  # clean_train_dataset_targets
                       test_dataset_with_transform,  # clean_test_dataset_with_transform
                       test_targets)  # clean_test_dataset_targets
        attack_trainer.benign_prepare = custom_benign_prepare
        
        try:
            # Execute training
            attack_trainer.stage1_non_training_data_prepare()
            attack_trainer.stage2_training()
            
            logging.info(f'Completed attack training step: {step_name}')
            
        except Exception as e:
            logging.error(f'Error in attack training step {step_name}: {e}')
            raise
        finally:
            # Restore original method
            attack_trainer.benign_prepare = original_benign_prepare

    def _save_overall_results(self):
        """
        Save the overall multi-step attack results to the main directory.
        """
        try:
            # Create a summary of all steps
            summary = {
                'multi_step_attack': True,
                'num_clean_steps': len(self.clean_steps),
                'attack_type': getattr(self.args, 'attack', None) or getattr(self.args, 'attack_type', 'unknown'),
                'clean_step_paths': [step.args.save_path for step in self.clean_steps],
                'attack_step_path': self.attack_step.args.save_path,
                'overall_save_path': self.original_save_path
            }
            
            # Save summary to the main directory
            summary_path = os.path.join(self.original_save_path, 'multi_step_summary.json')
            import json
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
            
            # Copy final results from attack step to main directory for easy access
            if hasattr(self, 'stage1_results'):
                import torch
                torch.save({
                    'stage1_results': self.stage1_results,
                    'net': self.net.state_dict() if hasattr(self, 'net') else None,
                    'multi_step_summary': summary
                }, os.path.join(self.original_save_path, 'multi_step_results.pt'))
            
            logging.info(f"Saved overall multi-step results to: {self.original_save_path}")
            
        except Exception as e:
            logging.warning(f"Failed to save overall results: {e}")



def main():
    """Main execution function for the general multi-step attack."""
    
    # Initialize the multi-step attack
    attack = MultiStepAttack()
    
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="General Multi-Step Attack Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --attack_type badnet --clean_steps 2 --dataset cifar10 --epochs 10
  %(prog)s --attack_type blended --clean_steps 3 --dataset mnist --epochs 5
  %(prog)s --attack_type sig --clean_steps 1 --dataset gtsrb --epochs 15
        """
    )
    
    # Add arguments from base classes
    parser = attack.set_args(parser)
    parser = attack.set_bd_args(parser)
    
    # First parse to get the attack type
    args, unknown = parser.parse_known_args()
    
    # Now add attack-specific arguments based on the attack type
    if hasattr(args, 'attack_type'):
        try:
            parser = MultiStepAttack.add_attack_specific_args(parser, args.attack_type)
        except Exception as e:
            logging.warning(f"Could not add attack-specific arguments for {args.attack_type}: {e}")
    
    # Parse again with all arguments
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)-8s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Log the configuration
    logging.info(f"Starting multi-step attack with {args.clean_pairs} clean pairs")
    logging.info(f"Attack type: {getattr(args, 'attack', None) or getattr(args, 'attack_type', 'badnet')}")
    logging.info(f"Train set clean percentage: {args.train_set_clean_percentage}")
    logging.info("Folder structure will be created as:")
    logging.info(f"  Main: {getattr(args, 'save_path', 'record/...')}")
    for i in range(args.clean_pairs):
        logging.info(f"    Clean pair {i+1}: train_{i}/")
        logging.info(f"      Step 0: train_{i}/step_0/")
        logging.info(f"      Step 1: train_{i}/step_1/")
    logging.info(f"    Attack steps:")
    logging.info(f"      Base: attack_base/ (trained from scratch on train_clean_part)")
    logging.info(f"      Base 2: attack_base_2/ (fine-tuned from attack_base on train_clean_part + train_attack_clean_part)")
    logging.info(f"      Clean: attack_clean/ (fine-tuned from attack_base_2 on full dataset WITHOUT poisoning)")
    logging.info(f"      Poisoned: attack_poisoned/ (fine-tuned from attack_base_2 on full dataset WITH poisoning)")
    logging.info(f"")
    logging.info(f"    Note: Both attack_clean and attack_poisoned use the same data split:")
    logging.info(f"          train_clean_part + train_attack_clean_part + train_attack_poisoned_part")
    logging.info(f"          The difference is that attack_poisoned has backdoor triggers in train_attack_poisoned_part")
    
    # Execute the attack
    try:
        logging.debug("Adding BD YAML configuration to args")
        attack.add_bd_yaml_to_args(args)
        
        logging.debug("Adding general YAML configuration to args")
        attack.add_yaml_to_args(args)
        
        logging.debug("Processing arguments")
        args = attack.process_args(args)
        
        logging.info("Preparing attack steps")
        attack.prepare(args)
        
        logging.info("Stage 1: Non-training data preparation")
        attack.stage1_non_training_data_prepare()
        
        logging.info("Stage 2: Multi-step training execution")
        attack.stage2_training()
        
        logging.info("Multi-step attack completed successfully!")
        
    except Exception as e:
        logging.error(f"Attack failed with error: {e}")
        raise



if __name__ == '__main__':
    # Check if we want to run in backward compatibility mode
    main()
