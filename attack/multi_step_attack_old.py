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
        parser.add_argument("--clean_steps", type=int, default=,
                           help="Number of clean training steps before the attack step")
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
            'wanet': ('attack.wanet', 'WaNet'),
            'ssba': ('attack.ssba', 'SSBA'),
            'inputaware': ('attack.inputaware', 'InputAware'),
            'ctrl': ('attack.ctrl', 'CTRL'),
            'lf': ('attack.lf', 'LF'),
            'lc': ('attack.lc', 'LC'),
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
                'wanet': ('attack.wanet', 'WaNet'),
                'ssba': ('attack.ssba', 'SSBA'),
                'inputaware': ('attack.inputaware', 'InputAware'),
                'ctrl': ('attack.ctrl', 'CTRL'),
                'lf': ('attack.lf', 'LF'),
                'lc': ('attack.lc', 'LC'),
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
        Prepare the multi-step attack by setting up clean steps and the final attack step.
        """
        super().prepare(args)
        
        # Store the original save path to create subfolders
        self.original_save_path = args.save_path
        self.original_save_folder_name = getattr(args, 'save_folder_name', None)
        
        self.training_steps = []
        
        # Add clean training steps
        for i in range(args.clean_steps):
            clean_step = NormalCase()
            
            # Create a copy of args with modified save path for this clean step
            step_args = deepcopy(args)
            step_save_name = f"clean_{i+1:02d}"  # clean_01, clean_02, etc.
            
            # Update save paths for this step
            step_args.save_path = os.path.join(self.original_save_path, step_save_name)
            
            # Keep save_folder_name but set it to the step-specific path  
            step_args.save_folder_name = step_save_name
            
            # Create the directory if it doesn't exist
            os.makedirs(step_args.save_path, exist_ok=True)
            
            logging.info(f"Created clean step {i+1} with save path: {step_args.save_path}")
            
            # Override the prepare method to use our custom save path
            original_prepare = clean_step.prepare
            def custom_prepare(custom_args):
                # Set the save path directly without calling generate_save_folder
                custom_args.save_path = step_args.save_path
                # Call parent's add_yaml_to_args and process_args but skip save path logic
                clean_step.add_yaml_to_args(custom_args)
                custom_args = clean_step.process_args(custom_args)
                # Save info.pickle directly to our path
                torch.save(custom_args.__dict__, custom_args.save_path + '/info.pickle')
                # Store args
                clean_step.args = custom_args
                return custom_args
            
            clean_step.prepare = custom_prepare
            clean_step.prepare(step_args)
            self.training_steps.append(clean_step)
        
        # Determine attack class
        if self.attack_class is None:
            attack_type = getattr(args, 'attack_type', 'badnet')
            self.attack_class = self._get_attack_class(attack_type)
        
        # Add the poisoned attack step
        poisoned_step = self.attack_class()
        
        # Create modified args with save path for the attack step
        attack_args = deepcopy(args)
        attack_save_name = "poisoned"
        
        # Update save paths for attack step
        attack_args.save_path = os.path.join(self.original_save_path, attack_save_name)
        
        # Keep save_folder_name but set it to the step-specific path
        attack_args.save_folder_name = attack_save_name
        
        # Create the directory if it doesn't exist
        os.makedirs(attack_args.save_path, exist_ok=True)
        
        logging.info(f"Created attack step with save path: {attack_args.save_path}")
        
        # Override the prepare method to use our custom save path
        original_prepare = poisoned_step.prepare
        def custom_prepare(custom_args):
            # Set the save path directly without calling generate_save_folder
            custom_args.save_path = attack_args.save_path
            # Call parent's add_yaml_to_args and process_args but skip save path logic
            poisoned_step.add_yaml_to_args(custom_args)
            custom_args = poisoned_step.process_args(custom_args)
            # Save info.pickle directly to our path
            torch.save(custom_args.__dict__, custom_args.save_path + '/info.pickle')
            # Store args
            poisoned_step.args = custom_args
            return custom_args
            
        poisoned_step.prepare = custom_prepare
        poisoned_step.prepare(attack_args)
        self.training_steps.append(poisoned_step)
        
        # Add set_data method to all training steps that don't have it
        for step in self.training_steps:
            if not hasattr(step, 'set_data'):
                step.set_data = self._create_set_data_method(step)

    @property
    def clean_steps(self):
        """Get all clean training steps (all except the last one)."""
        return self.training_steps[:-1]
    
    @property 
    def attack_step(self):
        """Get the attack step (last training step)."""
        return self.training_steps[-1]

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
        Execute stage1 for all training steps with improved logic.
        
        1. First execute the attack step's stage1 to get complete datasets (including backdoored data)
        2. Split those datasets into stratified parts  
        3. Set each step's data appropriately
        """
        logging.info("Multi-step stage1 start")

        # First, let the attack step (last step) prepare its data to get complete datasets
        attack_step = self.attack_step
        attack_step.stage1_non_training_data_prepare()
        
        # Get the complete datasets from the attack step
        if hasattr(attack_step, 'stage1_results'):
            clean_train_dataset_with_transform, \
            clean_test_dataset_with_transform, \
            bd_train_dataset, \
            bd_test_dataset = attack_step.stage1_results
        else:
            # Fallback to benign prepare if stage1_results not available
            train_dataset_without_transform, \
            train_img_transform, \
            train_label_transform, \
            test_dataset_without_transform, \
            test_img_transform, \
            test_label_transform, \
            clean_train_dataset_with_transform, \
            clean_train_dataset_targets, \
            clean_test_dataset_with_transform, \
            clean_test_dataset_targets \
                = self.benign_prepare()
            
            bd_train_dataset = clean_train_dataset_with_transform
            bd_test_dataset = clean_test_dataset_with_transform
        
        # Get the underlying datasets without transforms for splitting
        train_dataset_without_transform = clean_train_dataset_with_transform.wrapped_dataset if hasattr(clean_train_dataset_with_transform, 'wrapped_dataset') else clean_train_dataset_with_transform
        test_dataset_without_transform = clean_test_dataset_with_transform.wrapped_dataset if hasattr(clean_test_dataset_with_transform, 'wrapped_dataset') else clean_test_dataset_with_transform
        
        # Split only the training datasets into stratified parts
        # Test datasets should remain the same for all steps for fair comparison
        num_steps = len(self.training_steps)
        train_dataset_parts = self._split_dataset(train_dataset_without_transform, num_steps)
        clean_train_dataset_parts = self._split_dataset(train_dataset_without_transform, num_steps)
        
        # Split the backdoor dataset as well
        bd_dataset_without_transform = bd_train_dataset.wrapped_dataset if hasattr(bd_train_dataset, 'wrapped_dataset') else bd_train_dataset
        bd_train_dataset_parts = self._split_dataset(bd_dataset_without_transform, num_steps)
        
        # For the attack step, we'll use the full backdoored training dataset
        # (not split) to maintain the original poisoning ratio

        # Set data for each training step and override their benign_prepare method
        for i, training_step in enumerate(self.training_steps):
            # Set the split training datasets for this step
            training_step._split_train_dataset = train_dataset_parts[i] 
            training_step._split_clean_train_dataset = clean_train_dataset_parts[i]
            
            # Set the same test datasets for all steps (no splitting for test data)
            training_step._split_test_dataset = test_dataset_without_transform
            training_step._split_clean_test_dataset = clean_test_dataset_with_transform
            
            # For the attack step (last step), use the split backdoored datasets
            if i == len(self.training_steps) - 1:
                if bd_train_dataset is not None:
                    training_step._split_bd_train_dataset = bd_train_dataset_parts[i]
                # Use the full backdoor test dataset (no splitting)
                if bd_test_dataset is not None:
                    training_step._split_bd_test_dataset = bd_test_dataset
            
            # Override the benign_prepare method to return the split datasets
            training_step._original_benign_prepare = training_step.benign_prepare
            training_step.benign_prepare = self._create_benign_prepare_override(training_step, i)
            
            # Now call stage1 on each step (except attack step which was already called)
            if i < len(self.training_steps) - 1:  # Clean steps
                training_step.stage1_non_training_data_prepare()
            else:  # Attack step - update its stage1_results with split data
                if hasattr(training_step, '_split_bd_train_dataset') and hasattr(training_step, '_split_bd_test_dataset'):
                    # Apply proper transforms to split backdoor datasets
                    from utils.bd_dataset_v2 import dataset_wrapper_with_transform
                    
                    # Get the original transforms from the attack step
                    if hasattr(training_step, 'stage1_results') and len(training_step.stage1_results) >= 4:
                        original_clean_train = training_step.stage1_results[0]
                        original_bd_train = training_step.stage1_results[2]
                        original_bd_test = training_step.stage1_results[3]
                        
                        # Extract transforms from the wrapped datasets
                        if hasattr(original_bd_train, 'wrap_img_transform'):
                            train_img_transform = original_bd_train.wrap_img_transform
                            train_label_transform = original_bd_train.wrap_label_transform
                        else:
                            # Fallback to clean dataset transforms
                            train_img_transform = original_clean_train.wrap_img_transform
                            train_label_transform = original_clean_train.wrap_label_transform
                        
                        if hasattr(original_bd_test, 'wrap_img_transform'):
                            test_img_transform = original_bd_test.wrap_img_transform
                            test_label_transform = original_bd_test.wrap_label_transform
                        else:
                            # Fallback to clean dataset transforms
                            test_img_transform = original_clean_train.wrap_img_transform if hasattr(original_clean_train, 'wrap_img_transform') else train_img_transform
                            test_label_transform = original_clean_train.wrap_label_transform if hasattr(original_clean_train, 'wrap_label_transform') else train_label_transform
                        
                        # Wrap split backdoor datasets with proper transforms
                        split_bd_train_with_transform = dataset_wrapper_with_transform(
                            training_step._split_bd_train_dataset,
                            train_img_transform,
                            train_label_transform
                        )
                        
                        # For test dataset, use the original full test dataset (no splitting)
                        # The _split_bd_test_dataset actually contains the full dataset
                        split_bd_test_with_transform = training_step._split_bd_test_dataset
                        
                        # Update stage1_results with properly wrapped split datasets
                        training_step.stage1_results = (
                            training_step._split_clean_train_dataset,
                            training_step._split_clean_test_dataset,
                            split_bd_train_with_transform,
                            split_bd_test_with_transform
                        )
                    else:
                        training_step.stage1_results = (
                            training_step._split_clean_train_dataset,
                            training_step._split_clean_test_dataset,
                        None,
                        None
                    )

    def _split_dataset(self, dataset, num_splits):
        """
        Split a dataset into N non-overlapping stratified parts.
        
        Args:
            dataset: The dataset to split (torch Dataset or similar)
            num_splits (int): Number of parts to split the dataset into
            
        Returns:
            list: List of dataset parts, each containing approximately equal 
                  class distributions and non-overlapping samples
        """
        # Import necessary modules for dataset handling
        from utils.bd_dataset_v2 import get_labels
        from torchvision.datasets import DatasetFolder, ImageFolder
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
        
        logging.info(f"Splitting dataset with {total_samples} samples across {len(unique_classes)} classes into {num_splits} parts")
        
        # Create stratified indices for each split
        split_indices = [[] for _ in range(num_splits)]
        
        # For each class, split its samples across the different parts
        for class_label in unique_classes:
            class_indices = np.where(labels == class_label)[0]
            np.random.shuffle(class_indices)  # Shuffle to ensure randomness
            
            # Calculate how many samples per split for this class
            samples_per_split = len(class_indices) // num_splits
            remainder = len(class_indices) % num_splits
            
            start_idx = 0
            for split_idx in range(num_splits):
                # Add one extra sample to the first 'remainder' splits to distribute remainder
                split_size = samples_per_split + (1 if split_idx < remainder else 0)
                end_idx = start_idx + split_size
                
                split_indices[split_idx].extend(class_indices[start_idx:end_idx])
                start_idx = end_idx
        
        # Shuffle indices within each split to mix classes
        for split_idx in range(num_splits):
            np.random.shuffle(split_indices[split_idx])
            
        # Create dataset parts by copying and subsetting the original dataset
        dataset_parts = []
        for split_idx in range(num_splits):
            if hasattr(dataset, 'copy'):
                # If dataset has a copy method, use it
                dataset_part = dataset.copy()
            else:
                # Otherwise, use deepcopy
                dataset_part = copy.deepcopy(dataset)
            
            # Subset the dataset to only include indices for this split
            if hasattr(dataset_part, 'subset'):
                dataset_part.subset(split_indices[split_idx])
            else:
                # If no subset method, we need to create a torch Subset
                from torch.utils.data import Subset
                dataset_part = Subset(dataset, split_indices[split_idx])
            
            dataset_parts.append(dataset_part)
            
            # Log split information
            if hasattr(dataset_part, 'targets'):
                part_labels = np.array(dataset_part.targets)
            elif hasattr(dataset_part, 'wrapped_dataset') and hasattr(dataset_part.wrapped_dataset, 'targets'):
                part_labels = np.array(dataset_part.wrapped_dataset.targets)
            else:
                part_labels = np.array([labels[idx] for idx in split_indices[split_idx]])
            
            unique_part_classes, counts = np.unique(part_labels, return_counts=True)
            logging.debug(f"Split {split_idx}: {len(split_indices[split_idx])} samples, "
                         f"classes: {dict(zip(unique_part_classes, counts))}")
        
        return dataset_parts

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

    def stage2_training(self):
        """
        Execute the multi-step training process.
        """
        logging.info('Multi-step stage2 training start')
        
        for step_index, training_step in enumerate(self.training_steps):
            step_type = "clean" if step_index < len(self.training_steps) - 1 else "attack"
            logging.info(f'Starting training step {step_index + 1}/{len(self.training_steps)} ({step_type})')
            
            # Set up weights_path for model transfer (except for first step)
            if step_index > 0:
                previous_step_index = step_index - 1
                if previous_step_index < len(self.clean_steps):
                    # Previous step was clean
                    previous_save_path = self.clean_steps[previous_step_index].args.save_path
                else:
                    # Previous step was attack (shouldn't happen in our case)
                    previous_save_path = self.attack_step.args.save_path
                
                weights_path = os.path.join(previous_save_path, 'clean_model.pt')
                training_step.args.weights_path = weights_path
                logging.info(f'Set weights_path for step {step_index + 1}: {weights_path}')
            else:
                training_step.args.weights_path = None
                logging.info(f'Step {step_index + 1} starting with fresh weights')
            
            # Execute the training step
            training_step.stage2_training()
            
            logging.info(f'Completed training step {step_index + 1}/{len(self.training_steps)} ({step_type})')

    def _save_overall_results(self):
        """
        Save the overall multi-step attack results to the main directory.
        """
        try:
            # Create a summary of all steps
            summary = {
                'multi_step_attack': True,
                'num_clean_steps': len(self.clean_steps),
                'attack_type': getattr(self.args, 'attack_type', 'unknown'),
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
    logging.info(f"Starting multi-step attack with {args.clean_steps} clean steps")
    logging.info(f"Attack type: {getattr(args, 'attack_type', 'badnet')}")
    logging.info("Folder structure will be created as:")
    logging.info(f"  Main: {getattr(args, 'save_path', 'record/...')}")
    for i in range(args.clean_steps):
        logging.info(f"    Clean step {i+1}: clean_{i+1:02d}/")
    logging.info(f"    Attack step: poisoned/")
    
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
