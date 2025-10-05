"""
SVHN (Street View House Numbers) dataset implementation for the PoisoningFramework.

The SVHN dataset consists of 32x32 colored digit images.
- 10 classes (digits 0-9)
- Training set: 73,257 images
- Test set: 26,032 images
- Extra set: 531,131 additional (less difficult) images

This implementation uses torchvision's SVHN dataset.
"""

import os
import torch.utils.data as data
from torchvision.datasets import SVHN as TorchSVHN
from PIL import Image


class SVHN(data.Dataset):
    def __init__(self, data_root, train=True, transform=None, download=True):
        """
        Initialize SVHN dataset.
        
        Args:
            data_root (str): Root directory for the dataset
            train (bool): If True, use training set; if False, use test set
            transform: Transforms to apply to the images
            download (bool): If True, download the dataset if not found
        """
        super(SVHN, self).__init__()
        
        self.data_root = data_root
        self.transform = transform
        
        # Create directory if it doesn't exist
        if not os.path.exists(data_root):
            os.makedirs(data_root)
        
        # Use 'train' for training set and 'test' for test set
        split = 'train' if train else 'test'
        
        # Load the dataset using torchvision's SVHN
        self.svhn_dataset = TorchSVHN(
            root=data_root,
            split=split,
            transform=None,  # We'll apply transforms manually
            download=download
        )
        
        # Get all data and labels
        self.data = self.svhn_dataset.data
        self.labels = self.svhn_dataset.labels.tolist()
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        """
        Get item at index.
        
        Args:
            index (int): Index of the item
            
        Returns:
            tuple: (image, label) where image is PIL Image and label is int
        """
        # Get image and label
        image = self.data[index]
        label = self.labels[index]
        
        # Convert numpy array to PIL Image
        # SVHN data comes as (3, 32, 32) so we need to transpose to (32, 32, 3)
        image = image.transpose(1, 2, 0)
        image = Image.fromarray(image)
        
        # Apply transforms if provided
        if self.transform is not None:
            image = self.transform(image)
            
        return image, label
    
    def get_number_classes(self):
        """Return the number of classes in SVHN (10 digits)."""
        return 10
