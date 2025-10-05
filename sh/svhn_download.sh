#!/bin/bash

# SVHN Dataset Download Script
# The SVHN dataset will be automatically downloaded by torchvision when needed.
# This script is mainly for consistency and to create the directory structure.

echo "Setting up SVHN dataset directory..."

# Create the data directory for SVHN
mkdir -p ../data/svhn

echo "SVHN directory created at ../data/svhn"
echo "The dataset will be automatically downloaded when first accessed through torchvision."
echo "SVHN dataset info:"
echo "  - Format: 32x32 color images"
echo "  - Classes: 10 (digits 0-9)"
echo "  - Training samples: ~73,257"
echo "  - Test samples: ~26,032"
echo "  - Extra samples: ~531,131 (optional)"
echo "Setup completed!"
