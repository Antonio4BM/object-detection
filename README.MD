# Object Detection

A machine learning project for underwater object detection using deep learning techniques.

## Overview

This repository contains code for training and implementing object detection models, with a focus on underwater object detection. The project combines Jupyter notebooks for exploration and analysis with Python modules for model implementation and utilities.

## Repository Structure

- **`underwater_object_detection.ipynb`** - Main Jupyter notebook containing the complete workflow for underwater object detection, including data exploration, model training, and evaluation
- **`model.py`** - Core model architecture and definitions
- **`dataset.py`** - Dataset handling and data loading utilities
- **`utils.py`** - Utility functions for training, evaluation, and data processing
- **`.gitignore`** - Git configuration file

## Key Components

### Model
The `model.py` module contains the neural network architecture and model definitions for object detection tasks.

### Dataset
The `dataset.py` module handles dataset preparation, loading, and preprocessing for training and evaluation.

### Utilities
The `utils.py` module provides helper functions for:
- Training pipelines
- Model evaluation
- Data augmentation and processing
- Visualization

## Getting Started

### Prerequisites
- Python 3.11+
- Jupyter Notebook
- PyTorch
- NumPy, Pandas, Matplotlib

### Installation

1. Clone the repository:
```bash
git clone https://github.com/Antonio4BM/object-detection.git
cd object-detection