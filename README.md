# Deepfake Image Detection using CNNs and Vision-Language Models

A deep learning project focused on detecting AI-generated facial images using transfer learning and multimodal computer vision techniques. This project compares traditional CNN-based image classification against transformer-based vision-language models for real vs fake face detection.

---

## Overview

This project uses the Kaggle **140k Real and Fake Faces** dataset to classify facial images as either:

- Real
- AI-generated / Fake

The system was built using:
- PyTorch
- Torchvision
- HuggingFace Transformers
- Transfer Learning
- ResNet18
- Vision Transformers (ViT)

The project explores how traditional convolutional neural networks compare against modern transformer-based multimodal models in deepfake image recognition tasks.

---

## Features

- Transfer learning using pretrained ResNet18
- Deepfake image classification pipeline
- Data augmentation and normalization
- Training and validation loss visualization
- Confusion matrix evaluation
- Classification report generation
- Sample prediction visualization
- Vision-language model comparison using HuggingFace transformers
- GPU acceleration with CUDA support

---

## Dataset

Dataset used:

**140k Real and Fake Faces**  
https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces

Dataset contains:
- Real human face images
- AI-generated synthetic face images

---

## Technologies Used

- Python
- PyTorch
- Torchvision
- HuggingFace Transformers
- Scikit-learn
- Matplotlib
- Google Colab

---

## Model Architecture

### CNN Model
- Pretrained ResNet18 backbone
- Frozen feature extraction layers
- Custom fully connected classification head
- Dropout regularization
- Adam optimizer
- Learning rate scheduler

### Vision-Language Model
- Google Vision Transformer (ViT)
- Transformer-based image classification pipeline
- HuggingFace Transformers integration

---

## Training Configuration

| Parameter | Value |
|---|---|
| Backbone | ResNet18 |
| Image Size | 128×128 |
| Batch Size | 32 |
| Optimizer | Adam |
| Learning Rate | 0.0005 |
| Epochs | 5 |
| Loss Function | CrossEntropyLoss |

---

## Results

### CNN Model Performance

| Metric | Score |
|---|---|
| Accuracy | 70.9% |
| Precision | 66.8% |
| Recall | 83.4% |
| F1 Score | 74.1% |

The CNN model achieved strong recall performance for detecting real images while maintaining balanced overall classification performance. Training and validation loss curves demonstrated stable convergence across epochs.

---

## Evaluation Visualizations

The project includes:
- Training vs validation loss curves
- Confusion matrix analysis
- Sample prediction visualizations
- Real vs fake image comparisons

The confusion matrix showed that the model was more effective at identifying real images than fake images, highlighting the challenges of synthetic face detection.

---

## Vision-Language Model Comparison

The project also experimented with transformer-based vision-language models using HuggingFace pipelines. A Vision Transformer (ViT) model was used to evaluate how transformer architectures interpret generated facial images compared to CNN-based feature extraction approaches.

Harini Kuchibhotla

GitHub: https://github.com/harinijk
