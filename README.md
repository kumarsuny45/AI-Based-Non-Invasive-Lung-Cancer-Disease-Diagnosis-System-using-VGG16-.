# Lung Cancer Classification Using VGG16

A deep learning project for **lung cancer image classification** using **TensorFlow/Keras** and transfer learning with the **VGG16** model pre-trained on ImageNet.

## Overview

The project classifies lung images into different categories using:

* **VGG16** for feature extraction
* **Global Average Pooling**
* **Dense layer (512 neurons)**
* **Dropout (0.5)** to reduce overfitting
* **Softmax** output for multi-class classification

The model is first trained with the VGG16 layers frozen and then **fine-tuned** by unfreezing the deeper layers.

## Key Features

* Image resizing to **150 × 150**
* Batch size: **32**
* Data augmentation:

  * Random brightness
  * Random contrast
  * Horizontal & vertical flipping
* Transfer learning with ImageNet weights
* Fine-tuning with a low learning rate
* Model saved in `.keras` format

## Dataset Structure

```text
abc/
├── class_1/
├── class_2/
├── class_3/
└── ...
```

Each folder represents one class. Class names are detected automatically from the directory structure.

## 🚀 Installation

```bash
pip install tensorflow
```

## ▶️ Run the Project

Place the dataset in the `abc` directory and run:

```bash
python train.py
```

The trained model will be saved as:

```text
vgg16_lung_cancer_model.keras
```

## 📊 Evaluation

The model evaluates:

* Training Accuracy
* Validation Accuracy
* Test Accuracy
* Training and Validation Loss

> **Note:** In the current implementation, the validation and test datasets are loaded from the same validation split. For a proper 70/15/15 evaluation, an independent test set should be created.

## 🔮 Future Improvements

* Use a proper independent test dataset
* Add confusion matrix and F1-score
* Apply VGG16 `preprocess_input`
* Experiment with ResNet/EfficientNet
* Add Grad-CAM for model interpretability

## Disclaimer

This project is for **educational and research purposes only**. It is not intended to provide medical diagnosis or replace professional medical advice.



If you find this project useful, feel free to ⭐ the repository.
