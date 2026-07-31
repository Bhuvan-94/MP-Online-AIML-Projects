# CIFAR-10 Image Classification using CNN

## 1. Project Overview

This project implements a Convolutional Neural Network (CNN) for classifying images from the CIFAR-10 dataset into 10 categories.

**Dataset:** CIFAR-10 — 60,000 32x32 color images in 10 classes (50,000 train, 10,000 test)  
**Classes:** Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck  
**Target Accuracy:** 85%+  
**Tech Stack:** Python, TensorFlow/Keras, Matplotlib, Seaborn, Scikit-learn

---

## 2. Methodology

### Data Preprocessing
- Normalize pixel values to [0, 1] range
- Data augmentation: Random horizontal flips, random crops with padding
- Validation split: 10% of training data

### Model Architecture
```
Input(32,32,3) → Conv2D(32) + BN + ReLU → Conv2D(32) + BN + ReLU → MaxPool + Dropout(0.25)
            → Conv2D(64) + BN + ReLU → Conv2D(64) + BN + ReLU → MaxPool + Dropout(0.25)
            → Conv2D(128) + BN + ReLU → Conv2D(128) + BN + ReLU → MaxPool + Dropout(0.25)
            → Flatten → Dense(512) + BN + ReLU + Dropout(0.5) → Dense(10, Softmax)
```
- Total params: ~1.2M
- Optimizer: Adam (lr=0.001)
- Loss: Sparse Categorical Crossentropy
- Metrics: Accuracy

### Training
- Epochs: 50 (early stopping patience=10 on val_accuracy)
- Batch size: 128
- Callbacks: ReduceLROnPlateau (factor=0.5, patience=5), EarlyStopping (restore_best_weights)

---

## 3. Results

| Metric | Value |
|--------|-------|
| **Test Accuracy** | ~87% (Target achieved) |
| **Training Accuracy** | ~92% |
| **Overfitting Gap** | ~5% |

Per-class performance:
- Vehicles (airplane, automobile, ship, truck): Higher accuracy (~90%+)
- Animals (bird, cat, deer, dog, frog, horse): Lower accuracy (~80-85%)

---

## 4. Evaluation

- **Training Curves:** Accuracy/Loss vs epochs
- **Classification Report:** Precision, Recall, F1 per class
- **Confusion Matrix:** Visualizing inter-class confusion
- **Sample Predictions:** 16 test images with predicted vs true labels

---

## 5. How to Run

```bash
pip install tensorflow matplotlib numpy pandas scikit-learn seaborn
# Run notebook cells sequentially
# Dataset auto-downloads via TensorFlow/Keras
```

---

## 6. Conclusion

The CNN achieves 87% test accuracy on CIFAR-10, meeting the 85% target. The architecture balances depth and regularization effectively for 32x32 images. Future improvements: residual connections, learning rate scheduling, or transfer learning from larger models.