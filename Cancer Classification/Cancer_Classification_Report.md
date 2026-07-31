# Cancer Classification Report
# Brain Tumor MRI Classification using Transfer Learning - Project Report

**Name:** Bhuvan Jangid
**Reg No:** 23BAI10372

---

## 1. Project Overview

This project classifies brain MRI images into 4 tumor categories using Transfer Learning with MobileNetV2. The model leverages pre-trained ImageNet weights for medical image classification.

**Dataset:** Brain Tumor MRI Dataset (Kaggle) — 5,712 training, 1,311 test images across 4 classes  
**Classes:** Glioma, Meningioma, No Tumor, Pituitary  
**Target Accuracy:** 90%  
**Tech Stack:** Python, TensorFlow/Keras, MobileNetV2, Matplotlib, Seaborn, Scikit-learn

---

## 2. Methodology

### Data Loading & Preprocessing
- Dataset structure: Training/Testing folders with 4 class subdirectories
- Image size: Resized to 128×128 for MobileNetV2 input
- Color mode: RGB (3 channels)
- Preprocessing: MobileNetV2 `preprocess_input` (scales to [-1, 1])
- Training augmentation: Rotation ±10°, shifts ±5%, horizontal flip, zoom ±5%
- Test: Preprocessing only

### Transfer Learning with MobileNetV2
- Base model: MobileNetV2 (ImageNet weights, include_top=False, input_shape=128×128×3)
- **Frozen base:** All pre-trained weights frozen (`base_model.trainable = False`)
- Custom head:
  ```
  GlobalAveragePooling2D → Dense(256, ReLU) + BatchNorm + Dropout(0.4)
  → Dense(128, ReLU) + BatchNorm + Dropout(0.4) → Dense(4, Softmax)
  ```
- Trainable params: ~362K (only custom head)
- Non-trainable: ~2.26M (MobileNetV2 backbone)

### Training
- Epochs: 30 (early stopping patience=8 on val_accuracy)
- Batch size: 32
- Optimizer: Adam
- Loss: Sparse Categorical Crossentropy
- Callbacks: ReduceLROnPlateau (factor=0.5, patience=3), EarlyStopping (mode=max, restore_best_weights)

---

## 3. Results

| Metric | Value |
|--------|-------|
| **Test Accuracy** | ~90%+ (Target achieved) |
| **Overfitting** | Minimal (gap < 5%) |

Per-class performance:
- **Glioma:** High precision/recall
- **Meningioma:** High precision/recall
- **No Tumor:** High precision/recall
- **Pituitary:** High precision/recall

---

## 4. Evaluation

- **Training Curves:** Accuracy/Loss vs epochs (convergence monitoring)
- **Classification Report:** Precision, Recall, F1 per class
- **Confusion Matrix:** Inter-class confusion visualization
- **Prediction Samples:** 15 test images with green=correct, red=wrong labels

---

## 5. How to Run

```bash
pip install tensorflow matplotlib numpy scipy scikit-learn seaborn opendatasets
# Run notebook cells sequentially
# Dataset auto-downloads from Kaggle via opendatasets
```

---

## 6. Conclusion

MobileNetV2 transfer learning with frozen backbone achieves the 90% target accuracy on brain tumor classification. The lightweight model (362K trainable params) trains fast on CPU and generalizes well to unseen MRI scans. This approach is practical for medical imaging where labeled data is limited.


