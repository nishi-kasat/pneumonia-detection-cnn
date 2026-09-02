# Explainable Pneumonia Detection using CNN

This project implements an end-to-end Convolutional Neural Network (CNN) to classify chest X-ray images as **Normal** or **Pneumonia**, with **Grad-CAM explainability** to visualize image regions that influence the model's prediction.

## Problem Statement

Pneumonia detection from chest X-rays is a challenging medical imaging task. A high-performing classifier is more useful when its predictions can also be inspected. This project therefore combines binary CNN classification with visual explanations of the model's pneumonia score.

## Approach

- Downloaded the Chest X-Ray Images (Pneumonia) dataset using the Kaggle API
- Preprocessed and augmented chest X-ray images
- Built a CNN using Conv2D, MaxPooling, Dropout, Flatten, and Dense layers
- Trained the model for binary classification using sigmoid activation and binary cross-entropy
- Evaluated the classifier using accuracy, confusion matrix, precision, recall, and F1-score
- Added **Grad-CAM** to identify spatial regions contributing to the pneumonia prediction
- Visualized the original X-ray, Grad-CAM heatmap, and heatmap overlay

## Explainability

**Grad-CAM (Gradient-weighted Class Activation Mapping)** uses gradients flowing into the final convolutional feature maps to produce a coarse localization map for the target prediction.

For this project, the explanation pipeline is:

`Chest X-ray → CNN → Pneumonia probability → Grad-CAM → Heatmap → Overlay`

The explanation is intended to make model behavior easier to inspect and debug. It is not a clinical diagnosis and should not be treated as evidence that a highlighted region is medically causal.

## Project Structure

```text
.
├── Pneumonia_Detection_using_CNN.ipynb
├── explainability/
│   ├── gradcam.py
│   └── README.md
└── README.md
```

## Technologies Used

- Python
- TensorFlow / Keras
- Convolutional Neural Networks
- Grad-CAM / Explainable AI (XAI)
- NumPy
- OpenCV
- Matplotlib
- Scikit-learn
- Google Colab
- Kaggle API

## How to Run

1. Open the notebook in Google Colab.
2. Configure Kaggle API credentials.
3. Run the existing dataset preparation, training, and evaluation cells.
4. Import `make_gradcam_heatmap` and `overlay_gradcam` from `explainability/gradcam.py`.
5. Run the Grad-CAM example in `explainability/README.md` on a test X-ray.

## Future Work

- Compare Grad-CAM with LIME or SHAP
- Evaluate explanation stability across correctly and incorrectly classified images
- Compare explanations across Normal and Pneumonia samples
- Add a lightweight inference interface for uploading an X-ray and viewing the prediction with its explanation

> **Disclaimer:** This is an educational machine-learning project. Model predictions and Grad-CAM visualizations are not medical advice or a substitute for professional radiological assessment.
