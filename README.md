# Pneumonia Detection using Convolutional Neural Networks (CNN)

This project implements an end-to-end Convolutional Neural Network (CNN) to classify chest X-ray images as **Normal** or **Pneumonia** using deep learning techniques.

The model is built using TensorFlow/Keras and trained on a real-world medical imaging dataset. The project demonstrates the complete workflow from dataset loading to model evaluation.

---

##  Problem Statement
Pneumonia is a serious respiratory disease that requires timely diagnosis. Manual interpretation of chest X-rays can be time-consuming and requires expert radiologists. This project aims to automate pneumonia detection using Convolutional Neural Networks to assist in medical image analysis.

---

##  Approach
- Designed a deep CNN architecture using Conv2D, MaxPooling, Dropout, and Dense layers
- Applied image preprocessing and data augmentation to improve generalization
- Trained the model for binary classification using sigmoid activation and binary cross-entropy loss
- Evaluated model performance using accuracy, confusion matrix, precision, recall, and F1-score

---

##  Dataset
- Source: Kaggle – Chest X-Ray Images (Pneumonia)
- Classes: Normal, Pneumonia
- Image Type: Chest X-ray images
- Dataset is downloaded programmatically using the Kaggle API

---

##  Technologies Used
- Python
- TensorFlow & Keras
- Convolutional Neural Networks (CNN)
- Google Colab
- Kaggle API
- Matplotlib, NumPy, Scikit-learn

---

## How to Run
1. Open the notebook in Google Colab
2. Configure Kaggle API credentials
3. Run all cells sequentially to download the dataset, train the CNN model, and evaluate performance

> Note: Model training may take time depending on available GPU resources.

---

##  Results
The CNN model successfully learns discriminative features from chest X-ray images and achieves strong classification performance, demonstrating the effectiveness of CNNs in medical image analysis.

---

##  Conclusion
This project highlights the practical application of Convolutional Neural Networks in healthcare-related computer vision tasks and showcases an end-to-end deep learning workflow suitable for real-world scenarios.

---
