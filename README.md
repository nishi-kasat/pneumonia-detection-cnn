# Explainable Pneumonia Detection using CNN

An end-to-end **medical-imaging + Explainable AI (XAI)** project that classifies chest X-rays as **Normal** or **Pneumonia** and uses **Grad-CAM** to visualize image regions associated with the model's prediction.

> **Educational project:** predictions and explanations are not medical advice or a substitute for professional radiological assessment.

## Why this project is stronger than a standard CNN classifier

A classifier can report *what* it predicts without showing *where* the prediction is coming from. This project adds an interpretability layer so model behavior can be inspected alongside its prediction.

```text
Chest X-ray → Preprocessing / augmentation → CNN → Pneumonia probability
                                              ↓
                                           Grad-CAM
                                              ↓
                                  Heatmap + X-ray overlay
                                              ↓
                                  TP / TN / FP / FN analysis
```

## Key Features

- Binary chest X-ray classification with TensorFlow/Keras
- Data augmentation and normalized image preprocessing
- Confusion-matrix, precision, recall and F1 evaluation
- **Grad-CAM** explanations from the last convolutional layer
- Representative **true-positive, true-negative, false-positive and false-negative** analysis
- Explanation-stability check under a small brightness perturbation
- Reusable XAI utilities in `explainability/gradcam.py`
- Standalone analysis script in `xai_analysis.py`

## Dataset

The project uses the **Chest X-Ray Images (Pneumonia)** dataset by Paul Mooney on Kaggle. The notebook downloads the dataset at runtime rather than committing medical images to the repository.

## Grad-CAM

**Gradient-weighted Class Activation Mapping (Grad-CAM)** uses gradients of the target output with respect to convolutional feature maps to create a coarse spatial importance map.

For this binary classifier, the target is the positive **Pneumonia** output. The heatmap highlights regions that most influenced that score.

A Grad-CAM highlight is evidence of model sensitivity, **not proof of a medically causal lesion or pathology**. Explanations should be evaluated together with prediction errors and, ideally, expert annotations.

## XAI Error Analysis

`xai_analysis.py` provides a reproducible workflow:

1. Load the saved Keras model.
2. Load a bounded, deterministic subset of the test set.
3. Report precision, recall, F1 and ROC-AUC.
4. Save a confusion matrix.
5. Find representative TP/TN/FP/FN examples.
6. Generate Grad-CAM visualizations for each available category.
7. Apply a small brightness perturbation and compare heatmaps using cosine similarity.

Outputs are written to `outputs/xai/` and are not committed by default.

## Project Structure

```text
.
├── Pneumonia_Detection_using_CNN.ipynb   # training + evaluation notebook
├── explainability/
│   ├── gradcam.py                         # reusable Grad-CAM utilities
│   └── README.md                          # notebook integration guide
├── xai_analysis.py                        # XAI + error-analysis pipeline
├── README.md
└── .gitignore
```

## Tech Stack

**ML / DL:** Python, TensorFlow, Keras, CNNs, scikit-learn  
**XAI:** Grad-CAM, explainability/error analysis  
**Computer Vision:** OpenCV, NumPy, Matplotlib  
**Workflow:** Google Colab, Kaggle API, Git/GitHub

## Running the Project

### 1. Train the model

Open `Pneumonia_Detection_using_CNN.ipynb` in Google Colab and run the dataset preparation, preprocessing, training and evaluation cells.

**Never commit `kaggle.json`, API keys, tokens or other credentials.** Use Colab Secrets/environment variables or upload the credential only during a runtime session.

### 2. Run Grad-CAM

After loading the trained model:

```python
from explainability.gradcam import make_gradcam_heatmap, overlay_gradcam
```

See `explainability/README.md` for a complete visualization example.

### 3. Run reproducible XAI analysis

```bash
pip install tensorflow scikit-learn opencv-python matplotlib numpy
python xai_analysis.py \
  --model pneumonia_cnn_model.h5 \
  --data-dir /path/to/chest_xray \
  --output-dir outputs/xai
```

For Colab, use `/content/chest_xray` and the saved model path.

## Responsible XAI

This project treats explainability as a **debugging and model-inspection tool**, not as a claim of clinical validity. A stronger medical-AI evaluation would additionally require external validation, calibration, subgroup analysis, leakage checks, expert review, and clinically meaningful localization ground truth.

## Future Improvements

- Compare Grad-CAM against LIME and SHAP where technically appropriate
- Quantify explanation quality using expert/localization annotations
- Evaluate explanation stability under multiple controlled perturbations
- Add probability calibration and threshold analysis
- Add a lightweight inference UI for prediction + explanation
- Add automated tests and CI for the XAI utilities
