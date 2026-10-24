# Brain Tumor MRI Classifier

An end-to-end Flask application that classifies brain MRI images into **glioma**, **meningioma**, **pituitary tumor**, or **no tumor**. It uses handcrafted computer vision features and a Support Vector Machine (SVM), rather than a deep learning model.

> **Medical disclaimer:** This project is an educational demonstration, not a medical device. Its predictions are not a diagnosis and should not be used to make treatment decisions. Consult a qualified medical professional.

## Live demo

[Open the Brain Tumor Classifier](https://brain-tumor-classification-kf5a.onrender.com)

## How it works

1. The uploaded image is resized to 128 × 128 pixels and converted to grayscale.
2. The app extracts and concatenates three feature groups:
   - **HOG (Histogram of Oriented Gradients):** edge and shape information.
   - **LBP (Local Binary Patterns):** local texture information.
   - **GLCM (Gray-Level Co-occurrence Matrix):** contrast, dissimilarity, homogeneity, energy, correlation, and ASM.
3. A `StandardScaler` transforms the 8,124-element feature vector.
4. An SVM predicts the class and returns a confidence score.

## Features

- Upload or drag and drop PNG, JPG/JPEG, BMP, or TIFF MRI images.
- Preview the uploaded image and view the predicted class and confidence.
- Responsive dark interface with an inference loading animation.
- Upload limit of 8 MB and image dimension limit of 20 megapixels.
- Lightweight health endpoint at `/healthz`.
- Model artifacts load on the first prediction and are cached for later requests.

## Technology

- **Machine learning:** Python, scikit-learn, scikit-image, OpenCV, NumPy, Joblib
- **Web app:** Flask, HTML, CSS, JavaScript
- **Deployment:** Render, Gunicorn, Git LFS

## Project layout

```text
.
├── app.py
├── requirements.txt
├── requirements-training.txt
├── ml.ipynb
├── Best_BrainTumor_Model.pkl   # downloaded on first prediction if absent
├── Scaler.pkl                  # downloaded on first prediction if absent
├── LabelEncoder.pkl            # downloaded on first prediction if absent
├── Model_Comparison.csv
├── Predictions.csv
├── templates/
│   └── index.html
├── BrainTumorDataset/
│   ├── Training/
│   └── Testing/
├── render.yaml
└── DEPLOYMENT.md
