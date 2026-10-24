# 🧠 Brain Tumor MRI Classifier

An end-to-end Machine Learning web application that classifies brain MRI images into four categories: **Glioma, Meningioma, Pituitary Tumor, and No Tumor**. The application uses handcrafted computer vision features and a Support Vector Machine (SVM) classifier instead of a deep learning model.

<p align="center">
  <a href="https://brain-tumor-classification-kf5a.onrender.com">
    <img src="https://img.shields.io/badge/Live%20Demo-Open%20Application-7C3AED?style=for-the-badge&logo=render&logoColor=white" alt="Live Demo">
  </a>
  <img src="https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Flask-Web%20Application-000000?style=for-the-badge&logo=flask&logoColor=white" alt="Flask">
  <img src="https://img.shields.io/badge/Machine%20Learning-SVM-orange?style=for-the-badge" alt="Machine Learning">
</p>

## 🌐 Live Demo

**Try the application:** [Brain Tumor MRI Classifier](https://brain-tumor-classification-kf5a.onrender.com)

Upload a brain MRI image to see the model's predicted class and its associated confidence score.

## 📌 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Tumor Categories](#-tumor-categories)
- [How It Works](#-how-it-works)
- [Technology Stack](#-technology-stack)
- [Model Architecture](#-model-architecture)
- [Project Structure](#-project-structure)
- [Installation and Setup](#-installation-and-setup)
- [Running the Application](#-running-the-application)
- [API Health Check](#-api-health-check)
- [Deployment](#-deployment)
- [Future Improvements](#-future-improvements)
- [Medical Disclaimer](#-medical-disclaimer)
- [Author](#-author)

## 📖 Overview

The Brain Tumor MRI Classifier is a machine learning project designed to demonstrate image processing, feature extraction, classification, and web application deployment.

The application processes an uploaded MRI image, extracts handcrafted image features, scales the resulting feature vector, and uses a trained Support Vector Machine to predict one of four classes.

Unlike CNN-based approaches, this project uses traditional computer vision techniques:

- Histogram of Oriented Gradients (HOG)
- Local Binary Patterns (LBP)
- Gray-Level Co-occurrence Matrix (GLCM)
- Support Vector Machine (SVM)

The model is integrated into a Flask web application with a responsive interface for image upload and prediction.

## ✨ Features

- 🧠 Classification of four brain MRI image categories.
- 📤 Upload or drag and drop MRI images.
- 🖼️ Preview the uploaded image before prediction.
- 📊 Display the predicted class and confidence score.
- ⚙️ Extract HOG, LBP, and GLCM features.
- 🤖 Use a trained SVM classification model.
- 🎨 Responsive dark-themed user interface.
- ⏳ Loading animation during inference.
- 🔒 Validate image file types and upload sizes.
- 📏 Enforce an 8 MB upload limit and a 20-megapixel image dimension limit.
- 🚀 Deploy the application using Render and Gunicorn.
- ❤️ Provide a lightweight health-check endpoint at `/healthz`.
- 📦 Load and cache model artifacts when needed.

## 🎯 Tumor Categories

The classifier predicts one of the following categories:

| Class | Description |
|---|---|
| Glioma | A category of tumors originating from glial cells in the brain or central nervous system. |
| Meningioma | A tumor arising from the meninges, the membranes surrounding the brain and spinal cord. |
| Pituitary Tumor | A tumor originating in the pituitary gland. |
| No Tumor | The image is classified as belonging to the dataset's no-tumor category. |

**Note:** These are model output categories, not confirmed medical findings.

## ⚙️ How It Works

The application follows this image-classification pipeline:

1. **Image Upload:** The user uploads an MRI image through the web interface.
2. **Preprocessing:** The image is resized to `128 × 128` pixels and converted to grayscale.
3. **HOG Feature Extraction:** Extracts edge and gradient-orientation information.
4. **LBP Feature Extraction:** Captures local texture patterns.
5. **GLCM Feature Extraction:** Calculates texture statistics, including contrast, dissimilarity, homogeneity, energy, correlation, and angular second moment (ASM).
6. **Feature Concatenation:** Combines the extracted features into an 8,124-element feature vector.
7. **Feature Scaling:** Uses the trained `StandardScaler` to transform the features.
8. **Classification:** The trained SVM predicts the class.
9. **Result Display:** The application displays the predicted category and a confidence score.

### Workflow

```text
        Brain MRI Image
               |
               v
       Image Preprocessing
         128 x 128
          Grayscale
               |
               v
       Feature Extraction
        /      |      \
      HOG      LBP     GLCM
        \      |      /
               |
               v
      Feature Concatenation
        8,124 Features
               |
               v
       StandardScaler
               |
               v
         SVM Classifier
               |
               v
       Predicted Category
               |
               v
      Result and Confidence
```

## 🛠️ Technology Stack

### Machine Learning and Image Processing

- **Python** — Core programming language.
- **Scikit-learn** — SVM classifier, feature scaling, and machine learning utilities.
- **Scikit-image** — HOG and LBP feature extraction.
- **OpenCV** — Image preprocessing.
- **NumPy** — Numerical computations.
- **Joblib** — Saving and loading model artifacts.

### Web Development

- **Flask** — Python web framework.
- **HTML5** — Application structure.
- **CSS3** — Responsive interface and styling.
- **JavaScript** — Upload interactions and prediction interface.

### Deployment and Version Control

- **Render** — Web application hosting.
- **Gunicorn** — Production WSGI server.
- **Git and GitHub** — Version control and source code management.
- **Git LFS** — Storage of large model files when required.

## 🧪 Model Architecture

This project uses traditional machine learning rather than a deep neural network.

| Component | Purpose |
|---|---|
| Image size | 128 × 128 pixels |
| Image format for processing | Grayscale |
| HOG | Gradient and shape features |
| LBP | Local texture features |
| GLCM | Statistical texture features |
| Feature vector | 8,124 elements |
| Feature scaling | StandardScaler |
| Classification algorithm | Support Vector Machine |
| Output | Four-class prediction |

### Model Artifacts

The application uses the following serialized files:

- `Best_BrainTumor_Model.pkl` — Trained SVM model.
- `Scaler.pkl` — Fitted feature scaler.
- `LabelEncoder.pkl` — Class-label encoder.

If an artifact is absent, the application is configured to download it on the first prediction, provided the required download configuration and source are available.

**Important:** The scaler, feature extraction configuration, and classifier must match the versions used during training.

## 📂 Project Structure

```text
BrainTumorPredict/
│
├── app.py
│
├── requirements.txt
├── requirements-training.txt
│
├── ml.ipynb
│
├── Best_BrainTumor_Model.pkl
├── Scaler.pkl
├── LabelEncoder.pkl
│
├── Model_Comparison.csv
├── Predictions.csv
│
├── templates/
│   └── index.html
│
├── BrainTumorDataset/
│   ├── Training/
│   └── Testing/
│
├── render.yaml
├── DEPLOYMENT.md
└── README.md
```

**Note:** The `.pkl` files may be downloaded when required rather than committed to the repository. Dataset files and large model artifacts may be managed separately using appropriate storage or Git LFS.

## 💻 Installation and Setup

### Prerequisites

Install the following tools before running the project:

- Python 3.x compatible with the pinned project dependencies.
- Git.
- A code editor such as Visual Studio Code or PyCharm.

### 1. Clone the Repository

```bash
git clone https://github.com/Poojasankannavar/BrainTumorPredict.git
```

Navigate to the project directory:

```bash
cd BrainTumorPredict
```

### 2. Create a Virtual Environment

**Windows:**

```bash
python -m venv .venv
.venv\Scripts\activate
```

**macOS/Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For model training and notebook experiments, install the additional training dependencies if required:

```bash
pip install -r requirements-training.txt
```

### 4. Configure Model Artifacts

Ensure that the application can access the following files or their configured download sources:

```text
Best_BrainTumor_Model.pkl
Scaler.pkl
LabelEncoder.pkl
```

If the model artifacts are downloaded automatically, verify that the download URLs and hosting permissions are configured correctly.

## ▶️ Running the Application

Start the Flask application:

```bash
python app.py
```

Open the local application in your browser at:

```text
http://127.0.0.1:5000
```

Upload a supported MRI image and run the prediction.

For production hosting, use a WSGI server such as Gunicorn rather than Flask's development server.

Example:

```bash
gunicorn app:app
```

This command assumes that `app.py` exposes the Flask application through a variable named `app`.

## ❤️ API Health Check

The application provides a health-check endpoint:

```text
/healthz
```

When running locally, test it at:

```text
http://127.0.0.1:5000/healthz
```

The deployed endpoint is:

[https://brain-tumor-classification-kf5a.onrender.com/healthz](https://brain-tumor-classification-kf5a.onrender.com/healthz)

The endpoint can be used to check application availability. A successful health response does not necessarily confirm that all model artifacts are downloaded or that inference is working.

## 🚀 Deployment

The application is deployed on Render.

**Live application:** [Brain Tumor MRI Classifier](https://brain-tumor-classification-kf5a.onrender.com)

The project uses:

- Flask for request handling.
- Gunicorn as the production application server.
- Render for hosting.
- GitHub for source code.
- Git LFS or external storage for large artifacts, where configured.

Refer to [`DEPLOYMENT.md`](DEPLOYMENT.md) for project-specific deployment instructions.

### Deployment Checklist

- [ ] Verify `requirements.txt`.
- [ ] Configure the Render build and start commands.
- [ ] Confirm that all required model artifacts can be downloaded.
- [ ] Check the `/healthz` endpoint.
- [ ] Test predictions using supported image formats.
- [ ] Verify the upload-size and image-dimension limits.
- [ ] Confirm that the deployed application handles invalid files gracefully.

## 🔮 Future Improvements

Potential improvements include:

- Evaluate the classifier on a separate, representative test dataset.
- Report per-class precision, recall, F1-score, and a confusion matrix.
- Add automated unit and integration tests.
- Improve handling of corrupted, unsupported, and non-MRI images.
- Add structured logging and clearer error messages.
- Improve accessibility and mobile responsiveness.
- Compare traditional machine learning with CNN-based image classification.
- Add reproducible training scripts and model-version tracking.
- Improve deployment monitoring and model artifact management.

## ⚠️ Medical Disclaimer

This project is intended **for educational and research demonstration purposes only**. It is not a certified medical device and has not been established as suitable for clinical diagnosis.

Predictions and confidence scores may be incorrect or misleading. They must not be used to diagnose a condition, make treatment decisions, or replace professional medical evaluation. Always consult a qualified healthcare professional for medical concerns.

## 👩‍💻 Author

**Pooja Sankannavar**

MCA Student | Full-Stack Developer | Software Development and AI Enthusiast

- **GitHub:** [@Poojasankannavar](https://github.com/Poojasankannavar)
- **LinkedIn:** [Pooja Sankannavar](https://www.linkedin.com/in/pooja-sankannavar-8751a0318/)
- **Live Project:** [Brain Tumor MRI Classifier](https://brain-tumor-classification-kf5a.onrender.com)

---

⭐ If you find this project interesting, consider giving the repository a star!

*Built as a learning project exploring computer vision, traditional machine learning, Flask development, and deployment.*
