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
```

The model artifacts are hosted in the configured [Hugging Face model repository](https://huggingface.co/poojasank/BrainTumorModel). If they are missing locally, the app downloads them on the first prediction; internet access is therefore needed for that first prediction. See [DEPLOYMENT.md](DEPLOYMENT.md) for Render and Git LFS instructions.

## Run locally

### 1. Clone the repository

```bash
git clone https://github.com/jishnu395/Brain_Tumor_Model.git
cd Brain_Tumor_Model
```

### 2. Create an environment and install dependencies

Windows PowerShell:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 3. Start the app

```bash
python app.py
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000). The health check is available at [http://127.0.0.1:5000/healthz](http://127.0.0.1:5000/healthz).

## Train the model

The training notebook is `ml.ipynb`. It expects the dataset under `BrainTumorDataset/Training` and `BrainTumorDataset/Testing`, with one subfolder per tumor class in each directory. Install the training dependencies from `requirements-training.txt`, then run the notebook cells in order. Training and cross-validation may take considerable time and disk space.

## Deployment

This repository includes a Render Blueprint (`render.yaml`). For deployment setup, artifact handling, and Git LFS commands, see [DEPLOYMENT.md](DEPLOYMENT.md).

## Author

**Jishnu V** — Computer Science Engineering student

- GitHub: [Poojasankannavar](https://github.com/Poojasankannavar)
- LinkedIn: [Pooja Sankannavar](https://www.linkedin.com/in/pooja-sankannavar-8751a0318/)

## Potential next steps

- Compare the SVM with a CNN model.
- Generate downloadable PDF reports.
- Add Grad-CAM visualization and batch prediction.
- Document a REST API and add Docker/CI support.
- Explore patient history management with appropriate privacy safeguards.
