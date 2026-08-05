import logging
import os
from pathlib import Path
import tempfile
from threading import Lock
from uuid import uuid4

import cv2
import joblib
import numpy as np
import requests
from flask import Flask, jsonify, render_template, request, send_from_directory, url_for
from skimage.feature import graycomatrix, graycoprops, hog, local_binary_pattern
from werkzeug.utils import secure_filename

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"

MODEL_FILES = {
    "model": BASE_DIR / "Best_BrainTumor_Model.pkl",
    "scaler": BASE_DIR / "Scaler.pkl",
    "encoder": BASE_DIR / "LabelEncoder.pkl",
}

BASE_URL = "https://huggingface.co/poojasank/BrainTumorModel/resolve/main"
logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "tif", "tiff"}
EXPECTED_FEATURE_COUNT = 8124
_ARTIFACT_LOCK = Lock()
_LOADED_ARTIFACTS = None

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024
app.config["MAX_IMAGE_PIXELS"] = 20_000_000


def download_model_files():
    """Download missing model files atomically from the configured Hugging Face repo."""
    for name, path in MODEL_FILES.items():
        if path.exists():
            continue

        url = f"{BASE_URL}/{path.name}"
        logger.info("Downloading missing model artifact %s", path.name)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = None
        try:
            with requests.get(url, timeout=(15, 300), stream=True) as response:
                response.raise_for_status()
                with tempfile.NamedTemporaryFile(
                    mode="wb", dir=path.parent, prefix=f".{path.name}.", delete=False
                ) as temporary_file:
                    temporary_path = Path(temporary_file.name)
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            temporary_file.write(chunk)
            if temporary_path.stat().st_size == 0:
                raise RuntimeError(f"Downloaded model artifact {path.name} is empty.")
            os.replace(temporary_path, path)
        finally:
            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink()


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def extract_features(image_path):

    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError("Uploaded image is invalid.")

    height, width = image.shape[:2]
    if height * width > app.config["MAX_IMAGE_PIXELS"]:
        raise ValueError("Image dimensions are too large. Please upload an image under 20 megapixels.")

    gray = cv2.cvtColor(
        cv2.resize(image, (128, 128)),
        cv2.COLOR_BGR2GRAY
    )

    hog_features = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        feature_vector=True,
    )

    radius = 2
    n_points = 8 * radius

    lbp = local_binary_pattern(
        gray,
        n_points,
        radius,
        method="uniform",
    )

    lbp_features, _ = np.histogram(
        lbp.ravel(),
        bins=np.arange(0, n_points + 3),
        range=(0, n_points + 2),
    )

    lbp_features = lbp_features.astype(float)
    lbp_features /= (lbp_features.sum() + 1e-7)

    glcm = graycomatrix(
        gray,
        distances=[1],
        angles=[0],
        levels=256,
        symmetric=True,
        normed=True,
    )

    glcm_features = np.array([
        graycoprops(glcm, "contrast")[0, 0],
        graycoprops(glcm, "dissimilarity")[0, 0],
        graycoprops(glcm, "homogeneity")[0, 0],
        graycoprops(glcm, "energy")[0, 0],
        graycoprops(glcm, "correlation")[0, 0],
        graycoprops(glcm, "ASM")[0, 0],
    ])

    return np.concatenate(
        [hog_features, lbp_features, glcm_features]
    ).reshape(1, -1)


def load_artifacts():
    global _LOADED_ARTIFACTS
    if _LOADED_ARTIFACTS is not None:
        return _LOADED_ARTIFACTS

    with _ARTIFACT_LOCK:
        if _LOADED_ARTIFACTS is not None:
            return _LOADED_ARTIFACTS

        download_model_files()
        model = joblib.load(MODEL_FILES["model"])
        scaler = joblib.load(MODEL_FILES["scaler"])
        encoder = joblib.load(MODEL_FILES["encoder"])

        scaler_features = getattr(scaler, "n_features_in_", None)
        model_features = getattr(model, "n_features_in_", scaler_features)
        if scaler_features != EXPECTED_FEATURE_COUNT or model_features != scaler_features:
            raise RuntimeError("The model artifacts do not match the app's 8,124-feature extractor.")
        if not hasattr(model, "predict_proba"):
            raise RuntimeError("The configured model does not provide prediction probabilities.")

        _LOADED_ARTIFACTS = (model, scaler, encoder)
        return _LOADED_ARTIFACTS


@app.route("/", methods=["GET", "POST"])
def index():

    result = None
    error = None
    image_url = None

    if request.method == "POST":

        upload = request.files.get("image")

        if not upload or upload.filename == "":
            error = "Please choose an MRI image."

        elif not allowed_file(upload.filename):
            error = "Only PNG, JPG, JPEG, BMP, TIFF images are allowed."

        else:

            image_path = None
            try:
                UPLOAD_DIR.mkdir(exist_ok=True)
                filename = f"{uuid4().hex}_{secure_filename(upload.filename)}"
                image_path = UPLOAD_DIR / filename
                upload.save(image_path)

                # Decode before loading the large model so malformed uploads fail quickly.
                image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                if image is None:
                    raise ValueError("The selected file is not a valid image.")
                if image.shape[0] * image.shape[1] > app.config["MAX_IMAGE_PIXELS"]:
                    raise ValueError("Image dimensions are too large. Please upload an image under 20 megapixels.")

                model, scaler, encoder = load_artifacts()
                features = extract_features(image_path)
                features = scaler.transform(features)
                prediction = model.predict(features)
                label = encoder.inverse_transform(prediction)[0]
                confidence = float(np.max(model.predict_proba(features)) * 100)
                if not np.isfinite(confidence):
                    raise RuntimeError("The model returned an invalid confidence score.")

                result = {
                    "label": str(label),
                    "confidence": f"{confidence:.2f}"
                }
                image_url = url_for("uploaded_file", filename=filename)
            except ValueError as exc:
                if image_path is not None:
                    image_path.unlink(missing_ok=True)
                error = str(exc)
            except Exception:
                if image_path is not None:
                    image_path.unlink(missing_ok=True)
                logger.exception("Prediction failed")
                error = "The prediction could not be completed. Check that model files are available and try again."

    return render_template(
        "index.html",
        result=result,
        error=error,
        image_url=image_url,
    )


@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_DIR, filename)


@app.get("/healthz")
def healthz():
    """Lightweight process health check; model loading stays lazy until a prediction."""
    return jsonify(status="ok")


@app.errorhandler(413)
def too_large(_error):
    return render_template(
        "index.html",
        error="Image size must be less than 8 MB."
    ), 413


if __name__ == "__main__":
    app.run(debug=os.getenv("FLASK_DEBUG", "0").lower() in {"1", "true", "yes"})
