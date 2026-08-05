# Brain Tumor MRI Classifier

A Flask application that classifies an uploaded MRI image into one of the four classes used by the trained model: glioma, meningioma, no tumor, or pituitary. This is an educational demonstration and is not a medical diagnosis.

## Run locally in VS Code

1. Open this folder in VS Code and open **Terminal → New Terminal**.
2. Create and activate a virtual environment (PowerShell):

   ```powershell
   py -3 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

   If PowerShell blocks activation, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in that terminal, then activate again. Alternatively, choose `.venv\Scripts\python.exe` as the VS Code interpreter and run commands with it directly.

3. Install app dependencies:

   ```powershell
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```

4. Press **F5** and choose **Run Brain Tumor Classifier**, or start it in the terminal:

   ```powershell
   python app.py
   ```

5. Open <http://127.0.0.1:5000>. Upload a PNG, JPG/JPEG, BMP, or TIFF image up to 8 MB. The app also accepts drag and drop. The status endpoint is <http://127.0.0.1:5000/healthz>.

The exported files (`Best_BrainTumor_Model.pkl`, `Scaler.pkl`, and `LabelEncoder.pkl`) are used when present. If any are missing, the app downloads the missing artifact from the configured Hugging Face repository on the first prediction, so internet access is needed. Model files are large; keep all three beside `app.py` for offline startup and prediction.

## Train from the notebook

`ml.ipynb` expects the dataset folders at `BrainTumorDataset/Training` and `BrainTumorDataset/Testing`, each containing one folder per class. Install the additional packages in `requirements-training.txt`, select the same `.venv` interpreter in VS Code, then open the notebook and run its cells in order. Training and cross-validation can take substantial time and disk space.

## Deployment

See [DEPLOYMENT.md](DEPLOYMENT.md) for Render and Git LFS setup.
