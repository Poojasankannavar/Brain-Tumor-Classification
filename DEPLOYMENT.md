# Deploying the trained model to Render

The three exported files are required for prediction:

- `Best_BrainTumor_Model.pkl`
- `Scaler.pkl`
- `LabelEncoder.pkl`

They are exported by cell 26 of `ml.ipynb`. The SVM model is about 256 MB, larger than GitHub's 100 MB normal Git limit. The `.gitattributes` file configures pickle files for Git LFS, while `.gitignore` keeps the local binaries out of ordinary commits. Install and initialize Git LFS before explicitly adding the model artifacts:

Before pushing, run these commands from the repository root:

```powershell
git lfs install
git add .gitattributes .gitignore app.py requirements.txt requirements-training.txt DEPLOYMENT.md README.md .vscode
git add -f Best_BrainTumor_Model.pkl Scaler.pkl LabelEncoder.pkl
git commit -m "Deploy trained brain tumor model with Git LFS"
git push origin main
```

Confirm `git lfs ls-files` lists all three artifacts before pushing. If the artifacts are not available in the checkout, the app downloads them from the Hugging Face repository at first prediction; ensure the Render service has outbound access and enough disk space. Keep the installed scikit-learn version compatible with the one used to serialize the model.

Create or update the Render service from `render.yaml`, then open its **Manual Deploy** menu and select **Clear build cache & deploy**. Verify the build output checks out the LFS objects; the service must contain all three files beside `app.py` before the first prediction. The Blueprint pins Python 3.13.2 to match the local artifact runtime and checks `/healthz`.

The service uses an ephemeral filesystem, so uploaded MRI images are only retained long enough to display the result. They are not permanent uploads.
