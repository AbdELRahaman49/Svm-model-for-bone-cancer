# SVM Model for Bone Cancer Classification

A reproducible, classical-ML pipeline for binary bone-tumor classification (Benign vs Malignant) using an optimized, hand-crafted feature extractor and an SVM with calibrated probabilities. The pipeline includes conservative data augmentation, robust texture/shape statistics, PCA, SMOTE, tree-based feature selection, randomized hyperparameter search, calibration, and thorough reporting.

## Features

- Optimized feature extractor:
  - GLCM statistics at multiple distances/angles
  - Local Binary Patterns (two radii) with histogram normalization
  - Canny edge statistics
  - Intensity histogram (32 bins), entropy, robust percentiles
  - HOG (reduced complexity)
  - Internal feature cache for speed
- Data handling:
  - Grayscale normalization and resizing to 128×128
  - Albumentations with moderate probabilities for geometric/photometric transforms
  - Stratified train/test split
- Model pipeline:
  - StandardScaler → PCA (retain ≥ 80–95% variance; tuned) → SMOTE → ExtraTrees feature selection → SVC (RBF)
  - RandomizedSearchCV over a constrained, regularized space
  - CalibratedClassifierCV (sigmoid) for well-behaved probabilities
- Reporting:
  - Train/Test metrics: Accuracy, Balanced Accuracy, ROC AUC
  - Classification reports
  - Overfitting diagnostics (train–test deltas)
  - Confusion matrices figure (`confusion_matrices.png`)
- Persistence:
  - Saves `best_model.joblib`, `calibrated_model.joblib`, `feature_extractor.joblib`
  - Saves best parameters, top parameter rankings, and training metrics

## Repository Structure

Svm-model-for-bone-cancer/
├─ data/
│ ├─ Benign/ # .png/.jpg images
│ └─ Malignant/ # .png/.jpg images
│ └─ optimized_models/
│ ├─ best_model.joblib
│ ├─ calibrated_model.joblib
│ ├─ feature_extractor.joblib # large file warning: ~66 MB
│ ├─ best_params.txt
│ └─ parameter_rankings.txt
├─ svm_bone_cancer.py # your main script (example name)
├─ README.md
├─ .gitignore
└─ .gitattributes # optional; if using Git LFS


> The script expects `DATA_DIR = D:/Svmfinal/data` by default. Adjust paths at the top of `__main__`.

## Data Layout

Place your images as:

data/
Benign/
img_001.png
...
Malignant/
img_101.png

Supported extensions: `.png`, `.jpg`, `.jpeg`. Images are read as grayscale.

## Installation

Tested with Python 3.10+.

```bash
# Create and activate a virtual environment (Windows PowerShell)
python -m venv .venv
. .\.venv\Scripts\Activate.ps1

pip install -U pip
pip install numpy scipy scikit-learn scikit-image opencv-python-headless matplotlib seaborn joblib tqdm albumentations imbalanced-learn
```
If you use Jupyter for exploration:
```
pip install notebook
```
 ## Usage

Run the script directly (edit DATA_DIR if needed):
```
python svm_bone_cancer.py
```
What happens:

Loads images from data/Benign and data/Malignant.

Extracts features for each image.

Adds one augmented sample per original image (controlled probability).

Splits into train/test (stratified).

Runs RandomizedSearchCV to find robust SVM hyperparameters.

Calibrates probabilities.

Reports metrics, saves artifacts, and plots confusion matrices.

Key outputs:

confusion_matrices.png

data/optimized_models/best_model.joblib

data/optimized_models/calibrated_model.joblib

data/optimized_models/feature_extractor.joblib

data/optimized_models/best_params.txt

data/optimized_models/parameter_rankings.txt

data/optimized_models/training_metrics.txt

Important Notes About Large Files

feature_extractor.joblib is ~66 MB. GitHub warns above 50 MB and hard-blocks above 100 MB.

If you prefer a lean repo, either:

Use Git LFS for .joblib artifacts, or

Exclude data/optimized_models/ via .gitignore and publish trained artifacts in GitHub Releases with a small download script.

Example .gitignore entries:
```
.venv/
__pycache__/
*.pyc
data/optimized_models/
data/**/*.csv
data/**/*.npy
```

If you want LFS:
```
git lfs install
git lfs track "*.joblib"
git add .gitattributes
git commit -m "Track .joblib via Git LFS
```
 ## Configuration

Key toggles in code:

Augmentation probability: augment_prob in load_data(...) (default 0.4)

PCA retained variance: pca__n_components search range

SMOTE neighbors: k_neighbors=3 (more conservative)

SVC search space:

svc__C: loguniform(1e-1, 1e2)

svc__gamma: loguniform(1e-3, 1e0)

svc__tol: [1e-2, 1e-3]

These were chosen to stabilize training and reduce overfitting while keeping runtime reasonable.

 ## Reproducibility

Global np.random.seed(42)

Deterministic cross-validation via StratifiedKFold(..., random_state=42)

PCA, ExtraTrees, SMOTE, and SVC all seeded where exposed.

 ## Troubleshooting

“No images found”: verify data/Benign and data/Malignant exist and contain valid images.

“MemoryError” or very slow HOG: lower image size (default 128×128) or reduce HOG parameters.

Class imbalance: SMOTE is enabled; you can disable it or tune k_neighbors.

Overfitting warning in logs: consider narrowing C, increasing regularization, or raising the feature selector threshold.

 ## Citing / Attribution

If you use this pipeline in publications, please cite scikit-learn, scikit-image, Albumentations, and any datasets you employ.
