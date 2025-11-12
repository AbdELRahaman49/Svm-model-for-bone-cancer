import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern, hog
from skimage.measure import shannon_entropy
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, balanced_accuracy_score, \
    roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import ExtraTreesClassifier
from scipy.stats import uniform, loguniform, randint
import joblib
from tqdm import tqdm
from sklearn.calibration import CalibratedClassifierCV
import albumentations as A
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline as make_imb_pipeline
import time
import warnings
from sklearn.exceptions import ConvergenceWarning

# Suppress convergence warnings to reduce output clutter
warnings.filterwarnings("ignore", category=ConvergenceWarning)
np.random.seed(42)


class OptimizedFeatureExtractor:
    def __init__(self):
        self.feature_cache = {}

    def _calculate_glcm_features(self, image):
        distances = [1, 3]
        angles = [0, np.pi / 4, np.pi / 2]
        props = ['contrast', 'homogeneity', 'energy', 'correlation']
        features = []

        for dist in distances:
            for angle in angles:
                glcm = graycomatrix(image, [dist], [angle], 256, symmetric=True, normed=True)
                features.extend([graycoprops(glcm, prop)[0, 0] for prop in props])
        return np.array(features)

    def _calculate_texture_features(self, image):
        lbp_features = []
        for radius in [1, 3]:
            n_points = 8 * radius
            lbp = local_binary_pattern(image, n_points, radius, method='uniform')
            hist, _ = np.histogram(lbp, bins=n_points + 2, range=(0, n_points + 2))
            hist = hist / (hist.sum() + 1e-10)
            lbp_features.extend(hist)

        edges = cv2.Canny(image, 50, 150)
        lbp_features.extend([
            np.mean(edges),
            np.sum(edges > 0) / edges.size,
            np.std(edges)
        ])
        return np.array(lbp_features)

    def _calculate_histogram_features(self, image):
        hist = cv2.calcHist([image], [0], None, [32], [0, 256])  # Reduced bins from 48 to 32
        hist = hist / (hist.sum() + 1e-10)
        return hist.flatten()

    def _safe_divide(self, a, b, default=0.0):
        return a / b if b != 0 else default

    def __call__(self, image):
        cache_key = hash(image.tobytes())
        if cache_key in self.feature_cache:
            return self.feature_cache[cache_key]

        image = cv2.resize(image, (128, 128))
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)

        features = []
        features.extend(self._calculate_glcm_features(image))
        features.extend(self._calculate_texture_features(image))
        features.extend(self._calculate_histogram_features(image))

        median_val = np.median(image)
        p5 = np.percentile(image, 5)
        p25 = np.percentile(image, 25)
        p75 = np.percentile(image, 75)
        p95 = np.percentile(image, 95)

        features.extend([
            np.mean(image), np.std(image),
            median_val, shannon_entropy(image),
            self._safe_divide(p95 - p5, median_val, 0.0),
            self._safe_divide(p75 - p25, median_val, 0.0)
        ])

        # Reduced HOG complexity
        hog_features = hog(image, orientations=8, pixels_per_cell=(16, 16),
                           cells_per_block=(2, 2), feature_vector=True, channel_axis=None)
        features.extend(hog_features)

        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        self.feature_cache[cache_key] = np.array(features)
        return self.feature_cache[cache_key]


def load_data(image_folder, augment_prob=0.3):
    """
    Load and extract features from images with more conservative augmentation.
    """
    extractor = OptimizedFeatureExtractor()
    augmenter = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, p=0.3),  # Reduced rotation limit
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),  # Reduced probability
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),  # Reduced limits
    ], p=augment_prob)

    features = []
    labels = []
    original_images = {0: [], 1: []}

    # First pass: Load all original images and extract features
    for class_idx, class_name in enumerate(['Benign', 'Malignant']):
        class_path = os.path.join(image_folder, class_name)
        if not os.path.exists(class_path):
            print(f"Warning: Path {class_path} does not exist")
            continue

        print(f"Loading original {class_name} images...")
        for filename in tqdm(os.listdir(class_path), desc=f"Processing {class_name}"):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                try:
                    img_path = os.path.join(class_path, filename)
                    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if image is None or image.size == 0:
                        continue

                    # Store original images for later augmentation
                    original_images[class_idx].append(image)

                    # Extract features
                    feats = extractor(image)
                    if feats is not None and not np.isnan(feats).any():
                        features.append(feats)
                        labels.append(class_idx)
                except Exception as e:
                    print(f"Error processing {filename}: {str(e)}")
                    continue

    # Print original class counts
    benign_count = np.sum(np.array(labels) == 0)
    malignant_count = np.sum(np.array(labels) == 1)
    print(f"Original class distribution: Benign={benign_count}, Malignant={malignant_count}")

    # Second pass: Add augmented samples - one per original image
    print("Generating augmented samples...")
    for class_idx in [0, 1]:
        for i, image in enumerate(tqdm(original_images[class_idx],
                                       desc=f"Augmenting class {class_idx}")):
            augmented = augmenter(image=image)['image']
            aug_feats = extractor(augmented)
            if aug_feats is not None and not np.isnan(aug_feats).any():
                features.append(aug_feats)
                labels.append(class_idx)

    # Convert to numpy arrays
    features = np.array(features)
    labels = np.array(labels)

    # Final class distribution
    final_benign = np.sum(labels == 0)
    final_malignant = np.sum(labels == 1)
    print(f"Final class distribution: Benign={final_benign}, Malignant={final_malignant}")

    return features, labels, extractor


def create_optimized_pipeline():
    # Create pipeline with stronger regularization
    scaler = StandardScaler()
    pca = PCA(n_components=0.9, random_state=42)  # Retain more variance
    smote = SMOTE(random_state=42, k_neighbors=3)  # Fewer neighbors
    feature_selector = SelectFromModel(
        ExtraTreesClassifier(n_estimators=50, random_state=42, max_depth=10),  # Simpler feature selector
        threshold='1.25*median'  # More selective threshold
    )
    svm = SVC(
        kernel='rbf',
        class_weight='balanced',
        probability=True,
        random_state=42,
        cache_size=2000,
        max_iter=50000,
        tol=1e-2,
        shrinking=True,
        break_ties=False  # Simpler decision making
    )

    return make_imb_pipeline(
        scaler,
        pca,
        smote,
        feature_selector,
        svm
    )


def train_model(X_train, y_train):
    pipeline = create_optimized_pipeline()

    # More constrained parameter space
    param_distributions = {
        'svc__C': loguniform(1e-1, 1e2),  # Reduced upper bound
        'svc__gamma': loguniform(1e-3, 1e0),  # Narrower range
        'pca__n_components': uniform(0.8, 0.15),  # Higher minimum variance
        'selectfrommodel__threshold': ['1.25*median', '1.5*median'],  # More selective thresholds
        'svc__tol': [1e-2, 1e-3]
    }

    model = RandomizedSearchCV(
        pipeline,
        param_distributions=param_distributions,
        n_iter=25,  # Fewer iterations
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring='balanced_accuracy',
        n_jobs=-1,
        verbose=2,
        random_state=42,
        return_train_score=True
    )

    print("Starting randomized search with regularization...")
    start_time = time.time()
    model.fit(X_train, y_train)
    end_time = time.time()
    print(f"Search completed in {end_time - start_time:.2f} seconds")
    print(f"Best parameters: {model.best_params_}")
    print(f"Best score: {model.best_score_:.4f}")

    print("Calibrating model probabilities...")
    calibrated_model = CalibratedClassifierCV(
        model.best_estimator_,
        method='sigmoid',
        cv=5
    )
    calibrated_model.fit(X_train, y_train)

    return model, calibrated_model


def evaluate_model(model, calibrated_model, X_train, y_train, X_test, y_test):
    print("\n=== Best Parameters ===")
    print(model.best_params_)
    print(f"Best score: {model.best_score_:.4f}")

    print("\n=== Top 5 Parameter Combinations ===")
    indices = np.argsort(model.cv_results_['mean_test_score'])[::-1][:5]
    for rank, i in enumerate(indices, 1):
        print(
            f"Rank {rank}: {model.cv_results_['mean_test_score'][i]:.4f} - Parameters: {model.cv_results_['params'][i]}")

    # Test Set Performance
    print("\n=== Test Set Performance ===")
    y_test_pred = model.predict(X_test)
    y_test_proba = model.predict_proba(X_test)[:, 1]
    test_acc = accuracy_score(y_test, y_test_pred)
    test_bal_acc = balanced_accuracy_score(y_test, y_test_pred)
    test_auc = roc_auc_score(y_test, y_test_proba)

    print(f"Accuracy: {test_acc:.4f}")
    print(f"Balanced Accuracy: {test_bal_acc:.4f}")
    print(f"ROC AUC: {test_auc:.4f}")
    print(classification_report(y_test, y_test_pred, target_names=['Benign', 'Malignant']))

    # Training Set Performance
    print("\n=== Training Set Performance ===")
    y_train_pred = model.predict(X_train)
    y_train_proba = model.predict_proba(X_train)[:, 1]
    train_acc = accuracy_score(y_train, y_train_pred)
    train_bal_acc = balanced_accuracy_score(y_train, y_train_pred)
    train_auc = roc_auc_score(y_train, y_train_proba)

    print(f"Training Accuracy: {train_acc:.4f}")
    print(f"Training Balanced Accuracy: {train_bal_acc:.4f}")
    print(f"Training ROC AUC: {train_auc:.4f}")
    print(classification_report(y_train, y_train_pred, target_names=['Benign', 'Malignant']))

    # Check for overfitting
    print("\n=== Overfitting Analysis ===")
    acc_diff = train_acc - test_acc
    bal_acc_diff = train_bal_acc - test_bal_acc
    auc_diff = train_auc - test_auc

    print(f"Accuracy difference (train-test): {acc_diff:.4f}")
    print(f"Balanced Accuracy difference: {bal_acc_diff:.4f}")
    print(f"AUC difference: {auc_diff:.4f}")

    if acc_diff > 0.15 or bal_acc_diff > 0.15 or auc_diff > 0.15:
        print("WARNING: Possible overfitting detected! Consider adjusting regularization parameters.")
    else:
        print("Model shows reasonable generalization with limited overfitting.")

    # Class-specific performance analysis
    cm_test = confusion_matrix(y_test, y_test_pred)
    benign_specificity = cm_test[0, 0] / (cm_test[0, 0] + cm_test[0, 1]) if (cm_test[0, 0] + cm_test[0, 1]) > 0 else 0
    malignant_sensitivity = cm_test[1, 1] / (cm_test[1, 0] + cm_test[1, 1]) if (cm_test[1, 0] + cm_test[
        1, 1]) > 0 else 0

    print(f"\n=== Class-specific Performance ===")
    print(f"Benign Specificity: {benign_specificity:.4f}")
    print(f"Malignant Sensitivity: {malignant_sensitivity:.4f}")

    # Calibrated model performance
    print("\n=== Calibrated Test Set Performance ===")
    y_test_calib_pred = calibrated_model.predict(X_test)
    y_test_calib_proba = calibrated_model.predict_proba(X_test)[:, 1]
    print(f"Accuracy: {accuracy_score(y_test, y_test_calib_pred):.4f}")
    print(f"Balanced Accuracy: {balanced_accuracy_score(y_test, y_test_calib_pred):.4f}")
    print(f"ROC AUC: {roc_auc_score(y_test, y_test_calib_proba):.4f}")
    print(classification_report(y_test, y_test_calib_pred, target_names=['Benign', 'Malignant']))

    # Plot confusion matrices
    plot_confusion_matrices(y_train, y_train_pred, y_test, y_test_pred)


def plot_confusion_matrices(y_train, y_train_pred, y_test, y_test_pred):
    plt.figure(figsize=(16, 7))

    # Training confusion matrix
    plt.subplot(1, 2, 1)
    cm_train = confusion_matrix(y_train, y_train_pred)
    sns.heatmap(cm_train,
                annot=True, fmt='d', cmap='Blues',
                xticklabels=['Benign', 'Malignant'],
                yticklabels=['Benign', 'Malignant'])

    sensitivity_train = cm_train[1, 1] / (cm_train[1, 0] + cm_train[1, 1]) if (cm_train[1, 0] + cm_train[
        1, 1]) > 0 else 0
    specificity_train = cm_train[0, 0] / (cm_train[0, 0] + cm_train[0, 1]) if (cm_train[0, 0] + cm_train[
        0, 1]) > 0 else 0

    plt.title(f'Training Confusion Matrix\n'
              f'Accuracy: {accuracy_score(y_train, y_train_pred):.2%}\n'
              f'Sensitivity: {sensitivity_train:.2%}\n'
              f'Specificity: {specificity_train:.2%}')

    # Test confusion matrix
    plt.subplot(1, 2, 2)
    cm_test = confusion_matrix(y_test, y_test_pred)
    sns.heatmap(cm_test,
                annot=True, fmt='d', cmap='Blues',
                xticklabels=['Benign', 'Malignant'],
                yticklabels=['Benign', 'Malignant'])

    sensitivity_test = cm_test[1, 1] / (cm_test[1, 0] + cm_test[1, 1]) if (cm_test[1, 0] + cm_test[1, 1]) > 0 else 0
    specificity_test = cm_test[0, 0] / (cm_test[0, 0] + cm_test[0, 1]) if (cm_test[0, 0] + cm_test[0, 1]) > 0 else 0

    plt.title(f'Test Confusion Matrix\n'
              f'Accuracy: {accuracy_score(y_test, y_test_pred):.2%}\n'
              f'Sensitivity: {sensitivity_test:.2%}\n'
              f'Specificity: {specificity_test:.2%}')

    plt.tight_layout()
    plt.savefig('confusion_matrices.png')
    plt.show()


def save_model(model, calibrated_model, feature_extractor, save_dir, X_train, y_train):
    os.makedirs(save_dir, exist_ok=True)

    try:
        # Save models
        joblib.dump(model.best_estimator_, os.path.join(save_dir, 'best_model.joblib'))
        joblib.dump(calibrated_model, os.path.join(save_dir, 'calibrated_model.joblib'))
        joblib.dump(feature_extractor, os.path.join(save_dir, 'feature_extractor.joblib'))

        # Save parameters and rankings
        with open(os.path.join(save_dir, 'best_params.txt'), 'w') as f:
            f.write(str(model.best_params_))

        # Save top parameter combinations
        indices = np.argsort(model.cv_results_['mean_test_score'])[::-1][:10]
        with open(os.path.join(save_dir, 'parameter_rankings.txt'), 'w') as f:
            for rank, i in enumerate(indices, 1):
                f.write(
                    f"Rank {rank}: {model.cv_results_['mean_test_score'][i]:.4f} - Parameters: {model.cv_results_['params'][i]}\n")

        # Save training metrics
        y_train_pred = model.predict(X_train)
        with open(os.path.join(save_dir, 'training_metrics.txt'), 'w') as f:
            f.write(f"Training Accuracy: {accuracy_score(y_train, y_train_pred):.4f}\n")
            f.write(f"Training Balanced Accuracy: {balanced_accuracy_score(y_train, y_train_pred):.4f}\n")
            f.write(f"Training Classification Report:\n")
            f.write(classification_report(y_train, y_train_pred, target_names=['Benign', 'Malignant']))

        print(f"✅ All model components saved to {save_dir}")

    except Exception as e:
        print(f"❌ Error saving model components: {str(e)}")


if __name__ == "__main__":
    cv2.setNumThreads(os.cpu_count() or 1)

    DATA_DIR = r"D:/Svmfinal/data"
    SAVE_DIR = os.path.join(DATA_DIR, "optimized_models")

    total_start_time = time.time()

    try:
        print("Loading and preprocessing data...")
        data_start_time = time.time()

        # Use the improved load_data function with controlled augmentation
        X, y, feature_extractor = load_data(
            DATA_DIR,
            augment_prob=0.4  # Moderate augmentation probability
        )

        data_end_time = time.time()
        print(f"✅ Data processing completed in {data_end_time - data_start_time:.2f} seconds")
        print(f"✅ Loaded {X.shape[0]} samples with {X.shape[1]} features")
        print(f"Class distribution: {np.bincount(y)}")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y)  # Slightly larger test set

        print("\nTraining model with improved settings...")
        train_start_time = time.time()
        model, calibrated_model = train_model(X_train, y_train)
        train_end_time = time.time()
        print(f"✅ Model training completed in {train_end_time - train_start_time:.2f} seconds")

        print("\nEvaluating model...")
        evaluate_model(model, calibrated_model, X_train, y_train, X_test, y_test)

        print("\nSaving models and preprocessing components...")
        save_model(model, calibrated_model, feature_extractor, SAVE_DIR, X_train, y_train)
        print(f"Models and components saved to {SAVE_DIR}")

        total_end_time = time.time()
        print(f"\n✅ Total execution time: {total_end_time - total_start_time:.2f} seconds")

    except Exception as e:
        print(f"\n❌ Error: {str(e)}")