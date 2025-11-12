import os
import cv2
import numpy as np
import tensorflow as tf
import pickle
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern, hog
from skimage.measure import shannon_entropy
from sklearn.calibration import CalibratedClassifierCV

# =============================================
# 1. Define the OptimizedFeatureExtractor class
# =============================================
class OptimizedFeatureExtractor:
    def __init__(self):
        self.feature_cache = {}

    def _calculate_glcm_features(self, image):
        distances = [1, 3]
        angles = [0, np.pi/4, np.pi/2]
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
        hist = cv2.calcHist([image], [0], None, [32], [0, 256])
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

# Rest of your code remains the same...

# =============================================
# 2. Load All Components from .pkl Files
# =============================================
def load_components(base_path):
    """Load all model components from .pkl files"""
    with open(f'{base_path}/best_model.pkl', 'rb') as f:
        best_model = pickle.load(f)

    with open(f'{base_path}/calibrated_model.pkl', 'rb') as f:
        calibrated_model = pickle.load(f)

    with open(f'{base_path}/feature_extractor.pkl', 'rb') as f:
        feature_extractor = pickle.load(f)

    return best_model, calibrated_model, feature_extractor

# Load all components
best_model, calibrated_model, feature_extractor = load_components(
    'D:/Svmfinal/data/saved_model_pkl'
)

# =============================================
# 3. Create Combined TensorFlow Model
# =============================================
class CancerClassificationModel(tf.Module):
    def __init__(self, feature_extractor, model, calibrated_model=None):
        super(CancerClassificationModel, self).__init__()
        self.feature_extractor = feature_extractor
        self.model = model
        self.calibrated_model = calibrated_model
        self.use_calibrated = calibrated_model is not None

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, None, None], dtype=tf.uint8)])
    def preprocess(self, image):
        """Preprocess image (resize, normalize, etc.)"""
        image = tf.numpy_function(
            func=self._preprocess_numpy,
            inp=[image],
            Tout=tf.float32
        )
        image.set_shape([None, 128, 128])  # Set output shape
        return image

    def _preprocess_numpy(self, image):
        """Internal numpy-based preprocessing"""
        # Convert to grayscale if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        elif len(image.shape) == 3 and image.shape[2] == 1:
            image = image[:, :, 0]

        # Resize and normalize
        image = cv2.resize(image, (128, 128))
        return cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, 128, 128], dtype=tf.float32)])
    def extract_features(self, image):
        """Extract features using the feature extractor"""
        features = tf.numpy_function(
            func=self._extract_features_numpy,
            inp=[image],
            Tout=tf.float32
        )
        features.set_shape([None, best_model.n_features_in_])
        return features

    def _extract_features_numpy(self, image):
        """Internal numpy-based feature extraction"""
        if len(image.shape) == 3:  # Batch processing
            return np.array([self.feature_extractor(img) for img in image])
        return self.feature_extractor(image)

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, None], dtype=tf.float32)])
    def predict(self, features):
        """Make prediction from features"""
        probs = tf.numpy_function(
            func=self._predict_numpy,
            inp=[features],
            Tout=tf.float32
        )
        probs.set_shape([None, 2])  # Assuming binary classification
        return {'probabilities': probs}

    def _predict_numpy(self, features):
        """Internal numpy-based prediction"""
        if self.use_calibrated:
            return self.calibrated_model.predict_proba(features)
        return self.model.predict_proba(features)

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, None, None], dtype=tf.uint8)])
    def __call__(self, image):
        """End-to-end prediction from raw image"""
        # 1. Preprocess
        processed = self.preprocess(image)

        # 2. Extract features
        features = self.extract_features(processed)

        # 3. Predict
        return self.predict(features)

# =============================================
# 4. Create and Save the Combined Model
# =============================================
def save_combined_model(output_dir="combined_cancer_model"):
    """Create and save the complete model"""
    # Create the combined model
    combined_model = CancerClassificationModel(
        feature_extractor=feature_extractor,
        model=best_model,
        calibrated_model=calibrated_model
    )

    # Initialize with dummy data
    dummy_input = tf.constant(np.zeros((1, 256, 256, 1), dtype=np.uint8))
    _ = combined_model(dummy_input)

    # Save as SavedModel
    tf.saved_model.save(
        combined_model,
        output_dir,
        signatures={
            'serving_default': combined_model.__call__.get_concrete_function(
                tf.TensorSpec(shape=[None, None, None], dtype=tf.uint8)
            ),
            'preprocess': combined_model.preprocess.get_concrete_function(
                tf.TensorSpec(shape=[None, None, None], dtype=tf.uint8)
            ),
            'extract_features': combined_model.extract_features.get_concrete_function(
                tf.TensorSpec(shape=[None, 128, 128], dtype=tf.float32)
            ),
            'predict': combined_model.predict.get_concrete_function(
                tf.TensorSpec(shape=[None, None], dtype=tf.float32)
            )
        }
    )
    print(f"✅ Combined model saved to {output_dir}")

save_combined_model()

# =============================================
# 5. Verify the Combined Model
# =============================================
def test_combined_model(model_dir="combined_cancer_model"):
    """Test the end-to-end pipeline"""
    # Load model
    loaded_model = tf.saved_model.load(model_dir)
    infer = loaded_model.signatures['serving_default']

    # Create test image (single channel)
    test_image = np.random.randint(0, 256, (1, 256, 256, 1), dtype=np.uint8)

    # Make prediction
    output = infer(tf.constant(test_image))

    print("\n=== Test Results ===")
    print("Output probabilities:", output['probabilities'].numpy())

    # Compare with original components
    original_image = test_image[0, :, :, 0]
    features = feature_extractor(original_image)
    original_probs = calibrated_model.predict_proba([features])[0]

    print("Original probabilities:", original_probs)
    print("Difference:", np.abs(output['probabilities'].numpy()[0] - original_probs))

test_combined_model()

# =============================================
# 6. (Optional) Convert to TensorFlow Lite
# =============================================
def convert_combined_to_tflite(saved_model_dir="combined_cancer_model",
                             tflite_path="combined_model.tflite"):
    """Convert the combined model to TFLite"""
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    tflite_model = converter.convert()

    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
    print(f"✅ TFLite model saved to {tflite_path}")

convert_combined_to_tflite()