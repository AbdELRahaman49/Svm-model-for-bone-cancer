import os
import cv2
import numpy as np
import joblib
from matplotlib import pyplot as plt
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern, hog
from skimage.measure import shannon_entropy


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
        print(f"After GLCM: {len(features)} features")

        features.extend(self._calculate_texture_features(image))
        print(f"After texture: {len(features)} features")

        features.extend(self._calculate_histogram_features(image))
        print(f"After histogram: {len(features)} features")

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
        print(f"After statistical: {len(features)} features")

        # Reduced HOG complexity
        hog_features = hog(image, orientations=8, pixels_per_cell=(16, 16),
                           cells_per_block=(2, 2), feature_vector=True, channel_axis=None)
        features.extend(hog_features)
        print(f"After HOG: {len(features)} features")

        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        self.feature_cache[cache_key] = np.array(features)
        return self.feature_cache[cache_key]


class BoneCancerPredictor:
    def __init__(self, model_dir):
        """
        Initialize the predictor by loading all required components
        Args:
            model_dir: Directory containing the saved model components
        """
        self.model_dir = model_dir
        self.load_components()

    def load_components(self):
        """Load all the required model components"""
        try:
            # Try to load the feature extractor
            try:
                self.feature_extractor = joblib.load(os.path.join(self.model_dir, 'feature_extractor.joblib'))
                print(f"Feature extractor type: {type(self.feature_extractor)}")
                print(f"Feature extractor methods: {dir(self.feature_extractor)}")
            except Exception as fe:
                print(f"Warning: Could not load feature extractor: {str(fe)}")
                print("Creating a new feature extractor instance instead")
                self.feature_extractor = OptimizedFeatureExtractor()

            self.model = joblib.load(os.path.join(self.model_dir, 'best_model.joblib'))
            self.calibrated_model = joblib.load(os.path.join(self.model_dir, 'calibrated_model.joblib'))
            print("✅ All model components loaded successfully")
        except Exception as e:
            raise Exception(f"Error loading model components: {str(e)}")

    def preprocess_image(self, image_path):
        """
        Load and preprocess an image for prediction
        Args:
            image_path: Path to the image file
        Returns:
            Preprocessed image in grayscale
        """
        try:
            # Read image in grayscale
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise ValueError(f"Could not read image from {image_path}")

            # Basic validation
            if image.size == 0:
                raise ValueError("Image is empty")

            # Resize and normalize to match training preprocessing
            image = cv2.resize(image, (128, 128))
            image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)

            return image
        except Exception as e:
            raise Exception(f"Image preprocessing failed: {str(e)}")

    def test_feature_extraction(self, image_path):
        """Test feature extraction"""
        image = self.preprocess_image(image_path)
        print(f"Preprocessed image shape: {image.shape}")

        try:
            # Create a fresh extractor for testing
            test_extractor = OptimizedFeatureExtractor()
            test_features = test_extractor(image)
            print(f"Test extractor features shape: {test_features.shape}")
        except Exception as e:
            print(f"Test extractor failed: {str(e)}")

        # Try with loaded extractor
        try:
            features = self.feature_extractor(image)
            print(f"Loaded extractor features shape: {features.shape}")
            return features
        except Exception as e:
            print(f"Loaded extractor failed: {str(e)}")
            return None

    def predict(self, image_path, show_image=False):
        """
        Make a prediction on a single image
        Args:
            image_path: Path to the image file
            show_image: Whether to display the image with prediction
        Returns:
            Dictionary containing prediction results
        """
        try:
            # Preprocess the image
            image = self.preprocess_image(image_path)
            print(f"Preprocessed image shape: {image.shape}")

            # Extract features using the same extractor used in training
            print("Extracting features...")
            features = self.feature_extractor(image)
            print(f"Raw features shape: {features.shape}, type: {type(features)}")

            # Check if we have enough features
            if features.shape[0] != 1669:
                print(f"WARNING: Expected 1669 features but got {features.shape[0]}")
                # Try to fix the feature extraction
                test_extractor = OptimizedFeatureExtractor()
                features = test_extractor(image)
                print(f"Re-extracted features shape: {features.shape}")

                if features.shape[0] != 1669:
                    raise ValueError(
                        f"Feature extraction produced {features.shape[0]} features, but model expects 1669 features")

            features = features.reshape(1, -1)
            print(f"Reshaped features shape: {features.shape}")

            # Make predictions
            print("Making predictions...")
            prediction = self.model.predict(features)[0]
            probability = self.model.predict_proba(features)[0][1]  # Probability of malignant class

            # Calibrated predictions
            calib_prediction = self.calibrated_model.predict(features)[0]
            calib_probability = self.calibrated_model.predict_proba(features)[0][1]

            # Get class names
            class_names = ['Benign', 'Malignant']

            # Prepare results
            results = {
                'image_path': image_path,
                'prediction': class_names[prediction],
                'probability': float(probability),
                'calibrated_prediction': class_names[calib_prediction],
                'calibrated_probability': float(calib_probability),
                'confidence': self._get_confidence_level(max(probability, calib_probability))
            }

            # Optional image display
            if show_image:
                self.display_prediction(image, results)

            return results

        except Exception as e:
            import traceback
            traceback.print_exc()
            raise Exception(f"Prediction failed: {str(e)}")

    def _get_confidence_level(self, probability):
        """Determine confidence level based on probability"""
        if probability > 0.85 or probability < 0.15:
            return 'High'
        elif probability > 0.7 or probability < 0.3:
            return 'Medium'
        else:
            return 'Low'

    def display_prediction(self, image, results):
        """Display the image with prediction results"""
        plt.figure(figsize=(8, 8))
        plt.imshow(image, cmap='gray')

        # Prepare text to display
        text = f"Prediction: {results['prediction']}\n" \
               f"Probability: {results['probability']:.2%}\n" \
               f"Calibrated: {results['calibrated_prediction']}\n" \
               f"Calib. Prob: {results['calibrated_probability']:.2%}\n" \
               f"Confidence: {results['confidence']}"

        # Add text to the image
        plt.text(10, 30, text,
                 fontsize=12,
                 bbox=dict(facecolor='white', alpha=0.8))

        plt.title(f"Bone Cancer Classification - {os.path.basename(results['image_path'])}")
        plt.axis('off')
        plt.show()

    def predict_batch(self, image_dir, output_file=None):
        """
        Make predictions on all images in a directory
        Args:
            image_dir: Directory containing images to predict
            output_file: Optional file to save results (CSV format)
        Returns:
            List of prediction results
        """
        results = []
        valid_extensions = ('.png', '.jpg', '.jpeg')

        print(f"Processing images in {image_dir}...")
        for filename in os.listdir(image_dir):
            if filename.lower().endswith(valid_extensions):
                try:
                    image_path = os.path.join(image_dir, filename)
                    result = self.predict(image_path)
                    results.append(result)
                    print(f"Processed {filename}: {result['prediction']} ({result['probability']:.2%})")
                except Exception as e:
                    print(f"Error processing {filename}: {str(e)}")
                    continue

        # Save to file if requested
        if output_file and results:
            self.save_results(results, output_file)

        return results

    def save_results(self, results, output_file):
        """Save prediction results to a CSV file"""
        import csv
        try:
            with open(output_file, 'w', newline='') as csvfile:
                fieldnames = ['image_path', 'prediction', 'probability',
                              'calibrated_prediction', 'calibrated_probability', 'confidence']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                writer.writeheader()
                for result in results:
                    writer.writerow(result)

            print(f"✅ Results saved to {output_file}")
        except Exception as e:
            print(f"Error saving results: {str(e)}")


# Alternative direct implementation (bypassing saved feature extractor)
def direct_predict(model_dir, image_path):
    """
    Make a prediction using direct implementation without loading the feature extractor
    """
    print("\nTrying alternative direct prediction method...")
    try:
        # Create feature extractor
        extractor = OptimizedFeatureExtractor()

        # Load models
        model = joblib.load(os.path.join(model_dir, 'best_model.joblib'))
        calibrated_model = joblib.load(os.path.join(model_dir, 'calibrated_model.joblib'))

        # Load and preprocess image
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Could not read image from {image_path}")

        # Extract features
        features = extractor(image)
        print(f"Direct features shape: {features.shape}")
        features = features.reshape(1, -1)

        # Make predictions
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0][1]

        # Print results
        class_names = ['Benign', 'Malignant']
        print(f"Direct prediction: {class_names[prediction]} ({probability:.2%} confidence)")

        return {
            'prediction': class_names[prediction],
            'probability': probability
        }
    except Exception as e:
        print(f"❌ Direct prediction failed: {str(e)}")
        return None


if __name__ == "__main__":
    # Configuration - UPDATE THESE PATHS
    MODEL_DIR = r"D:\Svmfinal\data\optimized_models"  # Directory containing the model files
    TEST_IMAGE = r"D:\Svmfinal\data\image2 (369).png"  # Your test image path
    TEST_DIR = r"D:\Svmfinal\data\predictB"  # Optional: Directory with multiple test images

    # Initialize predictor
    try:
        predictor = BoneCancerPredictor(MODEL_DIR)

        # Test feature extraction first
        print("\nTesting feature extraction...")
        predictor.test_feature_extraction(TEST_IMAGE)

        # Try direct prediction as an alternative
        direct_result = direct_predict(MODEL_DIR, TEST_IMAGE)

        # Single image prediction
        print("\nMaking single prediction...")
        result = predictor.predict(TEST_IMAGE, show_image=True)
        print("\nPrediction Results:")
        print(f"Image: {os.path.basename(result['image_path'])}")
        print(f"Prediction: {result['prediction']} ({result['probability']:.2%} confidence)")
        print(
            f"Calibrated Prediction: {result['calibrated_prediction']} ({result['calibrated_probability']:.2%} confidence)")
        print(f"Confidence Level: {result['confidence']}")

        # Uncomment to process a directory of images
        # print("\nMaking batch predictions...")
        # batch_results = predictor.predict_batch(TEST_DIR, output_file="predictions.csv")

    except Exception as e:
        print(f"❌ Error: {str(e)}")

        # As a fallback, try direct prediction if the regular method failed
        if 'predictor' not in locals() or not hasattr(predictor, 'model'):
            direct_predict(MODEL_DIR, TEST_IMAGE)