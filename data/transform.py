import os
import pickle
from joblib import load

# Define paths
original_dir = r"D:\Svmfinal\data\optimized_models"  # Your original .joblib directory
new_dir = r"D:\Svmfinal\data\saved_model_pkl"  # New folder for .pkl files

# Create the new directory if it doesn't exist
os.makedirs(new_dir, exist_ok=True)

# List of .joblib files to convert
joblib_files = [
    "best_model.joblib",
    "feature_extractor.joblib",
    "calibrated_model.joblib"
]


# Define a temporary class for missing definitions
class OptimizedFeatureExtractor:
    """Temporary placeholder for missing class definition"""

    def __init__(self, *args, **kwargs):
        self.__dict__.update(kwargs)


# Add the class to __main__ module (needed for pickle)
import sys

setattr(sys.modules['__main__'], 'OptimizedFeatureExtractor', OptimizedFeatureExtractor)

# Convert each .joblib file to .pkl
for file in joblib_files:
    joblib_path = os.path.join(original_dir, file)
    pkl_filename = file.replace(".joblib", ".pkl")
    pkl_path = os.path.join(new_dir, pkl_filename)

    try:
        # Load the .joblib file
        data = load(joblib_path)

        # Save as .pkl
        with open(pkl_path, "wb") as f:
            pickle.dump(data, f)

        print(f"✅ Successfully converted {file} → {pkl_filename}")

    except AttributeError as e:
        print(f"⚠️ Warning: Could not fully convert {file} - missing attributes:")
        print(f"    {str(e)}")
        print("    The converted file may not work properly without the original class definitions")

    except Exception as e:
        print(f"❌ Failed to convert {file}: {str(e)}")
        continue

print("\n🎉 Conversion process completed!")
print(f"All converted files saved in: {new_dir}")
print("Note: Some models may require their original class definitions to work properly")