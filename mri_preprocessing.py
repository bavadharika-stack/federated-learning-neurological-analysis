import os
import numpy as np

# Path to your preprocessed MRI files
mri_dir = os.path.join("data", "preprocessed_mri")

x_mri = []
y_mri = []

for filename in os.listdir(mri_dir):
    if filename.endswith(".npy") and not filename.startswith(("x_mri", "y_mri")):
        filepath = os.path.join(mri_dir, filename)
        mri_array = np.load(filepath)

        # Accept shapes (128,128,128) or (128,128,128,1)
        if mri_array.shape == (128, 128, 128, 1):
            mri_array = np.squeeze(mri_array, axis=-1)
        elif mri_array.shape != (128, 128, 128):
            print(f"⚠️ Skipping due to invalid shape: {filename} → {mri_array.shape}")
            continue

        # Normalize: min-max to [0, 1]
        mri_array = (mri_array - np.min(mri_array)) / (np.max(mri_array) - np.min(mri_array) + 1e-5)

        x_mri.append(mri_array)

        # Dummy label logic (update this to your real labeling strategy)
        if "AD" in filename or "alzheimer" in filename.lower():
            y_mri.append(0)  # Alzheimer's
        else:
            y_mri.append(1)  # Control

        print(f"✅ Processed: {filename} → shape: {mri_array.shape}")

x_mri = np.array(x_mri)
y_mri = np.array(y_mri)

# Save to .npy
np.save(os.path.join(mri_dir, "x_mri.npy"), x_mri)
np.save(os.path.join(mri_dir, "y_mri.npy"), y_mri)

print(f"\n✅ Final x_mri shape: {x_mri.shape}")
print(f"✅ Final y_mri shape: {y_mri.shape}")
print(f"🧠 Unique labels used: {set(y_mri)}")
print("✅ Saved preprocessed MRI data successfully.")
