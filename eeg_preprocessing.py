import os
import numpy as np

input_dir = "data/preprocessed_eeg"
output_data_file = os.path.join(input_dir, "X_eeg.npy")
output_label_file = os.path.join(input_dir, "y_eeg.npy")

X = []
y = []

# ✅ Updated matching logic for filenames like: S109_S109R01.npy
for file in os.listdir(input_dir):
    if file.endswith(".npy") and not file.startswith("X") and not file.startswith("y"):
        file_path = os.path.join(input_dir, file)
        file_upper = file.upper()

        if "R01" in file_upper or "R02" in file_upper or "R03" in file_upper:
            label = 0  # Seizure
        elif "R04" in file_upper or "R05" in file_upper or "R06" in file_upper:
            label = 1  # Relaxed
        elif "R07" in file_upper or "R08" in file_upper or "R09" in file_upper:
            label = 2  # Normal
        else:
            print(f"⚠️ Skipping file (unknown label): {file}")
            continue

        eeg_data = np.load(file_path)
        X.append(eeg_data)
        y.append(label)

X = np.array(X)
y = np.array(y)

print(f"✅ Final EEG data shape: {X.shape}")
print(f"✅ Final EEG labels shape: {y.shape}")
print(f"🧠 Unique labels: {np.unique(y)}")

np.save(output_data_file, X)
np.save(output_label_file, y)

print("✅ EEG data and labels saved successfully.")
