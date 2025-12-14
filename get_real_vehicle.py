import os
import numpy as np
import scipy.io
import bz2
import shutil
from sklearn.datasets import load_svmlight_file

# ==========================================
# Script to Decompress & Convert SensIT Vehicle Data
# Input: vehicle.bz2 (Compressed)
# Output: vehicle.mat (Ready for Ditto)
# ==========================================

RAW_DATA_DIR = './raw_data'
MAT_FILE_PATH = os.path.join(RAW_DATA_DIR, 'vehicle.mat')
BZ2_FILE_PATH = os.path.join(RAW_DATA_DIR, 'vehicle.bz2')
TXT_FILE_PATH = os.path.join(RAW_DATA_DIR, 'vehicle_libsvm.txt')

def get_real_vehicle_data():
    if not os.path.exists(RAW_DATA_DIR):
        os.makedirs(RAW_DATA_DIR)

    # 1. Decompress .bz2 file
    print("1. Checking for vehicle.bz2...")
    
    # If text file doesn't exist but bz2 exists, decompress it
    if not os.path.exists(TXT_FILE_PATH):
        if os.path.exists(BZ2_FILE_PATH):
            print("   Found compressed file! Decompressing...")
            try:
                with bz2.open(BZ2_FILE_PATH, 'rb') as f_in:
                    with open(TXT_FILE_PATH, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                print("   Decompression complete! Created 'vehicle_libsvm.txt'")
            except Exception as e:
                print(f"   Error during decompression: {e}")
                return
        else:
            print(f"   Error: Cannot find '{BZ2_FILE_PATH}'")
            print("   Please rename your downloaded file to 'vehicle.bz2' and put it in 'raw_data' folder.")
            return
    else:
        print("   Text file already exists. Skipping decompression.")

    # 2. Read and Convert data
    print("2. Loading and converting data (Expect 100 features)...")
    
    try:
        data_X, data_y = load_svmlight_file(TXT_FILE_PATH, n_features=100)
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    data_X = data_X.toarray()
    
    # SensIT labels are 1, 2, 3. Binarize: Class 3 -> 1, Others -> 0
    data_y = np.where(data_y == 3, 1, 0)

    # 3. Shape Check
    print(f"   Data Shape: {data_X.shape}") 
    # Should be (78823, 100)

    # 4. Distribute to 23 users
    NUM_USER = 23
    samples_per_user = len(data_X) // NUM_USER
    
    final_X = np.zeros((NUM_USER, 1), dtype=object)
    final_Y = np.zeros((NUM_USER, 1), dtype=object)

    print(f"3. Distributing to {NUM_USER} users...")

    for i in range(NUM_USER):
        start_idx = i * samples_per_user
        end_idx = (i + 1) * samples_per_user
        if i == NUM_USER - 1: end_idx = len(data_X)

        user_x = data_X[start_idx:end_idx]
        user_y = data_y[start_idx:end_idx].reshape(1, -1)
        
        # Transpose to (Features, Samples)
        final_X[i, 0] = user_x.T 
        final_Y[i, 0] = user_y

    # 5. Save
    scipy.io.savemat(MAT_FILE_PATH, {'X': final_X, 'Y': final_Y})
    print(f"4. Success! Saved to: {MAT_FILE_PATH}")
    print("   Now please run 'python create_dataset.py'.")

if __name__ == "__main__":
    get_real_vehicle_data()