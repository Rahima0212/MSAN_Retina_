import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from tqdm import tqdm

# --- Helper Function to Handle Unicode Paths ---
def imread_safe(file_path, flags=cv2.IMREAD_GRAYSCALE):
    if not os.path.exists(file_path):
        return None
    try:
        with open(file_path, 'rb') as f:
            file_buffer = np.frombuffer(f.read(), np.uint8)
        img = cv2.imdecode(file_buffer, flags)
        return img
    except Exception as e:
        print(f"Error decoding image {file_path}: {e}")
        return None

def create_otsu_roi_mask(oct_image_path, save_path):
    try:
        img = imread_safe(oct_image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Warning: Could not read image/file not found: {oct_image_path}")
            return None
        img_blur = cv2.GaussianBlur(img, (5, 5), 0)
        _, img_otsu = cv2.threshold(img_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        img_roi = cv2.medianBlur(img_otsu, 5)
        is_success, im_buf_arr = cv2.imencode(".png", img_roi)
        if is_success:
            im_buf_arr.tofile(save_path)
            return save_path
        else:
            print(f"Error encoding ROI mask for {save_path}")
            return None
    except Exception as e:
        print(f"Error processing {oct_image_path}: {e}")
        return None

# --- Main Function to Build the Master CSV ---
def create_master_csv(base_dir, class_names, output_csv_name):
    roi_output_dir = os.path.join(os.path.dirname(base_dir), "generated_roi_masks")
    os.makedirs(roi_output_dir, exist_ok=True)
    print(f"ROI masks will be saved in: {roi_output_dir}")
    
    data = []
    label_map = {class_name: i for i, class_name in enumerate(class_names)}
    
    for class_name in class_names:
        class_dir = os.path.join(base_dir, class_name)
        if not os.path.isdir(class_dir):
            print(f"Warning: Directory not found: {class_dir}")
            continue
        print(f"Processing class: {class_name}")
        for patient_id in tqdm(os.listdir(class_dir)):
            patient_dir = os.path.join(class_dir, patient_id)
            if not os.path.isdir(patient_dir): continue
            for eye in os.listdir(patient_dir):
                eye_dir = os.path.join(patient_dir, eye)
                if not os.path.isdir(eye_dir): continue
                fundus_path, oct_path = None, None
                for f in os.listdir(eye_dir):
                    if "_Color_" in f and f.endswith(('.jpg', '.png', '.jpeg')):
                        fundus_path = os.path.join(eye_dir, f)
                    elif ("_B-scan_" in f or f == "original.jpg") and f.endswith(('.jpg', '.png', '.jpeg')):
                        oct_path = os.path.join(eye_dir, f)
                if fundus_path and oct_path:
                    oct_filename = os.path.basename(oct_path)
                    roi_filename = f"{patient_id}_{eye}_{os.path.splitext(oct_filename)[0]}_roi.png"
                    roi_save_path = os.path.join(roi_output_dir, roi_filename)
                    if not os.path.exists(roi_save_path):
                        create_otsu_roi_mask(oct_path, roi_save_path)
                    if os.path.exists(roi_save_path):
                        data.append({
                            "fundus_path": fundus_path, "oct_path": oct_path, "roi_path": roi_save_path,
                            "label": label_map[class_name], "class_name": class_name, "patient_id": patient_id
                        })
    df = pd.DataFrame(data)
    df.to_csv(output_csv_name, index=False)
    print(f"\nSuccessfully created {output_csv_name} with {len(df)} entries.")
    return df

# --- THIS IS THE RESTORED split_data FUNCTION ---
def split_data(df, csv_prefix):
    """Splits a dataframe by patient_id into train, validation, and test sets."""
    if df.empty:
        print(f"\nSkipping split for {csv_prefix} as the DataFrame is empty.")
        return
        
    print(f"\nSplitting data for {csv_prefix} into Train/Val/Test sets...")
    
    # First split: 80% train, 20% temp (for val/test)
    splitter = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=42)
    train_idx, temp_idx = next(splitter.split(df, groups=df['patient_id']))
    
    train_df = df.iloc[train_idx]
    temp_df = df.iloc[temp_idx]
    
    # Second split: 50% of temp for val, 50% for test (10% of total)
    splitter_val = GroupShuffleSplit(test_size=0.5, n_splits=1, random_state=42)
    val_idx, test_idx = next(splitter_val.split(temp_df, groups=temp_df['patient_id']))
    
    val_df = temp_df.iloc[val_idx]
    test_df = temp_df.iloc[test_idx]
    
    # Save the splits to the data/ folder
    train_df.to_csv(f"data/{csv_prefix}_train.csv", index=False)
    val_df.to_csv(f"data/{csv_prefix}_val.csv", index=False)
    test_df.to_csv(f"data/{csv_prefix}_test.csv", index=False)
    
    print(f"Total: {len(df)} | Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

# --- Main Execution Block ---
if __name__ == '__main__':
    DATASET_ROOT = "/jupyter/sods.user21/MSAN_Retina/latest/Dataset/Dataset" # Adjust if needed

    # --- Process Macula Model Data ---
    MACULA_ROOT = os.path.join(DATASET_ROOT, "Macula")
    MACULA_CLASSES = ["acute CSR", "chronic CSR", "ci-DME", "geographic_AMD", "Healthy", "neovascular_AMD"]
    macula_df = create_master_csv(MACULA_ROOT, MACULA_CLASSES, "data/macula_master.csv")
    if not macula_df.empty:
        print("\n--- Macula Model Stats ---")
        print(macula_df['class_name'].value_counts())
        # --- THIS CALLS THE RESTORED FUNCTION ---
        split_data(macula_df, "macula")

    # --- Process OD Model Data ---
    OD_ROOT = os.path.join(DATASET_ROOT, "OD")
    OD_CLASSES = ["Glaucoma", "Healthy"]
    od_df = create_master_csv(OD_ROOT, OD_CLASSES, "data/od_master.csv")
    if not od_df.empty:
        print("\n--- OD Model Stats ---")
        print(od_df['class_name'].value_counts())
        # --- THIS CALLS THE RESTORED FUNCTION ---
        split_data(od_df, "od")

    print("\n--- Data Preparation Complete ---")
    print("You now have 'macula_train.csv', 'macula_val.csv', and 'macula_test.csv'.")