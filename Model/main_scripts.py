import os
from helper_functions import (
    merge_chunks_and_aggregate_data_by_type,
    process_aggregated_files,
    add_features,
    load_model_data,
    split_data,
    train_random_forest,
    evaluate_model
)

# Define data chunks and output directory
data_chunks = {
    "Battery": ["Data/battery_mock_data_collection_agg.csv", "Data/battery_mock_data_collection_agg1.csv", "Data/battery_mock_data_collection_agg2.csv", "Data/battery_mock_data_collection_agg3.csv"],
    "System": ["Data/system_mock_data_collection_agg.csv", "Data/system_mock_data_collection_agg1.csv", "Data/system_mock_data_collection_agg2.csv", "Data/system_mock_data_collection_agg3.csv"],
    "Thermal": ["Data/thermal_mock_data_collection_agg 1.csv", "Data/thermal_mock_data_collection_agg1 1.csv"]
}

output_dir = "Output_files"

# Create the output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Created output directory: {output_dir}")

# --- Data Processing and Feature Engineering ---

# 1. Merge chunks and aggregate data by type
print("\n--- Step 1: Merging and Aggregating Data ---")
merge_chunks_and_aggregate_data_by_type(data_chunks, output_dir)

# 2. Process aggregated files (merge them into a single file)
print("\n--- Step 2: Processing Aggregated Files (Merging) ---")
process_aggregated_files(input_dir=output_dir, output_file_name='model_data.csv', operation='merge', join_on='sn_masked')

# 3. Add new features
print("\n--- Step 3: Adding New Features ---")
add_features(input_file=os.path.join(output_dir, "model_data.csv"), output_dir=output_dir)

# --- Machine Learning Model ---

# Define parameters for model
model_data_file = os.path.join(output_dir, "updated_model_data.csv")
target_column = "issue_x"
columns_to_remove = ["sn_masked", "batterysn", "ctnumber", "issue_x","issue_y","issue"]

# 1. Load model data
print("\n--- Step 4: Loading Model Data ---")
X, y = load_model_data(model_data_file, columns_to_remove,target_column)
print(f"Features (X) shape: {X.shape}")
print(f"Target (y) shape: {y.shape}")

# 2. Split data into training and testing sets
print("\n--- Step 5: Splitting Data ---")
X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=42)
print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

# 3. Train Random Forest model
print("\n--- Step 6: Training Random Forest Model ---")
model = train_random_forest(X_train, y_train, n_estimators=100)
print("Random Forest Model training complete.")

# 4. Evaluate the model
print("\n--- Step 7: Evaluating Model ---")
evaluate_model(model, X_test, y_test)