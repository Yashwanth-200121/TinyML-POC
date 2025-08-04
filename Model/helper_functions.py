import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def merge_chunks_and_aggregate_data_by_type(data_chunks, output_dir):
    """
    Merges CSV chunks for each type and writes final CSVs.

    Parameters:
    - data_chunks: dict
        A dictionary where keys are types and values are lists of CSV file paths.
    - output_dir: str
        Directory to save the final merged and aggregated CSVs.
    """
    for data_type, file_list in data_chunks.items():
        merged_df = pd.DataFrame()
        for file_path in file_list:
            try:
                df = pd.read_csv(file_path)
                merged_df = pd.concat([merged_df, df], ignore_index=True)
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
        
        groupby_cols = []
        agg_dict = {}

        if data_type == 'Battery':
            groupby_cols = ['sn_masked', 'batterysn', 'ctnumber', 'issue']
            agg_dict = {'designcapacity': 'median',
                        'fullchargecapacity': 'median',
                        'remainingcapacity': 'median',
                        'maxerrorpercentage': 'median',
                        'cyclecount': 'median',
                        'temperature': 'median',
                        'voltage': 'median',
                        'batterycurrent': 'median',
                        'designvoltage': 'median',
                        'cellvoltage1': 'median',
                        'cellvoltage2': 'median',
                        'cellvoltage3': 'median',
                        'cellvoltage4': 'median',
                        'chargerate': 'median',
                        'dischargerate': 'median'}
        elif data_type == 'System':
            groupby_cols = ['sn_masked', 'issue']
            agg_dict = {'max_maxcpu': 'median',
                        'max_avgcpu': 'median',
                        'avg_maxcpu': 'median',
                        'avg_avgcpu': 'median',
                        'max_maxram': 'median',
                        'max_avgram': 'median',
                        'avg_maxram': 'median',
                        'avg_avgram': 'median',
                        'avg_perctimewithinternet': 'mean',
                        'median_perctimewithinternet': 'median',
                        'avg_perctimeonbatteries': 'mean',
                        'median_perctimeonbatteries': 'median',
                        'ssmcount': 'median'}
        else: # Thermal
            groupby_cols = ['sn_masked', 'issue']
            agg_dict = {'Battery_med_temp': 'median',
                        'CPU_med_temp': 'median',
                        'Fan_med_temp': 'median',
                        'GPU_med_temp': 'median',
                        'GPU_10DE_med_temp': 'median',
                        'GPU_1002_med_temp': 'median',
                        'Battery_max_temp': 'median',
                        'CPU_max_temp': 'median',
                        'Fan_max_temp': 'median',
                        'GPU_max_temp': 'median',
                        'GPU_10DE_max_temp': 'median',
                        'GPU_1002_max_temp': 'median',
                        'Battery_min_temp': 'median',
                        'CPU_min_temp': 'median',
                        'Fan_min_temp': 'median',
                        'GPU_min_temp': 'median',
                        'GPU_10DE_min_temp': 'median',
                        'GPU_1002_min_temp': 'median'}
        
        try:
            aggregated_df = merged_df.groupby(groupby_cols).agg(agg_dict).reset_index()
        except Exception as e:
            print(f"Aggregation failed for {data_type}: {e}")
            continue

        # Removing duplicate sn_masked records due to sn_masked - batterysn combo
        if data_type == 'Battery':
            srnos_with_multiple_rows = aggregated_df['sn_masked'].value_counts()
            srnos_with_multiple_rows = srnos_with_multiple_rows[srnos_with_multiple_rows > 1].index.tolist()
            aggregated_df = aggregated_df[~aggregated_df['sn_masked'].isin(srnos_with_multiple_rows)]

        output_path = f"{output_dir}/{data_type}.csv"
        output_path_agg = f"{output_dir}/{data_type}_agg.csv"
        merged_df.to_csv(output_path, index=False)
        aggregated_df.to_csv(output_path_agg, index=False)
        print(f"Merged and Aggregated CSVs for {data_type} written to: {output_path}")


def process_aggregated_files(input_dir, output_file_name, operation, join_on=None):
    """
    Reads aggregated CSVs from a directory, performs a specified operation,
    and saves the output file.

    Parameters:
    - input_dir: str
        Directory containing aggregated CSV files.
    - output_file_name: str
        Name of the output CSV file to be saved (without path).
    - operation: str
        'concat' or 'merge'. If 'merge', join_on must be specified.
    - join_on: str or list, optional
        Column(s) to join on when performing merge (only used if operation='merge').
    """
    csv_files = [f for f in os.listdir(input_dir) if f.endswith("_agg.csv")]
    
    if not csv_files:
        raise ValueError(f"No aggregated CSV files found in {input_dir}.")

    dataframes = []
    for file in csv_files:
        df = pd.read_csv(os.path.join(input_dir, file))
        dataframes.append(df)

    if operation == 'concat':
        final_df = pd.concat(dataframes, ignore_index=True)
    elif operation == 'merge':
        if not join_on:
            raise ValueError("join_on parameter must be specified for merge operation.")
        
        final_df = dataframes[0]
        for df in dataframes[1:]:
            final_df = final_df.merge(df, on=join_on, how='inner')
            # Drop the duplicate columns from the right dataframe after merge
            final_df = final_df.loc[:,~final_df.columns.duplicated()]

    else:
        raise ValueError("Unsupported operation. Choose 'concat' or 'merge'.")

    output_path = os.path.join(input_dir, output_file_name)
    final_df.to_csv(output_path, index=False)
    print(f"Final model data written to: {output_path}")


def try_divide(numerator, denominator):
    return np.where((denominator == 0) | (pd.isnull(denominator)), np.nan, numerator / denominator)


def add_features(input_file, output_dir):
    """
    Loads data, adds new features, and saves the updated data.

    Parameters:
    - input_file: str
        Path to the input CSV file.
    - output_dir: str
        Directory to save the updated CSV.
    """
    data = pd.read_csv(input_file)
    data["batterycapacity__perc"] = np.round(
        try_divide(data["fullchargecapacity"], data["designcapacity"]) * 100, 2)

    data["designvoltage_volts_perc"] = np.round(
        try_divide(data["voltage"], data["designvoltage"]) * 100, 2)

    data["Fan_max_rpm"] = np.where(
        data["Fan_max_temp"].isin([0, 255]) | data["Fan_max_temp"].isna(), 0, (3932160 * 2) / (data["Fan_max_temp"] * 32))
    data["Fan_max_rpm"] = data["Fan_max_rpm"].round(2)
    
    output_path = os.path.join(output_dir, "updated_model_data.csv")
    data.to_csv(output_path, index=False)
    print(f"Updated model data written to: {output_path}")


def load_model_data(file_path: str, cols_to_remove: list, target_column: str):
    """Loads the CSV file and returns features and target."""
    df = pd.read_csv(file_path)
    X = df.drop(columns=cols_to_remove)
    y = df[target_column]
    return X, y

def split_data(X, y, test_size=0.2, random_state=42):
    """Splits data into train and test sets."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def train_random_forest(X_train, y_train, n_estimators=100):
    """Trains a Random Forest classifier."""
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """Prints accuracy, confusion matrix, and classification report."""
    y_pred = model.predict(X_test)
    print("\nModel Evaluation Metrics:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
