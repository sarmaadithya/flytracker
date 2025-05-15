import os
import tkinter as tk
from tkinter import filedialog
from tkinter.simpledialog import askstring, askfloat
import numpy as np
import pandas as pd
import re

def process_npz_files(directory, expected_rows, threshold_percentage):
    npz_files_below_threshold = []
    issue_counts = []

    for file in os.listdir(directory):
        if file.endswith('.npz'):
            file_path = os.path.join(directory, file)
            with np.load(file_path, allow_pickle=True) as npz:
                frame_count = len(npz[npz.files[0]])  # Assuming the first file key has the frame count
                if frame_count < expected_rows * (threshold_percentage / 100):
                    npz_files_below_threshold.append(file)
                
                issues = {'X': {'nan': 0, 'inf': 0}, 
                          'Y': {'nan': 0, 'inf': 0}, 
                          'ANGLE': {'nan': 0, 'inf': 0}}
                
                keys = npz.files
                X_key = [key for key in keys if re.match(r'X#wcentroid(\s+\(cm\))?$', key)][0]
                Y_key = [key for key in keys if re.match(r'Y#wcentroid(\s+\(cm\))?$', key)][0]
                ANGLE_key = [key for key in keys if 'ANGLE' in key][0]

                X = npz[X_key]
                Y = npz[Y_key]
                ANGLE = npz[ANGLE_key]

                # Count NaN and Inf values in X, Y, ANGLE
                issues['X']['nan'] = np.isnan(X).sum()
                issues['X']['inf'] = np.isinf(X).sum()
                issues['Y']['nan'] = np.isnan(Y).sum()
                issues['Y']['inf'] = np.isinf(Y).sum()
                issues['ANGLE']['nan'] = np.isnan(ANGLE).sum()
                issues['ANGLE']['inf'] = np.isinf(ANGLE).sum()

                total_issues = sum(issues[column][issue_type] for column in issues for issue_type in issues[column])
                if total_issues > 0:
                    issue_counts.append((file, issues['X']['nan'], issues['X']['inf'], issues['Y']['nan'], issues['Y']['inf'], issues['ANGLE']['nan'], issues['ANGLE']['inf']))

    return npz_files_below_threshold, issue_counts

def save_to_csv(file_list, directory, file_name, column_names):
    if file_list:
        df = pd.DataFrame(file_list, columns=column_names)
        output_file = os.path.join(directory, file_name)
        df.to_csv(output_file, index=False)
        print(f"CSV file '{file_name}' has been saved to {output_file}")
    else:
        print(f"No data to save for '{file_name}'.")

def main():
    root = tk.Tk()
    root.withdraw()  # Hide the main window

    directory = filedialog.askdirectory(title='Select Folder Containing NPZ Files')
    if directory:
        expected_rows = askstring("Input", "Enter the total expected number of rows for each NPZ:")
        threshold_percentage = askfloat("Input", "Enter the threshold percentage:")

        if expected_rows and threshold_percentage:
            expected_rows = int(expected_rows)
            npz_files_below_threshold, issue_counts = process_npz_files(directory, expected_rows, threshold_percentage)
            save_to_csv(npz_files_below_threshold, directory, 'npz_files_below_threshold.csv', ['npz_files'])
            save_to_csv(issue_counts, directory, 'npz_files_with_issues.csv', ['npz_file', 'X_nan', 'X_inf', 'Y_nan', 'Y_inf', 'ANGLE_nan', 'ANGLE_inf'])
        else:
            print("Operation cancelled. Please provide the expected rows and threshold percentage.")
    else:
        print("Operation cancelled. Please select a directory.")

if __name__ == "__main__":
    main()
