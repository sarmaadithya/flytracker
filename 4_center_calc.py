import os
import re
import numpy as np
import csv
from tkinter import filedialog
from tkinter import Tk

# Function to calculate the mean frame count spent in the center for each experiment
def calculate_mean_per_experiment(directory):
    experiment_data = {}

    # Regular expression pattern to match the file names
    pattern = re.compile(r"fixed_([A-Za-z0-9-+]+)_(\d+)_fish\d+.npz")

    # List all npz files in the given directory
    for file_name in os.listdir(directory):
        if file_name.endswith('.npz'):
            match = pattern.match(file_name)
            if match:
                first_sequence, second_sequence = match.groups()
                # Combine the first and second sequences to form the experiment ID
                experiment_id = f"{first_sequence}_{second_sequence}"
                # Print the file name being processed
                print(f"Processing file: {file_name}")
                # Load data from npz file
                data = np.load(os.path.join(directory, file_name))
                inside_center_area = data['inside_center_area']
                
                # Calculate and print the count of frames spent in the center
                center_frames_count = np.sum(inside_center_area == 1)
                print(f"Frames spent in center for {file_name}: {center_frames_count}")
                
                # Add the count of frames spent in the center to the respective experiment
                experiment_data.setdefault(experiment_id, []).append(np.sum(inside_center_area == 1))
            else:
                # Print file names that do not match the pattern (optional)
                print(f"Skipped file (does not match pattern): {file_name}")    
    # Calculate the mean frame count for each experiment
    means = {experiment: np.mean(counts) for experiment, counts in experiment_data.items()}
    return means

# Function to write the results to a CSV file
def write_results_to_csv(means, output_file):
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Experiment ID', 'Mean Frames Spent in Center'])
        for experiment_id, mean_count in means.items():
            writer.writerow([experiment_id, mean_count])

# Main function to select folder and process files
def process_folder():
    root = Tk()
    root.withdraw()  # Hide the main window
    directory = filedialog.askdirectory(title='Select Folder Containing NPZ Files')

    if directory:
        means = calculate_mean_per_experiment(directory)
        output_file = os.path.join(directory, 'experiment_means_in_center.csv')
        write_results_to_csv(means, output_file)
        print(f"Results written to {output_file}")
    else:
        print("No folder was selected.")

# Run the folder processing function
process_folder()
