import numpy as np
import os
import pandas as pd
import tkinter as tk
from tkinter import filedialog
import re
from tqdm import tqdm  # Import the tqdm library
from sklearn.linear_model import LinearRegression

#calculate speed and also enable a smoothing function
#set windows size to change smoothing window
#set FPS to the native video FPS
def calculate_speed(i, x_coords, y_coords, smoothing=False, window_size=15, fps=30):

    if i == 0:
        return 0

    if smoothing and i >= window_size:
        start_index = i - window_size + 1
        end_index = i + 1

        x_window = x_coords[start_index:end_index].copy()
        y_window = y_coords[start_index:end_index].copy()

        # Replace infinite values with the values from the previous frame
        for j in range(len(x_window)):
            if np.isinf(x_window[j]):
                x_window[j] = x_window[j-1] if j > 0 else 0
            if np.isinf(y_window[j]):
                y_window[j] = y_window[j-1] if j > 0 else 0

        time_window = np.arange(start_index, end_index).reshape(-1, 1)

        lr_x = LinearRegression().fit(time_window, x_window)
        lr_y = LinearRegression().fit(time_window, y_window)

        x1, x2 = lr_x.predict([[i-1], [i]])
        y1, y2 = lr_y.predict([[i-1], [i]])

    else:
        x1, y1 = x_coords[i-1], y_coords[i-1]
        x2, y2 = x_coords[i], y_coords[i]
        
        # Replace inf values with values from the previous frame
        if np.isinf(x2):
            x2 = x1
        if np.isinf(y2):
            y2 = y1

    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2) * fps

#T he average speed is calculated for all individuals within the same group categorized by experiment, replicate
def calculate_average_speed(results):
    final_results = []
    for experiment, exp_data in results.items():
        for replicate, frame_data in exp_data.items():
                total_speed = 0
                total_frames = 0
                
                for frame, speeds in frame_data.items():
                    # Remove inf values and consider only finite values
                    finite_speeds = [speed for speed in speeds if np.isfinite(speed)]
                    
                    total_speed += sum(finite_speeds)
                    total_frames += len(finite_speeds)

                # Calculate average speed for this replicate
                average_speed_per_replicate = total_speed / total_frames if total_frames else 0
                
                final_results.append({
                    "Experiment": experiment,
                    "Replicate": replicate,
                    "Average Speed": average_speed_per_replicate
                })

    df = pd.DataFrame(final_results)
    return df

### MAIN SCRIPT ###

# Create a root Tk window and hide it
root = tk.Tk()
root.withdraw()

# Ask the user for the source folder containing NPZ files
source_folder = filedialog.askdirectory(title="Select Source Folder with NPZ files")

# Ask the user for the destination folder to save CSV outputs
destination_folder = filedialog.askdirectory(title="Select Destination Folder for CSV outputs")

# Define the regular expression pattern to extract replicate number
replicate_pattern = re.compile(r'_\d+_')

# Recursively process NPZ files in the source folder and subfolders
for root_dir, subdirs, files in os.walk(source_folder):
    results = {}  # Initialize results dictionary
    
    for filename in tqdm(files, desc=f"Processing Files in {os.path.basename(root_dir)}"):
        if filename.endswith('.npz') and 'fish' in filename:
            parts = filename[:-4].split('_')
            experiment = parts[1]  # Extracting experiment number
            replicate = int(parts[2])  # Extracting replicate number

            # Load the .npz file
            data = np.load(os.path.join(root_dir, filename))
            
            # Extract X and Y coordinates of the centroid
            x_coords = data['X#wcentroid']
            y_coords = data['Y#wcentroid']
            
            # Calculate speed for each frame
            speed = [calculate_speed(i, x_coords, y_coords, smoothing=True, window_size=3, fps=30) for i in range(len(x_coords))]
            
            # Store the data in the nested dictionary
            if experiment not in results:
                results[experiment] = {}
            if replicate not in results[experiment]:
                results[experiment][replicate] = {}
            for frame, speed_value in enumerate(speed, start=1):
                if frame not in results[experiment][replicate]:
                    results[experiment][replicate][frame] = []
                results[experiment][replicate][frame].append(speed_value)


    # Calculate average speed per frame and replicate and save CSV output for each subfolder
    if results:
        df = calculate_average_speed(results)
        output_filename = os.path.join(destination_folder, f'{os.path.basename(root_dir)}_average_speed_per_replicate.csv')
        df.to_csv(output_filename)
        print("Average speed per replicate saved to", output_filename)

print("Processing complete.")
