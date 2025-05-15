import numpy as np
import os
import re
from collections import defaultdict
from tkinter import Tk
from tkinter import filedialog
import cv2
from shapely.geometry import Point, Polygon, MultiPoint
from scipy import interpolate

# Remember that this approach assumes the arena is a perfect rectangle and that the camera angle does not distort the image. 
def draw_arena(image_path, arena_width, crop_values):
    img = cv2.imread(image_path)
    h, w, _ = img.shape

    # Apply cropping values to the image
    x1 = int(w * crop_values[0])
    x2 = int(w * (1 - crop_values[2]))
    y1 = int(h * crop_values[1])
    y2 = int(h * (1 - crop_values[3]))
    cropped_img = img[y1:y2, x1:x2]

    # start from top right, and counter clockwise !!!

    arena_points = []

    def draw_polygon(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(arena_points) < 4:
                arena_points.append((x, y))
                cv2.circle(cropped_img, (x, y), 3, (0, 255, 0), -1)
                if len(arena_points) == 4:
                    cv2.polylines(cropped_img, [np.array(arena_points)], True, (0, 255, 0), 2)

    cv2.namedWindow('Select Rectangle Corners')
    cv2.setMouseCallback('Select Rectangle Corners', draw_polygon)

    while True:
        cv2.imshow('Select Rectangle Corners', cropped_img)
        if cv2.waitKey(1) & 0xFF == ord('q') or len(arena_points) == 4:
            break
    cv2.destroyAllWindows()

    if len(arena_points) != 4:
        print("Failed to get four points for the rectangle. Exiting.")
        return None, None

    # Calculate the pixel-to-cm ratio using the width (shortest side)
    widths = [np.linalg.norm(np.array(arena_points[i]) - np.array(arena_points[(i+1) % 4])) for i in range(4)]
    shortest_side_px = min(widths)
    calculated_centimeter_per_pixel = arena_width / shortest_side_px

    # Calculate the width reduction in pixels
    width_reduction_px = shortest_side_px / 2

    # Calculate the width and length of the arena
    width_of_arena_px = np.linalg.norm(np.array(arena_points[1]) - np.array(arena_points[0]))
    length_of_arena_px = np.linalg.norm(np.array(arena_points[3]) - np.array(arena_points[0]))

    # Calculate the reduction in pixels (half the width of the arena)
    width_reduction_px = width_of_arena_px / 2

    # Determine the scale factors based on the reduction in pixels
    width_scale_factor = 1 - (width_reduction_px / width_of_arena_px)
    length_scale_factor = 1 - (width_reduction_px / length_of_arena_px)

    # Calculate the center region
    center_point = np.mean(arena_points, axis=0)
    center_region_points = []

    for point in arena_points:
        vector = np.array(point) - center_point
        if point[0] > center_point[0]:  # Points on the right half
            new_x = center_point[0] + vector[0] * width_scale_factor
        else:  # Points on the left half
            new_x = center_point[0] + vector[0] * width_scale_factor
        
        if point[1] > center_point[1]:  # Points on the bottom half
            new_y = center_point[1] + vector[1] * length_scale_factor
        else:  # Points on the top half
            new_y = center_point[1] + vector[1] * length_scale_factor

        new_point = np.array([new_x, new_y])
        center_region_points.append(new_point.astype(int))



    # Draw the center region
    cv2.polylines(cropped_img, [np.array(center_region_points)], True, (0, 0, 255), 2)
    cv2.imshow('Arena with Center Region', cropped_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Save the image with the arena and center region drawn
    cv2.imwrite("image_with_center_region.png", cropped_img)

    print(f"Calculated centimeter per pixel: {calculated_centimeter_per_pixel}")

    return calculated_centimeter_per_pixel, np.array(center_region_points, dtype=int)

#Function to load npz file
def load_npz_file(file_path):
    with np.load(file_path, allow_pickle=True) as data:
        return {key: data[key] for key in data.keys()}

#Function to save combined npz file
def save_combined_npz(fly_data, file_path):
    np.savez(file_path, **fly_data)

def convert_cm_trex_to_real(cm, wrong_ratio, actual_ratio):
    # Convert from cm to pixels using the wrong ratio
    pixels = cm * wrong_ratio
    
    # Convert from pixels back to cm using the actual ratio
    cm_converted_back = pixels / actual_ratio
    
    return cm_converted_back


def fix_lost_tracking(experiment_data, max_gap_forward_fill=30, max_gap_interpolation=60):
    fixed_data = {}
    for fly_identifier, data in experiment_data.items():
        num_fixed = 0

        # Assume 'missing' is a boolean array indicating missing data
        missing = np.isinf(data['X#wcentroid']) | np.isinf(data['Y#wcentroid']) | np.isinf(data['midline_length']) | np.isinf(data['ANGLE'])
        missing_indices = np.where(missing)[0]

        if missing_indices.size > 0:
            num_fixed += len(missing_indices)  # Increment num_fixed by the number of missing data points

            # Convert angles to degrees and handle missing data
            data['angle_degrees'] = np.rad2deg(data['ANGLE'])

            # Convert angles to Cartesian coordinates for interpolation
            data['cos_angle'] = np.cos(np.deg2rad(data['angle_degrees']))
            data['sin_angle'] = np.sin(np.deg2rad(data['angle_degrees']))

            # Perform interpolation for missing data
            for column in ['X#wcentroid', 'Y#wcentroid', 'midline_length', 'cos_angle', 'sin_angle']:
                valid_indices = np.where(~missing)[0]
                spline = interpolate.InterpolatedUnivariateSpline(valid_indices, data[column][valid_indices])
                data[column][missing_indices] = spline(missing_indices)

            # Convert back to angles
            data['angle_degrees'] = np.rad2deg(np.arctan2(data['sin_angle'], data['cos_angle']))
            data['angle_degrees'] = data['angle_degrees'] % 360

            # Update 'ANGLE' with interpolated values
            data['ANGLE'] = np.deg2rad(data['angle_degrees'])
            del data['cos_angle'], data['sin_angle'], data['angle_degrees']

        print(f"For {fly_identifier}, fixed {num_fixed} instances of missing data.")
        data['missing'][:] = False  # Update the 'missing' array
        fixed_data[fly_identifier] = data

    return fixed_data



# Main Code

# Define values of experiment and used in TREX in cm
arena_width = 4
trex_cm_per_pixel = 0.00418
crop_values = [0,0,0,0]

# GUI to select files
root = Tk()
root.withdraw()

# Select npz files
file_paths = filedialog.askopenfilenames(title='Select npz Files', filetypes=[("NPZ files", "*.npz")])

# Since all npz files are assumed to be in the same directory, we can extract the directory like this:
npz_file_directory = os.path.dirname(file_paths[0])

if not file_paths:
    print("No files selected, exiting.")
    exit()

# Load background image and get pixel_to_cm_ratio
image_path = filedialog.askopenfilename(title="Select Background Image")
background_image = cv2.imread(image_path)
video_height, video_width, _ = background_image.shape

print(f"Video Width: {video_width} pixels")
print(f"Video Height: {video_height} pixels")

# Use the function with your image path and actual measurements
calculated_cm_per_pixel, center_area_points = draw_arena(image_path, arena_width, crop_values)

# Convert center area points to cm
center_area_points_cm = [(x * calculated_cm_per_pixel, y * calculated_cm_per_pixel) for x, y in center_area_points]
center_polygon = Polygon(center_area_points_cm)

# Parse filenames and load data
experiment_data = defaultdict(list)
experiments = {}

# Main processing and saving loop
for file_path in file_paths:
    filename = os.path.basename(file_path)
    match = re.match(r"([A-Za-z0-9-+]+_\d+)_fish(\d+).npz", filename)
    if match:
        experiment_id, individual_fly = match.groups()
        fly_identifier = f"{experiment_id}_fish{individual_fly}"  # Correctly formatted identifier

        data = load_npz_file(file_path)

        # Correct positions
        data['X#wcentroid'] = convert_cm_trex_to_real(data['X#wcentroid'], trex_cm_per_pixel, calculated_cm_per_pixel)
        data['Y#wcentroid'] = convert_cm_trex_to_real(data['Y#wcentroid'], trex_cm_per_pixel, calculated_cm_per_pixel)

        # Check if the individual is within the center area for each frame and assign 1 or 0
        inside_center_area = [1 if center_polygon.contains(Point(x, y)) else 0 
                              for x, y in zip(data['X#wcentroid'], data['Y#wcentroid'])]
        data['inside_center_area'] = np.array(inside_center_area, dtype=np.int8)

        # Store the modified data
        experiment_data[fly_identifier] = data

# Fix lost tracking and save fixed data
fixed_data = fix_lost_tracking(experiment_data)

for fly_identifier, fly_data in fixed_data.items():
    output_dict = {
        'X#wcentroid': fly_data['X#wcentroid'],
        'Y#wcentroid': fly_data['Y#wcentroid'],
        'frame': fly_data['frame'],
        'inside_center_area': fly_data['inside_center_area'],
        'missing': fly_data['missing'],
        'ANGLE': fly_data['ANGLE'],
        'midline_length': fly_data['midline_length']
    }

    output_file_path = os.path.join(npz_file_directory, f'fixed_{fly_identifier}.npz')
    save_combined_npz(output_dict, output_file_path)

print("Done.")

