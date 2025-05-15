# === IMPORTS ===
import os
import re
import numpy as np
import pandas as pd
from collections import defaultdict
from itertools import combinations
from scipy.interpolate import UnivariateSpline
import tkinter as tk
from tkinter import filedialog
import warnings

# === FILE READING FUNCTIONS ===
def read_csv_file(file_path):
    df = pd.read_csv(file_path)
    x_col = df.columns[df.columns.str.match(r'X#wcentroid(\s+\(cm\))?')][0]
    y_col = df.columns[df.columns.str.match(r'Y#wcentroid(\s+\(cm\))?')][0]
    angle_col = df.columns[df.columns.str.match(r'ANGLE')][0]
    midline_length_col = df.columns[df.columns.str.match(r'midline_length')][0]
    frame_col = df.columns[df.columns.str.match(r'frame')][0]
    return df[x_col].values, df[y_col].values, df['missing'].astype(bool).values, \
           df[angle_col].values, df[midline_length_col].values, df[frame_col].values

def read_npz_file(file_path):
    with np.load(file_path) as npz:
        keys = npz.files
        X = npz[[k for k in keys if re.match(r'X#wcentroid(\s+\(cm\))?$', k)][0]]
        Y = npz[[k for k in keys if re.match(r'Y#wcentroid(\s+\(cm\))?$', k)][0]]
        ANGLE = npz[[k for k in keys if 'ANGLE' in k][0]]
        midline_length = npz[[k for k in keys if 'midline_length' in k][0]]
        missing = npz[[k for k in keys if 'missing' in k][0]].astype(bool)
        frame_numbers = npz[[k for k in keys if 'frame' in k][0]]
    return X, Y, missing, ANGLE, midline_length, frame_numbers

# === INTERPOLATION LOGIC ===
def interpolate_gap(values, frame_numbers, gap_indices, method='spline'):
    valid = np.isfinite(values)
    x_valid = frame_numbers[valid]
    y_valid = values[valid]
    if len(x_valid) < 4:
        return values
    try:
        if method == 'spline':
            spline = UnivariateSpline(x_valid, y_valid, k=3, s=0)
            interpolated = spline(frame_numbers)
        elif method == 'nearest':
            interpolated = values.copy()
            for idx in gap_indices:
                nearest_idx = x_valid[np.argmin(np.abs(x_valid - frame_numbers[idx]))]
                interpolated[idx] = values[frame_numbers == nearest_idx][0]
        values[gap_indices] = interpolated[gap_indices]
    except Exception as e:
        warnings.warn(f"Interpolation failed: {e}")
    return values

def fill_missing_data_modified(missing_fly_data, frame_rate=30):
    max_short_gap = 4
    max_spline_gap = 15
    total_filled = 0
    total_skipped = 0
    skipped_log = []

    for fly_data in missing_fly_data:
        X, Y, ANGLE = fly_data["X"], fly_data["Y"], fly_data["ANGLE"]
        midline_length, missing, frame_numbers = fly_data["midline_length"], fly_data["missing"], fly_data["frame_numbers"]
        group_id, fly_num = fly_data["group"], fly_data["fly_number"]

        missing_inds = np.where(missing)[0]
        if len(missing_inds) == 0:
            continue
        gaps = np.split(missing_inds, np.where(np.diff(missing_inds) != 1)[0] + 1)

        for gap in gaps:
            gap_len = len(gap)
            if gap_len <= max_short_gap:
                method = 'nearest'
            elif gap_len <= max_spline_gap:
                method = 'spline'
            else:
                total_skipped += gap_len
                skipped_log.append((group_id[0], group_id[1], fly_num, int(frame_numbers[gap[0]]), gap_len))
                continue
            X = interpolate_gap(X, frame_numbers, gap, method)
            Y = interpolate_gap(Y, frame_numbers, gap, method)
            total_filled += gap_len

        ANGLE = interpolate_gap(ANGLE, frame_numbers, missing_inds, method='spline')

        valid_ml = np.where(~np.isnan(midline_length))[0]
        for gap in gaps:
            if len(gap) > max_spline_gap:
                continue
            for idx in gap:
                prev = valid_ml[valid_ml < idx]
                if len(prev) > 0:
                    midline_length[idx] = midline_length[prev[-1]]

        fly_data.update({
            "X": X, "Y": Y, "ANGLE": ANGLE, "midline_length": midline_length,
            "missing": np.zeros_like(missing, dtype=bool)
        })

    print(f"Filled {total_filled} missing values using interpolation.")
    print(f"Skipped {total_skipped} values due to long gaps (>15 frames).")
    return missing_fly_data, skipped_log

def save_skipped_log(skipped_log, output_dir):
    if skipped_log:
        df = pd.DataFrame(skipped_log, columns=['Genotype', 'Sample_Number', 'Fly_Number', 'Start_Frame', 'Gap_Length'])
        df.to_csv(os.path.join(output_dir, "skipped_missing_data_log.csv"), index=False)
        print(f"Saved skipped frames log with {len(skipped_log)} entries.")

# === ANGLE + DISTANCE CALCULATION ===
def calculate_angle_distance_diff_single(df_group):
    fly_numbers = sorted([int(col.split('_')[1]) for col in df_group.columns if col.startswith('X_')])
    angle_diffs, distance_diffs = [], []

    for fly1, fly2 in combinations(fly_numbers, 2):
        angle_diff = np.abs(df_group[f'ANGLE_{fly1}'] - df_group[f'ANGLE_{fly2}'])
        angle_diff = np.where(angle_diff > 180, 360 - angle_diff, angle_diff)
        angle_diff = (180 - angle_diff) / 2
        angle_diffs.append(pd.Series(angle_diff, name=f'angle_diff_{fly1}_{fly2}'))

        dist = np.sqrt(
            (df_group[f'X_{fly1}'] - df_group[f'X_{fly2}'])**2 +
            (df_group[f'Y_{fly1}'] - df_group[f'Y_{fly2}'])**2
        )
        distance_diffs.append(pd.Series(dist, name=f'distance_diff_{fly1}_{fly2}'))

    df_group = pd.concat([df_group] + angle_diffs + distance_diffs, axis=1)
    return df_group

# === SOCIAL INTERACTION ===
def analyze_social_interactions(df, midline_length_thresh, angle_diff_thresh):
    fly_numbers = sorted({int(col.split('_')[1]) for col in df.columns if col.startswith('X_')})
    interactions = []
    for i, j in combinations(fly_numbers, 2):
        dist = df[f'distance_diff_{i}_{j}']
        angle = df[f'angle_diff_{i}_{j}']
        cond = (dist < midline_length_thresh) & (angle.between(-angle_diff_thresh, angle_diff_thresh))
        groups = (cond != cond.shift()).cumsum()
        groups = groups[cond]
        long_groups = groups.value_counts()[groups.value_counts() > 45]
        for grp_id, dur in long_groups.items():
            start_frame = groups[groups == grp_id].index.min()
            interactions.append((i, j, start_frame, dur))
    return pd.DataFrame(interactions, columns=['Fly1', 'Fly2', 'starting_frame', 'duration'])

# === NEAREST FLY DISTANCE ===
def calculate_inter_individual_distances(df):
    fly_numbers = sorted({int(col.split('_')[1]) for col in df.columns if col.startswith('X_')})
    distances = []
    for i, j in combinations(fly_numbers, 2):
        X1, Y1 = df[f'X_{i}'], df[f'Y_{i}']
        X2, Y2 = df[f'X_{j}'], df[f'Y_{j}']
        dist = np.sqrt((X1 - X2)**2 + (Y1 - Y2)**2)
        distances.append(pd.Series(dist, name=f'distance_{i}_{j}'))
    return pd.concat([df['frame']] + distances, axis=1)

def calculate_nearest_fly_distance(df_group_dict):
    out_rows = []
    for group, df in df_group_dict.items():
        dists_df = calculate_inter_individual_distances(df)
        fly_pairs = [col for col in dists_df.columns if col.startswith('distance_')]
        avg_per_frame = dists_df[fly_pairs].mean(axis=1)
        avg_distance = avg_per_frame.mean()
        out_rows.append({
            'Genotype': group[0],
            'Sample_Number': group[1],
            'Average_Nearest_Fly_Distance': avg_distance
        })
    return pd.DataFrame(out_rows)

# === MAIN ===
def main():
    root = tk.Tk(); root.withdraw()
    data_dir = filedialog.askdirectory()
    output_dir = os.path.join(data_dir, 'output')
    os.makedirs(output_dir, exist_ok=True)

    data_files = [f for f in os.listdir(data_dir) if f.endswith((".npz", ".csv"))]
    df_group = defaultdict(list)
    all_interactions = []
    all_angles_distances = []
    missing_fly_data = []
    pixel_cm_ratio = 0.0042

    for filename in data_files:
        match = re.search(r"^[^_]+_([\w+]+)_(\d+)_fish(\d+)", filename)
        if not match:
            print(f"Skipping malformed filename: {filename}")
            continue
        genotype, sample_number, fly_number = match.groups()
        group = (genotype, int(sample_number))
        path = os.path.join(data_dir, filename)

        if filename.endswith(".npz"):
            X, Y, missing, ANGLE, midline_length, frame_numbers = read_npz_file(path)
        else:
            X, Y, missing, ANGLE, midline_length, frame_numbers = read_csv_file(path)

        ANGLE = abs(np.degrees(ANGLE))
        midline_length *= pixel_cm_ratio

        missing_fly_data.append({
            "group": group, "fly_number": int(fly_number),
            "X": X, "Y": Y, "missing": missing,
            "ANGLE": ANGLE, "midline_length": midline_length,
            "frame_numbers": frame_numbers
        })

    print("Interpolating missing data...")
    missing_fly_data, skipped_log = fill_missing_data_modified(missing_fly_data)
    save_skipped_log(skipped_log, output_dir)

    for fly in missing_fly_data:
        group, fn = fly["group"], fly["fly_number"]
        df = pd.DataFrame({
            "frame": fly["frame_numbers"],
            f"X_{fn}": fly["X"], f"Y_{fn}": fly["Y"],
            f"ANGLE_{fn}": fly["ANGLE"], f"midline_length_{fn}": fly["midline_length"]
        })
        df_group[group].append(df)

    for group, dfs in df_group.items():
        merged = dfs[0]
        for df in dfs[1:]:
            merged = pd.merge(merged, df, on="frame", how="outer")
        df_group[group] = calculate_angle_distance_diff_single(merged)
        df_group[group]['Genotype'], df_group[group]['Sample_Number'] = group
        all_angles_distances.append(df_group[group])

        interactions = analyze_social_interactions(df_group[group], midline_length_thresh=0.5, angle_diff_thresh=90)
        interactions['Genotype'], interactions['Sample_Number'] = group
        all_interactions.append(interactions)

    pd.concat(all_angles_distances).to_csv(os.path.join(output_dir, "all_angles_distances.csv"), index=False)
    pd.concat(all_interactions).to_csv(os.path.join(output_dir, "raw_social_interactions.csv"), index=False)

    nearest_df = calculate_nearest_fly_distance(df_group)
    nearest_df.to_csv(os.path.join(output_dir, "nearest_fly_distances.csv"), index=False)
    print("Saved average nearest-fly distances.")

    print(f"Analysis complete. Results saved to: {output_dir}")

if __name__ == "__main__":
    main()