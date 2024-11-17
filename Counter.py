import os
import json
import pandas as pd

def count_2_5s_windows_with_overlap(file_path):
    """
    Counts 2.5-second overlapping windows with 75% overlap in a single JSON file,
    ignoring gaps larger than 2.5 seconds.
    """
    with open(file_path, 'r') as f:
        data = json.load(f)

    all_timestamps = []
    for sensor, readings in data.items():
        df = pd.DataFrame(readings)

        if 't' not in df.columns:
            print(f"Skipping {file_path} - Missing 't' column in sensor data for {sensor}.")
            continue

        all_timestamps.extend(df['t'].tolist())

    if not all_timestamps:
        return 0

    all_timestamps = sorted(all_timestamps)
    window_count = 0
    start_time = all_timestamps[0]
    window_duration = 2500  # 2.5 seconds in milliseconds
    step_size = int(window_duration * 0.25)  # 25% of 2.5 seconds = 625 ms

    i = 0
    while start_time <= all_timestamps[-1]:
        end_time = start_time + window_duration
        # Check if there are timestamps within the current window
        window_data = [t for t in all_timestamps if start_time <= t < end_time]

        if window_data:
            window_count += 1
            start_time += step_size  # Move start by 625 ms for 75% overlap
        else:
            # If no data is found in the window, move start to the next available timestamp
            next_data_index = next((j for j in range(i, len(all_timestamps)) if all_timestamps[j] >= end_time), None)
            if next_data_index is not None:
                start_time = all_timestamps[next_data_index]
                i = next_data_index  # Update loop index
            else:
                break  # Exit loop if no more data is available after the gap

    return window_count

def count_2_5s_windows_in_folder_with_overlap(folder_path):
    """
    Counts all 2.5-second overlapping windows across all JSON files in a specified folder.
    """
    total_windows = 0
    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            file_path = os.path.join(folder_path, filename)
            file_window_count = count_2_5s_windows_with_overlap(file_path)
            print(f"{filename}: {file_window_count} windows")
            total_windows += file_window_count

    print(f"Total 2.5-second overlapping windows across all files: {total_windows}")
    return total_windows

# Example usage
folder_path = './SEGMENTED_DATA/CTL-PSANO/WALKING'  # Replace with the path to your folder
count_2_5s_windows_in_folder_with_overlap(folder_path)
