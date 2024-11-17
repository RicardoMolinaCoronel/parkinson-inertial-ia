import json
import os
from pathlib import Path
import copy

def read_turn_segments(txt_path):
    """
    Read the txt file containing turn segments and return a dictionary
    with filenames as keys and list of turn ranges as values.
    The txt file alternates between filename and segment lines.
    """
    turn_segments = {}
    
    with open(txt_path, 'r') as f:
        lines = [line.strip() for line in f.readlines()]
    
    # Remove empty lines
    lines = [line for line in lines if line]
    
    print("\nDebug: Reading turn segments file")
    print(f"Total lines after removing empty lines: {len(lines)}")
    
    # Process lines in pairs (filename followed by segments)
    for i in range(0, len(lines), 2):
        if i + 1 >= len(lines):  # Skip incomplete pairs at the end
            print(f"Warning: Skipping incomplete pair at end of file")
            break
            
        filename = lines[i].strip()
        segments_line = lines[i + 1].strip()
        
        print(f"\nProcessing entry:")
        print(f"Filename: '{filename}'")
        print(f"Segments: '{segments_line}'")
        
        if not filename or not segments_line:  # Skip empty lines
            print("Warning: Empty filename or segments line, skipping")
            continue
        
        segments = segments_line.split(',')
        turn_ranges = []
        
        for segment in segments:
            segment = segment.strip()
            if '-' in segment:
                try:
                    start, end = map(int, segment.split('-'))
                    turn_ranges.append((start, end))
                except ValueError as e:
                    print(f"Warning: Could not parse segment '{segment}' for file {filename}")
        
        turn_segments[filename] = turn_ranges
        print(f"Added {len(turn_ranges)} turn ranges for {filename}")
    
    print(f"\nTotal entries read: {len(turn_segments)}")
    print("Files in turn_segments dictionary:")
    for fname in sorted(turn_segments.keys()):
        print(f"  '{fname}'")
    
    return turn_segments

def is_in_turn_segment(t, turn_segments):
    """Check if a given timestamp is within any turn segment"""
    for start, end in turn_segments:
        if start <= t <= end:
            return True
    return False

def separate_imu_data(json_path, turn_segments):
    """
    Separate the IMU data into straight and turn segments
    Returns two lists: one for straight segments and one for turns
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Initialize containers for separated data
    straight_data = {
        "derecha": [],
        "izquierda": [],
        "espina_base": []
    }
    
    turn_data = []
    current_turn = None
    
    # Initialize a template for new turn segments
    turn_template = {
        "derecha": [],
        "izquierda": [],
        "espina_base": []
    }
    
    # Go through each measurement in derecha
    for sensor_data in data["derecha"]:
        t = sensor_data["t"]
        is_turn = is_in_turn_segment(t, turn_segments)
        
        if is_turn:
            # If we're not already in a turn, start a new one
            if current_turn is None:
                current_turn = copy.deepcopy(turn_template)
            
            # Add data to current turn
            current_turn["derecha"].append(sensor_data)
            
            # Find corresponding data points in other sensors
            matching_time = sensor_data["millis"]
            for izq_data in data["izquierda"]:
                if izq_data["millis"] == matching_time:
                    current_turn["izquierda"].append(izq_data)
                    break
            
            for esp_data in data["espina_base"]:
                if esp_data["millis"] == matching_time:
                    current_turn["espina_base"].append(esp_data)
                    break
                    
        else:
            # If we were in a turn and now we're not, save the turn
            if current_turn is not None:
                turn_data.append(current_turn)
                current_turn = None
            
            # Add data to straight segments
            straight_data["derecha"].append(sensor_data)
            
            # Find corresponding data points in other sensors
            matching_time = sensor_data["millis"]
            for izq_data in data["izquierda"]:
                if izq_data["millis"] == matching_time:
                    straight_data["izquierda"].append(izq_data)
                    break
                    
            for esp_data in data["espina_base"]:
                if esp_data["millis"] == matching_time:
                    straight_data["espina_base"].append(esp_data)
                    break
    
    # Don't forget to add the last turn if we're still in one
    if current_turn is not None:
        turn_data.append(current_turn)
    
    return straight_data, turn_data

def process_files(input_folder, walks_folder, turns_folder, txt_path):
    """
    Process all JSON files in the input folder and separate them according to the turn segments
    """
    # Create output directories if they don't exist
    Path(walks_folder).mkdir(parents=True, exist_ok=True)
    Path(turns_folder).mkdir(parents=True, exist_ok=True)
    
    # Read turn segments information
    turn_segments = read_turn_segments(txt_path)
    
    # Get list of all json files in input folder
    json_files = {file.stem: file for file in Path(input_folder).glob('*.json')}
    
    print(f"\nFound {len(json_files)} JSON files in input folder:")
    for fname in sorted(json_files.keys()):
        print(f"  '{fname}'")
    
    print(f"\nFound {len(turn_segments)} entries in turn segments file")
    
    # Check for files in turn_segments that don't exist in the input folder
    missing_files = []
    for filename in turn_segments:
        if filename not in json_files:
            missing_files.append(filename)
            print(f"Warning: {filename} found in turn segments file but not in input folder")
    
    print(f"Number of files missing from input folder: {len(missing_files)}")
    
    # Process each JSON file that has corresponding turn segments
    processed_files = 0
    skipped_files = 0
    
    for filename, json_file in json_files.items():
        if filename not in turn_segments:
            print(f"Skipping '{filename}' - not found in turn segments file")
            # Print closest matching filename for debugging
            close_matches = [f for f in turn_segments.keys() if filename in f or f in filename]
            if close_matches:
                print(f"  Closest matching filenames in turn segments: {close_matches}")
            skipped_files += 1
            continue
        
        print(f"Processing {filename}")
        
        try:
            # Separate the data
            straight_data, turn_data = separate_imu_data(json_file, turn_segments[filename])
            
            # Save straight segments in WALKING folder
            walking_path = Path(walks_folder) / f"{filename}.json"
            with open(walking_path, 'w') as f:
                json.dump(straight_data, f, indent=2)
            
            # Save turn segments in TURNS folder
            for i, turn in enumerate(turn_data):
                turn_path = Path(turns_folder) / f"{filename}_t{i+1}.json"
                with open(turn_path, 'w') as f:
                    json.dump(turn, f, indent=2)
            
            processed_files += 1
            print(f"Successfully processed {filename} - Created {len(turn_data)} turn files")
            
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
 
if __name__ == "__main__":
    # Configuration
    INPUT_FOLDER = "./Segmentado"     # Replace with your input folder path
    WALKING_FOLDER = "../../SEGMENTED_DATA/CTL-PSANO/WALKING"        # Replace with path to existing WALKING folder
    TURNS_FOLDER = "../../SEGMENTED_DATA/CTL-PSANO/TURNS"           # Replace with path to existing TURNS folder
    TURN_SEGMENTS_FILE = "../segmentarSano.txt"  # Replace with your txt file path
    
    # Process the files
    process_files(INPUT_FOLDER, WALKING_FOLDER, TURNS_FOLDER, TURN_SEGMENTS_FILE)