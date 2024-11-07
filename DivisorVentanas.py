import json
from collections import defaultdict
import os
from pathlib import Path

def segment_imu_data(folder_path):
   
    segmented_data = defaultdict(list)
    folder = Path(folder_path)


    json_files = list(folder.glob('*.json'))
    
    if not json_files:
        print(f"No JSON files found in {folder_path}")
        return segmented_data

    print(f"Found {len(json_files)} JSON files")
    
    for json_file in json_files:
        try:
            with open(json_file, 'r') as file:
                sensor_data = json.load(file)

            for sensor_name, readings in sensor_data.items():
                segments = []
                current_segment = []
                last_timestamp = None

                for reading in readings:
                    timestamp = reading['t']
                    if last_timestamp is None or timestamp - last_timestamp >= 1000:
                        if current_segment:
                            segments.append(current_segment)
                        current_segment = [reading]
                    else:
                        current_segment.append(reading)
                    last_timestamp = timestamp

                if current_segment:
                    segments.append(current_segment)

                segmented_data[sensor_name].extend(segments)
                
        except json.JSONDecodeError:
            print(f"Error: {json_file.name} is not a valid JSON file")
        except Exception as e:
            print(f"Error processing {json_file.name}: {str(e)}")

    print("\nProcessing complete!")
    print("Segments found per sensor:")
    for sensor_name, segments in segmented_data.items():
        print(f"{sensor_name}: {len(segments)} segments")
            
    return segmented_data

def save_segments(segmented_data, output_folder):
   
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for sensor_name, segments in segmented_data.items():
        output_file = output_path / f"{sensor_name}_segments.json"
        with open(output_file, 'w') as f:
            json.dump({
                'sensor_name': sensor_name,
                'total_segments': len(segments),
                'segments': segments
            }, f, indent=2)
        print(f"Saved segments for {sensor_name} to {output_file}")