import os
import json
import shutil
import argparse
from typing import List, Tuple

def load(file_path: str) -> List[Tuple[str, str]]:
    with open(file_path, 'r') as f:
        return json.load(f)

def extract_frames(sequence_dir: str, json_files: List[str], output_dirs: List[str]):
    for output_dir in output_dirs:
        os.makedirs(output_dir, exist_ok=True)

    for json_file, output_dir in zip(json_files, output_dirs):
        frame_data = load(json_file)

        for cam_id, frame_id in frame_data:
            alpha_map_src = os.path.join(sequence_dir, 'alpha_map-73fps', cam_id, frame_id)
            image_src = os.path.join(sequence_dir, 'images-2x-73fps', cam_id, frame_id)

            if os.path.exists(alpha_map_src):
                shutil.copy(alpha_map_src, os.path.join(output_dir, f"{cam_id}_{frame_id}"))
            else:
                print(f"no alpha map")

            if os.path.exists(image_src):
                shutil.copy(image_src, os.path.join(output_dir, f"{cam_id}_{frame_id}"))
            else:
                print(f"no img found")

        print(f"done extracting {json_file} --> saved to {output_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='extract frames & alpha maps based on json files')
    parser.add_argument('sequence_directory', type=str, help='path to seq dir')

    args = parser.parse_args()

    sequence_directory = args.sequence_directory

    json_files = [
        os.path.join(sequence_directory, 'reference_images_1.json'),
        os.path.join(sequence_directory, 'reference_images_10.json'),
        os.path.join(sequence_directory, 'reference_images_100.json'),
        os.path.join(sequence_directory, 'evaluation_images.json')
    ]

    output_dirs = [
        os.path.join(sequence_directory, 'reference_images_1'),
        os.path.join(sequence_directory, 'reference_images_10'),
        os.path.join(sequence_directory, 'reference_images_100'),
        os.path.join(sequence_directory, 'evaluate_images')
    ]

    extract_frames(sequence_directory, json_files, output_dirs)
