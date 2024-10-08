import os
import random
import json
import argparse
from typing import List, Dict, Tuple

def select_reference_and_evaluation_frames(all_frames: List[str], eval_mode: str) -> Dict[str, List[str]]:
    video_frames = random.sample(all_frames, min(100, len(all_frames)))
    few_frames = random.sample(video_frames, min(10, len(video_frames)))
    single_frame = random.sample(few_frames, 1)
    
    if eval_mode == 'single':
        reference_frames = single_frame
    elif eval_mode == 'few':
        reference_frames = few_frames
    elif eval_mode == 'video':
        reference_frames = video_frames

    evaluation_frames = [frame for frame in all_frames if frame not in reference_frames]
    
    return {
        'reference': reference_frames,
        'evaluation': evaluation_frames
    }

def save_json(output_dir: str, file_name: str, data: List[Tuple[str, str]]):
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, file_name)
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)

def create_reference_and_evaluation_jsons(sequence_dir: str):
    # assumption: output_dir = sequence_dir
    output_dir = sequence_dir
    images_dir = os.path.join(sequence_dir, 'images-2x-73fps')
    camera_views = os.listdir(images_dir)
    selected_camera_view = random.choice(camera_views)

    all_frames = [
        os.path.join(selected_camera_view, frame)
        for frame in os.listdir(os.path.join(images_dir, selected_camera_view))
    ]

    reference_1 = select_reference_and_evaluation_frames(all_frames, 'single')
    reference_10 = select_reference_and_evaluation_frames(all_frames, 'few')
    reference_100 = select_reference_and_evaluation_frames(all_frames, 'video')

    cam_id = selected_camera_view
    reference_images_1 = [(cam_id, os.path.basename(frame)) for frame in reference_1['reference']]
    reference_images_10 = [(cam_id, os.path.basename(frame)) for frame in reference_10['reference']]
    reference_images_100 = [(cam_id, os.path.basename(frame)) for frame in reference_100['reference']]

    if len(all_frames) < 300:
        evaluation_images = [(cam_id, os.path.basename(frame)) for frame in reference_100['evaluation']][:100]
    else:
        evaluation_images = [(cam_id, os.path.basename(frame)) for frame in reference_100['evaluation']][:200]

    save_json(output_dir, 'reference_images_1.json', reference_images_1)
    save_json(output_dir, 'reference_images_10.json', reference_images_10)
    save_json(output_dir, 'reference_images_100.json', reference_images_100)
    save_json(output_dir, 'evaluation_images.json', evaluation_images)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='create ref & eval json files')
    parser.add_argument('sequence_directory', type=str, help='path to seq dir')

    args = parser.parse_args()

    create_reference_and_evaluation_jsons(args.sequence_directory)
    print(f"saved evaluation_images.json, reference_images_1.json, reference_images_10.json, reference_images_100.json to {args.sequence_directory}")
