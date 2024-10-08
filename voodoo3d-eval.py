# TODO: alpha blending
# TODO: fid and fvd metrics

import torch
import click
import cv2
import glob
import os
import os.path as osp
from tqdm import tqdm
import yaml
import numpy as np
import random
from typing import List, Dict
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import lpips

from data_preprocessing.data_preprocess import DataPreprocessor
from models import get_model
from resources.consts import IMAGE_EXTS
from utils.image_utils import tensor2img

def tensor_from_path(img_path, max_size=512):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    h, w = img.shape[:2]
    if h > max_size or w > max_size:
        scale = max_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        img = cv2.resize(img, (new_w, new_h))
    
    img = np.transpose(img, (2, 0, 1))[None, :, :, :] / 255.
    img = (img * 2 - 1)
    return torch.from_numpy(img).float()

def select_reference_and_evaluation_frames(all_frames: List[str], eval_mode: str) -> Dict[str, List[str]]:
    if not all_frames:
        raise ValueError("no frames")

    if eval_mode not in ['single', 'few', 'video']:
        raise ValueError(f"invalid mode for eval: {eval_mode}")
    
    if eval_mode == 'video' and len(all_frames) < 100:
        raise ValueError(f"not enough frames for 'video' --> {len(all_frames)} found, need 100")
    
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

def calculate_metrics(generated_img, actual_img):
    generated_img = generated_img.astype(np.float32) / 255.0
    actual_img = actual_img.astype(np.float32) / 255.0
    
    mse = np.mean((generated_img - actual_img) ** 2)
    psnr_value = psnr(actual_img, generated_img, data_range=1.0)
    
    # adjust window size for ssim [oct06 modification]
    min_dim = min(generated_img.shape[0], generated_img.shape[1])
    win_size = min(7, min_dim)  # use 7 or smaller odd num [oct06 modification]
    if win_size % 2 == 0:
        win_size -= 1
    
    ssim_value = ssim(actual_img, generated_img, win_size=win_size, channel_axis=2, data_range=1.0)
    
    lpips_model = lpips.LPIPS(net='vgg')
    tensor_gen = torch.from_numpy(generated_img.transpose(2, 0, 1)).unsqueeze(0)
    # print(tensor_gen.shape)
    tensor_actual = torch.from_numpy(actual_img.transpose(2, 0, 1)).unsqueeze(0)
    lpips_value = lpips_model(tensor_gen, tensor_actual).item()
    
    return {
        'mse': mse,
        'psnr': psnr_value,
        'ssim': ssim_value,
        'lpips': lpips_value 
        # fid (ffhq)?
        # identity preservation wrt reference images(use face rec network to get feature vec to describe the face identity and see )
        # one video metrics (can be but dont have to be no overlapping 4 cameras of 100 frames each) -> jod
    }

@torch.no_grad()
@click.command()
@click.option('--source_root', type=str, required=True, help='Directory containing all frames')
@click.option('--config_path', type=str, required=True, help='Config path')
@click.option('--model_path', type=str, required=True, help='Model path')
@click.option('--save_root', type=str, required=True, help='Save root')
@click.option('--skip_preprocess', is_flag=True, help='Do not use preprocessing')
@click.option('--eval_mode', type=click.Choice(['single', 'few', 'video'], case_sensitive=False), required=True, help='Evaluation mode')
@click.option('--max_image_size', type=int, default=512, help='Maximum image dimension')

def main(source_root, config_path, model_path, save_root, skip_preprocess, eval_mode, max_image_size):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    processor = DataPreprocessor(device)

    assert osp.isdir(source_root), f"source root {source_root} is an invalid directory"

    # pick a random view
    frame_dirs = sorted(glob.glob(osp.join(source_root, 'frame_*')))
    if not frame_dirs:
        raise ValueError(f"no frame dir in {source_root}")

    random_frame = random.choice(frame_dirs)
    image_dir = osp.join(random_frame, 'images-2x-73fps') # hard coded
    if not osp.exists(image_dir):
        raise ValueError(f"images dir not found")

    camera_files = [f for f in os.listdir(image_dir) if f.startswith('cam_') and f.endswith('.png')]
    if not camera_files:
        raise ValueError(f"no cam im in {image_dir}")

    camera = random.choice(camera_files).split('_')[1].split('.')[0]

    # get all frames for selected view
    all_frames = []
    for frame_dir in frame_dirs:
        image_path = osp.join(frame_dir, 'images-2x-73fps', f'cam_{camera}.png') # add .png [oct05 modification]
        # print(image_path)
        if osp.exists(image_path):
            all_frames.append(image_path)

    if not all_frames:
        raise ValueError(f"no frames for camera {camera}")

    print(f"selected camera: {camera}")
    print(f"total frames: {len(all_frames)}")

    selected_frames = select_reference_and_evaluation_frames(all_frames, eval_mode)
    reference_paths = selected_frames['reference']
    evaluation_paths = selected_frames['evaluation']

    print(f'eval mode: {eval_mode}')
    print(f'num reference images: {len(reference_paths)}')
    print(f'num eval images: {len(evaluation_paths)}')

    print('\npreparing data...')
    reference_data = []
    for path in tqdm(reference_paths, desc="processing ref frames"):
        if not skip_preprocess:
            data = processor.from_path(path, device, keep_bg=False)
        else:
            data = {'image': tensor_from_path(path, max_size=max_image_size).to(device)}
        reference_data.append(data)

    # load model --> code taken from repo
    with open(config_path, 'r') as f:
        options = yaml.safe_load(f)
    model = get_model(options['model']).to(device)
    state_dict = torch.load(model_path, map_location='cpu')
    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    os.makedirs(save_root, exist_ok=True)
    metrics_list = []

    for idx, eval_path in enumerate(tqdm(evaluation_paths, desc="evaluating")):
        if not skip_preprocess:
            eval_data = processor.from_path(eval_path, device, keep_bg=False)
        else:
            eval_data = {'image': tensor_from_path(eval_path, max_size=max_image_size).to(device)}
        
        out = model(xs_data=reference_data[0], xd_data=eval_data)
        
        out_hr = tensor2img(out['image'], min_max=(-1, 1))
        actual_img = tensor2img(eval_data['image'][0], min_max=(-1, 1))
        
        try:
            metrics = calculate_metrics(out_hr, actual_img)
            metrics_list.append(metrics)
        except Exception as e:
            print(f"error calculating metrics for image {idx}: {str(e)}")
            print(f"image shape: {out_hr.shape}")
            continue
        
        save_path = osp.join(save_root, f'eval_{idx:04d}.png')
        cv2.imwrite(save_path, np.hstack((actual_img, out_hr)))

    # avg metrics 
    avg_metrics = {metric: np.mean([m[metric] for m in metrics_list]) for metric in metrics_list[0]}
    
    print("\naverage metrics:")
    for metric, value in avg_metrics.items():
        print(f"{metric}: {value:.4f}")

    # yaml
    with open(osp.join(save_root, 'metrics.yaml'), 'w') as f:
        yaml.dump(avg_metrics, f)

if __name__ == '__main__':
    main()
