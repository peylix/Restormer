## Restormer: Efficient Transformer for High-Resolution Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang
## https://arxiv.org/abs/2111.09881
##
## Adapted for AllWeather (Outdoor-Rain + Raindrop + Snow100K) evaluation.
## Metrics (PSNR / SSIM / MAE on Y channel, LPIPS, DISTS) follow the shared
## uniweather metrics module so numbers are comparable across baselines.
##
## Usage: cd AllWeather && python test.py --input_dir <path> --gt_dir <path> --weights <path>
## To evaluate all three weather test sets in one go, see test_all.sh.
## Extra deps: pip install lpips piq

import csv
import os
import re
import time
import argparse

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from natsort import natsorted
from glob import glob
from skimage import img_as_ubyte

import utils
from basicsr.models.archs.restormer_arch import Restormer
from measure import (METRIC_KEYS, PerceptualMetricComputer, build_summary,
                     compute_all_metrics, load_image_bgr_or_raise,
                     print_metric_summary)

parser = argparse.ArgumentParser(description='AllWeather Restoration using Restormer')

parser.add_argument('--input_dir', default='./Datasets/allweather/input', type=str,
                    help='Directory of input (degraded) images')
parser.add_argument('--gt_dir', default='./Datasets/allweather/gt_val', type=str,
                    help='Directory of ground truth images for evaluation')
parser.add_argument('--result_dir', default='./results/', type=str,
                    help='Directory for results (restored images + metrics.csv)')
parser.add_argument('--weights', default='./pretrained_models/allweather.pth', type=str,
                    help='Path to weights')
parser.add_argument('--no_save_images', action='store_true',
                    help='Skip saving restored images (metrics + CSV only)')

args = parser.parse_args()

####### Load yaml #######
yaml_file = 'Options/AllWeather_Restormer.yml'
import yaml

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

x = yaml.load(open(yaml_file, mode='r'), Loader=Loader)

s = x['network_g'].pop('type')
##########################

model_restoration = Restormer(**x['network_g'])

checkpoint = torch.load(args.weights)
model_restoration.load_state_dict(checkpoint['params'])
print("===>Testing using weights: ", args.weights)
model_restoration.cuda()
model_restoration = nn.DataParallel(model_restoration)
model_restoration.eval()

total_params = sum(p.numel() for p in model_restoration.parameters())
print(f"Model parameters: {total_params:,} ({total_params / 1e6:.2f} M)")

factor = 8

os.makedirs(args.result_dir, exist_ok=True)
img_save_dir = os.path.join(args.result_dir, 'restored')
if not args.no_save_images:
    os.makedirs(img_save_dir, exist_ok=True)

gt_files = natsorted(os.listdir(args.gt_dir))
gt_paths = {os.path.splitext(f)[0]: os.path.join(args.gt_dir, f) for f in gt_files}


def find_gt_path(input_basename):
    """Match an input image to its GT under the naming schemes of our test sets:
    - identical basenames (AllWeather / Snow100K: synthetic vs gt)
    - RainDrop:            '0_rain'          -> '0_clean'
    - Outdoor-Rain CVPR19: 'im_0001_s80_a04' -> 'im_0001'
    """
    if input_basename in gt_paths:
        return gt_paths[input_basename]
    if input_basename.endswith('_rain'):
        candidate = input_basename[:-len('_rain')] + '_clean'
        if candidate in gt_paths:
            return gt_paths[candidate]
    m = re.match(r'^(.+?)_s\d+_a\d+$', input_basename)
    if m and m.group(1) in gt_paths:
        return gt_paths[m.group(1)]
    return None


inp_files = natsorted(
    glob(os.path.join(args.input_dir, '*.png')) +
    glob(os.path.join(args.input_dir, '*.jpg')) +
    glob(os.path.join(args.input_dir, '*.jpeg'))
)

pairs = []
unmatched = []
for f in inp_files:
    base = os.path.splitext(os.path.basename(f))[0]
    gt_path = find_gt_path(base)
    if gt_path is None:
        unmatched.append(base)
    else:
        pairs.append((f, gt_path))

if unmatched:
    print(f"Warning: {len(unmatched)} input images have no matching GT, "
          f"e.g. {unmatched[:5]}")
print(f"Testing on {len(pairs)} images (matched with GT)")

metric_computer = PerceptualMetricComputer(device='cuda')
metric_lists = {k: [] for k in METRIC_KEYS}
per_image_rows = []
inference_times = []

with torch.no_grad():
    for file_, gt_file in tqdm(pairs):
        img = np.float32(utils.load_img(file_)) / 255.
        img = torch.from_numpy(img).permute(2, 0, 1)
        input_ = img.unsqueeze(0).cuda()

        # Padding in case images are not multiples of 8
        h, w = input_.shape[2], input_.shape[3]
        H, W = ((h + factor) // factor) * factor, ((w + factor) // factor) * factor
        padh = H - h if h % factor != 0 else 0
        padw = W - w if w % factor != 0 else 0
        input_ = F.pad(input_, (0, padw, 0, padh), 'reflect')

        torch.cuda.synchronize()
        tic = time.perf_counter()
        restored = model_restoration(input_)
        torch.cuda.synchronize()
        inference_times.append(time.perf_counter() - tic)

        restored = restored[:, :, :h, :w]
        restored = torch.clamp(restored, 0, 1).cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()

        basename = os.path.splitext(os.path.basename(file_))[0]
        restored_ubyte = img_as_ubyte(restored)
        if not args.no_save_images:
            utils.save_img(os.path.join(img_save_dir, basename + '.png'), restored_ubyte)

        # metrics operate on BGR uint8, same as the shared metrics module
        restored_bgr = cv2.cvtColor(restored_ubyte, cv2.COLOR_RGB2BGR)
        gt_bgr = load_image_bgr_or_raise(gt_file)

        cur = compute_all_metrics(restored_bgr, gt_bgr, metric_computer)
        for k in METRIC_KEYS:
            metric_lists[k].append(cur[k])
        per_image_rows.append([basename] + [cur[k] for k in METRIC_KEYS]
                              + [inference_times[-1]])

# per-image metrics for later analysis (e.g. per-domain breakdown)
csv_path = os.path.join(args.result_dir, 'metrics.csv')
with open(csv_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['image'] + METRIC_KEYS + ['inference_time_s'])
    writer.writerows(per_image_rows)

summary = build_summary(metric_lists)
print_metric_summary(summary, decimals=4, title="\nTesting set metrics (mean ± std):")
print(f"Total image:{len(per_image_rows)}")

print(f"Model parameters: {total_params:,} ({total_params / 1e6:.2f} M)")
# The first image is discarded from the timing stats: it includes CUDA/cuDNN
# warm-up and is far slower than steady-state inference.
if len(inference_times) > 1:
    steady = np.asarray(inference_times[1:])
    std = steady.std(ddof=1) if steady.size > 1 else 0.0
    print(f"Inference time per image (first image excluded): "
          f"{steady.mean() * 1000:.2f} ± {std * 1000:.2f} ms "
          f"({1.0 / steady.mean():.2f} FPS, averaged over {steady.size} images; "
          f"first image took {inference_times[0] * 1000:.2f} ms)")
elif inference_times:
    print(f"Inference time: only one image ({inference_times[0] * 1000:.2f} ms, "
          f"includes warm-up — not representative)")
if not args.no_save_images:
    print(f"Restored images: {img_save_dir}")
print(f"Per-image metrics: {csv_path}")
