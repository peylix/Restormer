## Restormer: Efficient Transformer for High-Resolution Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang
## https://arxiv.org/abs/2111.09881
##
## Adapted for AllWeather (Outdoor-Rain + Raindrop + Snow100K) evaluation.
## Metrics (PSNR / SSIM / MAE on Y channel, LPIPS, DISTS) follow the shared
## uniweather metrics module so numbers are comparable across baselines.
##
## Usage: cd AllWeather && python test.py --input_dir <path> --gt_dir <path> --weights <path>
## Extra deps: pip install lpips piq

import csv
import os
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

factor = 8

img_save_dir = os.path.join(args.result_dir, 'restored')
os.makedirs(img_save_dir, exist_ok=True)

gt_files = natsorted(os.listdir(args.gt_dir))
gt_paths = {os.path.splitext(f)[0]: os.path.join(args.gt_dir, f) for f in gt_files}

inp_files = natsorted(
    glob(os.path.join(args.input_dir, '*.png')) +
    glob(os.path.join(args.input_dir, '*.jpg'))
)
inp_files = [f for f in inp_files if os.path.splitext(os.path.basename(f))[0] in gt_paths]

print(f"Testing on {len(inp_files)} images (matched with GT)")

metric_computer = PerceptualMetricComputer(device='cuda')
metric_lists = {k: [] for k in METRIC_KEYS}
per_image_rows = []

with torch.no_grad():
    for file_ in tqdm(inp_files):
        img = np.float32(utils.load_img(file_)) / 255.
        img = torch.from_numpy(img).permute(2, 0, 1)
        input_ = img.unsqueeze(0).cuda()

        # Padding in case images are not multiples of 8
        h, w = input_.shape[2], input_.shape[3]
        H, W = ((h + factor) // factor) * factor, ((w + factor) // factor) * factor
        padh = H - h if h % factor != 0 else 0
        padw = W - w if w % factor != 0 else 0
        input_ = F.pad(input_, (0, padw, 0, padh), 'reflect')

        restored = model_restoration(input_)
        restored = restored[:, :, :h, :w]
        restored = torch.clamp(restored, 0, 1).cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()

        basename = os.path.splitext(os.path.basename(file_))[0]
        restored_ubyte = img_as_ubyte(restored)
        utils.save_img(os.path.join(img_save_dir, basename + '.png'), restored_ubyte)

        # metrics operate on BGR uint8, same as the shared metrics module
        restored_bgr = cv2.cvtColor(restored_ubyte, cv2.COLOR_RGB2BGR)
        gt_bgr = load_image_bgr_or_raise(gt_paths[basename])

        cur = compute_all_metrics(restored_bgr, gt_bgr, metric_computer)
        for k in METRIC_KEYS:
            metric_lists[k].append(cur[k])
        per_image_rows.append([basename] + [cur[k] for k in METRIC_KEYS])

# per-image metrics for later analysis (e.g. per-domain breakdown)
csv_path = os.path.join(args.result_dir, 'metrics.csv')
with open(csv_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['image'] + METRIC_KEYS)
    writer.writerows(per_image_rows)

summary = build_summary(metric_lists)
print_metric_summary(summary, decimals=4, title="\nTesting set metrics (mean ± std):")
print(f"Total image:{len(per_image_rows)}")
print(f"Restored images: {img_save_dir}")
print(f"Per-image metrics: {csv_path}")
