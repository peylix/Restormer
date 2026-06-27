## Restormer: Efficient Transformer for High-Resolution Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang
## https://arxiv.org/abs/2111.09881
##
## Adapted for AllWeather (Outdoor-Rain + Raindrop + Snow100K) evaluation
## Usage: cd AllWeather && python test.py --input_dir <path> --gt_dir <path> --weights <path>

import numpy as np
import os
import argparse
from tqdm import tqdm

import torch.nn as nn
import torch
import torch.nn.functional as F
import utils

from natsort import natsorted
from glob import glob
from basicsr.models.archs.restormer_arch import Restormer
from skimage import img_as_ubyte

parser = argparse.ArgumentParser(description='AllWeather Restoration using Restormer')

parser.add_argument('--input_dir', default='./Datasets/allweather/input', type=str,
                    help='Directory of input (degraded) images')
parser.add_argument('--gt_dir', default='./Datasets/allweather/gt_val', type=str,
                    help='Directory of ground truth images for evaluation')
parser.add_argument('--result_dir', default='./results/', type=str,
                    help='Directory for results')
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

os.makedirs(args.result_dir, exist_ok=True)

gt_files = natsorted(os.listdir(args.gt_dir))
gt_basenames = set(os.path.splitext(f)[0] for f in gt_files)

inp_files = natsorted(
    glob(os.path.join(args.input_dir, '*.png')) +
    glob(os.path.join(args.input_dir, '*.jpg'))
)
inp_files = [f for f in inp_files if os.path.splitext(os.path.basename(f))[0] in gt_basenames]

print(f"Testing on {len(inp_files)} images (matched with GT)")

psnr_val_rgb = []
ssim_val_rgb = []

with torch.no_grad():
    for file_ in tqdm(inp_files):
        torch.cuda.ipc_collect()
        torch.cuda.empty_cache()

        img = np.float32(utils.load_img(file_)) / 255.
        img = torch.from_numpy(img).permute(2, 0, 1)
        input_ = img.unsqueeze(0).cuda()

        h, w = input_.shape[2], input_.shape[3]
        H, W = ((h + factor) // factor) * factor, ((w + factor) // factor) * factor
        padh = H - h if h % factor != 0 else 0
        padw = W - w if w % factor != 0 else 0
        input_ = F.pad(input_, (0, padw, 0, padh), 'reflect')

        restored = model_restoration(input_)
        restored = restored[:, :, :h, :w]
        restored = torch.clamp(restored, 0, 1).cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()

        basename = os.path.splitext(os.path.basename(file_))[0]
        utils.save_img(os.path.join(args.result_dir, basename + '.png'), img_as_ubyte(restored))

        gt_ext = None
        for ext in ['.png', '.jpg']:
            if os.path.exists(os.path.join(args.gt_dir, basename + ext)):
                gt_ext = ext
                break
        if gt_ext is not None:
            gt_img = np.float32(utils.load_img(os.path.join(args.gt_dir, basename + gt_ext))) / 255.
            gt_img = gt_img[:h, :w]
            restored_clip = np.clip(restored, 0, 1)

            psnr = utils.calculate_psnr(img_as_ubyte(restored_clip), img_as_ubyte(gt_img))
            ssim = utils.calculate_ssim(img_as_ubyte(restored_clip), img_as_ubyte(gt_img))
            psnr_val_rgb.append(psnr)
            ssim_val_rgb.append(ssim)

if psnr_val_rgb:
    psnr_avg = sum(psnr_val_rgb) / len(psnr_val_rgb)
    ssim_avg = sum(ssim_val_rgb) / len(ssim_val_rgb)
    print(f"PSNR: {psnr_avg:.4f} dB | SSIM: {ssim_avg:.4f} | Images: {len(psnr_val_rgb)}")
else:
    print("No GT images found for evaluation. Only restored images were saved.")
