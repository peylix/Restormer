## Restormer for AllWeather dataset
## Generate meta_info files for training and validation
## Usage: python AllWeather/generate_meta_info.py --data_dir /path/to/allweather

import os
import argparse
from natsort import natsorted


def generate_meta_info(data_dir, output_dir):
    gt_dir = os.path.join(data_dir, 'gt')
    gt_val_dir = os.path.join(data_dir, 'gt_val')
    input_dir = os.path.join(data_dir, 'input')

    for d in [gt_dir, gt_val_dir, input_dir]:
        assert os.path.isdir(d), f'Directory not found: {d}'

    input_files = set(os.listdir(input_dir))

    train_meta = os.path.join(output_dir, 'meta_info_train.txt')
    gt_files = natsorted(os.listdir(gt_dir))
    missing = [f for f in gt_files if f not in input_files]
    if missing:
        print(f'Warning: {len(missing)} GT files have no matching input: {missing[:5]}...')
    with open(train_meta, 'w') as f:
        for fname in gt_files:
            if fname in input_files:
                f.write(f'{fname}\n')
    print(f'Train meta_info: {len(gt_files) - len(missing)} pairs -> {train_meta}')

    val_meta = os.path.join(output_dir, 'meta_info_val.txt')
    gt_val_files = natsorted(os.listdir(gt_val_dir))
    missing_val = [f for f in gt_val_files if f not in input_files]
    if missing_val:
        print(f'Warning: {len(missing_val)} GT_val files have no matching input: {missing_val[:5]}...')
    with open(val_meta, 'w') as f:
        for fname in gt_val_files:
            if fname in input_files:
                f.write(f'{fname}\n')
    print(f'Val meta_info: {len(gt_val_files) - len(missing_val)} pairs -> {val_meta}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to allweather dataset root (containing input/, gt/, gt_val/)')
    parser.add_argument('--output_dir', type=str, default='./AllWeather/Datasets',
                        help='Output directory for meta_info files')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    generate_meta_info(args.data_dir, args.output_dir)
