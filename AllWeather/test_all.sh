#!/usr/bin/env bash
## Evaluate a trained AllWeather Restormer on the three weather-specific test sets:
##   rain+fog : CVPR19RainTrain/test  (Outdoor-Rain)
##   snow     : Snow100K-testset ... Snow100K-L
##   raindrop : raindrop_data/test_a
##
## Usage: bash AllWeather/test_all.sh [DATA_ROOT] [WEIGHTS] [extra test.py args...]
##   DATA_ROOT defaults to ~/autodl-tmp (directory containing the datasets below)
##   WEIGHTS   defaults to ./pretrained_models/allweather.pth
## Example:
##   bash AllWeather/test_all.sh ~/autodl-tmp ../experiments/AllWeather_Restormer/models/net_g_latest.pth --no_save_images

set -e
cd "$(dirname "$0")"

DATA_ROOT=${1:-$HOME/autodl-tmp}
WEIGHTS=${2:-./pretrained_models/allweather.pth}
EXTRA_ARGS=("${@:3}")

echo "==== [1/3] Rain+Fog: CVPR19 Outdoor-Rain (CVPR19RainTrain/test) ===="
python test.py \
    --input_dir "$DATA_ROOT/CVPR19RainTrain/test/data" \
    --gt_dir "$DATA_ROOT/CVPR19RainTrain/test/gt" \
    --result_dir ./results/rain \
    --weights "$WEIGHTS" "${EXTRA_ARGS[@]}"

echo ""
echo "==== [2/3] Snow: Snow100K-L ===="
python test.py \
    --input_dir "$DATA_ROOT/Snow100K-testset/jdway/GameSSD/overlapping/test/Snow100K-L/synthetic" \
    --gt_dir "$DATA_ROOT/Snow100K-testset/jdway/GameSSD/overlapping/test/Snow100K-L/gt" \
    --result_dir ./results/snow \
    --weights "$WEIGHTS" "${EXTRA_ARGS[@]}"

echo ""
echo "==== [3/3] Raindrop: raindrop_data/test_a ===="
python test.py \
    --input_dir "$DATA_ROOT/raindrop_data/test_a/data" \
    --gt_dir "$DATA_ROOT/raindrop_data/test_a/gt" \
    --result_dir ./results/raindrop \
    --weights "$WEIGHTS" "${EXTRA_ARGS[@]}"

echo ""
echo "All three test sets done. Results under AllWeather/results/{rain,snow,raindrop}/"
