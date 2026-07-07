## Metric helpers for AllWeather evaluation.
## Adapted from the shared uniweather metrics module (measure.py / image_utils.py)
## so that Restormer baseline numbers use the exact same implementations:
##   - PSNR / SSIM / MAE on the Y channel (BGR uint8 inputs, range [0, 255])
##   - LPIPS (AlexNet, via `pip install lpips`)
##   - DISTS (via `pip install piq`)
##
## Standalone usage (evaluate a folder of saved results against GT):
##   python measure.py --results_dir ./results/restored --gt_dir /path/to/gt_val

import os

import cv2
import numpy as np
import torch

from metrics import calculate_psnr, calculate_ssim, to_y_channel


def load_image_bgr_or_raise(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return img


def crop_to_common_shape(img1, img2):
    """Crop two HWC images to their common top-left region."""
    h = min(img1.shape[0], img2.shape[0])
    w = min(img1.shape[1], img2.shape[1])
    return img1[:h, :w], img2[:h, :w]


def calculate_mae(img1, img2, test_y_channel=True):
    """Mean absolute error on [0, 1] scale."""
    assert img1.shape == img2.shape
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    if test_y_channel:
        img1 = to_y_channel(img1)
        img2 = to_y_channel(img2)
    return float(np.mean(np.abs(img1 - img2)) / 255.0)


def _bgr_to_rgb_tensor(img_bgr, device):
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
    return tensor


class PerceptualMetricComputer:
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._lpips_model = None
        self._dists_model = None

    def _get_lpips_model(self):
        if self._lpips_model is None:
            try:
                import lpips
            except ImportError as exc:
                raise ImportError("LPIPS requires `pip install lpips`.") from exc
            self._lpips_model = lpips.LPIPS(net="alex").to(self.device).eval()
        return self._lpips_model

    def _get_dists_model(self):
        if self._dists_model is None:
            try:
                from piq import DISTS
            except ImportError as exc:
                raise ImportError("DISTS requires `pip install piq`.") from exc
            self._dists_model = DISTS().to(self.device).eval()
        return self._dists_model

    @torch.no_grad()
    def calculate_lpips(self, img1, img2):
        model = self._get_lpips_model()
        pred = _bgr_to_rgb_tensor(img1, self.device) * 2.0 - 1.0
        gt = _bgr_to_rgb_tensor(img2, self.device) * 2.0 - 1.0
        return float(model(pred, gt).item())

    @torch.no_grad()
    def calculate_dists(self, img1, img2):
        model = self._get_dists_model()
        pred = _bgr_to_rgb_tensor(img1, self.device)
        gt = _bgr_to_rgb_tensor(img2, self.device)
        return float(model(pred, gt).item())


def compute_all_metrics(res_bgr, gt_bgr, metric_computer, test_y_channel=True):
    """Compute all five metrics for one BGR uint8 image pair."""
    res_bgr, gt_bgr = crop_to_common_shape(res_bgr, gt_bgr)
    return {
        "psnr": calculate_psnr(res_bgr, gt_bgr, test_y_channel=test_y_channel),
        "ssim": calculate_ssim(res_bgr, gt_bgr, test_y_channel=test_y_channel),
        "mae": calculate_mae(res_bgr, gt_bgr, test_y_channel=test_y_channel),
        "lpips": metric_computer.calculate_lpips(res_bgr, gt_bgr),
        "dists": metric_computer.calculate_dists(res_bgr, gt_bgr),
    }


METRIC_KEYS = ["psnr", "ssim", "mae", "lpips", "dists"]


def summarize_metric(values):
    arr = np.asarray(values, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return float("nan"), float("nan")
    mean = float(finite.mean())
    std = float(finite.std(ddof=1)) if finite.size > 1 else 0.0
    return mean, std


def format_mean_std(mean, std, decimals=4):
    return f"{mean:.{decimals}f} ± {std:.{decimals}f}"


def print_metric_summary(summary, decimals=4, title=None):
    if title:
        print(title)
    for label, key in [
        ("PSNR", "psnr"),
        ("SSIM", "ssim"),
        ("MAE", "mae"),
        ("LPIPS", "lpips"),
        ("DISTS", "dists"),
    ]:
        print("%s: %s" % (label.ljust(5), format_mean_std(
            summary[f"{key}_mean"], summary[f"{key}_std"], decimals=decimals
        )))


def build_summary(metric_lists):
    """metric_lists: dict of key -> list of per-image values."""
    summary = {}
    for name in METRIC_KEYS:
        mean, std = summarize_metric(metric_lists[name])
        summary[f"{name}_mean"] = mean
        summary[f"{name}_std"] = std
        summary[name] = metric_lists[name]
    return summary


def evaluate_folder_pair(results_path, gt_path, verbose=True, test_y_channel=True, device=None):
    imgs_name = sorted(os.listdir(results_path))
    gts_name = sorted(os.listdir(gt_path))
    assert len(imgs_name) == len(gts_name), (
        f"Image count mismatch: results={len(imgs_name)}, gt={len(gts_name)}"
    )

    metric_computer = PerceptualMetricComputer(device=device)
    metric_lists = {k: [] for k in METRIC_KEYS}

    for i in range(len(imgs_name)):
        res = load_image_bgr_or_raise(os.path.join(results_path, imgs_name[i]))
        gt = load_image_bgr_or_raise(os.path.join(gt_path, gts_name[i]))
        if verbose:
            print("image:%s, gt:%s" % (imgs_name[i], gts_name[i]))

        cur = compute_all_metrics(res, gt, metric_computer, test_y_channel=test_y_channel)
        for k in METRIC_KEYS:
            metric_lists[k].append(cur[k])

        if verbose:
            print(
                "PSNR=%.4f, SSIM=%.4f, MAE=%.4f, LPIPS=%.4f, DISTS=%.4f"
                % (cur["psnr"], cur["ssim"], cur["mae"], cur["lpips"], cur["dists"])
            )

    summary = build_summary(metric_lists)
    print_metric_summary(summary, decimals=4, title="Testing set metrics (mean ± std):")
    print("Total image:%d" % len(imgs_name))
    return summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate saved results against GT")
    parser.add_argument("--results_dir", type=str, required=True,
                        help="Directory of restored images")
    parser.add_argument("--gt_dir", type=str, required=True,
                        help="Directory of ground truth images")
    parser.add_argument("--per_image", action="store_true",
                        help="Print per-image metrics")
    args = parser.parse_args()
    evaluate_folder_pair(args.results_dir, args.gt_dir, verbose=args.per_image)
