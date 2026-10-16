"""Frozen gray-energy sampling and measured five-pulse encoding. CPU only."""

import math
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from scipy.ndimage import gaussian_filter
from skimage.feature import hog

DIM = 28224


def load_clip(root, clip):
    """Read every frame from original images; no external pixel/feature cache."""
    rgb, nir = [], []
    y0, y1, x0, x1 = clip["rgb_box"]
    for rp, ip in zip(clip["rgb"], clip["nir"]):
        with Image.open(root / rp) as im:
            image = np.asarray(im.convert("RGB").resize((224, 168), Image.Resampling.BOX))
        image = np.asarray(
            Image.fromarray(image[y0:y1, x0:x1]).resize((128, 128), Image.Resampling.BILINEAR)
        )
        rgb.append(image[:, ::-1].transpose(2, 0, 1))
        with Image.open(root / ip) as im:
            nir.append(np.asarray(im).copy())
    rgb, nir = np.stack(rgb), np.stack(nir)
    if rgb.shape != (clip["n_frames"], 3, 128, 128) or nir.shape != (clip["n_frames"], 171, 224):
        raise ValueError(
            f"Unexpected image sizes for clip {clip['clip_id']}: {rgb.shape}, {nir.shape}"
        )
    return rgb, nir


def geometry(frames, roi, params):
    x = torch.from_numpy(np.array(frames, dtype=np.float32))[None]
    h, w = x.shape[2:]
    if roi is not None:
        side = roi["side"]
        theta = torch.tensor(
            [
                [
                    [side / w, 0.0, 2 * roi["center_x"] / w - 1],
                    [0.0, side / h, 2 * roi["center_y"] / h - 1],
                ]
            ],
            dtype=torch.float32,
        )
        grid = F.affine_grid(theta, (1, 40, 128, 128), align_corners=False)
        x = F.grid_sample(x, grid, align_corners=False, padding_mode="zeros")
    angle, scale, tx, ty = params
    if angle != 0.0 or scale != 1.0 or tx != 0.0 or ty != 0.0:
        h, w = x.shape[2:]
        c, s = math.cos(math.radians(angle)), math.sin(math.radians(angle))
        theta = torch.tensor(
            [[[c / scale, -s * h / w / scale, tx], [s * w / h / scale, c / scale, ty]]],
            dtype=torch.float32,
        )
        grid = F.affine_grid(theta, x.shape, align_corners=False)
        x = F.grid_sample(x, grid, align_corners=False, padding_mode="zeros")
    return x[0].numpy()


def augmentation(clip_id, view):
    table = [[0.0, 1.0, 0.0, 0.0]]
    rng = np.random.default_rng([20261012, clip_id, 1])
    for _ in range(2):
        table.append(
            [
                rng.uniform(-7.0, 7.0),
                rng.uniform(0.90, 1.10),
                rng.uniform(-0.04, 0.04),
                rng.uniform(-0.04, 0.04),
            ]
        )
    return table[view]


def sampling_plan(full, clip):
    i, n = clip["clip_id"], clip["n_frames"]
    aligned = geometry(full[1], clip["nir_roi"], augmentation(i, 0))
    image = gaussian_filter(aligned[:, 8:120, 8:120], sigma=(0, 1, 1), mode="reflect", truncate=4.0)
    energy = np.r_[0.0, np.abs(np.diff(image.astype(np.float64), axis=0)).mean((1, 2)) / 65535]
    fallback = float(energy.sum()) <= 1e-12
    bounds = (
        [int(v) for v in np.rint(np.linspace(0, n - 1, 9))]
        if fallback
        else [0]
        + [
            int(np.searchsorted(np.cumsum(energy) / energy.sum(), k / 8, side="left"))
            for k in range(1, 8)
        ]
        + [n - 1]
    )
    plan = np.zeros((3, 40), np.int16)
    for view in range(3):
        if fallback:
            j = clip["center_start"]
            plan[view] = np.arange(j, j + 40)
            continue
        rng = np.random.default_rng([20261014, i, view, 1, 74])
        centers = [
            (bounds[k] + bounds[k + 1]) // 2
            if view == 0 or clip["split"] != "train"
            else int(rng.integers(bounds[k], bounds[k + 1] + 1))
            for k in range(8)
        ]
        starts = np.clip(np.array(centers) - 2, 0, n - 5)
        plan[view] = (starts[:, None] + np.arange(5)).ravel()
    index = plan.reshape(3, 8, 5)
    assert (np.diff(index, axis=2) == 1).all() and (np.diff(index[:, :, 0], axis=1) >= 0).all()
    assert plan.min() >= 0 and plan.max() < n
    return plan, {
        "energy": energy.tolist(),
        "boundaries": bounds,
        "fallback": fallback,
        "unique_frames": [int(len(np.unique(p))) for p in plan],
    }


def selected_frames(full, clip, indices, view):
    rgb, nir = full
    params = augmentation(clip["clip_id"], view)
    a = geometry(
        np.array(rgb[indices], dtype=np.float32).reshape(120, 128, 128), None, params
    ).reshape(40, 3, 128, 128)
    b = geometry(nir[indices], clip["nir_roi"], params)
    return np.concatenate([a, b[:, None]], axis=1)


def encode(frames, clip_id, view, response):
    thresholds = response["thresholds"][None, :, None, None]
    bits = (frames > thresholds).reshape(8, 5, 4, 128, 128)
    codes = (
        (bits * np.array([16, 8, 4, 2, 1], np.int16)[None, :, None, None, None])
        .sum(1)
        .astype(np.int16)
    )
    result = []
    for ch in range(4):
        mu, sample = response["mean_nA"][ch], response["samples_nA"][ch]
        q = codes[:, ch]
        rng = np.random.default_rng([20261012, clip_id, view, 0, ch])
        out = mu[q]
        k = rng.integers(0, sample.shape[0], size=q.shape)
        out = out + (sample[k, q] - out)
        result.append(out.astype(np.float32))
    return np.stack(result, axis=1)


def hog_currents(currents):
    a = np.asarray(currents)
    if a.shape != (8, 4, 128, 128) or not np.isfinite(a).all():
        raise ValueError("temporal_nA must be finite and shaped (8,4,128,128).")
    x = torch.from_numpy(np.array(a, dtype=np.float32))
    x = ((x - x.mean((0, 2, 3), keepdim=True)) / (x.std((0, 2, 3), keepdim=True) + 1e-6)).numpy()[
        :, :, 8:120, 8:120
    ]
    x = gaussian_filter(x, sigma=(0, 0, 1, 1), mode="reflect", truncate=4.0)
    rgb, nir = [], []
    kw = dict(
        orientations=9,
        pixels_per_cell=(14, 14),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        transform_sqrt=False,
    )
    for t in range(8):
        rgb.append(hog(x[t, :3].transpose(1, 2, 0), channel_axis=-1, **kw))
        nir.append(hog(x[t, 3], channel_axis=None, **kw))
    return np.concatenate([np.stack(rgb).ravel(), np.stack(nir).ravel()]).astype(np.float32)


def feature_for(full, clip, indices, view, response):
    currents = encode(selected_frames(full, clip, indices, view), clip["clip_id"], view, response)
    return hog_currents(currents), currents
