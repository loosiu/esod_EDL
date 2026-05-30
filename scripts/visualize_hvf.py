"""Visualize Heat (H), Vacuity (V), and Fused (F) maps for trained dual models.

Outputs per image:
  - Side-by-side panel [input | H | V | F] (same colormap for H/V/F so they are visually comparable)
  - H vs V histogram overlay (addresses professor's "두 맵의 분포를 히스토그램으로 비교" check)

Also saves an aggregate histogram across N images to runs/visualize/<name>/hist_aggregate.png.

Supports both:
  - 3-ch Segmenter (3-B variants: BCE heat + EDL Dirichlet)
  - 4-ch Segmenter (3-A Full TMC: Dirichlet heat + Dirichlet EDL)

Example:
  ESOD_FUSION_MODE=noisy_or_vac ESOD_VAC_NORM=minmax_img \
    python scripts/visualize_hvf.py \
        --weights runs/train/dual_noisy_or/weights/best.pt \
        --data data/visdrone.yaml \
        --n-images 10 \
        --name 3B_a_noisy_or
"""
import argparse
import os
import sys
from pathlib import Path

# Allow running from project root or scripts/
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from models.experimental import attempt_load
from utils.datasets import create_dataloader, norm_imgs
from utils.general import check_dataset, check_img_size, increment_path, colorstr
from utils.torch_utils import select_device


def _to_uint8(img_tensor):
    """Convert [3,H,W] float tensor (already normalized for model) back to uint8 HxWx3 for matplotlib."""
    img = img_tensor.detach().cpu().float()
    # Reverse mean/std-like normalization (we just clamp/scale from min/max for display)
    img = (img - img.min()) / (img.max() - img.min() + 1e-6)
    img = (img * 255).clamp(0, 255).numpy().astype(np.uint8)
    return img.transpose(1, 2, 0)


def extract_maps(pred_masks_0):
    """Return dict of (H, V, F) numpy arrays for the first batch element.

    pred_masks_0: [B,C,H,W] tensor (Segmenter output).
    """
    pm = pred_masks_0.detach().float()
    C = pm.shape[1]

    if C == 1:
        # Plain heat-only baseline (Segmenter,[1])
        H = pm[0, 0].sigmoid().cpu().numpy()
        V = None
        Fm = H
    elif C == 2:
        # EDL-only (Segmenter,[2])
        e = F.softplus(pm)
        alpha = e + 1.0
        S = alpha.sum(dim=1)  # [B,H,W]
        p_obj = (alpha[:, 1] / S).cpu().numpy()[0]
        vacuity = (2.0 / S).cpu().numpy()[0]
        H = p_obj          # EDL "heat" interpretation = expected p_obj
        V = vacuity
        Fm = vacuity       # EDL-only uses vacuity as the patch-score map
    elif C == 3:
        # 3-B dual: ch0 = BCE heat logit, ch1-2 = EDL Dirichlet
        from models.common import parse_dual_3ch, _get_fusion_mode
        parsed = parse_dual_3ch(pm, fusion_mode=_get_fusion_mode())
        H = parsed['heat_p'].cpu().numpy()[0]
        V = parsed['vacuity'].cpu().numpy()[0]
        Fm = parsed['mask_pred'].cpu().numpy()[0]
    elif C == 4:
        # 3-A Full TMC dual: ch0-1 = Heat Dirichlet, ch2-3 = EDL Dirichlet
        from models.common import parse_dual_4ch, _get_fusion_mode
        parsed = parse_dual_4ch(pm, fusion_mode=_get_fusion_mode())
        H = parsed['heat_p'].cpu().numpy()[0]
        V = parsed['edl_vac'].cpu().numpy()[0]
        Fm = parsed['mask_pred'].cpu().numpy()[0]
    else:
        raise ValueError(f"Unsupported Segmenter channels: {C}")

    return {'H': H, 'V': V, 'F': Fm, 'C': C}


def plot_panel(img_uint8, maps, out_path, title):
    """Plot input | H | V | F in one row + H/V histogram overlay below."""
    H, V, Fm = maps['H'], maps['V'], maps['F']
    fig = plt.figure(figsize=(20, 6))
    gs = fig.add_gridspec(2, 4, height_ratios=[3, 1.2])

    ax0 = fig.add_subplot(gs[0, 0]); ax0.imshow(img_uint8); ax0.set_title('input'); ax0.axis('off')
    # Same vmin/vmax 0..1 across maps so a pixel value of 0.5 looks identical in H/V/F
    cmap = 'viridis'
    ax1 = fig.add_subplot(gs[0, 1])
    im1 = ax1.imshow(H, vmin=0, vmax=1, cmap=cmap)
    ax1.set_title(f'H  (mean={H.mean():.3f}, p50={np.median(H):.3f})'); ax1.axis('off')
    fig.colorbar(im1, ax=ax1, fraction=0.046)

    if V is not None:
        ax2 = fig.add_subplot(gs[0, 2])
        im2 = ax2.imshow(V, vmin=0, vmax=1, cmap=cmap)
        ax2.set_title(f'V  (mean={V.mean():.3f}, p50={np.median(V):.3f})'); ax2.axis('off')
        fig.colorbar(im2, ax=ax2, fraction=0.046)
    else:
        ax2 = fig.add_subplot(gs[0, 2]); ax2.axis('off'); ax2.set_title('V (n/a)')

    ax3 = fig.add_subplot(gs[0, 3])
    im3 = ax3.imshow(Fm, vmin=0, vmax=1, cmap=cmap)
    ax3.set_title(f'F  (mean={Fm.mean():.3f}, p50={np.median(Fm):.3f})'); ax3.axis('off')
    fig.colorbar(im3, ax=ax3, fraction=0.046)

    # Histogram overlay (H vs V) — addresses professor's distribution-comparison check
    axh = fig.add_subplot(gs[1, :])
    bins = np.linspace(0, 1, 51)
    axh.hist(H.flatten(), bins=bins, alpha=0.5, label='H', color='C0', density=True)
    if V is not None:
        axh.hist(V.flatten(), bins=bins, alpha=0.5, label='V', color='C3', density=True)
    axh.hist(Fm.flatten(), bins=bins, alpha=0.4, label='F', color='C2', histtype='step', linewidth=2, density=True)
    axh.set_xlim(0, 1); axh.set_xlabel('value'); axh.set_ylabel('density')
    axh.set_title(f'H vs V vs F distribution  (V saturated near 1.0 → KL/EDL diagnosis)')
    axh.legend()

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def aggregate_histogram(all_H, all_V, all_F, out_path, title):
    fig, ax = plt.subplots(figsize=(10, 5))
    bins = np.linspace(0, 1, 81)
    ax.hist(all_H, bins=bins, alpha=0.5, label=f'H (n={len(all_H)})', color='C0', density=True)
    if len(all_V) > 0:
        ax.hist(all_V, bins=bins, alpha=0.5, label=f'V (n={len(all_V)})', color='C3', density=True)
    ax.hist(all_F, bins=bins, alpha=0.4, label=f'F (n={len(all_F)})', color='C2',
            histtype='step', linewidth=2, density=True)
    ax.set_xlim(0, 1); ax.set_xlabel('value'); ax.set_ylabel('density')
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main(opt):
    device = select_device(opt.device, batch_size=1)
    model = attempt_load(opt.weights, map_location=device).float().eval()
    gs = max(int(model.stride.max()), 32)
    imgsz = check_img_size(opt.img_size, s=gs)

    with open(opt.data) as f:
        data = yaml.safe_load(f)
    check_dataset(data)

    # Use a minimal dummy opt for create_dataloader
    class _DLOpt: pass
    dlo = _DLOpt(); dlo.single_cls = False
    task = opt.task
    dataloader = create_dataloader(data[task], imgsz, 1, gs, dlo,
                                   pad=0., rect=True, prefix=colorstr(f'{task}: '))[0]

    save_dir = increment_path(Path('runs/visualize') / opt.name, exist_ok=opt.exist_ok)
    save_dir.mkdir(parents=True, exist_ok=True)

    all_H, all_V, all_F = [], [], []
    fusion_mode = os.environ.get('ESOD_FUSION_MODE', 'dempster')
    vac_norm    = os.environ.get('ESOD_VAC_NORM', 'minmax_img')
    print(f'[viz] fusion_mode={fusion_mode}  vac_norm={vac_norm}  save_dir={save_dir}')

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= opt.n_images:
                break
            img, _targets, _masks, _w, paths, _shapes = batch
            img = img.to(device).float()
            img = norm_imgs(img, model)

            # Run model with hm_only=True to get only Segmenter output cheaply
            (_, _), pred_masks = model(img, hm_only=True)
            maps = extract_maps(pred_masks[0])
            img_uint8 = _to_uint8(img[0])

            stem = Path(paths[0]).stem
            out = save_dir / f'{i:02d}_{stem}.png'
            plot_panel(img_uint8, maps, out,
                       title=f'[{i}] {stem}  |  C={maps["C"]}  fusion={fusion_mode}  vac_norm={vac_norm}')

            all_H.append(maps['H'].flatten())
            if maps['V'] is not None:
                all_V.append(maps['V'].flatten())
            all_F.append(maps['F'].flatten())
            print(f'  saved {out.name}')

    all_H = np.concatenate(all_H) if all_H else np.array([])
    all_V = np.concatenate(all_V) if all_V else np.array([])
    all_F = np.concatenate(all_F) if all_F else np.array([])

    aggregate_histogram(all_H, all_V, all_F, save_dir / 'hist_aggregate.png',
                        title=f'Aggregate H/V/F over {opt.n_images} val images  |  fusion={fusion_mode}  vac_norm={vac_norm}')
    # Numeric summary for the prof
    def pct(x):
        if x.size == 0: return None
        return {p: float(np.quantile(x, p/100.)) for p in (10, 25, 50, 75, 90)}
    summary = {
        'fusion_mode': fusion_mode, 'vac_norm': vac_norm,
        'n_images': int(opt.n_images),
        'H_percentiles': pct(all_H),
        'V_percentiles': pct(all_V),
        'F_percentiles': pct(all_F),
    }
    with open(save_dir / 'hist_summary.yaml', 'w') as f:
        yaml.safe_dump(summary, f, default_flow_style=False)
    print(f'\n[viz] summary:\n{yaml.safe_dump(summary, default_flow_style=False)}')
    print(f'[viz] done → {save_dir}/')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, required=True)
    parser.add_argument('--data', type=str, default='data/visdrone.yaml')
    parser.add_argument('--img-size', type=int, default=1536)
    parser.add_argument('--task', type=str, default='val')
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--n-images', type=int, default=10)
    parser.add_argument('--name', type=str, default='hvf')
    parser.add_argument('--exist-ok', action='store_true')
    main(parser.parse_args())
