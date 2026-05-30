"""On-the-fly COCO-format size-stratified mAP (AP_s / AP_m / AP_l).

Builds GT JSON from the dataloader's labels (no external annotation file required) and runs
pycocotools COCOeval to get the COCO-standard size breakdown:
    small  : area <  32**2 = 1024
    medium : 32**2 <= area < 96**2 = 9216
    large  : area >= 96**2

Use from test.py:

    from utils.coco_size_eval import build_coco_gt_from_dataloader, run_coco_size_eval
    coco_gt = build_coco_gt_from_dataloader(dataloader, names)
    metrics = run_coco_size_eval(coco_gt, jdict)
    # metrics = {'AP': ..., 'AP50': ..., 'AP75': ..., 'AP_s': ..., 'AP_m': ..., 'AP_l': ...,
    #            'AR_s': ..., 'AR_m': ..., 'AR_l': ...}
"""
import io
import json
import contextlib
from pathlib import Path

import numpy as np


def _load_label_file(label_path, img_w, img_h):
    """Read a YOLO-format label file and return list of (cls, x_min, y_min, w_pix, h_pix) in pixels."""
    labels = []
    if not label_path.exists():
        return labels
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls = int(float(parts[0]))
            xc = float(parts[1]) * img_w
            yc = float(parts[2]) * img_h
            w  = float(parts[3]) * img_w
            h  = float(parts[4]) * img_h
            x_min = xc - w / 2.0
            y_min = yc - h / 2.0
            labels.append((cls, x_min, y_min, w, h))
    return labels


def _label_path_for(img_path: str) -> Path:
    """YOLO convention: <root>/images/<name>.jpg → <root>/labels/<name>.txt
    Replaces the LAST '/images/' in the path with '/labels/' and swaps extension to .txt."""
    p = str(img_path)
    if '/images/' in p:
        idx = p.rfind('/images/')
        p2 = p[:idx] + '/labels/' + p[idx + len('/images/'):]
    else:
        p2 = p
    p2 = Path(p2)
    return p2.with_suffix('.txt')


def build_coco_gt_from_dataloader(dataloader, class_names):
    """Walk the dataloader's underlying dataset and build a COCO-format GT dict.

    dataloader.dataset must expose .img_files and .shapes (YOLOv5/ESOD convention).
    Returns the dict (not yet a pycocotools object).
    """
    ds = dataloader.dataset
    img_files = list(ds.img_files)
    shapes = getattr(ds, 'shapes', None)  # array of (w, h) per image at orig resolution

    images, annotations = [], []
    ann_id = 1
    for img_id, img_path in enumerate(img_files, start=1):
        # Resolve image size
        if shapes is not None:
            try:
                w_orig, h_orig = int(shapes[img_id - 1][0]), int(shapes[img_id - 1][1])
            except Exception:
                w_orig = h_orig = None
        else:
            w_orig = h_orig = None
        if w_orig is None:
            from PIL import Image
            with Image.open(img_path) as im:
                w_orig, h_orig = im.size

        images.append({
            'id': img_id,
            'file_name': Path(img_path).name,
            'width':  int(w_orig),
            'height': int(h_orig),
        })

        for cls, x, y, w, h in _load_label_file(_label_path_for(img_path), w_orig, h_orig):
            annotations.append({
                'id': ann_id,
                'image_id': img_id,
                'category_id': int(cls) + 1,        # COCO category ids are 1-based
                'bbox': [float(x), float(y), float(w), float(h)],
                'area': float(w * h),
                'iscrowd': 0,
                'segmentation': [],
            })
            ann_id += 1

    categories = [{'id': i + 1, 'name': n} for i, n in enumerate(class_names)]
    return {
        'images': images,
        'annotations': annotations,
        'categories': categories,
        '_image_id_by_path': {str(p): i + 1 for i, p in enumerate(img_files)},
    }


def _normalize_jdict_for_cocoeval(jdict, image_id_by_path, image_id_by_stem):
    """jdict entries from test.py use image_id = stem (filename without ext) by default.
    We need numeric image_ids matching the GT built above."""
    out = []
    for d in jdict:
        d = dict(d)  # shallow copy
        # If image_id is already int and present in GT, keep it
        if isinstance(d.get('image_id'), int) and d['image_id'] in image_id_by_path.values():
            out.append(d); continue
        # Otherwise look up by stem
        stem = str(d.get('image_id'))
        if stem in image_id_by_stem:
            d['image_id'] = image_id_by_stem[stem]
            d['category_id'] = int(d['category_id']) + 1   # 0-indexed → 1-indexed COCO
            out.append(d)
        # else: silently drop (image not in GT)
    return out


def run_coco_size_eval(coco_gt_dict, jdict, iou_type='bbox'):
    """Run pycocotools COCOeval on built GT + predictions.

    Returns a dict with summary stats.
    """
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    image_id_by_path = coco_gt_dict.pop('_image_id_by_path', {})
    image_id_by_stem = {Path(p).stem: i for p, i in image_id_by_path.items()}

    # Save GT to a temp file so COCO() can load it
    import tempfile
    with tempfile.NamedTemporaryFile('w', suffix='.json', delete=False) as f:
        json.dump(coco_gt_dict, f)
        gt_path = f.name

    preds = _normalize_jdict_for_cocoeval(jdict, image_id_by_path, image_id_by_stem)
    with tempfile.NamedTemporaryFile('w', suffix='.json', delete=False) as f:
        json.dump(preds, f)
        pred_path = f.name

    coco_gt = COCO(gt_path)
    coco_dt = coco_gt.loadRes(pred_path)
    coco_eval = COCOeval(coco_gt, coco_dt, iou_type)
    coco_eval.evaluate()
    coco_eval.accumulate()
    # Capture summarize() text output
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        coco_eval.summarize()
    text = buf.getvalue()
    print(text)

    s = coco_eval.stats  # 12-vector standard COCO summary
    return {
        'AP':    float(s[0]),
        'AP50':  float(s[1]),
        'AP75':  float(s[2]),
        'AP_s':  float(s[3]),
        'AP_m':  float(s[4]),
        'AP_l':  float(s[5]),
        'AR_1':  float(s[6]),
        'AR_10': float(s[7]),
        'AR_100':float(s[8]),
        'AR_s':  float(s[9]),
        'AR_m':  float(s[10]),
        'AR_l':  float(s[11]),
        'summary_text': text,
    }
