"""Convert TrackNetV3 (tracking network only, bg_mode=concat, seq_len=8) to Core ML.

Input  'frames'  : (1, 27, 288, 512) float, RGB/255: [median background, 8 frames], channels-first.
Output 'heatmaps': (1, 8, 288, 512) float in 0..1 (sigmoid), one heatmap per frame.

Then checks the converted model against PyTorch on real frames: same preprocessing
as the author's Video_IterableDataset, compares the detected shuttle positions.

  ../.venv-coreml/bin/python convert_tracknet.py VIDEO
"""
import sys
from pathlib import Path

import coremltools as ct
import cv2
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
VENDOR = ROOT / '04_原型/vendor/TrackNetV3'
sys.path.insert(0, str(VENDOR))
from model import TrackNet  # noqa: E402

H, W, SEQ = 288, 512, 8
OUT = Path(__file__).resolve().parents[1] / 'HawkEye/Resources/TrackNetV3.mlpackage'


def load_torch():
    ck = torch.load(ROOT / '04_原型/models/tracknetv3/TrackNet_best.pt', map_location='cpu', weights_only=True)
    assert ck['param_dict']['seq_len'] == SEQ and ck['param_dict']['bg_mode'] == 'concat'
    m = TrackNet((SEQ + 1) * 3, SEQ)
    m.load_state_dict(ck['model'], strict=True)
    return m.eval()


def prep(frames_bgr, median_rgb):
    """Author preprocessing: RGB, PIL resize to 512x288, channels-first, /255."""
    chans = [np.moveaxis(np.array(Image.fromarray(median_rgb.astype('uint8')).resize((W, H))), -1, 0)]
    for f in frames_bgr:
        chans.append(np.moveaxis(np.array(Image.fromarray(f[..., ::-1]).resize((W, H))), -1, 0))
    return (np.concatenate(chans, 0)[None].astype(np.float32) / 255.0)


def peak(heat, th=0.5):
    """Centre of the largest above-threshold blob (author's predict_location), or None."""
    m = (heat > th).astype(np.uint8)
    if not m.any():
        return None
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x, y, w, h = max((cv2.boundingRect(c) for c in cnts), key=lambda r: r[2] * r[3])
    return (x + w / 2, y + h / 2)


def main():
    model = load_torch()
    example = torch.rand(1, (SEQ + 1) * 3, H, W)
    traced = torch.jit.trace(model, example)
    ml = ct.convert(traced, inputs=[ct.TensorType(name='frames', shape=example.shape)],
                    outputs=[ct.TensorType(name='heatmaps')], convert_to='mlprogram',
                    compute_precision=ct.precision.FLOAT16, minimum_deployment_target=ct.target.iOS17)
    ml.short_description = 'TrackNetV3 shuttlecock heatmaps (bg_mode=concat, seq_len=8)'
    ml.save(str(OUT))
    print('saved', OUT)
    if len(sys.argv) < 2:
        return
    cap = cv2.VideoCapture(sys.argv[1])
    frames = []
    while len(frames) < 400:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    median = np.median(np.stack(frames[::6]), 0)[..., ::-1]
    agree = both = only_t = only_c = 0
    dists = []
    for s in range(0, len(frames) - SEQ, SEQ):
        x = prep(frames[s:s + SEQ], median)
        with torch.no_grad():
            ht = model(torch.from_numpy(x)).numpy()[0]
        hc = ml.predict({'frames': x})['heatmaps'][0]
        for a, b in zip(ht, hc):
            pa, pb = peak(a), peak(b)
            if pa and pb:
                both += 1
                d = np.hypot(pa[0] - pb[0], pa[1] - pb[1])
                dists.append(d)
                agree += d <= 2
            elif pa:
                only_t += 1
            elif pb:
                only_c += 1
    n = len(dists)
    print(f'frames compared: {SEQ * (len(frames) // SEQ - 1)}; detected by both {both}, torch only {only_t}, coreml only {only_c}')
    if n:
        print(f'position difference (512x288 px): median {np.median(dists):.2f}, max {np.max(dists):.2f}; within 2 px {agree}/{n}')


if __name__ == '__main__':
    main()
