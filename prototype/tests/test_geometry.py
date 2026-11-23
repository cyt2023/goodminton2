"""Synthetic checks: a known camera views the court; clicks and detections get
known noise. Checks the ground mapping, the distortion round trip, the signed
distance, and that the 95% ellipse covers about 95% of true positions when the
noise model is right.

Run:  ../../04_原型/.venv-vision/bin/python -m pytest tests -q   (from prototype/)
  or: python tests/test_geometry.py
"""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import calib  # noqa: E402
import court  # noqa: E402
import uncertainty  # noqa: E402

W_IMG, H_IMG = 3840, 2160
K = np.array([[2900.0, 0, W_IMG / 2], [0, 2900.0, H_IMG / 2], [0, 0, 1]])


def camera(height=1.8):
    """Camera beside the near-left corner, outside the court, looking at it."""
    C = np.array([-1.5, -2.0, height])
    target = np.array([court.W / 2, court.L * 0.45, 0])
    z = (target - C) / np.linalg.norm(target - C)
    x = np.cross(z, [0, 0, 1.0]); x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.vstack([x, y, z])
    return R, -R @ C


def to_img(R, t, ground, dist=None):
    P = np.c_[np.asarray(ground, float), np.zeros(len(ground))]
    px, _ = cv2.projectPoints(P.reshape(-1, 1, 3), cv2.Rodrigues(R)[0], t, K,
                              np.zeros(5) if dist is None else dist)
    return px.reshape(-1, 2)


def record_for(R, t, sigma, rng, dist=None):
    kp = court.default_keypoints()
    names = [n for n in kp if n.split('/')[1] in ('doubles_left', 'doubles_right', 'centre')
             and n.split('/')[0] in ('back_near', 'short_service_near', 'short_service_far', 'back_far')]
    img = to_img(R, t, [kp[n] for n in names], dist) + rng.normal(0, sigma, (len(names), 2))
    return {'court_points': {n: p.tolist() for n, p in zip(names, img)}, 'image_size': [W_IMG, H_IMG]}


def test_mapping_noise_free():
    R, t = camera()
    rng = np.random.default_rng(0)
    cal = calib.build(record_for(R, t, 0.0, rng), click_sigma_px=0.01, n_samples=20)
    g = np.array([[0.0, 0.0], [6.1, 13.4], [0.46, 10.0]])
    est = calib.project(cal['H'], to_img(R, t, g))
    assert np.abs(est - g).max() < 1e-6


def test_distortion_round_trip():
    dist = np.array([0.08, -0.05, 0, 0, 0])
    intr = (K, dist)
    p = np.array([[100.0, 200.0], [1900, 1000], [3700, 2100]])
    assert np.abs(calib.distort(calib.undistort(p, intr), intr) - p).max() < 0.05


def test_mapping_with_distortion():
    dist = np.array([0.08, -0.05, 0, 0, 0])
    R, t = camera()
    rng = np.random.default_rng(0)
    import json, tempfile
    with tempfile.NamedTemporaryFile('w', suffix='.json', delete=False) as f:
        json.dump({'K': K.tolist(), 'dist': dist.tolist()}, f)
    cal = calib.build(record_for(R, t, 0.0, rng, dist), intrinsics_file=f.name, click_sigma_px=0.01, n_samples=20)
    g = np.array([[0.2, 1.0], [5.9, 12.0]])
    est = calib.project(cal['H'], calib.undistort(to_img(R, t, g, dist), cal['intrinsics']))
    assert np.abs(est - g).max() < 1e-3


def test_signed_distance():
    poly = court.boundary('singles')
    assert abs(uncertainty.signed_distance((0.46, 5.0), poly)) < 1e-12        # on the outer edge -> 0 (IN)
    assert uncertainty.signed_distance((0.45, 5.0), poly) < 0
    assert abs(uncertainty.signed_distance((3.05, 13.41), poly) + 0.01) < 1e-9


def test_coverage_95():
    """True point should fall inside the 95% ellipse ~95% of the time."""
    R, t = camera()
    sigma_click, sigma_det = 1.5, 2.0
    rng = np.random.default_rng(42)
    poly = court.boundary('singles')
    inside, trials = 0, 200
    for k in range(trials):
        cal = calib.build(record_for(R, t, sigma_click, rng), click_sigma_px=sigma_click,
                          n_samples=200, seed=k)
        g = np.array([rng.uniform(0.3, 5.8), rng.uniform(0.5, 13.0)])
        px = to_img(R, t, [g])[0] + rng.normal(0, sigma_det, 2)
        r = uncertainty.locate(cal, px, uncertainty.pixel_cov(sigma_det), poly, n=1000, seed=k)
        diff = g - np.array(r['court_xy_m'])
        m2 = diff @ np.linalg.inv(np.array(r['court_cov_m2'])) @ diff
        inside += m2 <= 5.991
    rate = inside / trials
    print(f'95% ellipse coverage on synthetic data: {rate:.3f}')
    assert 0.88 <= rate <= 0.99


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn()
            print('ok', name)
