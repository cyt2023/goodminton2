"""Ground-plane calibration (image pixels -> court metres) with uncertainty.

A single camera can only locate a point that lies on the floor, so the ground
mapping is a homography fitted to clicked line intersections. Calibration
uncertainty is represented by Monte Carlo homography samples obtained by
re-fitting after perturbing every clicked point with the click noise.
"""
import json
from pathlib import Path

import cv2
import numpy as np

import court


def load_intrinsics(file):
    if not file:
        return None
    v = json.loads(Path(file).read_text())
    return np.asarray(v['K'], float), np.asarray(v['dist'], float).ravel()


def undistort(points, intr):
    pts = np.asarray(points, float).reshape(-1, 2)
    if intr is None:
        return pts
    K, dist = intr
    return cv2.undistortPoints(pts.reshape(-1, 1, 2), K, dist, P=K).reshape(-1, 2)


def distort(points, intr):
    """Inverse of undistort: map ideal pixel coordinates back to the raw image."""
    pts = np.asarray(points, float).reshape(-1, 2)
    if intr is None:
        return pts
    K, dist = intr
    norm = np.c_[pts, np.ones(len(pts))] @ np.linalg.inv(K).T
    out, _ = cv2.projectPoints(norm.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), K, dist)
    return out.reshape(-1, 2)


def court_to_image(cal, court_pts):
    """Court metres -> raw image pixels (including lens distortion if known).
    Court points beyond the camera's horizon have no image; they come back NaN."""
    pts = np.asarray(court_pts, float).reshape(-1, 2)
    w = np.c_[pts, np.ones(len(pts))] @ cal['H_inv'][2]
    ok = np.sign(w) == cal['wc_sign']
    out = np.full((len(pts), 2), np.nan)
    if ok.any():
        out[ok] = distort(project(cal['H_inv'], pts[ok]), cal['intrinsics'])
    return out


def project(Hm, pts):
    pts = np.asarray(pts, float).reshape(-1, 2)
    p = np.c_[pts, np.ones(len(pts))] @ Hm.T
    return p[:, :2] / p[:, 2:3]


def fit(img_pts, court_pts):
    """Image -> court homography. Fitted in the court -> image direction so the
    least-squares error is measured in pixels, where the click noise lives."""
    Hci, _ = cv2.findHomography(np.asarray(court_pts, float), np.asarray(img_pts, float), method=0)
    if Hci is None or not np.isfinite(Hci).all() or abs(np.linalg.det(Hci)) < 1e-12:
        raise ValueError('Degenerate court calibration: check that clicked points are not collinear')
    Hm = np.linalg.inv(Hci)
    return Hm / Hm[2, 2]


def build(calib_record, intrinsics_file=None, click_sigma_px=1.5, n_samples=400, seed=0):
    """Fit H from a saved click record and draw Monte Carlo samples of H."""
    keypoints = court.load_keypoints(calib_record.get('court_file'))
    names = [n for n in calib_record['court_points'] if n in keypoints]
    unknown = [n for n in calib_record['court_points'] if n not in keypoints]
    if unknown:
        raise ValueError(f'Unknown keypoint names: {unknown}')
    if len(names) < 4:
        raise ValueError('Need at least 4 clicked court keypoints (6+ recommended, spread over the court)')
    intr = load_intrinsics(intrinsics_file)
    img = undistort([calib_record['court_points'][n] for n in names], intr)
    world = np.asarray([keypoints[n] for n in names], float)
    Hm = fit(img, world)
    Hinv = np.linalg.inv(Hm)
    reproj = project(Hinv, world)
    resid_px = np.linalg.norm(reproj - img, axis=1)
    dof = 2 * len(names) - 8
    sigma_est = float(np.sqrt((resid_px ** 2).sum() / dof)) if dof > 0 else None
    # Use the larger of the prior click noise and the residual-based estimate.
    sigma = max(click_sigma_px, sigma_est or 0.0)
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(n_samples):
        try:
            samples.append(fit(img + rng.normal(0, sigma, img.shape), world))
        except ValueError:
            continue
    w_sign = float(np.sign(np.median(np.c_[img, np.ones(len(img))] @ Hm[2])))
    wc_sign = float(np.sign(np.median(np.c_[world, np.ones(len(world))] @ Hinv[2])))
    return {
        'H': Hm, 'H_inv': Hinv, 'H_samples': np.asarray(samples), 'intrinsics': intr, 'w_sign': w_sign, 'wc_sign': wc_sign,
        'names': names, 'residual_px': resid_px.tolist(),
        'residual_rms_px': float(np.sqrt((resid_px ** 2).mean())),
        'click_sigma_prior_px': click_sigma_px, 'click_sigma_estimated_px': sigma_est,
        'click_sigma_used_px': sigma, 'n_points': len(names),
        'image_size': calib_record.get('image_size'),
    }


def ground_side(cal, pixels):
    """+1 for pixels below the floor's horizon (same side as the clicked court
    points), -1 above it. Points above the horizon cannot lie on the floor."""
    pts = undistort(pixels, cal['intrinsics'])
    w = np.c_[pts, np.ones(len(pts))] @ cal['H'][2]
    return np.sign(w) * cal['w_sign']


def summary(cal):
    return {k: cal[k] for k in ['names', 'residual_px', 'residual_rms_px', 'click_sigma_prior_px',
                                'click_sigma_estimated_px', 'click_sigma_used_px', 'n_points']} | {
        'H_image_to_court_m': cal['H'].tolist(), 'n_H_samples': int(len(cal['H_samples'])),
        'intrinsics_used': cal['intrinsics'] is not None}
