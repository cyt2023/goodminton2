"""Propagate pixel and calibration uncertainty to a landing call.

Monte Carlo: each sample combines one calibration homography sample with one
pixel-noise sample (detection noise plus within-frame motion), projects it to
the court, and the fraction of samples inside the court polygon is P(in).
The call is IN / OUT when P(in) is beyond 1 - alpha / alpha, else ABSTAIN.
"""
import numpy as np

import calib


def pixel_cov(det_sigma_px, velocity_px_per_frame=None):
    cov = np.eye(2) * det_sigma_px ** 2
    if velocity_px_per_frame is not None:
        v = np.asarray(velocity_px_per_frame, float)
        # Contact happens somewhere within one frame interval: uniform along v.
        cov += np.outer(v, v) / 12.0
    return cov


def signed_distance(p, poly):
    """Distance (m) from p to the rectangle boundary; positive inside."""
    xs, ys = [q[0] for q in poly], [q[1] for q in poly]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    x, y = p
    if x0 <= x <= x1 and y0 <= y <= y1:
        return min(x - x0, x1 - x, y - y0, y1 - y)
    dx = max(x0 - x, 0, x - x1)
    dy = max(y0 - y, 0, y - y1)
    return -float(np.hypot(dx, dy))


def nearest_side(p, poly):
    xs, ys = [q[0] for q in poly], [q[1] for q in poly]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    d = {'左边线': abs(p[0] - x0), '右边线': abs(x1 - p[0]), '近端底线': abs(p[1] - y0), '远端底线': abs(y1 - p[1])}
    return min(d, key=d.get)


def locate(cal, pixel, cov_px, poly, contact_radius_m=0.0, alpha=0.05, n=4000, seed=1):
    rng = np.random.default_rng(seed)
    Hs = cal['H_samples']
    raw = rng.multivariate_normal(np.asarray(pixel, float), cov_px, size=n)
    above = float((calib.ground_side(cal, raw) < 0).mean())
    pix = calib.undistort(raw, cal['intrinsics'])
    ground = np.empty((n, 2))
    for i in range(n):
        ground[i] = calib.project(Hs[i % len(Hs)], pix[i:i + 1])[0]
    nominal = calib.project(cal['H'], calib.undistort([pixel], cal['intrinsics']))[0]
    d = np.array([signed_distance(g, poly) for g in ground]) + contact_radius_m
    p_in = float((d >= 0).mean())
    call = 'IN' if p_in >= 1 - alpha else 'OUT' if p_in <= alpha else 'ABSTAIN'
    reason = None
    if above > 0.01:
        # Pixel noise reaches the floor's horizon: the ground position is unbounded.
        call, reason = 'ABSTAIN', f'{above:.0%} of samples at or above the floor horizon'
    elif call == 'ABSTAIN':
        reason = 'uncertainty region crosses the boundary'
    cov = np.cov(ground.T)
    d_nom = signed_distance(nominal, poly) + contact_radius_m
    return {
        'court_xy_m': nominal.tolist(), 'court_cov_m2': cov.tolist(),
        'court_sigma_m': np.sqrt(np.diag(cov)).tolist(),
        'signed_distance_m': float(d_nom), 'signed_distance_sigma_m': float(d.std()),
        'nearest_boundary': nearest_side(nominal, poly),
        'p_in': p_in, 'call': call, 'abstain_reason': reason, 'alpha': alpha, 'n_samples': n,
        'fraction_above_horizon': above,
    }


def ellipse_points(mean, cov, k=2.4477, m=48):
    """95% (chi2, 2 dof) confidence ellipse outline in court metres."""
    vals, vecs = np.linalg.eigh(np.asarray(cov))
    vals = np.clip(vals, 0, None)
    t = np.linspace(0, 2 * np.pi, m)
    circle = np.c_[np.cos(t), np.sin(t)] * (k * np.sqrt(vals))
    return np.asarray(mean) + circle @ vecs.T
