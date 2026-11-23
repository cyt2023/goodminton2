"""Synthetic 2D tracks for the landing detector and its rally-context check.

Run:  ../../04_原型/.venv-vision/bin/python tests/test_landing.py   (from prototype/)
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import landing  # noqa: E402

FPS, WIDTH, N = 30, 1280, 300
REST = np.array([500.0, 500.0])


def track():
    return np.full((N, 2), np.nan)


def fly(xy, f0, f1, p0, p1):
    for f in range(f0, f1 + 1):
        xy[f] = p0 + (p1 - p0) * (f - f0) / (f1 - f0)


def rest(xy, f0, f1, p=REST):
    rng = np.random.default_rng(f0)
    for f in range(f0, f1 + 1):
        xy[f] = p + rng.normal(0, 0.7, 2)


def only(xy):
    found, _ = landing.find_landings(xy, FPS, WIDTH)
    assert len(found) == 1, found
    return found[0]


def test_rally_landing_with_occlusion_and_false_detection():
    xy = track()
    fly(xy, 50, 70, np.array([800.0, 300.0]), np.array([550.0, 450.0]))   # flight, then occluded
    xy[78] = [100, 250]                                                    # false detection elsewhere
    rest(xy, 85, 160)
    lnd = only(xy)
    ctx = lnd['rally_context']
    assert ctx['is_rally_landing'], ctx
    assert lnd['contact_frame'] == 85
    assert lnd['last_flight_frame'] == 70
    v = np.linalg.norm(lnd['approach_velocity_px_per_frame'])
    assert 10 < v < 16, v          # flight speed, not the jump to the false detection


def test_broken_rest_is_one_landing():
    """Rest interrupted by an occlusion while a stray detection appears elsewhere
    (seen on the smoke clip with the Core ML tracker): still one landing."""
    xy = track()
    fly(xy, 50, 70, np.array([800.0, 300.0]), np.array([550.0, 450.0]))
    rest(xy, 85, 100)
    xy[105] = [100, 250]                                  # stray far detection during the gap
    rest(xy, 110, 160)
    found, _ = landing.find_landings(xy, FPS, WIDTH)
    assert len(found) == 1, [(l['contact_frame'], l['rest_frame_end']) for l in found]
    assert found[0]['rest_frame_end'] >= 160 - 1


def test_contact_is_at_rest_point_not_while_descending():
    xy = track()
    fly(xy, 50, 80, np.array([700.0, 300.0]), REST)    # visible all the way down
    rest(xy, 81, 160)
    lnd = only(xy)
    assert np.linalg.norm(np.asarray(lnd['contact_px_centre']) - REST) <= 2 * max(3, 0.004 * WIDTH)
    assert lnd['rally_context']['is_rally_landing']


def test_quick_scoop_is_not_rally():
    xy = track()
    fly(xy, 50, 80, np.array([700.0, 300.0]), REST)
    rest(xy, 81, 92)                                    # 0.4 s on the floor
    fly(xy, 93, 110, REST, np.array([450.0, 300.0]))   # scooped up
    lnd = only(xy)
    ctx = lnd['rally_context']
    assert ctx['checks']['flight'] and not ctx['checks']['rest'], ctx
    assert not ctx['is_rally_landing']


def test_short_drop_is_not_rally():
    xy = track()
    fly(xy, 60, 66, np.array([500.0, 430.0]), np.array([500.0, 490.0]))   # dropped from the hand
    xy[40] = [900, 200]                                                    # unrelated far detection
    rest(xy, 67, 160)
    found, _ = landing.find_landings(xy, FPS, WIDTH)
    assert all(not l['rally_context']['is_rally_landing'] for l in found), found


def test_juggle_apex_is_not_rally():
    """Shuttle tossed up on the racket and hanging near the apex (as at frame ~196
    of the smoke clip): it looks still for ~0.3 s, then falls again."""
    xy = track()
    apex = np.array([500.0, 330.0])
    fly(xy, 40, 55, REST, apex)
    rest(xy, 56, 66, apex)
    fly(xy, 67, 80, apex, REST)
    found, _ = landing.find_landings(xy, FPS, WIDTH)
    at_apex = [l for l in found if np.linalg.norm(np.asarray(l['rest_px']) - apex) < 10]
    assert at_apex, found
    assert all(not l['rally_context']['is_rally_landing'] for l in at_apex), at_apex


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn()
            print('ok', name)
