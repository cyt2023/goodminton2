"""Field-session logic: truth from reference clicks, candidate merging, phone offset,
psychometric fit.

Run:  ../../04_原型/.venv-vision/bin/python tests/test_session.py   (from prototype/)
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import audio  # noqa: E402
import report  # noqa: E402
import session  # noqa: E402

# Reference image: line drawn from x=500 to x=540 (40 px = 40 mm, so 1 mm/px).


def test_truth_court_on_left():
    # Court is on the left, so the outer edge is x=540. Cork spans 520..546: overlaps the line -> in.
    r = session.truth_from_clicks({'line_a': [500, 300], 'line_b': [540, 300], 'cork_a': [520, 301], 'cork_b': [546, 301]}, 'left')
    assert r['label'] == 'in' and abs(r['edge_mm'] - 20) < 1e-6, r
    assert abs(r['centre_mm'] - 7) < 1e-6 and abs(r['cork_width_mm'] - 26) < 1e-6, r


def test_truth_out_and_click_order_irrelevant():
    # Cork fully beyond the outer edge (x 545..571): out by 5 mm, whatever order the edges are clicked in.
    a = session.truth_from_clicks({'line_a': [540, 300], 'line_b': [500, 300], 'cork_a': [571, 300], 'cork_b': [545, 300]}, 'left')
    assert a['label'] == 'out' and abs(a['edge_mm'] + 5) < 1e-6, a


def test_truth_court_on_right():
    # Court on the right: outer edge is x=500. Cork 480..499.5 lies outside, 0.5 mm away.
    r = session.truth_from_clicks({'line_a': [500, 300], 'line_b': [540, 300], 'cork_a': [480, 300], 'cork_b': [499.5, 300]}, 'right')
    assert r['label'] == 'out' and abs(r['edge_mm'] + 0.5) < 1e-6, r
    # Touching the outer edge exactly counts as in.
    r = session.truth_from_clicks({'line_a': [500, 300], 'line_b': [540, 300], 'cork_a': [474, 300], 'cork_b': [500, 300]}, 'right')
    assert r['label'] == 'in' and abs(r['edge_mm']) < 1e-6, r


def test_truth_scale_from_line_width():
    # Same geometry, line 20 px wide -> 2 mm/px.
    r = session.truth_from_clicks({'line_a': [500, 300], 'line_b': [520, 300], 'cork_a': [525, 300], 'cork_b': [538, 300]}, 'left')
    assert abs(r['edge_mm'] + 10) < 1e-6 and abs(r['cork_width_mm'] - 26) < 1e-6, r


def test_candidates_link_shouts_and_keep_unlinked():
    machine = [{'id': 'm100', 't_sys': 10.0, 'sources': ['machine']}]
    shouts = [{'t': 10.9, 'strength': 2, 'mics': ['ref']}, {'t': 30.0, 'strength': 2, 'mics': ['sys']}]
    c = session.build_candidates(machine, shouts)
    assert [x['id'] for x in c] == ['m100', 's30000'], c
    assert c[0]['sources'] == ['machine', 'shout'] and c[0]['shouts'] == [10.9]
    assert abs(c[1]['t_sys'] - (30.0 - session.SHOUT_REACTION_S)) < 1e-9


def test_merge_shouts_two_mics():
    m = session.merge_shouts([{'t': 5.0, 'strength': 1}], [{'t': 7.1, 'strength': 3}, {'t': 20.0, 'strength': 1}], offset=2.0)
    assert len(m) == 2 and m[0]['mics'] == ['ref', 'sys'] and m[0]['strength'] == 3, m


def test_offset_on_synthetic_clicks():
    sr = 16000
    rng = np.random.default_rng(0)
    x = rng.normal(0, 0.01, sr * 30).astype(np.float32)
    for t in rng.uniform(1, 28, 25):                     # racket-hit-like clicks
        i = int(t * sr)
        x[i:i + 200] += rng.normal(0, 0.5, 200) * np.exp(-np.arange(200) / 40)
    shift = 3.27                                         # reference started 3.27 s later
    y = x[int(shift * sr):] + rng.normal(0, 0.01, len(x) - int(shift * sr)).astype(np.float32)
    off, conf = audio.estimate_offset(x, y, sr)
    assert abs(off + shift) < 0.006 and conf > 3, (off, conf)


def test_psychometric_fit_recovers_parameters():
    rng = np.random.default_rng(1)
    d = rng.uniform(-150, 150, 400)
    said_in = rng.uniform(size=400) < report.norm.cdf((d - (-10)) / 30)
    mu, sg, ok = report.fit_psychometric(d, said_in)
    assert ok and abs(mu + 10) < 8 and abs(sg - 30) < 10, (mu, sg)


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn()
            print('ok', name)
