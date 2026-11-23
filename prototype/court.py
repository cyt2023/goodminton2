"""Badminton court geometry in metres.

Coordinate system (ground plane, z = 0):
  x across the court, 0 .. 6.10 (outer edge of one doubles sideline to the other)
  y along the court, 0 .. 13.40 (outer edge of one back boundary line to the other)
  The net is at y = 6.70.

Dimensions follow the BWF Laws of Badminton: all court dimensions are measured
to the outer edges of the lines, lines are 40 mm wide, and lines belong to the
area they bound, so a shuttle touching a boundary line is IN. The in/out
boundary is therefore the OUTER edge of each boundary line.

Calibration keypoints are the intersections of line CENTRELINES, because that
is the point a person can click most repeatably. Measure the real court with a
tape before experiments and override these values in a court JSON if needed.
"""
import json
from pathlib import Path

W, L = 6.10, 13.40
NET_Y = L / 2
LINE = 0.04
H = LINE / 2  # half line width

# Outer edges (towards out-of-court) of each line, BWF Laws.
SINGLES_INSET = 0.46      # doubles sideline outer edge -> singles sideline outer edge
LONG_SERVICE_INSET = 0.76  # back line outer edge -> doubles long service line far edge
SHORT_SERVICE_FROM_NET = 1.98  # net -> outer edge of short service line

# Centreline coordinates of each line.
X_CENTRES = {
    'doubles_left': H, 'singles_left': SINGLES_INSET + H, 'centre': W / 2,
    'singles_right': W - SINGLES_INSET - H, 'doubles_right': W - H,
}
Y_CENTRES = {
    'back_near': H, 'long_service_near': LONG_SERVICE_INSET - H,
    'short_service_near': NET_Y - SHORT_SERVICE_FROM_NET + H,
    'short_service_far': NET_Y + SHORT_SERVICE_FROM_NET - H,
    'long_service_far': L - LONG_SERVICE_INSET + H, 'back_far': L - H,
}


def default_keypoints():
    """All 30 line-centreline intersections (the centre line runs from each back
    line to the short service line, so it meets every cross line)."""
    return {f'{yn}/{xn}': (x, y) for yn, y in Y_CENTRES.items() for xn, x in X_CENTRES.items()}


def boundary(mode):
    """In-court polygon (outer edges of boundary lines) for a rally, not a serve."""
    if mode == 'doubles':
        x0, x1 = 0.0, W
    elif mode == 'singles':
        x0, x1 = SINGLES_INSET, W - SINGLES_INSET
    else:
        raise ValueError("mode must be 'singles' or 'doubles'")
    return [(x0, 0.0), (x1, 0.0), (x1, L), (x0, L)]


def line_segments():
    """Centreline segments of every painted line, for drawing overlays."""
    segs = []
    for y in Y_CENTRES.values():
        segs.append(((H, y), (W - H, y)))
    for name, x in X_CENTRES.items():
        if name == 'centre':
            segs.append(((x, Y_CENTRES['back_near']), (x, Y_CENTRES['short_service_near'])))
            segs.append(((x, Y_CENTRES['short_service_far']), (x, Y_CENTRES['back_far'])))
        else:
            segs.append(((x, H), (x, L - H)))
    return segs


def load_keypoints(court_file=None):
    if court_file is None:
        return default_keypoints()
    value = json.loads(Path(court_file).read_text())
    return {k: tuple(v) for k, v in value['keypoints_m'].items()}


EDGES = ('left', 'right', 'near', 'far')
EDGE_CN = {'left': '左边线', 'right': '右边线', 'near': '近端底线', 'far': '远端底线'}


def edge_distance(p, mode, edge):
    """Signed distance (m) from court point p to one boundary edge (outer edge of the
    line), positive on the court side. Edges as in nearest_side(): left/right seen
    from the near end (y=0) looking at the net."""
    (x0, y0), (x1, _), _, (_, y1) = boundary(mode)
    x, y = float(p[0]), float(p[1])
    return {'left': x - x0, 'right': x1 - x, 'near': y - y0, 'far': y1 - y}[edge]
