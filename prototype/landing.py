"""Find shuttle landings in a single-camera 2D track.

A single camera cannot see height, so a landing is recognised from what
happens after it: the shuttle arrives from flight and then stays (almost)
still on the floor. The contact frame is the first observed frame at which
the arriving shuttle is close to its resting position. Only TrackNet's
observed detections are used as evidence; InpaintNet completions are ignored.

Not every resting shuttle ends a rally: players juggle it on the racket, drop
it and scoop it up between points. rally_context() keeps a candidate only with
a coherent flight before the rest and a long enough rest after it.

TrackNet returns the centre of the shuttle, which at contact is several cm
above the floor; projecting it through the ground homography shifts the point
away from the camera. refine_contact() therefore looks for the lowest pixel of
the shuttle blob (the cork touching the floor) and reports which was used.
"""
import cv2
import numpy as np


def load_track(traj):
    n = traj['frame_count']
    xy = np.full((n, 2), np.nan)
    for f in traj['frames']:
        if f.get('raw_visible') and f.get('raw_xy'):
            xy[f['frame_index']] = f['raw_xy']
    return xy


def flight_chains(xy, fps, width, max_gap_s=0.15, max_speed=2.5):
    """Split observed detections into chains a single moving shuttle could form:
    consecutive detections at most max_gap_s apart and moving at most
    max_speed image widths per second. Isolated false detections end up as
    chains of one or two points. Returns a list of frame-index arrays."""
    idx = np.flatnonzero(~np.isnan(xy[:, 0]))
    if len(idx) == 0:
        return []
    max_gap = max(1, int(round(max_gap_s * fps)))
    v_max = max_speed * width / fps                       # px per frame
    chains, cur = [], [idx[0]]
    for a, b in zip(idx[:-1], idx[1:]):
        if b - a <= max_gap and np.linalg.norm(xy[b] - xy[a]) <= v_max * (b - a):
            cur.append(b)
        else:
            chains.append(np.array(cur))
            cur = [b]
    chains.append(np.array(cur))
    return chains


def rally_context(xy, fps, width, s, p_rest, r_still, r_far, chains,
                  arrive_s=1.5, min_flight_dets=5, min_flight_disp=0.08, min_rest_s=0.6):
    """Evidence that a resting shuttle ended a rally rather than being dropped,
    juggled on the racket or picked up between points.

    flight: a chain of >= min_flight_dets detections that moved >= min_flight_disp
            image widths toward the rest point and ended within arrive_s before the
            rest began, close enough to the rest point to be reached in the gap
            (players often occlude the last part of the flight);
    rest:   after contact the shuttle stays at the rest point >= min_rest_s
            (a racket-juggle apex or a quick scoop does not)."""
    back = int(round(arrive_s * fps))
    flight = None
    for ch in reversed(chains):
        pre = ch[(ch < s) & (ch >= s - back)]
        if len(pre) == 0:
            continue
        d = np.linalg.norm(xy[pre] - p_rest, axis=1)
        moving = pre[d > r_far]
        if len(moving) < min_flight_dets:
            continue
        end = moving[-1]
        disp = float(np.linalg.norm(xy[moving] - xy[end], axis=1).max())   # extent of the flight
        tail = moving[-3:]
        v = (xy[tail[-1]] - xy[tail[0]]) / max(1, tail[-1] - tail[0])
        gap = s - end
        reach = float(np.linalg.norm(v)) * gap * 1.5 + r_far
        if disp >= min_flight_disp * width and np.linalg.norm(xy[end] - p_rest) <= reach:
            flight = {'start_frame': int(moving[0]), 'end_frame': int(end), 'n_detections': int(len(moving)),
                      'displacement_px': round(disp, 1), 'end_velocity_px_per_frame': v.tolist(),
                      'gap_to_rest_frames': int(gap)}
            break
    # Contact: first detection after the flight that is at the rest point.
    after = flight['end_frame'] if flight else s - 1
    vis = np.flatnonzero(~np.isnan(xy[after + 1:, 0])) + after + 1
    at_rest = [f for f in vis if np.linalg.norm(xy[f] - p_rest) <= 2 * r_still]
    contact = int(at_rest[0]) if at_rest else int(s)
    # Rest: until the shuttle is seen away from the rest point in two consecutive detections.
    last_near, away = contact, 0
    for f in vis[vis >= contact]:
        if np.linalg.norm(xy[f] - p_rest) <= 2 * r_still:
            last_near, away = f, 0
        else:
            away += 1
            if away >= 2:
                break
    rest_s = float((last_near - contact) / fps)
    checks = {'flight': flight is not None, 'rest': bool(rest_s >= min_rest_s)}
    reasons = []
    if not checks['flight']:
        reasons.append('没有找到落地前的连续飞行轨迹')
    if not checks['rest']:
        reasons.append(f'落地后只静止了 {rest_s:.2f} s（< {min_rest_s} s），像颠球或马上被捡起')
    return contact, {'is_rally_landing': all(checks.values()), 'checks': checks, 'reasons': reasons,
                     'flight': flight, 'rest_s': round(rest_s, 3)}


def find_landings(xy, fps, width, still_s=0.3, still_frac=0.8, arrive_s=1.5, **rally_kw):
    """Candidates: the shuttle comes to rest after being seen elsewhere in the
    preceding arrive_s. Each candidate carries a rally_context verdict."""
    n = len(xy)
    k = max(3, int(round(still_s * fps)))
    r_still = max(3.0, 0.004 * width)        # px: shuttle counts as not moving
    r_far = 8 * r_still                       # px: shuttle clearly elsewhere (in flight)
    vis = ~np.isnan(xy[:, 0])
    still = np.zeros(n, bool)
    rest = np.full((n, 2), np.nan)
    for s in range(n - k + 1):
        w = xy[s:s + k][vis[s:s + k]]
        if len(w) < still_frac * k:
            continue
        med = np.median(w, axis=0)
        if np.linalg.norm(w - med, axis=1).max() <= r_still:
            still[s] = True
            rest[s] = med
    chains = flight_chains(xy, fps, width)
    landings, s = [], 0
    back = int(round(arrive_s * fps))
    while s < n:
        if not still[s]:
            s += 1
            continue
        e = s
        while e + 1 < n and still[e + 1] and np.linalg.norm(rest[e + 1] - rest[s]) <= r_still:
            e += 1
        p_rest = rest[s]
        lo = max(0, s - back)
        far = [f for f in range(lo, s) if vis[f] and np.linalg.norm(xy[f] - p_rest) > r_far]
        last_end = landings[-1]['rest_frame_end'] if landings else -1
        if far and far[-1] > last_end:       # a new arrival, not the same shuttle again
            contact, ctx = rally_context(xy, fps, width, s, p_rest, r_still, r_far, chains,
                                         arrive_s=arrive_s, **rally_kw)
            # The same landing again: the rest was briefly broken (occlusion, a stray detection
            # elsewhere) but the shuttle arrived with the same flight, or with none, at the same spot.
            if landings:
                prev = landings[-1]
                pf = prev['rally_context']['flight']
                same_flight = ctx['flight'] and pf and ctx['flight']['end_frame'] == pf['end_frame']
                same_spot = not ctx['flight'] and np.linalg.norm(p_rest - np.asarray(prev['rest_px'])) <= r_far
                if same_flight or same_spot:
                    prev['rest_frame_end'] = int(e + k - 1)
                    s = e + k
                    continue
            if ctx['flight']:
                last_flight = ctx['flight']['end_frame']
                v = np.asarray(ctx['flight']['end_velocity_px_per_frame'])
            else:                             # no coherent flight: fall back to the last far detection
                last_flight = far[-1]
                v = (xy[contact] - xy[last_flight]) / max(1, contact - last_flight)
            landings.append({
                'contact_frame': contact, 'contact_time_s': round(contact / fps, 4),
                'last_flight_frame': int(last_flight), 'rest_frame_start': int(s), 'rest_frame_end': int(e + k - 1),
                'contact_px_centre': xy[contact].tolist(), 'rest_px': p_rest.tolist(),
                'post_contact_drift_px': float(np.linalg.norm(p_rest - xy[contact])),
                'approach_velocity_px_per_frame': v.tolist(),
                'frames_flight_to_contact': int(contact - last_flight),
                'rally_context': ctx,
            })
        s = e + k
    return landings, {'still_window_frames': k, 'r_still_px': r_still, 'r_far_px': r_far,
                      'arrive_s': arrive_s, **rally_kw}


def read_frame(video, index):
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def refine_contact(video, landing, width):
    """Lowest pixel of the shuttle blob near the TrackNet centre at contact.

    Background is the frame at the last in-flight detection (shuttle elsewhere).
    Returns (pixel, method)."""
    cx, cy = landing['contact_px_centre']
    cur, bg = read_frame(video, landing['contact_frame']), read_frame(video, landing['last_flight_frame'])
    if cur is None or bg is None:
        return [cx, cy], 'centre (frames unreadable)'
    r = int(max(12, 0.02 * width))
    x0, y0 = int(max(0, cx - r)), int(max(0, cy - r))
    x1, y1 = int(min(cur.shape[1], cx + r)), int(min(cur.shape[0], cy + r))
    diff = cv2.absdiff(cv2.cvtColor(cur[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY),
                       cv2.cvtColor(bg[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY))
    _, mask = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    nlab, lab, stats, _ = cv2.connectedComponentsWithStats(mask)
    if nlab <= 1:
        return [cx, cy], 'centre (no blob)'
    # Component containing / nearest to the TrackNet centre.
    ys, xs = np.nonzero(lab)
    d = (xs + x0 - cx) ** 2 + (ys + y0 - cy) ** 2
    comp = lab[ys[d.argmin()], xs[d.argmin()]]
    area = stats[comp, cv2.CC_STAT_AREA]
    if area < 4 or area > 0.5 * mask.size:
        return [cx, cy], f'centre (blob area {area} implausible)'
    by, bx = np.nonzero(lab == comp)
    bottom = by.max()
    sel = by >= bottom - 1
    return [float(bx[sel].mean() + x0), float(bottom + y0)], 'blob bottom'
