"""Review outputs: zoomed landing crops, replay clips, top-down court map, index.html."""
import html
import json
from pathlib import Path

import cv2
import numpy as np

import calib
import court
import uncertainty

CALL_COLOR = {'IN': (80, 200, 80), 'OUT': (60, 60, 230), 'ABSTAIN': (0, 200, 255)}
CALL_CN = {'IN': '界内', 'OUT': '出界', 'ABSTAIN': '无法判定'}


def boundary_edges(poly):
    """Outer edges (the in/out boundary) as dense polylines in court metres."""
    pts = list(poly) + [poly[0]]
    return [np.linspace(pts[i], pts[i + 1], 60) for i in range(4)]


def polyline(img, px, closed, color, thick):
    """Draw a polyline, skipping points without an image (NaN) and splitting there."""
    px = np.asarray(px, float)
    ok = np.isfinite(px).all(axis=1)
    if closed and ok.all():
        cv2.polylines(img, [np.clip(px, -1e5, 1e5).astype(np.int32)], True, color, thick, cv2.LINE_AA)
        return
    run = []
    for p, good in zip(px, ok):
        if good:
            run.append(p)
        elif run:
            if len(run) > 1:
                cv2.polylines(img, [np.clip(run, -1e5, 1e5).astype(np.int32)], False, color, thick, cv2.LINE_AA)
            run = []
    if len(run) > 1:
        cv2.polylines(img, [np.clip(run, -1e5, 1e5).astype(np.int32)], False, color, thick, cv2.LINE_AA)


def draw_court(img, cal, poly, thick=1):
    for a, b in court.line_segments():
        polyline(img, calib.court_to_image(cal, np.linspace(a, b, 40)), False, (255, 0, 255), thick)
    for edge in boundary_edges(poly):
        polyline(img, calib.court_to_image(cal, edge), False, (0, 255, 255), thick)


def draw_landing(img, cal, res, pixel):
    col = CALL_COLOR[res['call']]
    polyline(img, calib.court_to_image(cal, uncertainty.ellipse_points(res['court_xy_m'], res['court_cov_m2'])), True, col, 1)
    cv2.drawMarker(img, tuple(int(v) for v in pixel), col, cv2.MARKER_CROSS, 10, 1, cv2.LINE_AA)


def crop(frame, cal, poly, res, pixel, half=90, scale=4):
    img = frame.copy()
    h, w = img.shape[:2]
    x, y = int(pixel[0]), int(pixel[1])
    x0, y0, x1, y1 = max(0, x - half), max(0, y - half), min(w, x + half), min(h, y + half)
    sub = cv2.resize(img[y0:y1, x0:x1], None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    # Draw after upscaling so lines stay thin: shift/scale the calibration overlay.
    def tf(px):
        return (np.asarray(px, float) - [x0, y0]) * scale
    for edge in boundary_edges(poly):
        polyline(sub, tf(calib.court_to_image(cal, edge)), False, (0, 255, 255), 1)
    col = CALL_COLOR[res['call']]
    ell = calib.court_to_image(cal, uncertainty.ellipse_points(res['court_xy_m'], res['court_cov_m2']))
    polyline(sub, tf(ell), True, col, 2)
    cv2.drawMarker(sub, tuple(int(v) for v in tf([pixel])[0]), col, cv2.MARKER_CROSS, 20, 2, cv2.LINE_AA)
    return sub


def court_map(landings, poly, px_per_m=60, margin=1.0):
    W, L = court.W, court.L
    size = (int((W + 2 * margin) * px_per_m), int((L + 2 * margin) * px_per_m))
    img = np.full((size[1], size[0], 3), (58, 107, 45), np.uint8)

    def tf(p):
        p = np.asarray(p, float).reshape(-1, 2)
        return np.c_[(p[:, 0] + margin) * px_per_m, (L - p[:, 1] + margin) * px_per_m].astype(np.int32)
    for a, b in court.line_segments():
        cv2.line(img, *map(tuple, tf([a, b])), (255, 255, 255), max(1, int(court.LINE * px_per_m)))
    cv2.polylines(img, [tf(poly)], True, (0, 255, 255), 1)
    cv2.line(img, *map(tuple, tf([(-0.3, court.NET_Y), (W + 0.3, court.NET_Y)])), (200, 200, 200), 1)
    for i, lnd in enumerate(landings, 1):
        r = lnd['result']
        col = CALL_COLOR[r['call']]
        cv2.polylines(img, [tf(uncertainty.ellipse_points(r['court_xy_m'], r['court_cov_m2']))], True, col, 1)
        c = tuple(tf([r['court_xy_m']])[0])
        cv2.circle(img, c, 4, col, -1)
        cv2.putText(img, str(i), (c[0] + 6, c[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    cv2.putText(img, 'near end (y=0)', (5, size[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    return img


def open_writer(path, fps, size):
    for fourcc in ('avc1', 'mp4v'):
        w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*fourcc), fps, size)
        if w.isOpened():
            return w
    raise RuntimeError('No usable MP4 encoder in this OpenCV build')


def clip(video, out, cal, poly, xy, lnd, fps, before_s=1.0, after_s=0.6, slow=4):
    f0 = max(0, lnd['contact_frame'] - int(before_s * fps))
    f1 = min(len(xy) - 1, lnd['contact_frame'] + int(after_s * fps))
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
    writer = None
    for f in range(f0, f1 + 1):
        ok, frame = cap.read()
        if not ok:
            break
        draw_court(frame, cal, poly)
        for g in range(max(f0, f - 8), f + 1):          # short trail of observed detections
            if not np.isnan(xy[g, 0]):
                cv2.circle(frame, (int(xy[g, 0]), int(xy[g, 1])), 3, (0, 165, 255), -1)
        if f >= lnd['contact_frame']:
            draw_landing(frame, cal, lnd['result'], lnd['contact_px'])
        cv2.putText(frame, f'frame {f}  x1/{slow}', (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        if writer is None:
            writer = open_writer(out, fps / slow, (frame.shape[1], frame.shape[0]))
        writer.write(frame)
    cap.release()
    if writer:
        writer.release()


def card(i, lnd, note=''):
    r = lnd['result']
    return f"""<div class="card"><h3>#{i} · {CALL_CN[r['call']]} · P(界内)={r['p_in']:.3f}</h3>{note}
<p>时间 {lnd['contact_time_s']:.2f}s（第 {lnd['contact_frame']} 帧）· 场地坐标 ({r['court_xy_m'][0]:.3f}, {r['court_xy_m'][1]:.3f}) m ·
离{html.escape(r['nearest_boundary'])} {r['signed_distance_m']*1000:+.0f} ± {r['signed_distance_sigma_m']*1000:.0f} mm（正=界内）·
落点像素来源：{html.escape(lnd['contact_px_method'])}{(' · 弃权原因：' + html.escape(r['abstain_reason'])) if r.get('abstain_reason') else ''}</p>
<img src="{lnd['crop_file']}"><video src="{lnd['clip_file']}" controls muted loop></video></div>"""


def index_html(run_dir, landings, meta, non_rally=()):
    rows = [card(i, lnd) for i, lnd in enumerate(landings, 1)]
    extra = [card(i, lnd, '<p class="why">不算回合落点：' + html.escape('；'.join(lnd['rally_context']['reasons'])) + '</p>')
             for i, lnd in enumerate(non_rally, len(landings) + 1)]
    extra_html = (f'<h2>不像回合落点的候选（{len(extra)}）</h2><p>颠球、丢球、回合之间捡球等。没有计入上面的落点，'
                  f'也没画在俯视图上；有争议时可以在这里回看。</p>' + ''.join(extra)) if extra else ''
    page = f"""<!doctype html><html lang="zh"><head><meta charset="utf-8"><title>落点回看</title><style>
body{{font:14px system-ui;margin:16px;background:#fafafa;color:#222}} .card{{background:#fff;border:1px solid #ddd;border-radius:8px;padding:12px;margin:12px 0}}
img,video{{max-width:48%;vertical-align:top;margin-right:1%}} .map{{float:right;max-width:240px;margin-left:16px}} .why{{color:#a15c00}}
</style></head><body><img class="map" src="court_map.png"><h1>落点回看</h1>
<p>{html.escape(meta)}</p><p>黄色线 = 判定边界（边界线外沿）；彩色椭圆 = 95% 误差范围；绿 = 界内，红 = 出界，橙 = 无法判定。</p>
{''.join(rows) or '<p>没有检测到落点。</p>'}{extra_html}</body></html>"""
    (Path(run_dir) / 'index.html').write_text(page, encoding='utf-8')
