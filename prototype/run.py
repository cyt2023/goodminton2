"""Single-phone landing prototype.

  python run.py analyze VIDEO --run RUN_DIR [--mode singles|doubles]
      Track the shuttle (TrackNetV3, skipped if RUN_DIR/trajectory.json exists),
      find landings, call IN / OUT / ABSTAIN with uncertainty, write review files.
  python run.py measure --run RUN_DIR
      Map clicked measurement points (static accuracy test) to court metres.

Calibrate first with calibrate_server.py, which writes RUN_DIR/calibration.json.
"""
import argparse
import json
from pathlib import Path

import cv2

import calib
import court
import landing
import render
import track
import uncertainty


def load_cal(run, args):
    record = json.loads((run / 'calibration.json').read_text())
    cal = calib.build(record, record.get('intrinsics_file'), click_sigma_px=args.click_sigma)
    return record, cal


def analyze(args):
    run = Path(args.run)
    record, cal = load_cal(run, args)
    traj_file = run / 'trajectory.json'
    if not traj_file.exists():
        track.run(args.video, run)
    traj = json.loads(traj_file.read_text())
    if record.get('image_size') and record['image_size'] != [traj['source_width'], traj['source_height']]:
        raise SystemExit('Calibration frame size differs from the tracked video size')
    fps, width = traj['nominal_fps'], traj['source_width']
    xy = landing.load_track(traj)
    found, params = landing.find_landings(xy, fps, width, min_rest_s=args.min_rest_s)
    poly = court.boundary(args.mode)
    # A resting shuttle above the floor's horizon (net tape, wall, far background) or
    # far outside the court is not a floor landing on this court: keep it, but separately.
    rejected, kept = [], []
    for lnd in found:
        px = lnd['contact_px_centre']
        if calib.ground_side(cal, [px])[0] < 0:
            lnd['reject_reason'] = 'above the floor horizon in the image'
        else:
            g = calib.project(cal['H'], calib.undistort([px], cal['intrinsics']))[0]
            if uncertainty.signed_distance(g, court.boundary('doubles')) < -args.max_outside:
                lnd['reject_reason'] = f'more than {args.max_outside} m outside the court ({g[0]:.1f}, {g[1]:.1f})'
        (rejected if 'reject_reason' in lnd else kept).append(lnd)
    for lnd in kept:
        if args.no_refine:
            px, method = lnd['contact_px_centre'], 'centre (refinement disabled)'
        else:
            px, method = landing.refine_contact(args.video, lnd, width)
        # Without a blob bottom the centre-to-cork offset is unknown: inflate the noise.
        det = args.det_sigma if method == 'blob bottom' else args.det_sigma_fallback
        cov = uncertainty.pixel_cov(det, lnd['approach_velocity_px_per_frame'] if args.motion_term else None)
        lnd.update(contact_px=px, contact_px_method=method, pixel_cov=cov.tolist(),
                   result=uncertainty.locate(cal, px, cov, poly, args.contact_radius, args.alpha))
    # Rally context: candidates without a flight before / a rest after are not rally
    # landings (juggling, dropping, scooping between points). They are still analysed
    # and shown separately so a disputed one can be reviewed.
    if args.no_rally_filter:
        found, non_rally = kept, []
    else:
        found = [l for l in kept if l['rally_context']['is_rally_landing']]
        non_rally = [l for l in kept if not l['rally_context']['is_rally_landing']]
    for i, lnd in enumerate(found + non_rally, 1):
        frame = landing.read_frame(args.video, lnd['contact_frame'])
        lnd['crop_file'], lnd['clip_file'] = f'landing_{i:03d}.jpg', f'landing_{i:03d}.mp4'
        cv2.imwrite(str(run / lnd['crop_file']), render.crop(frame, cal, poly, lnd['result'], lnd['contact_px']))
        render.clip(args.video, run / lnd['clip_file'], cal, poly, xy, lnd, fps)
    cv2.imwrite(str(run / 'court_map.png'), render.court_map(found, poly))
    out = {'video': str(Path(args.video).resolve()), 'mode': args.mode, 'fps': fps,
           'calibration': calib.summary(cal), 'detector_params': params,
           'settings': {k: getattr(args, k) for k in ['alpha', 'click_sigma', 'det_sigma', 'det_sigma_fallback',
                                                       'contact_radius', 'motion_term', 'no_refine',
                                                       'no_rally_filter', 'min_rest_s']},
           'landings': found, 'non_rally_candidates': non_rally, 'rejected_candidates': rejected}
    (run / 'landings.json').write_text(json.dumps(out, ensure_ascii=False, indent=2))
    meta = (f"{Path(args.video).name} · {args.mode} · {fps:.1f} fps · 标定 {cal['n_points']} 点，"
            f"RMS {cal['residual_rms_px']:.2f} px · 检出 {len(found)} 个落点"
            f"；另有 {len(non_rally)} 个不像回合落点的候选（见页面下方），"
            f"{len(rejected)} 个静止候选不在地面或远离球场（见 landings.json）")
    render.index_html(run, found, meta, non_rally)
    calls = [l['result']['call'] for l in found]
    print(f"{len(found)} landings: " + ', '.join(f'{c}={calls.count(c)}' for c in ['IN', 'OUT', 'ABSTAIN']))
    print(f"{len(non_rally)} non-rally candidates, {len(rejected)} off-floor candidates")
    print(f"Review: {run / 'index.html'}")


def measure(args):
    run = Path(args.run)
    record, cal = load_cal(run, args)
    poly = court.boundary(args.mode)
    cov = uncertainty.pixel_cov(args.det_sigma)
    rows = {}
    for name, px in record.get('measurement_points', {}).items():
        rows[name] = {'pixel': px} | uncertainty.locate(cal, px, cov, poly, args.contact_radius, args.alpha)
        r = rows[name]
        print(f"{name}: ({r['court_xy_m'][0]:.4f}, {r['court_xy_m'][1]:.4f}) m  "
              f"sigma=({r['court_sigma_m'][0]*1000:.1f}, {r['court_sigma_m'][1]*1000:.1f}) mm  "
              f"d={r['signed_distance_m']*1000:+.1f} mm to {r['nearest_boundary']}  {r['call']} p_in={r['p_in']:.3f}")
    (run / 'measurements.json').write_text(json.dumps(
        {'calibration': calib.summary(cal), 'det_sigma_px': args.det_sigma, 'points': rows}, ensure_ascii=False, indent=2))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    for name in ('analyze', 'measure'):
        p = sub.add_parser(name)
        if name == 'analyze':
            p.add_argument('video')
            p.add_argument('--det-sigma-fallback', type=float, default=6.0,
                           help='px noise when the cork contact pixel could not be refined')
            p.add_argument('--no-motion-term', dest='motion_term', action='store_false')
            p.add_argument('--no-refine', action='store_true')
            p.add_argument('--no-rally-filter', action='store_true',
                           help='keep candidates without rally context (no flight before / short rest after)')
            p.add_argument('--min-rest-s', type=float, default=0.6,
                           help='s the shuttle must stay at rest after contact to count as a rally landing')
            p.add_argument('--max-outside', type=float, default=3.0,
                           help='m beyond the doubles boundary after which a candidate is rejected')
        p.add_argument('--run', required=True)
        p.add_argument('--mode', choices=['singles', 'doubles'], default='singles')
        p.add_argument('--alpha', type=float, default=0.05, help='abstain if alpha < P(in) < 1-alpha')
        p.add_argument('--click-sigma', type=float, default=1.5, help='prior px noise of clicked keypoints')
        p.add_argument('--det-sigma', type=float, default=2.0, help='px noise of the contact pixel')
        p.add_argument('--contact-radius', type=float, default=0.0,
                       help='m added to the boundary for the cork contact patch (0 = point contact)')
    args = ap.parse_args()
    analyze(args) if args.cmd == 'analyze' else measure(args)


if __name__ == '__main__':
    main()
