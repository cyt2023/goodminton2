"""Field session: system phone + floor-level reference phone + players' spoken calls.

  python session.py init RUN --system A.mov --reference B.mov --line right --ref-court-side left [--mode singles]
  python session.py prepare RUN      # align the phones by sound, find shouts, build landing candidates
  python session.py annotate RUN     # web page: read the truth on the reference video, label the call
  python session.py report RUN       # pilot report against the go/no-go questions

RUN is the directory used by calibrate_server.py and run.py (calibration.json,
landings.json). prepare works without landings.json too: then candidates come
only from shouts and from landings added by hand on the annotation page.

Truth comes from the reference phone lying on the floor, looking along the
target line. On the contact frame the annotator clicks both edges of the line
and both edges of the cork at the cork's depth; the painted line width (40 mm)
gives the scale there. The cork width (25-28 mm by the Laws) is then a check
on the reading.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np

import audio
import court

LINE_WIDTH_MM = court.LINE * 1000
LINK_BEFORE_S, LINK_AFTER_S = 0.3, 3.0      # a shout this close after a landing is its call
SHOUT_REACTION_S = 0.7                      # landing time assumed for a shout without a landing
NEAR_LINE_M = 1.0                           # machine landings this close to the target line


def load(run):
    return json.loads((Path(run) / 'session.json').read_text())


def save_json(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2))
    os.replace(tmp, path)


def load_annotations(run):
    f = Path(run) / 'annotations.json'
    return json.loads(f.read_text()) if f.exists() else {}


def truth_from_clicks(clicks, interior_side):
    """Signed distances from four clicks on the reference contact frame.

    clicks: {'line_a', 'line_b', 'cork_a', 'cork_b'} as [x, y] pixels.
    interior_side: 'left' or 'right', the court side of the line in the reference image.
    Distances are measured along the axis through the two line-edge clicks, positive
    towards the court. Returns None if the line clicks coincide."""
    A, B = (np.asarray(clicks[k], float) for k in ('line_a', 'line_b'))
    C1, C2 = (np.asarray(clicks[k], float) for k in ('cork_a', 'cork_b'))
    w = np.linalg.norm(B - A)
    if w < 1e-6:
        return None
    u = (B - A) / w
    # Orient u to point out of the court.
    if (interior_side == 'left') != (u[0] > 0):
        u = -u
    a = lambda P: float((P - A) @ u)
    outer = max(a(A), a(B))
    mm = LINE_WIDTH_MM / w
    ca, cb = a(C1), a(C2)
    edge_mm = (outer - min(ca, cb)) * mm          # innermost cork edge to the line's outer edge
    centre_mm = (outer - (ca + cb) / 2) * mm
    return {'edge_mm': round(edge_mm, 1), 'centre_mm': round(centre_mm, 1),
            'cork_width_mm': round(abs(ca - cb) * mm, 1), 'line_width_px': round(w, 1),
            'mm_per_px': round(mm, 3), 'label': 'in' if edge_mm >= 0 else 'out'}


def video_info(path):
    import cv2
    cap = cv2.VideoCapture(str(path))
    fps, n = cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    if not fps:
        raise SystemExit(f'OpenCV cannot read {path}')
    return fps, n / fps


def cmd_init(args):
    run = Path(args.run)
    run.mkdir(parents=True, exist_ok=True)
    for v in (args.system, args.reference):
        if not Path(v).exists():
            raise SystemExit(f'Not found: {v}')
    s = {'system_video': str(Path(args.system).resolve()), 'reference_video': str(Path(args.reference).resolve()),
         'line': args.line, 'mode': args.mode, 'ref_court_side': args.ref_court_side,
         'offset_s': None, 'offset_confidence': None, 'offset_manual': False}
    f = run / 'session.json'
    if f.exists():
        old = json.loads(f.read_text())
        if old.get('offset_manual'):                 # keep a hand-tuned offset
            s.update({k: old[k] for k in ('offset_s', 'offset_confidence', 'offset_manual')})
    save_json(f, s)
    print(f'Session written: {f}')


def machine_candidates(run, s):
    f = Path(run) / 'landings.json'
    if not f.exists():
        return [], 'landings.json not found: no machine candidates (run run.py analyze first to add them)'
    d = json.loads(f.read_text())
    if Path(d['video']).resolve() != Path(s['system_video']).resolve():
        raise SystemExit(f"landings.json is for {d['video']}, not the session's system video")
    out = []
    for group in ('landings', 'non_rally_candidates'):
        for l in d.get(group, []):
            r = l['result']
            dist = court.edge_distance(r['court_xy_m'], s['mode'], s['line'])
            if abs(dist) > NEAR_LINE_M:
                continue
            out.append({'id': f"m{l['contact_frame']}", 't_sys': l['contact_time_s'], 'sources': ['machine'],
                        'machine': {'call': r['call'], 'p_in': r['p_in'], 'edge_distance_m': round(dist, 4),
                                    'sigma_m': r['signed_distance_sigma_m'], 'court_xy_m': r['court_xy_m'],
                                    'rally': group == 'landings', 'contact_frame': l['contact_frame'],
                                    'abstain_reason': r.get('abstain_reason')}})
    return out, f'{len(out)} machine landings within {NEAR_LINE_M} m of the target line'


def merge_shouts(shouts_sys, shouts_ref, offset):
    """Shout times on the system clock, merging the two microphones (0.4 s)."""
    ts = sorted([(e['t'], e, 'sys') for e in shouts_sys] + [(e['t'] - offset, e, 'ref') for e in shouts_ref])
    merged = []
    for t, e, mic in ts:
        if merged and t - merged[-1]['t'] < 0.4:
            merged[-1]['mics'].add(mic)
            merged[-1]['strength'] = max(merged[-1]['strength'], e['strength'])
            continue
        merged.append({'t': round(t, 3), 'strength': e['strength'], 'mics': {mic}})
    for m in merged:
        m['mics'] = sorted(m['mics'])
    return merged


def build_candidates(machine, shouts):
    cands = [dict(c) for c in machine]
    free = []
    for sh in shouts:
        best = None
        for c in cands:
            dt = sh['t'] - c['t_sys']
            if -LINK_BEFORE_S <= dt <= LINK_AFTER_S and (best is None or dt < best[0]):
                best = (dt, c)
        if best:
            best[1].setdefault('shouts', []).append(sh['t'])
            if 'shout' not in best[1]['sources']:
                best[1]['sources'].append('shout')
        else:
            free.append(sh)
    for sh in free:
        cands.append({'id': f"s{int(round(sh['t'] * 1000))}", 't_sys': round(max(0.0, sh['t'] - SHOUT_REACTION_S), 3),
                      'sources': ['shout'], 'shouts': [sh['t']]})
    return sorted(cands, key=lambda c: c['t_sys'])


def cmd_prepare(args):
    run = Path(args.run)
    s = load(run)
    for key, name in (('system_video', '系统机'), ('reference_video', '参照机')):
        fps, dur = video_info(s[key])
        print(f'{name}: {fps:.1f} fps, {dur / 60:.1f} min')
        if key == 'reference_video' and fps < 100:
            print('  Warning: the reference video is not high-frame-rate. A slow-motion clip shared without '
                  '"All Photos Data"/original is re-rendered at 30 fps and slowed down, which breaks the alignment. '
                  'Export the original 240 fps file.')
    wav_s = audio.extract_wav(s['system_video'], run / 'audio_system.wav')
    wav_r = audio.extract_wav(s['reference_video'], run / 'audio_reference.wav')
    xs, sr = audio.read_wav(wav_s)
    xr, _ = audio.read_wav(wav_r)
    if not s.get('offset_manual') or args.reset_offset:
        off, conf = audio.estimate_offset(xs, xr, sr, args.max_lag)
        s.update(offset_s=round(off, 4), offset_confidence=round(conf, 2), offset_manual=False)
        save_json(run / 'session.json', s)
    print(f"Offset: t_ref = t_sys + {s['offset_s']:.3f} s (confidence {s['offset_confidence']}"
          f"{', set by hand' if s.get('offset_manual') else ''})")
    if s['offset_confidence'] is not None and s['offset_confidence'] < 2:
        print('  Warning: the sound alignment is ambiguous; check and adjust it on the annotation page.')
    shouts = merge_shouts(audio.detect_shouts(xs, sr), audio.detect_shouts(xr, sr), s['offset_s'])
    machine, msg = machine_candidates(run, s)
    print(msg)
    cands = build_candidates(machine, shouts)
    # Keep landings added by hand on the annotation page.
    old = run / 'candidates.json'
    if old.exists():
        manual = [c for c in json.loads(old.read_text())['candidates'] if 'manual' in c['sources']]
        cands = sorted(cands + manual, key=lambda c: c['t_sys'])
    save_json(old, {'shouts': shouts, 'candidates': cands})
    n_sh = sum(1 for c in cands if c['sources'] == ['shout'])
    print(f'{len(shouts)} shouts; {len(cands)} candidates ({n_sh} found only by a shout)')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('init')
    p.add_argument('run')
    p.add_argument('--system', required=True, help='system phone video (the one calibrated)')
    p.add_argument('--reference', required=True, help='floor-level phone looking along the target line')
    p.add_argument('--line', required=True, choices=court.EDGES,
                   help='target line; left/right seen from the near end (y=0) looking at the net')
    p.add_argument('--ref-court-side', required=True, choices=['left', 'right'],
                   help='on which side of the line the court is, in the reference image')
    p.add_argument('--mode', choices=['singles', 'doubles'], default='singles')
    p = sub.add_parser('prepare')
    p.add_argument('run')
    p.add_argument('--max-lag', type=float, default=None, help='largest phone offset to consider (s)')
    p.add_argument('--reset-offset', action='store_true', help='discard a hand-set offset')
    p = sub.add_parser('annotate')
    p.add_argument('run')
    p.add_argument('--port', type=int, default=8771)
    p = sub.add_parser('report')
    p.add_argument('run')
    args = ap.parse_args()
    if args.cmd == 'init':
        cmd_init(args)
    elif args.cmd == 'prepare':
        cmd_prepare(args)
    elif args.cmd == 'annotate':
        import annotate_server
        annotate_server.serve(Path(args.run), args.port)
    else:
        import report
        report.write(Path(args.run))


if __name__ == '__main__':
    main()
