"""Pilot report for a field session (python session.py report RUN).

Answers the six feasibility questions of 场测指南_v1.md from the annotations:
near-line yield, truth readability, call capture, candidate recall, machine
error, and phone alignment. Writes report.html, report.json and, when there are
enough near-line calls, psychometric.png.
"""
import html
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

import court
import session

NEAR_MM = 100
BINS = [(-1e9, -100), (-100, -25), (-25, 0), (0, 25), (25, 100), (100, 1e9)]
BIN_CN = ['出界 >100 mm', '出界 25–100', '出界 0–25', '界内 0–25', '界内 25–100', '界内 >100 mm']
THRESH = {'yield_per_10min': 15, 'readable': 0.8, 'cork_median': (20, 35), 'call_capture': 0.9}


def video_minutes(path):
    cap = cv2.VideoCapture(str(path))
    n, fps = cap.get(cv2.CAP_PROP_FRAME_COUNT), cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return n / fps / 60 if fps else None


def fit_psychometric(d_mm, said_in):
    """MLE of P(call in | d) = Phi((d - mu) / sigma). d > 0 is inside the court."""
    d, y = np.asarray(d_mm, float), np.asarray(said_in, float)

    def nll(p):
        mu, ls = p
        q = np.clip(norm.cdf((d - mu) / np.exp(ls)), 1e-6, 1 - 1e-6)
        return -(y * np.log(q) + (1 - y) * np.log(1 - q)).sum()
    r = minimize(nll, [0.0, np.log(30.0)], method='Nelder-Mead')
    return float(r.x[0]), float(np.exp(r.x[1])), bool(r.success)


def rows(run):
    s = session.load(run)
    cands = {c['id']: c for c in json.loads((Path(run) / 'candidates.json').read_text())['candidates']}
    ann = session.load_annotations(run)
    out = []
    for cid, a in ann.items():
        c = cands.get(cid, {'sources': ['unknown']})
        t = a.get('truth') or {}
        out.append({'id': cid, 't_sys': a.get('t_sys'), 'status': a.get('status'), 'sources': c.get('sources', []),
                    'machine': c.get('machine'), 'truth_label': t.get('label'), 'truth_method': t.get('method'),
                    'edge_mm': t.get('edge_mm'), 'centre_mm': t.get('centre_mm'), 'cork_width_mm': t.get('cork_width_mm'),
                    'call': (a.get('call') or {}).get('label'), 'caller': (a.get('call') or {}).get('caller')})
    return s, cands, out


def in_bin(x, b):
    return x is not None and b[0] <= x < b[1]


def write(run):
    run = Path(run)
    s, cands, R = rows(run)
    minutes = video_minutes(s['system_video'])
    land = [r for r in R if r['status'] == 'landing']
    known = [r for r in land if r['truth_label'] in ('in', 'out')]
    measured = [r for r in known if r['edge_mm'] is not None]
    near = [r for r in measured if abs(r['edge_mm']) <= NEAR_MM]
    readable = len(known) / len(land) if land else None
    corks = [r['cork_width_mm'] for r in measured if r['cork_width_mm'] is not None]
    with_call = [r for r in known if r['call'] in ('in', 'out', 'unsure')]
    call_capture = len(with_call) / len(known) if known else None
    # Calls against truth.
    decided = [r for r in known if r['call'] in ('in', 'out')]
    call_acc = np.mean([r['call'] == r['truth_label'] for r in decided]) if decided else None
    by_bin = []
    for b, name in zip(BINS, BIN_CN):
        rr = [r for r in measured if in_bin(r['edge_mm'], b)]
        dec = [r for r in rr if r['call'] in ('in', 'out')]
        by_bin.append({'bin': name, 'n': len(rr), 'n_decided': len(dec),
                       'correct': int(sum(r['call'] == r['truth_label'] for r in dec)),
                       'unsure': sum(r['call'] == 'unsure' for r in rr), 'no_call': sum(r['call'] in (None, 'none') for r in rr)})
    psy = None
    pts = [(r['edge_mm'], r['call'] == 'in') for r in measured if r['call'] in ('in', 'out') and abs(r['edge_mm']) <= 300]
    if len(pts) >= 15 and 0 < sum(p[1] for p in pts) < len(pts):
        mu, sg, ok = fit_psychometric(*zip(*pts))
        psy = {'n': len(pts), 'mu_mm': round(mu, 1), 'sigma_mm': round(sg, 1), 'converged': ok}
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5.5, 3.2), dpi=120)
        d = np.array([p[0] for p in pts]); y = np.array([p[1] for p in pts], float)
        ax.scatter(d, y + np.random.default_rng(0).uniform(-0.03, 0.03, len(y)), s=12, alpha=0.6, color='#333')
        xs = np.linspace(min(-150, d.min()), max(150, d.max()), 200)
        ax.plot(xs, norm.cdf((xs - mu) / sg), color='#c0392b')
        ax.axvline(0, color='#999', lw=0.8)
        ax.set_xlabel('truth: cork inner edge to line outer edge (mm, + = in)')
        ax.set_ylabel('called "in"')
        ax.set_title(f'n={len(pts)}  mu={mu:.0f} mm  sigma={sg:.0f} mm', fontsize=9)
        fig.tight_layout(); fig.savefig(run / 'psychometric.png'); plt.close(fig)
    # Machine.
    found_by = {'machine': sum('machine' in r['sources'] for r in known),
                'shout_only': sum(r['sources'] == ['shout'] for r in known),
                'manual': sum('manual' in r['sources'] for r in known)}
    mach = [r for r in measured if r['machine'] and r['centre_mm'] is not None]
    err = [r['machine']['edge_distance_m'] * 1000 - r['centre_mm'] for r in mach]
    mach_calls = [r for r in known if r['machine']]
    m_dec = [r for r in mach_calls if r['machine']['call'] != 'ABSTAIN']
    machine = {'n_with_truth': len(mach_calls), 'n_measured': len(mach),
               'abstain_rate': round(1 - len(m_dec) / len(mach_calls), 3) if mach_calls else None,
               'accuracy_when_called': round(float(np.mean([(r['machine']['call'] == 'IN') == (r['truth_label'] == 'in') for r in m_dec])), 3) if m_dec else None,
               'abs_error_mm_median': round(float(np.median(np.abs(err))), 1) if err else None,
               'abs_error_mm_p95': round(float(np.percentile(np.abs(err), 95)), 1) if err else None,
               'bias_mm': round(float(np.mean(err)), 1) if err else None}
    false_by_src = {}
    for r in R:
        if r['status'] == 'not_landing':
            for src in r['sources']:
                false_by_src[src] = false_by_src.get(src, 0) + 1
    yield_10 = len(near) / minutes * 10 if minutes else None
    res = {'session': s, 'video_minutes': round(minutes, 2) if minutes else None,
           'candidates': len(cands), 'annotated': len(R), 'landings': len(land), 'truth_known': len(known),
           'truth_measured': len(measured), 'near_line': len(near), 'near_line_per_10min': round(yield_10, 1) if yield_10 else None,
           'readable_rate': round(readable, 3) if readable is not None else None,
           'cork_width_median_mm': round(float(np.median(corks)), 1) if corks else None,
           'call_capture_rate': round(call_capture, 3) if call_capture is not None else None,
           'call_accuracy': round(float(call_acc), 3) if call_acc is not None else None,
           'calls_by_distance': by_bin, 'psychometric': psy, 'found_by': found_by,
           'false_candidates_by_source': false_by_src, 'machine': machine}
    session.save_json(run / 'report.json', res)
    (run / 'report.html').write_text(page(res), encoding='utf-8')
    print(f"Report: {run / 'report.html'}")
    return res


def verdict(ok):
    if ok == 'record':
        return '只记录'
    return '—（数据不够）' if ok is None else ('✅ 过' if ok else '❌ 不过')


def page(r):
    s, m = r['session'], r['machine']
    pct = lambda x: '—' if x is None else f'{x * 100:.0f}%'
    num = lambda x, u='': '—' if x is None else f'{x}{u}'
    cork_ok = None if r['cork_width_median_mm'] is None else THRESH['cork_median'][0] <= r['cork_width_median_mm'] <= THRESH['cork_median'][1]
    q = [
        ('1 近线球够不够多', f"{r['near_line']} 个距线 ±{NEAR_MM} mm 内的落点，录像 {num(r['video_minutes'])} 分钟，每 10 分钟 {num(r['near_line_per_10min'])} 个",
         f"每 10 分钟 ≥ {THRESH['yield_per_10min']}（喂球段）", None if r['near_line_per_10min'] is None else r['near_line_per_10min'] >= THRESH['yield_per_10min']),
        ('2 参照机能不能读出真值', f"落点中能读出真值 {pct(r['readable_rate'])}；球托宽度读数中位数 {num(r['cork_width_median_mm'], ' mm')}",
         '≥ 80%；中位数 20–35 mm', None if r['readable_rate'] is None else (r['readable_rate'] >= THRESH['readable'] and cork_ok is not False)),
        ('3 喊判收得到吗', f"有真值的落点中听到喊判 {pct(r['call_capture_rate'])}", '≥ 90%',
         None if r['call_capture_rate'] is None else r['call_capture_rate'] >= THRESH['call_capture']),
        ('4 候选找得全吗', f"有真值的落点：机器检出 {r['found_by']['machine']}，只靠喊声 {r['found_by']['shout_only']}，手动添加 {r['found_by']['manual']}；"
         f"误报：{', '.join(f'{k} {v}' for k, v in r['false_candidates_by_source'].items()) or '无'}", '记录', 'record'),
        ('5 机器有多准', f"有真值的机器落点 {m['n_with_truth']} 个；弃权 {pct(m['abstain_rate'])}；判定时正确 {pct(m['accuracy_when_called'])}；"
         f"距离误差中位数 {num(m['abs_error_mm_median'], ' mm')}，95 分位 {num(m['abs_error_mm_p95'], ' mm')}，偏差 {num(m['bias_mm'], ' mm')}（{m['n_measured']} 个有测量真值）", '记录', 'record'),
        ('6 两部手机能不能对齐', f"声音对齐置信度 {num(s.get('offset_confidence'))}，偏移 {num(s.get('offset_s'), ' s')}{'（手动调过）' if s.get('offset_manual') else ''}",
         '置信度 ≥ 3 且不需要手动调', None if s.get('offset_confidence') is None else (s['offset_confidence'] >= 3 and not s.get('offset_manual'))),
    ]
    trs = ''.join(f'<tr><td>{html.escape(a)}</td><td>{html.escape(b)}</td><td>{html.escape(c)}</td><td>{verdict(d)}</td></tr>' for a, b, c, d in q)
    bins = ''.join(f"<tr><td>{b['bin']}</td><td>{b['n']}</td><td>{b['correct']}/{b['n_decided']}</td><td>{b['unsure']}</td><td>{b['no_call']}</td></tr>" for b in r['calls_by_distance'])
    psy = r['psychometric']
    psy_html = (f"<p>心理测量曲线（P(喊内) = Φ((d − μ)/σ)）：n = {psy['n']}，μ = {psy['mu_mm']} mm（负数表示偏向判内），σ = {psy['sigma_mm']} mm。"
                f"样本少时只作参考。</p><img src='psychometric.png' style='max-width:520px'>") if psy else '<p>近线喊判少于 15 个，或者全是同一种判定，暂不拟合心理测量曲线。</p>'
    return f"""<!doctype html><html lang="zh"><head><meta charset="utf-8"><title>场测报告</title><style>
body{{font:14px system-ui;margin:20px;max-width:1000px;color:#222}} table{{border-collapse:collapse;margin:8px 0}} td,th{{border:1px solid #ccc;padding:5px 8px;text-align:left;vertical-align:top}} th{{background:#f3f3f3}}
</style></head><body><h1>场测报告</h1>
<p>目标线：{court.EDGE_CN[s['line']]} · {'单打' if s['mode'] == 'singles' else '双打'} · 候选 {r['candidates']} 个，已标 {r['annotated']} 个，其中落点 {r['landings']} 个，有真值 {r['truth_known']} 个（点击测量 {r['truth_measured']} 个）。</p>
<h2>六个可行性问题</h2><table><tr><th>问题</th><th>结果</th><th>门槛（暂定）</th><th>结论</th></tr>{trs}</table>
<h2>球员喊判 vs 真值</h2><p>喊"内/出"的整体正确率：{pct(r['call_accuracy'])}。下表只含点击测量过距离的落点（距离 = 球托内沿到线外沿）。</p>
<table><tr><th>真值距离</th><th>个数</th><th>喊对/喊了内或出</th><th>喊"不知道"</th><th>没喊</th></tr>{bins}</table>
{psy_html}
<p style="color:#666">原始数据：annotations.json、candidates.json、report.json。</p></body></html>"""
