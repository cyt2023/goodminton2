"""Annotation page for a field session (python session.py annotate RUN).

For each landing candidate: step through the system video and the reference
video around it, mark the contact frame and click the line / cork edges on the
reference frame (truth), listen to the call and label it, mark false candidates.
Frames are decoded server-side with OpenCV, so any codec OpenCV reads (iPhone
HEVC .mov included) works in any browser, frame-exact, including 240 fps.
"""
import json
import threading
import time
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

import audio
import court
import session

MAX_FRAMES = 480
SYS_MAX_W = 1280


class Frames:
    """Decoded frame windows, a few kept in memory."""

    def __init__(self, videos, keep=6):
        self.videos, self.keep = videos, keep
        self.cache = OrderedDict()
        self.lock = threading.Lock()

    def window(self, v, t0, t1):
        key = f'{v}:{t0:.3f}:{t1:.3f}'
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
                return key, self.cache[key]
            cap = cv2.VideoCapture(self.videos[v])
            if not cap.isOpened():
                raise RuntimeError(f'Cannot open {self.videos[v]}')
            cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, t0) * 1000)
            times, jpgs, size = [], [], None
            while len(jpgs) < MAX_FRAMES:
                ok, frame = cap.read()
                if not ok:
                    break
                t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
                if t > t1:
                    break
                if t < t0 - 1e-3:
                    continue
                if v == 'sys' and frame.shape[1] > SYS_MAX_W:
                    f = SYS_MAX_W / frame.shape[1]
                    frame = cv2.resize(frame, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
                size = [frame.shape[1], frame.shape[0]]
                times.append(round(t, 4))
                jpgs.append(cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tobytes())
            cap.release()
            win = {'times': times, 'jpgs': jpgs, 'size': size}
            self.cache[key] = win
            while len(self.cache) > self.keep:
                self.cache.popitem(last=False)
            return key, win

    def frame(self, key, k):
        with self.lock:
            win = self.cache.get(key)
            return win['jpgs'][k] if win and 0 <= k < len(win['jpgs']) else None


def serve(run, port):
    run = Path(run)
    s = session.load(run)
    cand_file = run / 'candidates.json'
    if not cand_file.exists():
        raise SystemExit('Run "session.py prepare" first')
    frames = Frames({'sys': s['system_video'], 'ref': s['reference_video']})
    wavs = {}
    for v, name in (('sys', 'audio_system.wav'), ('ref', 'audio_reference.wav')):
        if (run / name).exists():
            wavs[v] = audio.read_wav(run / name)
    write_lock = threading.Lock()

    def state():
        return {'session': session.load(run), 'candidates': json.loads(cand_file.read_text())['candidates'],
                'annotations': session.load_annotations(run), 'line_cn': court.EDGE_CN[s['line']]}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def send(self, body, ctype, code=200):
            if isinstance(body, str):
                body = body.encode('utf-8')
            self.send_response(code)
            self.send_header('Content-Type', ctype)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(body)

        def json(self, obj, code=200):
            self.send(json.dumps(obj, ensure_ascii=False), 'application/json; charset=utf-8', code)

        def do_GET(self):
            u = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(u.query).items()}
            try:
                if u.path == '/':
                    self.send(PAGE, 'text/html; charset=utf-8')
                elif u.path == '/api/state':
                    self.json(state())
                elif u.path == '/api/window':
                    key, win = frames.window(q['v'], float(q['t0']), float(q['t1']))
                    self.json({'id': key, 'times': win['times'], 'size': win['size']})
                elif u.path == '/frame':
                    jpg = frames.frame(q['id'], int(q['k']))
                    self.send(jpg, 'image/jpeg') if jpg else self.send('gone', 'text/plain', 404)
                elif u.path == '/audio':
                    if q['v'] not in wavs:
                        return self.send('no audio', 'text/plain', 404)
                    x, sr = wavs[q['v']]
                    a, b = int(max(0, float(q['t0'])) * sr), int(max(0, float(q['t1'])) * sr)
                    self.send(audio.wav_bytes(x[a:b], sr), 'audio/wav')
                else:
                    self.send('not found', 'text/plain', 404)
            except Exception as e:                     # report to the page instead of a silent 500
                self.json({'error': f'{type(e).__name__}: {e}'}, 500)

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers.get('Content-Length', 0))) or b'{}')
            try:
                with write_lock:
                    if self.path == '/api/annotate':
                        ann = session.load_annotations(run)
                        rec = body['data']
                        truth = rec.get('truth') or {}
                        clicks = truth.get('clicks') or {}
                        if truth.get('method') == 'clicks':
                            if not all(k in clicks for k in ('line_a', 'line_b', 'cork_a', 'cork_b')):
                                return self.json({'error': '点击真值需要线的两边和球托的两边，共 4 个点'}, 400)
                            r = session.truth_from_clicks(clicks, s['ref_court_side'])
                            if r is None:
                                return self.json({'error': '线的两个边点重合了'}, 400)
                            truth.update(r)
                        rec['truth'] = truth
                        rec['saved_at'] = time.strftime('%Y-%m-%d %H:%M:%S')
                        ann[body['id']] = rec
                        session.save_json(run / 'annotations.json', ann)
                        self.json({'ok': True, 'record': rec})
                    elif self.path == '/api/add':
                        d = json.loads(cand_file.read_text())
                        t = round(float(body['t_sys']), 3)
                        c = {'id': f'u{int(round(t * 1000))}', 't_sys': t, 'sources': ['manual']}
                        if any(x['id'] == c['id'] for x in d['candidates']):
                            return self.json({'error': '这个时间已经有一个手动添加的落点'}, 400)
                        d['candidates'] = sorted(d['candidates'] + [c], key=lambda x: x['t_sys'])
                        session.save_json(cand_file, d)
                        self.json({'ok': True, 'candidate': c})
                    elif self.path == '/api/offset':
                        cur = session.load(run)
                        cur.update(offset_s=round(float(body['offset_s']), 4), offset_manual=True)
                        session.save_json(run / 'session.json', cur)
                        self.json({'ok': True})
                    else:
                        self.send('not found', 'text/plain', 404)
            except Exception as e:
                self.json({'error': f'{type(e).__name__}: {e}'}, 500)

    print(f'Annotation page: http://127.0.0.1:{port}  (Ctrl+C to stop)')
    ThreadingHTTPServer(('127.0.0.1', port), Handler).serve_forever()


PAGE = r"""<!doctype html><html lang="zh"><head><meta charset="utf-8"><title>场测标注</title><style>
*{box-sizing:border-box} body{margin:0;font:13px system-ui;color:#222;background:#f4f4f4;display:flex;flex-direction:column;height:100vh}
#top{padding:6px 10px;background:#fff;border-bottom:1px solid #ddd;display:flex;gap:14px;align-items:center;flex-wrap:wrap}
#wrap{flex:1;display:flex;min-height:0}
#list{width:190px;overflow:auto;background:#fff;border-right:1px solid #ddd}
.it{padding:4px 8px;border-bottom:1px solid #eee;cursor:pointer;white-space:nowrap} .it.sel{background:#ffeaa0} .it .tag{font-size:11px;color:#666}
.done{color:#2a7a2a} .skip{color:#999}
#views{flex:1;display:flex;flex-direction:column;min-width:0}
.view{flex:1;display:flex;flex-direction:column;min-height:0;border-bottom:1px solid #ddd;background:#222}
.view canvas{flex:1;min-height:0;width:100%;display:block;cursor:crosshair}
.bar{background:#fff;padding:4px 8px;display:flex;gap:8px;align-items:center} .bar input[type=range]{flex:1}
#side{width:300px;overflow:auto;background:#fff;border-left:1px solid #ddd;padding:8px}
h4{margin:10px 0 4px} button{margin:2px 2px 2px 0;padding:4px 8px;border:1px solid #bbb;background:#f8f8f8;border-radius:4px;cursor:pointer}
button.on{background:#ffd54d;border-color:#c9a000} .res{background:#f0f7f0;padding:6px;border-radius:4px;margin-top:4px;white-space:pre-wrap}
.err{color:#b00} .mach{background:#eef3ff;padding:6px;border-radius:4px;white-space:pre-wrap} kbd{font-size:11px;color:#666}
</style></head><body>
<div id="top"><b>场测标注</b><span id="info"></span>
<span>参照机时间 = 系统机时间 + <b id="off"></b> s
<button data-d="-0.1">−0.1</button><button data-d="-0.01">−0.01</button><button data-d="0.01">+0.01</button><button data-d="0.1">+0.1</button>
<button id="saveoff">保存偏移</button></span>
<span>跳到系统机 <input id="goto" size="6" placeholder="秒"> s <button id="gobtn">看</button></span>
<label><input type="checkbox" id="todo"> 只看未标</label><span id="prog"></span><span id="msg" class="err"></span></div>
<div id="wrap"><div id="list"></div>
<div id="views">
 <div class="view"><canvas id="cs"></canvas><div class="bar">系统机 <button id="ps">▶</button><input type="range" id="rs" min="0" value="0"><span id="ts"></span>
   <button id="addhere">在这一帧添加落点</button></div></div>
 <div class="view" style="flex:1.4"><canvas id="cr"></canvas><div class="bar">参照机 <button id="pr">▶</button><input type="range" id="rr" min="0" value="0"><span id="tr"></span>
   <label><input type="checkbox" id="follow" checked> 跟随系统机</label></div></div>
</div>
<div id="side">
 <div id="mach" class="mach"></div>
 <h4>这是一个落点吗</h4>
 <button data-st="landing">是落点</button><button data-st="not_landing">不是（误报）</button><button data-st="skip">跳过</button>
 <h4>真值（参照机）</h4>
 <div>先在参照机上找到球托触地的那一帧 <button id="mark">标为触地帧</button> <span id="ctime"></span></div>
 <div>再依次点击（滚轮放大，拖动平移，右键撤销当前点）：</div>
 <button data-ck="line_a">线的一边</button><button data-ck="line_b">线的另一边</button><button data-ck="cork_a">球托一边</button><button data-ck="cork_b">球托另一边</button>
 <div style="font-size:12px;color:#666">在球托触地的那一行，点线的两条边，再点球托左右两边。</div>
 <div>或者目测：<button data-tv="in">明显界内</button><button data-tv="out">明显出界</button><button data-tv="unreadable">看不清</button></div>
 <div id="truth" class="res"></div>
 <h4>喊判 <kbd>1 内 · 2 出 · 3 不知道 · 0 没喊</kbd></h4>
 <button data-call="in">内</button><button data-call="out">出</button><button data-call="unsure">不知道</button><button data-call="none">没喊</button>
 <div>谁喊的：<button data-who="receiver">接球方</button><button data-who="other">其他人</button></div>
 <div><button id="as">▶ 系统机声音</button><button id="ar">▶ 参照机声音</button></div>
 <h4>备注</h4><input id="note" style="width:100%">
 <div style="margin-top:10px"><button id="save">保存</button><button id="next"><b>保存并下一个</b> <kbd>Enter</kbd></button></div>
 <div style="margin-top:8px;font-size:12px;color:#666">键盘：← → 参照机逐帧，Shift+← → 系统机逐帧，空格 播放/暂停参照机。</div>
</div></div>
<script>
let S, cur = null, rec = null, sysW = null, refW = null, off = 0, clickKey = null;
const $ = id => document.getElementById(id);
const fmt = t => t == null ? '' : t.toFixed(3) + ' s';
const CALL_CN = {in:'内', out:'出', unsure:'不知道', none:'没喊'};
const ST_CN = {landing:'落点', not_landing:'误报', skip:'跳过'};
async function api(path, body){ const r = await fetch(path, body ? {method:'POST', body:JSON.stringify(body)} : {});
  const v = await r.json(); if (v.error){ $('msg').textContent = v.error; throw new Error(v.error); } $('msg').textContent=''; return v; }

// ---- zoomable frame view ----
function View(canvas, slider, label, kind){
  const v = {canvas, g: canvas.getContext('2d'), img: new Image(), s:1, ox:0, oy:0, win:null, k:0, play:null};
  v.fit = () => { canvas.width = canvas.clientWidth; canvas.height = canvas.clientHeight;
    if (!v.win || !v.win.size) return v.draw(); const [w,h] = v.win.size; v.s = Math.min(canvas.width/w, canvas.height/h); v.ox=(canvas.width-w*v.s)/2; v.oy=(canvas.height-h*v.s)/2; v.draw(); };
  v.draw = () => { const g = v.g; g.setTransform(1,0,0,1,0,0); g.fillStyle='#222'; g.fillRect(0,0,canvas.width,canvas.height);
    if (!v.win) return; g.setTransform(v.s,0,0,v.s,v.ox,v.oy); if (v.img.complete && v.img.naturalWidth) g.drawImage(v.img,0,0);
    if (kind === 'ref' && rec && rec.truth && rec.truth.clicks){ const r = 7/v.s; g.lineWidth = 1.5/v.s;
      for (const [n,p] of Object.entries(rec.truth.clicks)){ g.strokeStyle = n.startsWith('line') ? '#0ff' : '#f0f';
        g.beginPath(); g.moveTo(p[0]-r,p[1]); g.lineTo(p[0]+r,p[1]); g.moveTo(p[0],p[1]-r); g.lineTo(p[0],p[1]+r); g.stroke(); } } };
  v.show = k => { if (!v.win || !v.win.times.length) return; v.k = Math.max(0, Math.min(v.win.times.length-1, k)); slider.value = v.k;
    v.img.src = `/frame?id=${encodeURIComponent(v.win.id)}&k=${v.k}`; label.textContent = fmt(v.win.times[v.k]) + `  (${v.k+1}/${v.win.times.length})`; };
  v.img.onload = v.draw;
  v.t = () => v.win && v.win.times.length ? v.win.times[v.k] : null;
  v.seek = t => { if (!v.win) return; let best = 0; v.win.times.forEach((x,i) => { if (Math.abs(x-t) < Math.abs(v.win.times[best]-t)) best = i; }); v.show(best); };
  v.load = async (t0, t1) => { label.textContent = '解码中…'; v.win = await api(`/api/window?v=${kind}&t0=${t0.toFixed(3)}&t1=${t1.toFixed(3)}`);
    slider.max = Math.max(0, v.win.times.length-1); v.fit(); if (!v.win.times.length) label.textContent = '这个时间段没有画面（偏移不对或超出录像）'; };
  slider.oninput = () => { v.show(+slider.value); if (kind === 'sys' && $('follow').checked) ref.seek(sys.t() + off); };
  let drag = null, moved = false;
  canvas.onwheel = e => { e.preventDefault(); const r = canvas.getBoundingClientRect(), mx = e.clientX-r.left, my = e.clientY-r.top;
    const f = e.deltaY < 0 ? 1.2 : 1/1.2; v.ox = mx-(mx-v.ox)*f; v.oy = my-(my-v.oy)*f; v.s *= f; v.draw(); };
  canvas.onmousedown = e => { drag = [e.clientX, e.clientY, v.ox, v.oy]; moved = false; };
  canvas.onmousemove = e => { if (!drag) return; const dx = e.clientX-drag[0], dy = e.clientY-drag[1];
    if (Math.abs(dx)+Math.abs(dy) > 3) moved = true; if (moved){ v.ox = drag[2]+dx; v.oy = drag[3]+dy; v.draw(); } };
  canvas.onmouseup = e => { const was = drag; drag = null; if (!was || moved || e.button !== 0 || kind !== 'ref' || !clickKey || !rec) return;
    const r = canvas.getBoundingClientRect(); const p = [(e.clientX-r.left-v.ox)/v.s, (e.clientY-r.top-v.oy)/v.s].map(x => Math.round(x*10)/10);
    if (!v.win.size || p[0] < 0 || p[1] < 0 || p[0] > v.win.size[0] || p[1] > v.win.size[1]) return;
    rec.truth = rec.truth && rec.truth.method === 'clicks' ? rec.truth : {method:'clicks', clicks:{}, contact_ref_t: rec.truth && rec.truth.contact_ref_t};
    rec.truth.clicks[clickKey] = p; const order = ['line_a','line_b','cork_a','cork_b'];
    const nxt = order.find(k => !rec.truth.clicks[k]); clickKey = nxt || null; paint(); v.draw(); };
  canvas.oncontextmenu = e => { e.preventDefault(); if (kind === 'ref' && rec && rec.truth && rec.truth.clicks && clickKey){ delete rec.truth.clicks[clickKey]; v.draw(); paint(); } };
  v.toggle = btn => { if (v.play){ clearInterval(v.play); v.play = null; btn.textContent='▶'; return; }
    btn.textContent='⏸'; v.play = setInterval(() => { if (v.k >= v.win.times.length-1){ clearInterval(v.play); v.play=null; btn.textContent='▶'; return; }
      v.show(v.k+1); if (kind === 'sys' && $('follow').checked) ref.seek(sys.t() + off); }, kind === 'ref' ? 40 : 60); };
  return v;
}
const sys = View($('cs'), $('rs'), $('ts'), 'sys'), ref = View($('cr'), $('rr'), $('tr'), 'ref');
$('ps').onclick = () => sys.toggle($('ps')); $('pr').onclick = () => ref.toggle($('pr'));
window.onresize = () => { sys.fit(); ref.fit(); };

// ---- candidates ----
function list(){ const L = $('list'); L.innerHTML = ''; let done = 0;
  for (const c of S.candidates){ const a = S.annotations[c.id]; if (a) done++;
    if ($('todo').checked && a && c.id !== (cur && cur.id)) continue;
    const d = document.createElement('div'); d.className = 'it' + (cur && c.id === cur.id ? ' sel' : '');
    const st = a ? (a.status === 'landing' ? `<span class="done">✓ ${a.call ? CALL_CN[a.call.label] || '' : ''}</span>` : `<span class="skip">${ST_CN[a.status]}</span>`) : '○';
    d.innerHTML = `${st} ${c.t_sys.toFixed(2)} s <span class="tag">${c.sources.map(x => ({machine:'机', shout:'喊', manual:'手'})[x]).join('·')}</span>`;
    d.onclick = () => open(c); L.appendChild(d); }
  $('prog').textContent = `已标 ${done}/${S.candidates.length}`; }
async function open(c){ cur = c; rec = JSON.parse(JSON.stringify(S.annotations[c.id] || {status:null, truth:null, call:null, note:''}));
  clickKey = null; list(); paint();
  const wide = c.sources.length === 1 && c.sources[0] !== 'machine';
  const t = c.t_sys, tr = t + off;
  await sys.load(Math.max(0, t - (wide ? 2.0 : 1.5)), t + (wide ? 1.5 : 1.0)); sys.seek(t);
  await ref.load(Math.max(0, tr - (wide ? 1.0 : 0.6)), tr + (wide ? 0.8 : 0.6));
  if (rec.truth && rec.truth.contact_ref_t != null) ref.seek(rec.truth.contact_ref_t); else ref.seek(tr); }
function paint(){
  const m = cur && cur.machine; $('mach').textContent = !cur ? '' : (m ? `机器：${({IN:'界内',OUT:'出界',ABSTAIN:'无法判定'})[m.call]}  P(界内)=${m.p_in.toFixed(3)}\n到${S.line_cn} ${(m.edge_distance_m*1000).toFixed(0)} ± ${(m.sigma_m*1000).toFixed(0)} mm（正=界内）${m.rally ? '' : '\n回合过滤：不像回合落点'}${m.abstain_reason ? '\n弃权原因：' + m.abstain_reason : ''}`
    : '机器没有在这里检出落点' ) + (cur && cur.shouts ? `\n喊声：${cur.shouts.map(x => x.toFixed(2)).join(', ')} s` : '');
  document.querySelectorAll('[data-st]').forEach(b => b.classList.toggle('on', rec && rec.status === b.dataset.st));
  document.querySelectorAll('[data-call]').forEach(b => b.classList.toggle('on', rec && rec.call && rec.call.label === b.dataset.call));
  document.querySelectorAll('[data-who]').forEach(b => b.classList.toggle('on', rec && rec.call && rec.call.caller === b.dataset.who));
  document.querySelectorAll('[data-ck]').forEach(b => { b.classList.toggle('on', clickKey === b.dataset.ck);
    b.textContent = b.textContent.replace(/ ✓$/, '') + (rec && rec.truth && rec.truth.clicks && rec.truth.clicks[b.dataset.ck] ? ' ✓' : ''); });
  document.querySelectorAll('[data-tv]').forEach(b => b.classList.toggle('on', rec && rec.truth && rec.truth.method !== 'clicks' && rec.truth.label === b.dataset.tv));
  $('ctime').textContent = rec && rec.truth && rec.truth.contact_ref_t != null ? fmt(rec.truth.contact_ref_t) : '';
  const tr = rec && rec.truth; let txt = '';
  if (tr && tr.edge_mm != null) txt = `${tr.label === 'in' ? '界内' : '出界'}：球托内沿到线外沿 ${tr.edge_mm} mm，球托中心 ${tr.centre_mm} mm\n球托宽度读数 ${tr.cork_width_mm} mm（规则 25–28 mm${tr.cork_width_mm < 20 || tr.cork_width_mm > 35 ? '，⚠ 读数可疑' : ''}）`;
  else if (tr && tr.method === 'clicks') txt = '已点 ' + Object.keys(tr.clicks || {}).length + '/4，保存时计算';
  else if (tr && tr.label) txt = {in:'目测界内', out:'目测出界', unreadable:'看不清'}[tr.label];
  $('truth').textContent = txt; $('note').value = rec ? rec.note || '' : '';
  $('off').textContent = off.toFixed(3); }
document.querySelectorAll('[data-st]').forEach(b => b.onclick = () => { rec.status = b.dataset.st; paint(); });
document.querySelectorAll('[data-call]').forEach(b => b.onclick = () => setCall(b.dataset.call));
function setCall(x){ if (!rec) return; rec.call = Object.assign({caller:'receiver'}, rec.call || {}, {label:x}); if (!rec.status) rec.status = 'landing'; paint(); }
document.querySelectorAll('[data-who]').forEach(b => b.onclick = () => { if (!rec) return; rec.call = Object.assign({label:'none'}, rec.call || {}, {caller:b.dataset.who}); paint(); });
document.querySelectorAll('[data-ck]').forEach(b => b.onclick = () => { if (!rec) return; clickKey = b.dataset.ck;
  if (!rec.truth || rec.truth.method !== 'clicks') rec.truth = {method:'clicks', clicks:{}, contact_ref_t: rec.truth && rec.truth.contact_ref_t};
  if (rec.truth.contact_ref_t == null) rec.truth.contact_ref_t = ref.t(); paint(); });
document.querySelectorAll('[data-tv]').forEach(b => b.onclick = () => { if (!rec) return;
  rec.truth = {method: b.dataset.tv === 'unreadable' ? 'unreadable' : 'visual', label: b.dataset.tv, contact_ref_t: rec.truth && rec.truth.contact_ref_t};
  if (!rec.status) rec.status = 'landing'; clickKey = null; paint(); ref.draw(); });
$('mark').onclick = () => { if (!rec) return; rec.truth = rec.truth || {method:'clicks', clicks:{}}; rec.truth.contact_ref_t = ref.t(); paint(); };
$('note').oninput = () => { if (rec) rec.note = $('note').value; };
function playAudio(v){ if (!cur) return; const t = v === 'sys' ? cur.t_sys : cur.t_sys + off;
  new Audio(`/audio?v=${v}&t0=${(t-0.5).toFixed(3)}&t1=${(t+3.0).toFixed(3)}`).play(); }
$('as').onclick = () => playAudio('sys'); $('ar').onclick = () => playAudio('ref');
async function save(){ if (!cur || !rec) return false;
  if (!rec.status){ $('msg').textContent = '先选"是落点 / 不是 / 跳过"'; return false; }
  rec.t_sys = cur.t_sys; if (document.activeElement) document.activeElement.blur();
  const v = await api('/api/annotate', {id: cur.id, data: rec}); S.annotations[cur.id] = v.record; rec = JSON.parse(JSON.stringify(v.record)); paint(); list(); return true; }
$('save').onclick = save;
async function saveNext(){ if (!(await save())) return; const i = S.candidates.findIndex(c => c.id === cur.id);
  const nxt = S.candidates.slice(i+1).find(c => !$('todo').checked || !S.annotations[c.id]); if (nxt) open(nxt); }
$('next').onclick = saveNext;
$('addhere').onclick = async () => { const t = sys.t(); if (t == null) return; const v = await api('/api/add', {t_sys: t});
  S.candidates.push(v.candidate); S.candidates.sort((a,b) => a.t_sys - b.t_sys); open(v.candidate); };
document.querySelectorAll('#top [data-d]').forEach(b => b.onclick = () => { off = Math.round((off + parseFloat(b.dataset.d))*1000)/1000; paint(); if (cur) open(cur); });
$('saveoff').onclick = async () => { await api('/api/offset', {offset_s: off}); $('msg').textContent = '偏移已保存'; };
$('todo').onchange = list;
$('gobtn').onclick = async () => { const t = parseFloat($('goto').value); if (isNaN(t)) return;
  cur = null; rec = null; list(); paint();
  await sys.load(Math.max(0, t - 1.5), t + 1.5); sys.seek(t); await ref.load(Math.max(0, t + off - 1.0), t + off + 1.0); ref.seek(t + off); };
document.onkeydown = e => { if (e.target.tagName === 'INPUT' && e.target.type === 'text') return;
  const keys = {'1':'in','2':'out','3':'unsure','0':'none'};
  if (keys[e.key]){ setCall(keys[e.key]); return; }
  if (e.key === 'Enter'){ e.preventDefault(); saveNext(); return; }
  if (e.key === ' '){ e.preventDefault(); ref.toggle($('pr')); return; }
  if (e.key === 'ArrowRight' || e.key === 'ArrowLeft'){ e.preventDefault(); const d = e.key === 'ArrowRight' ? 1 : -1;
    if (e.shiftKey){ sys.show(sys.k + d); if ($('follow').checked) ref.seek(sys.t() + off); } else ref.show(ref.k + d); } };
(async () => { S = await api('/api/state'); off = S.session.offset_s || 0;
  const s = S.session; $('info').textContent = `目标线：${S.line_cn} · ${s.mode === 'singles' ? '单打' : '双打'} · 参照机里场内在线的${s.ref_court_side === 'left' ? '左' : '右'}侧`;
  paint(); list(); if (S.candidates.length) open(S.candidates.find(c => !S.annotations[c.id]) || S.candidates[0]); else $('msg').textContent = '没有候选：可以在系统机画面里找到落点后点"在这一帧添加落点"'; })();
</script></body></html>"""
