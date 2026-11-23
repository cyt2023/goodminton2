"""Local click tool for court calibration.

Usage:
  python calibrate_server.py VIDEO --out RUN_DIR [--time 2.0] [--port 8770]

Opens http://127.0.0.1:PORT . Pick a keypoint on the court map, click it in the
frame (mouse wheel zooms, drag pans), repeat for 6+ points spread over the
court, then Save. The server fits the homography, writes RUN_DIR/calibration.json
and draws the projected court lines back on the frame so you can check the fit.
Optional "measurement points" (e.g. the cork contact of a shuttle placed for a
static accuracy test) can be clicked in the same page.
"""
import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np

import calib
import court

LABEL_Y = {'back_near': '近端底线', 'long_service_near': '近端双打后发球线', 'short_service_near': '近端前发球线',
           'short_service_far': '远端前发球线', 'long_service_far': '远端双打后发球线', 'back_far': '远端底线'}
LABEL_X = {'doubles_left': '左双打边线', 'singles_left': '左单打边线', 'centre': '中线',
           'singles_right': '右单打边线', 'doubles_right': '右双打边线'}


def grab_frame(video, t):
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
    ok, frame = cap.read()
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if not ok:
        raise SystemExit(f'Cannot read a frame at {t}s from {video}')
    return frame, fps


PAGE = r"""<!doctype html><html lang="zh"><head><meta charset="utf-8"><title>场地标定</title>
<style>
body{margin:0;font:14px system-ui;display:flex;height:100vh;background:#111;color:#eee}
#side{width:300px;padding:10px;overflow:auto;background:#1b1b1b}
#main{flex:1;position:relative;overflow:hidden}
canvas{position:absolute;left:0;top:0;cursor:crosshair}
.kp{padding:3px 6px;margin:1px 0;border-radius:4px;cursor:pointer}
.kp.sel{background:#2a5bd7}.kp.done{color:#7fdc8a}
button{margin:4px 2px;padding:6px 10px}
#msg{white-space:pre-wrap;font-size:12px;color:#ccc}
svg{background:#2d6b3a;display:block;margin:6px 0}
</style></head><body>
<div id="side">
<b>1. 选点 → 2. 在画面上点击</b><br>
<small>近端 = y=0 那一端（建议选离这部手机近的一端）；左右按"站在近端看向球网"。滚轮缩放，拖动平移，右键删除当前点。</small>
<svg id="map" width="150" height="320" viewBox="-0.4 -0.4 6.9 14.2"></svg>
<div><label><input type="radio" name="mode" value="kp" checked> 场地关键点</label>
<label><input type="radio" name="mode" value="meas"> 测量点</label></div>
<div id="list"></div>
<div id="measbox" style="display:none">测量点名称 <input id="mname" value="P1" size="8"></div>
<button id="save">保存并拟合</button>
<div id="msg"></div>
</div>
<div id="main"><canvas id="c"></canvas></div>
<script>
const KP = __KEYPOINTS__, LABELS = __LABELS__, SEGS = __SEGS__;
let pts = __EXISTING__.court_points || {}, meas = __EXISTING__.measurement_points || {};
let sel = Object.keys(KP)[0], overlay = null;
const img = new Image(); img.src = '/frame.jpg';
const c = document.getElementById('c'), g = c.getContext('2d');
let s = 1, ox = 0, oy = 0, drag = null, moved = false;
function fit(){ c.width = c.parentElement.clientWidth; c.height = c.parentElement.clientHeight;
  s = Math.min(c.width/img.width, c.height/img.height); ox = 0; oy = 0; draw(); }
img.onload = fit; window.onresize = fit;
function toImg(e){ const r = c.getBoundingClientRect(); return [(e.clientX-r.left-ox)/s, (e.clientY-r.top-oy)/s]; }
function draw(){
  g.setTransform(1,0,0,1,0,0); g.clearRect(0,0,c.width,c.height);
  g.setTransform(s,0,0,s,ox,oy); g.drawImage(img,0,0);
  const r = 6/s; g.lineWidth = 1.5/s;
  if (overlay){ g.strokeStyle='rgba(255,60,200,.9)'; for (const [a,b] of overlay){ g.beginPath(); g.moveTo(a[0],a[1]); g.lineTo(b[0],b[1]); g.stroke(); } }
  for (const [n,p] of Object.entries(pts)){ g.strokeStyle = n==sel?'#ff0':'#0f0'; g.beginPath();
    g.moveTo(p[0]-r,p[1]); g.lineTo(p[0]+r,p[1]); g.moveTo(p[0],p[1]-r); g.lineTo(p[0],p[1]+r); g.stroke(); }
  for (const [n,p] of Object.entries(meas)){ g.strokeStyle='#0ff'; g.beginPath(); g.arc(p[0],p[1],r,0,7); g.stroke();
    g.fillStyle='#0ff'; g.font=(12/s)+'px sans-serif'; g.fillText(n,p[0]+r,p[1]-r); }
}
c.onwheel = e => { e.preventDefault(); const r=c.getBoundingClientRect(), mx=e.clientX-r.left, my=e.clientY-r.top;
  const k = e.deltaY<0?1.2:1/1.2; ox = mx-(mx-ox)*k; oy = my-(my-oy)*k; s*=k; draw(); };
c.onmousedown = e => { drag = [e.clientX, e.clientY, ox, oy]; moved = false; };
c.onmousemove = e => { if (!drag) return; const dx=e.clientX-drag[0], dy=e.clientY-drag[1];
  if (Math.abs(dx)+Math.abs(dy)>3) moved = true; if (moved){ ox=drag[2]+dx; oy=drag[3]+dy; draw(); } };
c.onmouseup = e => { const was = drag; drag = null; if (!was || moved || e.button!==0) return;
  const p = toImg(e).map(v=>Math.round(v*10)/10);
  if (p[0] < 0 || p[1] < 0 || p[0] > img.width || p[1] > img.height) return;  // outside the frame
  if (mode()==='kp'){ pts[sel]=p; }
  else { meas[document.getElementById('mname').value || ('P'+(Object.keys(meas).length+1))]=p;
    const m=document.getElementById('mname'); const n=parseInt(m.value.replace(/\D/g,''))||0; m.value='P'+(n+1); }
  overlay=null; render(); };
c.oncontextmenu = e => { e.preventDefault(); if (mode()==='kp') delete pts[sel]; overlay=null; render(); };
function mode(){ return document.querySelector('input[name=mode]:checked').value; }
document.querySelectorAll('input[name=mode]').forEach(r=>r.onchange=()=>{
  document.getElementById('measbox').style.display = mode()==='meas'?'block':'none'; });
function render(){
  const L = document.getElementById('list'); L.innerHTML='';
  for (const n of Object.keys(KP)){ const d=document.createElement('div');
    d.className='kp'+(n==sel?' sel':'')+(pts[n]?' done':''); d.textContent=(pts[n]?'✓ ':'')+LABELS[n];
    d.onclick=()=>{ sel=n; render(); }; L.appendChild(d); }
  const m = document.getElementById('map'); let h='';
  for (const [a,b] of SEGS) h+=`<line x1="${a[0]}" y1="${13.4-a[1]}" x2="${b[0]}" y2="${13.4-b[1]}" stroke="#fff" stroke-width=".05"/>`;
  h+=`<line x1="-0.3" y1="6.7" x2="6.4" y2="6.7" stroke="#ccc" stroke-width=".08" stroke-dasharray=".2"/>`;
  for (const n of Object.keys(pts)) h+=`<circle cx="${KP[n][0]}" cy="${13.4-KP[n][1]}" r=".15" fill="#0f0"/>`;
  h+=`<circle cx="${KP[sel][0]}" cy="${13.4-KP[sel][1]}" r=".3" fill="none" stroke="#ff0" stroke-width=".1"/>`;
  h+=`<text x="2.2" y="14.1" font-size=".6" fill="#fff">近端 y=0</text>`;
  m.innerHTML=h; draw();
}
document.getElementById('save').onclick = async () => {
  const r = await fetch('/save',{method:'POST',body:JSON.stringify({court_points:pts, measurement_points:meas})});
  const v = await r.json(); document.getElementById('msg').textContent = v.message; overlay = v.overlay_px || null; draw(); };
render();
</script></body></html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('video')
    ap.add_argument('--out', required=True, help='run directory (created if missing)')
    ap.add_argument('--time', type=float, default=2.0, help='frame time in seconds')
    ap.add_argument('--port', type=int, default=8770)
    ap.add_argument('--intrinsics', help='optional intrinsics.json from intrinsics.py')
    ap.add_argument('--court-file', help='optional measured court keypoints JSON')
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    frame, fps = grab_frame(args.video, args.time)
    ok, jpg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
    calib_file = out / 'calibration.json'
    existing = json.loads(calib_file.read_text()) if calib_file.exists() else {}
    keypoints = court.load_keypoints(args.court_file)
    labels = {n: f"{LABEL_Y.get(n.split('/')[0], n.split('/')[0])} × {LABEL_X.get(n.split('/')[1], n.split('/')[1])}"
              for n in keypoints}
    page = (PAGE.replace('__KEYPOINTS__', json.dumps(keypoints))
            .replace('__LABELS__', json.dumps(labels, ensure_ascii=False))
            .replace('__SEGS__', json.dumps(court.line_segments()))
            .replace('__EXISTING__', json.dumps(existing))).encode()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def send(self, body, ctype, code=200):
            self.send_response(code)
            self.send_header('Content-Type', ctype)
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == '/':
                self.send(page, 'text/html; charset=utf-8')
            elif self.path == '/frame.jpg':
                self.send(jpg.tobytes(), 'image/jpeg')
            else:
                self.send(b'not found', 'text/plain', 404)

        def do_POST(self):
            if self.path != '/save':
                return self.send(b'not found', 'text/plain', 404)
            body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            record = {
                'video': str(Path(args.video).resolve()), 'frame_time_s': args.time, 'fps': fps,
                'image_size': [frame.shape[1], frame.shape[0]],
                'court_file': args.court_file, 'intrinsics_file': args.intrinsics,
                'court_points': body['court_points'], 'measurement_points': body['measurement_points'],
            }
            calib_file.write_text(json.dumps(record, ensure_ascii=False, indent=2))
            try:
                cal = calib.build(record, args.intrinsics)
            except ValueError as e:
                return self.send(json.dumps({'message': f'已保存点位，但无法拟合：{e}'}, ensure_ascii=False).encode(),
                                 'application/json')
            overlay = []
            for a, b in court.line_segments():
                seg = np.linspace(a, b, 30)
                px = calib.court_to_image(cal, seg)
                overlay += [[px[i].tolist(), px[i + 1].tolist()] for i in range(len(px) - 1)
                            if np.isfinite(px[i:i + 2]).all()]
            (out / 'calibration_fit.json').write_text(json.dumps(calib.summary(cal), indent=2))
            worst = max(zip(cal['residual_px'], cal['names']))
            msg = (f"已保存 {calib_file}\n关键点 {cal['n_points']} 个，重投影 RMS {cal['residual_rms_px']:.2f} px，"
                   f"最大 {worst[0]:.2f} px（{labels[worst[1]]}）\n"
                   f"估计点击噪声 {cal['click_sigma_estimated_px']}，采用 {cal['click_sigma_used_px']:.2f} px\n"
                   "请检查粉色投影线是否与真实场地线重合。")
            self.send(json.dumps({'message': msg, 'overlay_px': overlay}, ensure_ascii=False).encode(),
                      'application/json')

    print(f'Open http://127.0.0.1:{args.port}  (Ctrl+C to stop)')
    ThreadingHTTPServer(('127.0.0.1', args.port), Handler).serve_forever()


if __name__ == '__main__':
    main()
