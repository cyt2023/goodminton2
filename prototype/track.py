"""Run the project's existing TrackNetV3 + InpaintNet adapter on one video.

Reuses 04_原型/specialist_tracknet.py and its virtualenv unchanged; the output
is RUN_DIR/trajectory.json (per-frame raw detections and completed coordinates).
"""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

PROTO = Path(__file__).resolve().parents[2] / '04_原型'
PYTHON = PROTO / '.venv-vision/bin/python'
ADAPTER = PROTO / 'specialist_tracknet.py'


def sha256(file):
    h = hashlib.sha256()
    with open(file, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def run(video, out_dir):
    out_dir = Path(out_dir).resolve()   # the adapter runs with cwd = 04_原型
    out_dir.mkdir(parents=True, exist_ok=True)
    config = {'source_file': str(Path(video).resolve()), 'specialist_folder': str(out_dir.resolve()),
              'source_sha256': sha256(video)}
    cfg = out_dir / 'tracknet_config.json'
    cfg.write_text(json.dumps(config, ensure_ascii=False, indent=2))
    proc = subprocess.run([str(PYTHON), str(ADAPTER), '--config', str(cfg)], cwd=str(PROTO))
    if proc.returncode != 0:
        raise SystemExit(f'TrackNetV3 failed with exit code {proc.returncode}')
    return out_dir / 'trajectory.json'


if __name__ == '__main__':
    print(run(sys.argv[1], sys.argv[2]))
