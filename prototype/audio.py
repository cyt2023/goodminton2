"""Audio for a field session: extract the soundtrack, align two phones, find shouts.

The phones are not synchronised. Both microphones hear the same hall (racket
hits, shoes, voices), so the offset between the recordings is found by cross-
correlating their onset envelopes. This is only used to pair up the same
landing in both videos, where tens of ms of sound travel time do not matter;
it is not used for triangulation.

Shouts ("内" / "出" / "不知道") are voiced, last ~0.15-1 s and carry most energy
in 250-3000 Hz, whereas racket hits are short broadband clicks. detect_shouts()
only proposes times; the call itself is labelled by a person listening to it.
"""
import subprocess
import wave
from pathlib import Path

import numpy as np
from scipy.signal import butter, find_peaks, sosfiltfilt

SR = 16000


def extract_wav(video, out_wav):
    """Mono 16 kHz WAV via macOS afconvert (no ffmpeg needed). Cached."""
    out_wav = Path(out_wav)
    if out_wav.exists() and out_wav.stat().st_mtime >= Path(video).stat().st_mtime:
        return out_wav
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(['afconvert', '-f', 'WAVE', '-d', f'LEI16@{SR}', '-c', '1', str(video), str(out_wav)],
                          capture_output=True, text=True)
    if proc.returncode or not out_wav.exists():
        raise RuntimeError(f'Could not extract audio from {video}: {proc.stderr.strip() or "afconvert failed"}')
    return out_wav


def read_wav(path):
    with wave.open(str(path), 'rb') as w:
        if w.getsampwidth() != 2 or w.getnchannels() != 1:
            raise ValueError(f'{path}: expected mono 16-bit WAV')
        x = np.frombuffer(w.readframes(w.getnframes()), '<i2').astype(np.float32) / 32768
        return x, w.getframerate()


def wav_bytes(x, sr):
    """Encode float samples as a 16-bit WAV file in memory (for the browser)."""
    import io
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype('<i2').tobytes())
    return buf.getvalue()


def _log_energy(x, sr, band, hop_s=0.01, win_s=0.02):
    sos = butter(3, band, btype='bandpass', fs=sr, output='sos')
    y = sosfiltfilt(sos, x)
    hop, win = int(hop_s * sr), int(win_s * sr)
    n = max(0, (len(y) - win) // hop + 1)
    if n == 0:
        return np.zeros(0)
    frames = np.lib.stride_tricks.as_strided(y, (n, win), (y.strides[0] * hop, y.strides[0]))
    return np.log(np.sqrt((frames ** 2).mean(axis=1)) + 1e-6)


def onset_envelope(x, sr, hop_s=0.01):
    """Positive change of broadband log energy (hits, claps, shouts), 100 Hz."""
    e = _log_energy(x, sr, (400, 7000), hop_s)
    d = np.maximum(np.diff(e, prepend=e[:1]), 0)
    return (d - d.mean()) / (d.std() + 1e-9)


def estimate_offset(x_sys, x_ref, sr, max_lag_s=None, hop_s=0.01):
    """Offset such that t_ref = t_sys + offset. Returns (offset_s, confidence).

    confidence = correlation peak divided by the best peak more than 0.5 s away
    (> ~3 means one clear alignment; ~1 means ambiguous)."""
    a, b = onset_envelope(x_sys, sr, hop_s), onset_envelope(x_ref, sr, hop_s)
    n = len(a) + len(b) - 1
    nfft = 1 << (n - 1).bit_length()
    c = np.fft.irfft(np.fft.rfft(b, nfft) * np.conj(np.fft.rfft(a, nfft)), nfft)
    lags = np.r_[np.arange(0, len(b)), np.arange(-(len(a) - 1), 0)]
    c = np.r_[c[:len(b)], c[nfft - (len(a) - 1):]]
    if max_lag_s is not None:
        keep = np.abs(lags) <= max_lag_s / hop_s
        c, lags = c[keep], lags[keep]
    i = int(np.argmax(c))
    far = np.abs(lags - lags[i]) > 0.5 / hop_s
    second = c[far].max() if far.any() else 0.0
    conf = float(c[i] / second) if second > 0 else float('inf')
    # Sub-hop refinement by a parabola through the peak.
    off = float(lags[i])
    if 0 < i < len(c) - 1:
        den = c[i - 1] - 2 * c[i] + c[i + 1]
        if den < 0:
            off += 0.5 * (c[i - 1] - c[i + 1]) / den
    return off * hop_s, conf


def detect_shouts(x, sr, min_s=0.12, max_s=1.5, hop_s=0.01):
    """Times (s) and strengths of voiced bursts that may be calls.

    Speech-band (250-3000 Hz) energy rising clearly above its local 3 s median
    and staying up for min_s..max_s seconds."""
    e = _log_energy(x, sr, (250, 3000), hop_s)
    if len(e) == 0:
        return []
    k = int(3.0 / hop_s)
    # Running median via strided windows on a decimated copy (fast enough for an hour).
    step = 10
    dec = e[::step]
    pad = np.pad(dec, (k // step // 2, k // step // 2), mode='edge')
    win = np.lib.stride_tricks.sliding_window_view(pad, k // step // 2 * 2 + 1)
    base = np.repeat(np.median(win, axis=1), step)[:len(e)]
    nov = e - base
    mad = np.median(np.abs(nov - np.median(nov))) * 1.4826 + 1e-6
    thr = max(0.8, 3 * mad)
    up = nov > thr
    events, i = [], 0
    while i < len(up):
        if not up[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(up) and (up[j + 1] or (j + 3 < len(up) and up[j + 2:j + 4].any())):
            j += 1                                        # bridge gaps < 30 ms inside a word
        dur = (j - i + 1) * hop_s
        if min_s <= dur <= max_s:
            events.append({'t': round(i * hop_s, 3), 'duration_s': round(dur, 3),
                           'strength': round(float(nov[i:j + 1].max()), 2)})
        i = j + 1
    return events
