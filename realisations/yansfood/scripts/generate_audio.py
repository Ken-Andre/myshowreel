#!/usr/bin/env python3
"""generate_audio.py -- Synthesises a 15.00s original score @ 120 BPM with drop & SFX."""
import numpy as np, math, wave
from scipy.signal import butter, lfilter

SR = 48000
DUR = 15.00
N = int(SR * DUR)
BEAT = 0.500  # 120 BPM

L = np.zeros(N, np.float64)
R = np.zeros(N, np.float64)
DUCK = np.ones(N, np.float64)

def idx(t): return int(round(t * SR))

def add(sig, t, pan=0.0, gain=1.0):
    i = idx(t)
    if i >= N: return
    n = min(len(sig), N - i)
    cl, cr = math.cos((pan + 1) * math.pi / 4), math.sin((pan + 1) * math.pi / 4)
    L[i:i + n] += sig[:n] * gain * cl
    R[i:i + n] += sig[:n] * gain * cr

def hp(sig, f, order=2):
    b, a = butter(order, min(f / (SR / 2), 0.99), btype='high')
    return lfilter(b, a, sig)

def lp(sig, f, order=2):
    b, a = butter(order, min(f / (SR / 2), 0.99), btype='low')
    return lfilter(b, a, sig)

def bp(sig, lo, hi, order=2):
    b, a = butter(order, [lo / (SR / 2), min(hi / (SR / 2), 0.99)], btype='band')
    return lfilter(b, a, sig)

rng = np.random.default_rng(42)
def noise(n): return rng.standard_normal(n)

# ----------------- DRUMS & PERCS
def kick(t, gain=1.0):
    n = idx(0.38)
    tt = np.arange(n) / SR
    f = 170 * np.exp(-tt / 0.04) + 48
    ph = 2 * np.pi * np.cumsum(f) / SR
    body = np.sin(ph) * np.exp(-tt / 0.15)
    click = hp(noise(idx(0.01)), 2500) * np.exp(-np.arange(idx(0.01)) / SR / 0.003)
    sig = body
    sig[:len(click)] += click * 0.45
    add(sig, t, 0.0, gain)
    # ducking
    i = idx(t)
    m = min(idx(0.28), N - i)
    if m > 0:
        tt_duck = np.arange(m) / SR
        DUCK[i:i + m] *= (0.45 + 0.55 * (1 - np.exp(-tt_duck / 0.10)))

def sub_drop(t, gain=1.1):
    n = idx(1.2)
    tt = np.arange(n) / SR
    f = 120 * np.exp(-tt / 0.35) + 38
    ph = 2 * np.pi * np.cumsum(f) / SR
    body = np.sin(ph) * np.exp(-tt / 0.8)
    add(body, t, 0.0, gain)

def sfx_click(t):
    n = idx(0.04)
    tt = np.arange(n) / SR
    f = 2200 * np.exp(-tt / 0.008)
    ph = 2 * np.pi * np.cumsum(f) / SR
    sig = np.sin(ph) * np.exp(-tt / 0.01) * 0.4
    add(sig, t, 0.1, 0.8)

def sfx_glass(t):
    n = idx(0.7)
    tt = np.arange(n) / SR
    sig = (np.sin(2 * np.pi * 1760 * tt) + 0.4 * np.sin(2 * np.pi * 3520 * tt)) * np.exp(-tt / 0.22)
    add(sig, t, -0.2, 0.6)

def sfx_order(t):
    # Two-tone chime (F#5 -> B5)
    n1 = idx(0.12)
    tt1 = np.arange(n1) / SR
    s1 = np.sin(2 * np.pi * 740 * tt1) * np.exp(-tt1 / 0.08)
    n2 = idx(0.35)
    tt2 = np.arange(n2) / SR
    s2 = np.sin(2 * np.pi * 987 * tt2) * np.exp(-tt2 / 0.15)
    add(s1, t, 0.0, 0.7)
    add(s2, t + 0.08, 0.0, 0.9)

def sfx_delivered(t):
    # Major triad fanfare (E5 - G#5 - B5 - E6)
    chord = [659, 830, 987, 1318]
    for k, freq in enumerate(chord):
        n = idx(0.4)
        tt = np.arange(n) / SR
        s = np.sin(2 * np.pi * freq * tt) * np.exp(-tt / 0.18)
        add(s, t + k * 0.04, (k - 1.5) * 0.3, 0.6)

def sfx_whoosh(t, dur=0.4):
    n = idx(dur)
    tt = np.arange(n) / SR
    raw = noise(n)
    filt = bp(raw, 300, 3200) * np.sin(np.pi * tt / dur) ** 2
    add(filt, t, 0.0, 0.7)

def synth_bass(t, freq, dur=0.22, gain=0.6):
    n = idx(dur)
    tt = np.arange(n) / SR
    sig = np.sin(2 * np.pi * freq * tt) + 0.3 * np.sin(2 * np.pi * freq * 2 * tt)
    env = np.sin(np.pi * tt / dur) ** 0.5
    add(sig * env, t, 0.0, gain)

# ----------------- TIMELINE SCORING (15.00s @ 120 BPM)
# Beats 01 to 23 (0.00s to 11.50s): EDM groove with downbeats
for b in range(23):
    t_beat = b * BEAT
    kick(t_beat, 0.9 if b % 2 == 0 else 0.75)
    # Bass notes
    if b >= 4 and b < 23:
        root = 55.0 if (b // 4) % 2 == 0 else 43.65
        synth_bass(t_beat, root, 0.22, 0.5)

# Clics & Iris
sfx_click(1.20)
sfx_whoosh(1.30, 0.3)
sfx_whoosh(1.95, 0.2)

# DROP AT 3.50s (Beat 08)
sub_drop(3.50, 1.3)
kick(3.50, 1.2)
sfx_whoosh(3.35, 0.35)

# Glass at 4.00s
sfx_glass(4.00)

# Breakdown at 11.50s - 12.75s: No kicks, subtle ambient swell
pad_n = idx(1.5)
pad_tt = np.arange(pad_n) / SR
pad_sig = lp(noise(pad_n), 800) * np.sin(np.pi * pad_tt / 1.5) * 0.35
add(pad_sig, 11.50, 0.0, 0.5)

# Order at 12.75s - 14.00s
sfx_click(12.75)
sfx_whoosh(12.75, 0.35) # Flood
sfx_order(13.00)        # Commande confirmée
sfx_delivered(13.75)    # Commande prête

# Wall & Iris snap
sfx_whoosh(14.20, 0.3)
sfx_click(14.50)

# FINAL RETURN SLAM at 14.75s (Beat 30)
sub_drop(14.75, 1.4)
kick(14.75, 1.3)
sfx_glass(14.80)

# ----------------- MASTERING (-14 LUFS LOUDNORM STYLE)
mus = np.stack([L, R])
mus *= DUCK[None, :]

# Soft clipping & compression
mus = np.tanh(mus * 1.3) / np.tanh(1.3)
pk = np.abs(mus).max()
if pk > 0: mus *= 0.92 / pk

# Fade out tail
fade_start = idx(14.85)
fade_len = N - fade_start
mus[:, fade_start:] *= np.linspace(1.0, 0.0, fade_len) ** 2

pcm = (mus.T * 32767).astype(np.int16)
out_path = "realisations/yansfood/audio/yansfood_score.wav"
with wave.open(out_path, "wb") as w:
    w.setnchannels(2)
    w.setsampwidth(2)
    w.setframerate(SR)
    w.writeframes(pcm.tobytes())

print(f"✅ Generated {out_path} ({DUR}s @ 48kHz, peak: {float(np.abs(mus).max()):.2f})")
