"""audio.py -- 15.0 s original score, 120 BPM, locked to the visual cut grid (downbeat t=1.0s)."""
import numpy as np, math
from scipy.signal import lfilter, butter

SR = 48000
DUR = 15.0
N = int(SR * DUR)
T0 = 1.0            # first kick / downbeat
BEAT = 0.5          # 120 bpm

L = np.zeros(N, np.float64)
R = np.zeros(N, np.float64)
DUCK = np.ones(N, np.float64)

def idx(t): return int(round(t * SR))

def add(sig, t, pan=0.0, gain=1.0):
    i = idx(t)
    if i >= N: return
    n = min(len(sig), N - i)
    g = gain
    cl, cr = math.cos((pan + 1) * math.pi / 4), math.sin((pan + 1) * math.pi / 4)
    L[i:i + n] += sig[:n] * g * cl
    R[i:i + n] += sig[:n] * g * cr

def duck_at(t, depth=0.55, dur=0.30):
    i = idx(t)
    n = idx(dur)
    if i >= N: return
    e = np.ones(n)
    e[:int(0.02 * SR)] = np.linspace(1 - depth, 1.0, int(0.02 * SR))[:len(e[:int(0.02*SR)])] if False else e[:int(0.02*SR)]
    # attack: instant dip then release
    m = min(n, N - i)
    tt = np.arange(m) / SR
    DUCK[i:i + m] *= (1 - depth) + depth * (1 - np.exp(-tt / 0.11))

def bp(sig, lo, hi, order=2):
    b, a = butter(order, [lo / (SR / 2), min(hi / (SR / 2), 0.99)], btype='band')
    return lfilter(b, a, sig)

def hp(sig, f, order=2):
    b, a = butter(order, f / (SR / 2), btype='high')
    return lfilter(b, a, sig)

def lp(sig, f, order=2):
    b, a = butter(order, min(f / (SR / 2), 0.99), btype='low')
    return lfilter(b, a, sig)

rng = np.random.default_rng(1234)

def noise(n): return rng.standard_normal(n)

# ------------------------------------------------------------------ drums
def kick(t, gain=1.0):
    n = idx(0.42)
    tt = np.arange(n) / SR
    f = 165 * np.exp(-tt / 0.045) + 46
    ph = 2 * np.pi * np.cumsum(f) / SR
    body = np.sin(ph) * np.exp(-tt / 0.16)
    click = hp(noise(idx(0.012)), 2500) * np.exp(-np.arange(idx(0.012)) / SR / 0.004)
    s = body * 1.0
    s[:len(click)] += click * 0.7
    add(s * np.exp(-tt / 0.30), t, 0.0, gain)
    duck_at(t, 0.5, 0.34)

def hat(t, gain=0.16, open_=False):
    dur = 0.16 if open_ else 0.045
    n = idx(dur)
    s = hp(noise(n), 7000) * np.exp(-np.arange(n) / SR / (0.09 if open_ else 0.016))
    add(s, t, rng.uniform(-0.35, 0.35), gain)

def snare(t, gain=0.5):
    n = idx(0.22)
    tt = np.arange(n) / SR
    s = bp(noise(n), 1200, 7000) * np.exp(-tt / 0.09)
    s += np.sin(2 * np.pi * 190 * tt) * np.exp(-tt / 0.05) * 0.5
    add(s, t, 0.0, gain)

def clap(t, gain=0.42):
    n = idx(0.30)
    tt = np.arange(n) / SR
    s = bp(noise(n), 900, 4200) * np.exp(-tt / 0.11)
    for d in (0.0, 0.012, 0.024):
        i = idx(d)
        s[i:] += bp(noise(n - i), 1000, 5000)[:n - i] * np.exp(-tt[:n - i] / 0.05) * 0.6
    add(s, t, 0.12, gain)

# ------------------------------------------------------------------ tonal
def bass_note(t, f, dur=0.24, gain=0.5):
    n = idx(dur + 0.1)
    tt = np.arange(n) / SR
    s = np.sin(2 * np.pi * f * tt) + 0.35 * np.sin(2 * np.pi * f * 2 * tt) + 0.12 * np.sin(2 * np.pi * f * 3 * tt)
    e = np.minimum(tt / 0.006, 1) * np.exp(-tt / (dur * 0.9))
    s = lp(s * e, 420)
    add(s, t, 0.0, gain)

def pluck(t, f, dur=0.30, gain=0.11, pan=0.0, cutoff=2600):
    n = idx(dur)
    tt = np.arange(n) / SR
    s = np.sin(2 * np.pi * f * tt) + 0.5 * np.sin(2 * np.pi * f * 2.01 * tt) + 0.25 * np.sin(2 * np.pi * f * 3.99 * tt)
    e = np.exp(-tt / (dur * 0.42))
    s = lp(s * e, cutoff)
    add(s, t, pan, gain)

def pad_chord(t, freqs, dur=2.0, gain=0.055):
    n = idx(dur)
    tt = np.arange(n) / SR
    s = np.zeros(n)
    for k, f in enumerate(freqs):
        for det in (-0.7, 0.8):
            s += np.sin(2 * np.pi * (f + det) * tt + k) * 0.5
    e = np.minimum(tt / 0.6, 1) * np.minimum((dur - tt) / 0.7, 1)
    s = lp(s * e, 1500)
    add(s, t, -0.2, gain); add(s, t, 0.2, gain * 0.9)

def riser(t, dur=0.8, gain=0.30):
    n = idx(dur)
    tt = np.arange(n) / SR
    s = noise(n)
    f = 300 * np.exp(tt / dur * 2.6)
    s = bp(s, 200, 9000)
    e = (tt / dur) ** 2.2
    s = s * e
    s += np.sin(2 * np.pi * np.cumsum(200 * np.exp(tt / dur * 2.2)) / SR) * e * 0.4
    add(s, t, 0.0, gain)

def whoosh(t, dur=0.35, gain=0.22):
    n = idx(dur)
    tt = np.arange(n) / SR
    s = bp(noise(n), 400, 6000)
    e = np.sin(np.pi * tt / dur) ** 1.5
    add(s * e, t, rng.uniform(-0.4, 0.4), gain)

def impact(t, gain=0.85, boom_f=42):
    n = idx(1.1)
    tt = np.arange(n) / SR
    f = boom_f * 3 * np.exp(-tt / 0.06) + boom_f
    s = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-tt / 0.42)
    s += lp(noise(n), 900) * np.exp(-tt / 0.10) * 0.8
    s += hp(noise(n), 3000) * np.exp(-tt / 0.03) * 0.5
    add(s, t, 0.0, gain)

def ping(t, f=1560, gain=0.16):
    n = idx(0.9)
    tt = np.arange(n) / SR
    s = np.sin(2 * np.pi * f * tt) * np.exp(-tt / 0.30) + np.sin(2 * np.pi * f * 2.76 * tt) * np.exp(-tt / 0.12) * 0.4
    add(s, t, 0.3, gain); add(s, t, -0.3, gain * 0.8)

def stutter(t, dur=0.05, gain=0.3, f=900):
    n = idx(dur)
    tt = np.arange(n) / SR
    s = np.sign(np.sin(2 * np.pi * f * tt)) * 0.4 + bp(noise(n), 600, 5000) * 0.6
    add(s, t, rng.uniform(-0.6, 0.6), gain)

# ================================================================= arrange
CHORDS = {  # bar -> (bass root, pad freqs, arp set)
    0: (55.00, [220.0, 261.63, 329.63, 440.0], [440.0, 523.25, 659.25, 880.0]),
    1: (55.00, [220.0, 261.63, 329.63, 440.0], [440.0, 659.25, 523.25, 880.0]),
    2: (43.65, [174.61, 220.0, 261.63, 349.23], [349.23, 440.0, 523.25, 698.46]),
    3: (65.41, [196.0, 261.63, 293.66, 392.0], [392.0, 523.25, 587.33, 783.99]),
    4: (49.00, [196.0, 246.94, 293.66, 392.0], [392.0, 493.88, 587.33, 783.99]),
    5: (43.65, [174.61, 220.0, 261.63, 349.23], [349.23, 523.25, 440.0, 698.46]),
    6: (55.00, [220.0, 261.63, 329.63, 440.0], [440.0, 523.25, 659.25, 880.0]),
}
CUTS = [1.0, 2.0, 4.0, 6.5, 9.0, 11.5, 13.5]

# opening: cold air + riser into first kick
add(lp(noise(idx(1.0)), 500) * np.linspace(0, 1, idx(1.0)) ** 2, 0.0, 0.0, 0.10)
riser(0.30, 0.70, 0.26)

for b in range(28):                       # 1.0 .. 14.5
    t = T0 + b * BEAT
    if t > 13.5: break
    kick(t, 1.0 if b % 4 in (0, 2) else 0.92)
for b in range(26):                       # hats from bar 2
    t = T0 + 1.0 + b * 0.25
    if t >= 11.5: break
    hat(t, 0.15 if b % 2 else 0.09, open_=(b % 8 == 7))
for b in range(24):                       # snare/clap backbeat from 4.0
    t = 4.0 + b * 0.5
    if t >= 11.5: break
    if b % 2 == 1:
        clap(t, 0.40) if b % 4 == 3 else snare(t, 0.34)
# 16th hats in montage
for b in range(16):
    t = 11.5 + b * 0.125
    if t < 13.4: hat(t, 0.12 if b % 4 else 0.16)

for bar in range(7):
    bs = T0 + bar * 2.0
    root, padf, arpf = CHORDS[bar]
    if bar >= 1:
        pad_chord(bs, padf, 2.0, 0.05 if bar < 5 else 0.06)
    if bs >= 2.0:
        for e in range(8):
            tt = bs + e * 0.25
            if tt > 13.4: break
            f = root if e % 4 != 3 else root * 1.5
            bass_note(tt, f, 0.22, 0.46)
    if bs >= 4.0:
        dens = 2 if bs < 9.0 else 1
        for e in range(0, 16, dens):
            tt = bs + e * 0.125
            if tt > 13.4: break
            f = arpf[(e * 3 + bar) % 4] * (2 if (e % 8 == 6) else 1)
            pluck(tt, f, 0.26, 0.085, pan=(-0.5 if e % 2 else 0.5), cutoff=1800 + 900 * math.sin(tt))

# scene impacts + whooshes
for i, ct in enumerate(CUTS):
    whoosh(ct - 0.34, 0.34, 0.20)
    if i == 0: impact(ct, 0.95); ping(ct + 0.02, 1560, 0.14)
    elif i == len(CUTS) - 1: impact(ct, 1.0, 38); ping(ct + 0.05, 2093, 0.16)
    else: impact(ct, 0.5, 50)
# extra punches on type slams
impact(0.98, 0.55, 60)
for tt in (4.8, 5.6, 6.42):
    stutter(tt, 0.04, 0.22, rng.uniform(500, 1400))
# montage glitches
for k in range(14):
    tt = 11.5 + k * 0.145
    if tt < 13.4:
        stutter(tt, 0.035, 0.26, rng.uniform(400, 2000))
riser(12.6, 0.9, 0.34)
# outro shimmer
pad_chord(13.6, [220.0, 329.63, 440.0, 659.25], 1.4, 0.075)
ping(13.9, 1318, 0.10)
ping(14.2, 1760, 0.08)

# ------------------------------------------------------------------ master
mus = np.stack([L, R])
mus *= DUCK[None, :]
# gentle glue compression
mono = mus.mean(0)
env = np.maximum.accumulate(np.abs(mono)[::-1])[::-1]
gain = 1.0 / (1.0 + 0.9 * np.clip(env - 0.35, 0, 2))
mus *= gain[None, :]
# soft clip + normalise
mus = np.tanh(mus * 1.25) / np.tanh(1.25)
pk = np.abs(mus).max()
mus *= 0.89 / pk
# fade tail
fade = np.ones(N)
fade[idx(14.55):] = np.linspace(1, 0, N - idx(14.55)) ** 1.4
fade[:idx(0.02)] = np.linspace(0, 1, idx(0.02))
mus *= fade[None, :]

pcm = (mus.T * 32767).astype(np.int16)
import wave
with wave.open("/home/user/showreel/out/audio.wav", "wb") as w:
    w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
    w.writeframes(pcm.tobytes())
print("audio.wav written", pcm.shape, "peak", float(np.abs(mus).max()))
