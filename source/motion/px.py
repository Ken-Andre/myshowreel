"""px.py -- vectorised particle systems: curl-noise flow field, attractors, shockwaves, splat rendering."""
import numpy as np, cv2, math
from .engine import W, H, CX, CY, clamp, seg


class Flow:
    """divergence-free-ish trig stream field"""
    def __init__(self, seed=3, gain=62000.0):
        r = np.random.default_rng(seed)
        self.k = r.uniform(0.0009, 0.0026, (4, 2))
        self.w = r.uniform(0.20, 0.60, (4,))
        self.a = r.uniform(0.6, 1.5, (4,))
        self.ph = r.uniform(0, 6.28, (4,))
        self.gain = gain

    def vel(self, x, y, t):
        arg = (self.k[:, 0][None, :] * x[:, None] + self.k[:, 1][None, :] * y[:, None]
               + self.w[None, :] * t + self.ph[None, :])
        c = np.cos(arg)
        ddy = (self.a[None, :] * self.k[:, 1][None, :] * c).sum(1)
        ddx = (self.a[None, :] * self.k[:, 0][None, :] * c).sum(1)
        vx = ddy * self.gain
        vy = -ddx * self.gain
        # gentle swirl around centre (px/s)
        rx = x - CX; ry = y - CY
        d = np.sqrt(rx * rx + ry * ry) + 1e-3
        sw = 150.0 * np.exp(-d / 900.0)
        vx += -ry / d * sw
        vy += rx / d * sw
        return vx, vy


class Particles:
    def __init__(self, n=4600, seed=11, hw=W // 2, hh=H // 2):
        r = np.random.default_rng(seed)
        self.n = n
        self.hw, self.hh = hw, hh
        self.p = np.stack([r.uniform(0, W, n), r.uniform(0, H, n)], 1).astype(np.float32)
        self.v = np.zeros((n, 2), np.float32)
        self.sz = r.uniform(0.9, 2.8, n).astype(np.float32)
        self.tint = (r.random(n) * 3).astype(np.int8)       # 0 cyan 1 lime 2 cream
        self.buf = np.zeros((hh, hw), np.float32)
        self.trail = np.zeros((hh, hw), np.float32)
        self.target = None
        self.flow = Flow()

    # ---------------------------------------------------------------- motion
    def step_flow(self, t, dt, damp=0.88, speed=1.0):
        vx, vy = self.flow.vel(self.p[:, 0], self.p[:, 1], t)
        self.v[:, 0] = self.v[:, 0] * damp + vx * (1 - damp) * speed
        self.v[:, 1] = self.v[:, 1] * damp + vy * (1 - damp) * speed
        self.p += self.v * dt
        self._wrap()

    def set_targets_from_mask(self, mask, rng=7):
        """sample n target points weighted by mask alpha"""
        r = np.random.default_rng(rng)
        ys, xs = np.nonzero(mask > 0.42)
        if len(xs) == 0:
            self.target = np.stack([np.full(self.n, CX), np.full(self.n, CY)], 1)
            return
        idx = r.integers(0, len(xs), self.n)
        self.target = np.stack([xs[idx].astype(np.float32), ys[idx].astype(np.float32)], 1)
        self.target += r.normal(0, 1.1, (self.n, 2)).astype(np.float32)

    def set_targets_ring(self, R=330.0, cx=CX, cy=CY, turns=1.0, rng=5):
        r = np.random.default_rng(rng)
        a = np.linspace(0, 2 * math.pi * turns, self.n, endpoint=False) + r.normal(0, 0.004, self.n)
        rad = R + r.normal(0, 3.0, self.n)
        self.target = np.stack([cx + np.cos(a) * rad, cy + np.sin(a) * rad * 0.92], 1).astype(np.float32)

    def step_attract(self, dt, k, damp=0.80, jitter=0.35, flowmix=0.0):
        t = self.target
        if t is None: return
        fx = (t[:, 0] - self.p[:, 0]) * k
        fy = (t[:, 1] - self.p[:, 1]) * k
        if flowmix > 0:
            vx, vy = self.flow.vel(self.p[:, 0], self.p[:, 1], 0.0)
            fx += vx * flowmix * 0.02
            fy += vy * flowmix * 0.02
        self.v[:, 0] = self.v[:, 0] * damp + fx * dt * 60.0
        self.v[:, 1] = self.v[:, 1] * damp + fy * dt * 60.0
        self.v += np.random.default_rng(0).normal(0, jitter, (self.n, 2)).astype(np.float32)
        self.p += self.v * dt * 60.0 * 0.016 * 60.0 * 0.016  # scaled
        self.p += self.v * 0.0
        self._wrap(off=False)

    def step_attract_px(self, dt, k, damp=0.82, jitter=0.30):
        """per-frame velocity in px/frame"""
        t = self.target
        if t is None: return
        self.v[:, 0] = self.v[:, 0] * damp + (t[:, 0] - self.p[:, 0]) * k
        self.v[:, 1] = self.v[:, 1] * damp + (t[:, 1] - self.p[:, 1]) * k
        self.v += np.random.default_rng(1).normal(0, jitter, (self.n, 2)).astype(np.float32) * 0.4
        self.p += self.v
        self._wrap(off=False)

    def shock(self, cx, cy, power=26.0, falloff=520.0):
        rx = self.p[:, 0] - cx; ry = self.p[:, 1] - cy
        d = np.sqrt(rx * rx + ry * ry) + 6.0
        f = power * np.exp(-d / falloff)
        self.v[:, 0] += rx / d * f
        self.v[:, 1] += ry / d * f

    def swirl(self, amt=6.0):
        rx = self.p[:, 0] - CX; ry = self.p[:, 1] - CY
        d = np.sqrt(rx * rx + ry * ry) + 1e-3
        self.v[:, 0] += -ry / d * amt
        self.v[:, 1] += rx / d * amt

    def _wrap(self, off=True):
        p = self.p
        if off:
            p[:, 0] = np.mod(p[:, 0], W)
            p[:, 1] = np.mod(p[:, 1], H)
        else:
            p[:, 0] = np.clip(p[:, 0], -40, W + 40)
            p[:, 1] = np.clip(p[:, 1], -40, H + 40)

    # -------------------------------------------------------------- render
    def splat(self, decay=0.80, gain=1.0, size_scale=1.0, streak=0.0):
        hw, hh = self.hw, self.hh
        b = self.trail
        b *= decay
        if streak > 0:
            for k in (1, 2):
                f = k / 2.0
                self._splat_pts(self.p[:, 0] - self.v[:, 0] * streak * f,
                                self.p[:, 1] - self.v[:, 1] * streak * f,
                                self.sz * gain * (0.55 ** k))
        self._splat_pts(self.p[:, 0], self.p[:, 1], self.sz * gain)
        return b

    def _splat_pts(self, X, Y, wgt):
        hw, hh = self.hw, self.hh
        x = X * 0.5
        y = Y * 0.5
        x0 = np.floor(x).astype(np.int32)
        y0 = np.floor(y).astype(np.int32)
        fx = (x - x0).astype(np.float32)
        fy = (y - y0).astype(np.float32)
        w = wgt
        ok = (x0 >= 0) & (x0 < hw - 1) & (y0 >= 0) & (y0 < hh - 1)
        xi = x0[ok]; yi = y0[ok]; fxi = fx[ok]; fyi = fy[ok]; wi = w[ok]
        b = self.trail
        np.add.at(b, (yi, xi), (1 - fxi) * (1 - fyi) * wi)
        np.add.at(b, (yi, xi + 1), fxi * (1 - fyi) * wi)
        np.add.at(b, (yi + 1, xi), (1 - fxi) * fyi * wi)
        np.add.at(b, (yi + 1, xi + 1), fxi * fyi * wi)
        return b

    def tint_split(self):
        """returns three full-res masks (cyan, lime, cream) from current trail buffer"""
        up = cv2.resize(self.trail, (W, H), interpolation=cv2.INTER_LINEAR)
        return up

    def density(self, sel=None):
        if sel is None:
            return cv2.resize(self.trail, (W, H), interpolation=cv2.INTER_LINEAR)
        return None

    def mask_by_tint(self, tint_id):
        sel = (self.tint == tint_id)
        b = np.zeros((self.hh, self.hw), np.float32)
        x = (self.p[sel, 0] * 0.5); y = (self.p[sel, 1] * 0.5)
        x0 = np.clip(np.floor(x).astype(np.int32), 0, self.hw - 1)
        y0 = np.clip(np.floor(y).astype(np.int32), 0, self.hh - 1)
        np.add.at(b, (y0, x0), self.sz[sel] * 1.5)
        b = cv2.GaussianBlur(b, (0, 0), 1.15)
        return cv2.resize(b, (W, H), interpolation=cv2.INTER_LINEAR)
