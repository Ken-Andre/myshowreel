"""
engine.py -- core motion-graphics engine.
Vectorised numpy/cv2 compositor + PIL vector rasteriser + variable-font type engine.
"""
import math, os, numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont

# ----------------------------------------------------------------- constants
W, H = 1920, 1080
FPS = 60
DUR = 15.0
NF = int(round(DUR * FPS))            # 900
CX, CY = W / 2.0, H / 2.0
FONT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "assets", "fonts"))

FONTS = {
    "archivo":  ("Archivo[wdth_wght].ttf", ("Weight", "Width")),
    "anton":    ("Anton-Regular.ttf", None),
    "ablack":   ("ArchivoBlack-Regular.ttf", None),
    "bebas":    ("BebasNeue-Regular.ttf", None),
    "syne":     ("Syne[wght].ttf", ("Weight",)),
    "grotesk":  ("SpaceGrotesk[wght].ttf", ("Weight",)),
    "mono":     ("IBMPlexMono-Bold.ttf", None),
    "monosb":   ("IBMPlexMono-SemiBold.ttf", None),
    "dmmono":   ("DMMono-Regular.ttf", None),
    "plexcond": ("IBMPlexSansCondensed-Bold.ttf", None),
}
AX_RANGE = {"archivo": ((100, 900), (62, 125)), "syne": ((400, 800),), "grotesk": ((300, 700),)}

CREAM  = np.array([0.965, 0.953, 0.918], np.float32)
WHITE  = np.array([1.0, 1.0, 1.0], np.float32)
LIME   = np.array([0.70, 1.00, 0.16], np.float32)
MAG    = np.array([1.00, 0.16, 0.47], np.float32)
CYAN   = np.array([0.18, 0.90, 1.00], np.float32)
ORANGE = np.array([1.00, 0.45, 0.13], np.float32)
VIOLET = np.array([0.55, 0.35, 1.00], np.float32)
INK    = np.array([0.020, 0.021, 0.031], np.float32)
INK2   = np.array([0.055, 0.058, 0.078], np.float32)
STEEL  = np.array([0.42, 0.45, 0.52], np.float32)

# ------------------------------------------------------------------- maths
def clamp(v, a=0.0, b=1.0):
    if isinstance(v, np.ndarray):
        return np.clip(v, a, b)
    return a if v < a else (b if v > b else v)

def npclamp(a, lo=0.0, hi=1.0):
    return np.clip(a, lo, hi)

def lerp(a, b, t):
    return a + (b - a) * t

def seg(t, a, b):
    """normalised 0..1 progress of t inside [a,b]"""
    if b <= a:
        return 1.0 if t >= b else 0.0
    return clamp((t - a) / (b - a))

def mix(a, b, t):
    """scalar or array lerp"""
    return a + (b - a) * t

def smooth(t):
    return t * t * (3 - 2 * t)

def smoother(t):
    return t * t * t * (t * (t * 6 - 15) + 10)

# -- easings
def e_linear(t): return clamp(t)
def e_in_quad(t): t = clamp(t); return t * t
def e_out_quad(t): t = clamp(t); return 1 - (1 - t) ** 2
def e_in_cubic(t): t = clamp(t); return t ** 3
def e_out_cubic(t): t = clamp(t); return 1 - (1 - t) ** 3
def e_inout_cubic(t):
    t = clamp(t)
    return 4 * t ** 3 if t < 0.5 else 1 - ((-2 * t + 2) ** 3) / 2
def e_out_quart(t): t = clamp(t); return 1 - (1 - t) ** 4
def e_inout_quart(t):
    t = clamp(t)
    return 8 * t ** 4 if t < 0.5 else 1 - ((-2 * t + 2) ** 4) / 2
def e_out_quint(t): t = clamp(t); return 1 - (1 - t) ** 5
def e_out_expo(t):
    t = clamp(t)
    return 1.0 if t >= 1 else 1 - 2 ** (-9 * t)
def e_in_expo(t):
    t = clamp(t)
    return 0.0 if t <= 0 else 2 ** (9 * t - 9)
def e_inout_expo(t):
    t = clamp(t)
    if t <= 0: return 0.0
    if t >= 1: return 1.0
    return 0.5 * 2 ** (18 * t - 9) if t < 0.5 else 0.5 * (2 - 2 ** (9 - 18 * t))
def e_out_back(t, s=2.2):
    t = clamp(t); t = t - 1
    return 1 + t * t * ((s + 1) * t + s)
def e_in_back(t, s=2.2):
    t = clamp(t)
    return t * t * ((s + 1) * t - s)
def e_out_elastic(t, k=0.42):
    t = clamp(t)
    if t <= 0: return 0.0
    if t >= 1: return 1.0
    return 2 ** (-10 * t) * math.sin((t - k / 4) * (2 * math.pi) / k) + 1
def spring(t, damp=6.0, freq=9.0):
    """critically-ish damped spring, settles to 1"""
    if t <= 0: return 0.0
    return 1 - math.exp(-damp * t) * math.cos(freq * t)
def whip(t, over=0.12):
    """fast out with slight overshoot then settle"""
    t = clamp(t)
    return e_out_expo(t) * (1 + over * math.sin(math.pi * t) * (1 - t))

_bcache = {}
def cubic_bezier(x1, y1, x2, y2):
    key = (x1, y1, x2, y2)
    if key in _bcache: return _bcache[key]
    def f(t):
        t = clamp(t)
        lo, hi = 0.0, 1.0
        u = t
        for _ in range(18):
            x = 3 * (1 - u) ** 2 * u * x1 + 3 * (1 - u) * u * u * x2 + u ** 3
            if abs(x - t) < 1e-5: break
            if x < t: lo = u
            else: hi = u
            u = (lo + hi) / 2
        return 3 * (1 - u) ** 2 * u * y1 + 3 * (1 - u) * u * u * y2 + u ** 3
    _bcache[key] = f
    return f

def hold(v, t0, t1):
    return 1.0 if t0 <= t <= t1 else 0.0

def pulse(t, center, width, ease=e_out_expo):
    """0->1->0 bump around center"""
    d = abs(t - center)
    if d >= width: return 0.0
    return 1.0 - ease(d / width)

def beat_phase(t, bpm=120.0, t0=1.0):
    b = (t - t0) * bpm / 60.0
    return b, b - math.floor(b)

def rng(seed):
    r = np.random.default_rng(seed)
    return r

# --------------------------------------------------------------- buffers
def new_rgb(w=W, h=H): return np.zeros((h, w, 3), np.float32)
def new_a(w=W, h=H): return np.zeros((h, w), np.float32)
def solid(color, w=W, h=H): return np.broadcast_to(np.asarray(color, np.float32), (h, w, 3)).copy()

def paste_mask(dst, mask, x, y, gain=1.0):
    """dst(h,w,3) += mask(h,w)*gain  (region-clipped, float32)"""
    mh, mw = mask.shape
    x0 = int(round(x)); y0 = int(round(y))
    sx0 = max(0, -x0); sy0 = max(0, -y0)
    dx0 = max(0, x0);  dy0 = max(0, y0)
    dx1 = min(dst.shape[1], x0 + mw); dy1 = min(dst.shape[0], y0 + mh)
    if dx1 <= dx0 or dy1 <= dy0: return
    sx1 = sx0 + (dx1 - dx0); sy1 = sy0 + (dy1 - dy0)
    sub = mask[sy0:sy1, sx0:sx1]
    if sub.dtype != np.float32: sub = sub.astype(np.float32)
    d = dst[dy0:dy1, dx0:dx1]
    if d.ndim == 3:
        d += (sub * gain)[..., None]
    else:
        d += sub * gain

def blit(canvas, mask, x, y, color, mode="norm", opacity=1.0):
    """composite a coloured mask onto canvas at top-left (x,y)."""
    if opacity <= 0.001: return
    mh, mw = mask.shape
    x0 = int(round(x)); y0 = int(round(y))
    sx0 = max(0, -x0); sy0 = max(0, -y0)
    dx0 = max(0, x0);  dy0 = max(0, y0)
    dx1 = min(W, x0 + mw); dy1 = min(H, y0 + mh)
    if dx1 <= dx0 or dy1 <= dy0: return
    sx1 = sx0 + (dx1 - dx0); sy1 = sy0 + (dy1 - dy0)
    a = mask[sy0:sy1, sx0:sx1].astype(np.float32) * opacity
    reg = canvas[dy0:dy1, dx0:dx1]
    c = np.asarray(color, np.float32)
    if mode == "add":
        reg += a[..., None] * c
    elif mode == "sub":
        reg -= a[..., None] * c
    elif mode == "mul":
        reg *= 1.0 - a[..., None] * (1.0 - c)
    else:
        aa = a[..., None]
        reg *= (1 - aa)
        reg += aa * c

# ------------------------------------------------------- mask transforms
def transform_mask(mask, scale=1.0, rot=0.0, shear=0.0, pad=1.45, interp=cv2.INTER_LINEAR):
    """scale/rotate/shear about the mask centre; returns (out_mask, ow, oh) centred box."""
    mask = mask.astype(np.float32)
    h, w = mask.shape
    cx, cy = w / 2.0, h / 2.0
    d = math.hypot(w, h) * max(pad, abs(scale) * 1.02)
    ow = int(math.ceil(d / 2) * 2) + 2
    oh = ow
    M = cv2.getRotationMatrix2D((cx, cy), math.degrees(rot), scale)
    if abs(shear) > 1e-6:
        M = np.array([[M[0, 0] + shear * M[1, 0], M[0, 1] + shear * M[1, 1], M[0, 2]],
                      [M[1, 0], M[1, 1], M[1, 2]]], np.float64)
    M[0, 2] += ow / 2.0 - cx
    M[1, 2] += oh / 2.0 - cy
    out = cv2.warpAffine(mask, M, (ow, oh), flags=interp, borderValue=0.0)
    return out, ow, oh

def warp_center(canvas, mask, cx, cy, color, mode="norm", scale=1.0, rot=0.0,
                shear=0.0, opacity=1.0, pad=1.45):
    """place a mask centred at (cx,cy) with scale/rot/shear."""
    if opacity <= 0.001: return
    if abs(scale - 1.0) < 0.002 and abs(rot) < 1e-4 and abs(shear) < 1e-5:
        m = mask.astype(np.float32)
        blit(canvas, m, cx - m.shape[1] / 2.0, cy - m.shape[0] / 2.0, color, mode, opacity)
        return
    m, ow, oh = transform_mask(mask, scale, rot, shear, pad)
    blit(canvas, m, cx - ow / 2.0, cy - oh / 2.0, color, mode, opacity)

def resize_mask(mask, w, h, interp=cv2.INTER_AREA):
    return cv2.resize(mask.astype(np.float32), (int(w), int(h)), interpolation=interp)

def dilate(mask, r):
    if r <= 0: return mask.astype(np.float32)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(r * 2) + 1, int(r * 2) + 1))
    return cv2.dilate(mask.astype(np.float32), k)

def stroke_of(mask, r):
    return np.clip(dilate(mask, r) - mask, 0, 1)

def blur_mask(mask, sigma):
    if sigma <= 0.05: return mask.astype(np.float32)
    return cv2.GaussianBlur(mask.astype(np.float32), (0, 0), sigma)

# --------------------------------------------------------- vector (PIL SS)
class Vec:
    """supersampled PIL vector layer -> float32 mask at canvas res"""
    def __init__(self, w=W, h=H, ss=2):
        self.w, self.h, self.ss = w, h, ss
        self.img = Image.new("L", (w * ss, h * ss), 0)
        self.d = ImageDraw.Draw(self.img)

    def line(self, p0, p1, width=2.0, fill=255):
        s = self.ss
        self.d.line([p0[0] * s, p0[1] * s, p1[0] * s, p1[1] * s], fill=int(fill), width=max(1, int(round(width * s))))

    def poly(self, pts, width=2.0, fill=255, joint="curve", close=False):
        if len(pts) < 2: return
        s = self.ss
        p = [(x * s, y * s) for x, y in pts]
        if close: p = p + [p[0]]
        self.d.line(p, fill=int(fill), width=max(1, int(round(width * s))), joint=joint)

    def circle(self, c, r, width=2.0, fill=255, outline=True, a0=0.0, a1=2 * math.pi):
        s = self.ss
        if abs((a1 - a0) - 2 * math.pi) < 1e-6 and outline:
            self.d.ellipse([ (c[0]-r)*s, (c[1]-r)*s, (c[0]+r)*s, (c[1]+r)*s ],
                           outline=int(fill), width=max(1, int(round(width * s))))
        elif outline:
            n = max(8, int(abs(a1 - a0) * r * 1.2))
            pts = [(c[0] + r * math.cos(a0 + (a1 - a0) * i / n), c[1] + r * math.sin(a0 + (a1 - a0) * i / n)) for i in range(n + 1)]
            self.poly(pts, width, fill)
        else:
            self.d.pieslice([(c[0]-r)*s, (c[1]-r)*s, (c[0]+r)*s, (c[1]+r)*s],
                            math.degrees(a0), math.degrees(a1), fill=int(fill))

    def rect(self, x, y, w, h, fill=255, radius=0, outline=None, width=2.0):
        s = self.ss
        box = [x * s, y * s, (x + w) * s, (y + h) * s]
        if outline is not None:
            self.d.rounded_rectangle(box, radius=radius * s, outline=int(fill), width=max(1, int(round(width * s))))
        else:
            self.d.rounded_rectangle(box, radius=radius * s, fill=int(fill))

    def text(self, xy, s, key, size, fill=255, anchor="la", stroke=0, stroke_fill=255,
             axes=None, outline_only=False):
        sc = self.ss
        f = TY.font(key, size * sc, axes)
        self.d.text((xy[0] * sc, xy[1] * sc), s, font=f,
                    fill=0 if outline_only else int(fill), anchor=anchor,
                    stroke_width=int(round(stroke * sc)), stroke_fill=int(stroke_fill))

    def result(self):
        a = np.asarray(self.img, np.float32) * (1.0 / 255.0)
        if self.ss != 1:
            a = cv2.resize(a, (self.w, self.h), interpolation=cv2.INTER_AREA)
        return a

# ------------------------------------------------------------- beziers
def cubic_pt(p0, p1, p2, p3, t):
    u = 1 - t
    return (u*u*u*p0[0] + 3*u*u*t*p1[0] + 3*u*t*t*p2[0] + t*t*t*p3[0],
            u*u*u*p0[1] + 3*u*u*t*p1[1] + 3*u*t*t*p2[1] + t*t*t*p3[1])

def cubic_samples(p0, p1, p2, p3, n=240):
    t = np.linspace(0, 1, n)
    u = 1 - t
    x = u**3*p0[0] + 3*u**2*t*p1[0] + 3*u*t**2*p2[0] + t**3*p3[0]
    y = u**3*p0[1] + 3*u**2*t*p1[1] + 3*u*t**2*p2[1] + t**3*p3[1]
    return np.stack([x, y], 1)

def arclen(pts):
    d = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    c = np.concatenate([[0], np.cumsum(d)])
    return c / max(c[-1], 1e-9)

def at_arclen(pts, cum, s):
    """point at normalised arclength s (constant speed)"""
    i = np.searchsorted(cum, clamp(s))
    i = min(max(i, 1), len(pts) - 1)
    t = (s - cum[i - 1]) / max(cum[i] - cum[i - 1], 1e-9)
    return pts[i - 1] * (1 - t) + pts[i] * t

# ------------------------------------------------------------------- 3D
def rot3(ax=0.0, ay=0.0, az=0.0):
    cx, sx = math.cos(ax), math.sin(ax)
    cy, sy = math.cos(ay), math.sin(ay)
    cz, sz = math.cos(az), math.sin(az)
    Rx = np.array([[1,0,0],[0,cx,-sx],[0,sx,cx]], np.float32)
    Ry = np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]], np.float32)
    Rz = np.array([[cz,-sz,0],[sz,cz,0],[0,0,1]], np.float32)
    return Rz @ Ry @ Rx

def project(pts, R, cx=CX, cy=CY, fov=1400.0, dist=4.0, scale=1.0):
    p = (R @ pts.T).T
    z = p[:, 2] / max(scale, 1e-6) + dist
    f = fov / np.maximum(z, 0.25)
    return np.stack([p[:, 0] * f * scale + cx, p[:, 1] * f * scale + cy], 1), f, p[:, 2]

# ------------------------------------------------------------ type engine
class Typer:
    def __init__(self, budget=520):
        self.cache = {}
        self.order = []
        self.budget = budget
        self.fcache = {}
        self.str_cache = {}

    def _path(self, key):
        return os.path.join(FONT_DIR, FONTS[key][0])

    def font(self, key, size, axes=None):
        size = int(round(size))
        ck = (key, size, axes)
        f = self.fcache.get(ck)
        if f is None:
            f = ImageFont.truetype(self._path(key), size)
            axnames = FONTS[key][1]
            if axnames and axes:
                rngs = AX_RANGE.get(key)
                vals = []
                for i, nm in enumerate(axnames):
                    v = axes[i] if i < len(axes) else None
                    if v is None: continue
                    if rngs:
                        lo, hi = rngs[i]
                        v = clamp(v, lo, hi)
                    vals.append(float(v))
                if vals:
                    try: f.set_variation_by_axes(vals)
                    except Exception: pass
            self.fcache[ck] = f
        return f

    def glyph(self, key, size, ch, axes=None, ss=2):
        sb = int(round(size / 3.0) * 3)
        ab = None
        if axes:
            ab = tuple(int(round(a / (25 if i == 0 else 8.0)) ) * (25 if i == 0 else 8) for i, a in enumerate(axes))
        ck = (key, sb, ab, ch)
        hit = self.cache.get(ck)
        if hit is not None:
            return hit
        f = self.font(key, sb * ss, axes)
        try:
            bb = f.getbbox(ch)
        except Exception:
            bb = (0, 0, sb, sb)
        l, t, r, b = bb
        w = max(1, r - l); h = max(1, b - t)
        img = Image.new("L", (w + 6, h + 6), 0)
        ImageDraw.Draw(img).text((3 - l, 3 - t), ch, font=f, fill=255)
        m = np.asarray(img, np.float32) * (1.0 / 255.0)
        m = cv2.resize(m, None, fx=1.0 / ss, fy=1.0 / ss, interpolation=cv2.INTER_AREA)
        adv = f.getlength(ch) / ss
        entry = (m.astype(np.float16), adv, l / ss, t / ss, w / ss, h / ss)
        self.cache[ck] = entry
        self.order.append(ck)
        if len(self.order) > self.budget:
            old = self.order.pop(0)
            self.cache.pop(old, None)
        return entry

    def layout(self, key, size, text, axes=None, tracking=0.0):
        """returns (glyphs list, total width). glyph = dict(m, adv, lb, tb, w, h, pen)"""
        ck = (key, int(round(size/3.0)*3), tuple(int(round(a/(25 if i==0 else 8.0)))*(25 if i==0 else 8) for i,a in enumerate(axes)) if axes else None,
              text, round(tracking, 3))
        hit = self.str_cache.get(ck)
        if hit is not None: return hit
        glyphs = []
        pen = 0.0
        for ch in text:
            m, adv, lb, tb, gw, gh = self.glyph(key, size, ch, axes)
            glyphs.append(dict(m=m, adv=adv, lb=lb, tb=tb, w=gw, h=gh, pen=pen, ch=ch))
            pen += adv + tracking * size
        out = (glyphs, pen - tracking * size if glyphs else 0.0)
        if len(self.str_cache) < 400:
            self.str_cache[ck] = out
        return out

    def string_mask(self, key, size, text, axes=None, tracking=0.0, stroke=0.0, ss=2,
                    outline_only=False, joint="curve"):
        """whole-string raster (for labels / outline type)"""
        f = self.font(key, size * ss, axes)
        sw = int(round(stroke * ss))
        try:
            bb = f.getbbox(text, stroke_width=sw)
        except Exception:
            bb = (0, 0, 10, 10)
        l, t, r, b = bb
        w = max(1, r - l); h = max(1, b - t)
        img = Image.new("L", (w + 8, h + 8), 0)
        ImageDraw.Draw(img).text((4 - l, 4 - t), text, font=f,
                                 fill=0 if outline_only else 255,
                                 stroke_width=sw, stroke_fill=255,
                                 stroke_joint=joint if outline_only else "curve")
        m = np.asarray(img, np.float32) * (1.0 / 255.0)
        m = cv2.resize(m, None, fx=1.0/ss, fy=1.0/ss, interpolation=cv2.INTER_AREA)
        return m, w / ss, h / ss

    def draw_glyphs(self, canvas, key, size, text, cx, cy, color, axes=None, tracking=0.0,
                    mode="norm", glyph_fn=None, valign="center", align="center", anchor=None):
        """per-glyph animated type. glyph_fn(i, n, ch) -> dict(dx,dy,scale,rot,shear,op,color,mode)"""
        glyphs, tw = self.layout(key, size, text, axes, tracking)
        n = len(glyphs)
        if n == 0: return tw
        ax = cx - tw / 2.0 if align == "center" else (cx - tw if align == "right" else cx)
        if anchor is not None: ax = anchor
        if valign == "center":
            tops = [g["tb"] for g in glyphs]
            bots = [g["tb"] + g["h"] for g in glyphs]
            base = cy - (min(tops) + max(bots)) / 2.0
        elif valign == "top":
            base = cy - min(g["tb"] for g in glyphs)
        else:
            base = cy
        for i, g in enumerate(glyphs):
            p = dict(dx=0.0, dy=0.0, scale=1.0, rot=0.0, shear=0.0, op=1.0, color=color, mode=mode)
            if glyph_fn is not None:
                p.update(glyph_fn(i, n, g["ch"]) or {})
            if p["op"] <= 0.002 or abs(p["scale"]) < 0.004: continue
            m = g["m"]
            gcx = ax + g["pen"] + g["lb"] + g["w"] / 2.0
            gcy = base + g["tb"] + g["h"] / 2.0
            warp_center(canvas, m, gcx + p["dx"], gcy + p["dy"], p["color"], p["mode"],
                        p["scale"], p["rot"], p["shear"], p["op"])
            # keep pen advancing (glyph positions are absolute, no reflow)
        return tw

    def draw_string(self, canvas, key, size, text, cx, cy, color, axes=None, tracking=0.0,
                    mode="norm", opacity=1.0, align="center", stroke=0.0, scale=1.0, rot=0.0,
                    outline_only=False):
        m, w, h = self.string_mask(key, size, text, axes, tracking if tracking else 0.0, stroke,
                                   outline_only=outline_only)
        if tracking:
            # tracking applied per glyph via layout instead
            glyphs, tw = self.layout(key, size, text, axes, tracking)
            ax = cx - tw / 2.0 if align == "center" else cx
            for g in glyphs:
                blit(canvas, g["m"].astype(np.float32), ax + g["pen"] + g["lb"], cy + g["tb"], color, mode, opacity)
            return tw
        x = cx - w / 2.0 if align == "center" else (cx - w if align == "right" else cx)
        if abs(scale - 1) > 0.002 or abs(rot) > 1e-4:
            warp_center(canvas, m, x + w / 2.0, cy + h / 2.0, color, mode, scale, rot, opacity=opacity)
        else:
            blit(canvas, m.astype(np.float32), x, cy, color, mode, opacity)
        return w

TY = Typer()

# tracking-aware layout helper: reposition pen per glyph
def layout_tracked(key, size, text, axes=None, tracking=0.0):
    glyphs, tw = TY.layout(key, size, text, axes, 0.0)
    if tracking:
        pen = 0.0
        for g in glyphs:
            g = dict(g)
            g["pen"] = pen
            pen += g["adv"] + tracking * size
        glyphs = [dict(g, pen=p) for g, p in zip(glyphs, [0.0] + list(np.cumsum([gg["adv"] + tracking*size for gg in glyphs])[:-1]))]
        tw = pen - tracking * size
    return glyphs, tw

# --------------------------------------------------------------- fx
def _down(a, f):
    return cv2.resize(a, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)

def _up(a, w, h):
    return cv2.resize(a, (w, h), interpolation=cv2.INTER_LINEAR)

def bloom(rgb, threshold=0.62, intensity=0.85, radius=0.5, soft=0.35):
    lum = rgb.max(axis=2)
    k = np.clip((lum - threshold) / max(soft, 1e-3), 0, 1)
    bright = rgb * (k[..., None] ** 1.4)
    small = _down(bright, 0.25)
    sig = max(1.0, radius * 12.0)
    small = cv2.GaussianBlur(small, (0, 0), sig)
    small = cv2.GaussianBlur(small, (0, 0), sig * 0.5)
    big = _up(small, W, H)
    return rgb + big * intensity

def bloom_mask(m, intensity=1.0, radius=0.5):
    small = _down(m.astype(np.float32), 0.25)
    sig = max(1.0, radius * 12.0)
    small = cv2.GaussianBlur(small, (0, 0), sig)
    return _up(small, W, H) * intensity

def chromatic(rgb, amount, cx=CX, cy=CY):
    """radial RGB split"""
    if abs(amount) < 0.05: return rgb
    out = np.empty_like(rgb)
    for i, k in enumerate((1.0, 1.0 + amount * 0.45, 1.0 + amount)):
        M = cv2.getRotationMatrix2D((cx, cy), 0, k)   # scale about (cx,cy) == radial split
        out[:, :, i] = cv2.warpAffine(rgb[:, :, i], M, (W, H),
                                      flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return out

def zoom_blur(rgb, amount, cx=CX, cy=CY, steps=5):
    if amount <= 0.001: return rgb
    acc = np.zeros_like(rgb)
    wsum = 0.0
    for i in range(steps):
        t = i / (steps - 1)
        s = 1.0 + amount * t
        wgt = (1.0 - t * 0.7)
        M = cv2.getRotationMatrix2D((cx, cy), 0, s)
        acc += cv2.warpAffine(rgb, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE) * wgt
        wsum += wgt
    return acc / wsum

def dir_blur(mask, angle_deg, dist, steps=6):
    """directional smear of a single-channel layer"""
    if dist < 0.6: return mask.astype(np.float32)
    a = math.radians(angle_deg)
    dx, dy = math.cos(a), math.sin(a)
    acc = np.zeros_like(mask, dtype=np.float32)
    h, w = mask.shape
    M0 = np.float32([[1, 0, 0], [0, 1, 0]])
    tot = 0.0
    for i in range(steps):
        t = (i / (steps - 1) - 0.5) * dist
        M = M0.copy(); M[0, 2] = dx * t; M[1, 2] = dy * t
        acc += cv2.warpAffine(mask.astype(np.float32), M, (w, h), flags=cv2.INTER_LINEAR) * (1 - abs(i / (steps - 1) - 0.5))
        tot += (1 - abs(i / (steps - 1) - 0.5))
    return acc / max(tot, 1e-6)

_vig = None
def vignette(rgb, amount=0.42, power=2.2, tint=INK):
    global _vig
    if _vig is None:
        y, x = np.mgrid[0:H, 0:W].astype(np.float32)
        d = np.sqrt(((x - CX) / (W * 0.62)) ** 2 + ((y - CY) / (H * 0.62)) ** 2)
        _vig = np.clip(d, 0, 1.6) ** power
    v = _vig * amount
    rgb *= (1 - v)[..., None]
    rgb += np.asarray(tint, np.float32) * (v * 0.55)[..., None]
    return rgb

_scan = None
def scanlines(rgb, amount=0.10, freq=3.0, roll=0.0):
    global _scan
    y = np.arange(H, dtype=np.float32)
    s = 0.5 + 0.5 * np.sin((y * freq * math.pi / 3.0) + roll)
    rgb *= (1 - amount * s)[:, None, None]
    return rgb

_grain = {}
def grain(rgb, t, amount=0.045, mono=True):
    h, w = rgb.shape[:2]
    key = (h, w)
    tiles = _grain.get(key)
    if tiles is None:
        r = np.random.default_rng(7)
        tiles = [r.standard_normal((h + 40, w + 40)).astype(np.float32) for _ in range(6)]
        if len(_grain) < 4:
            _grain[key] = tiles
    g = tiles[int(t * 24) % 6]
    oy = int((t * 137.3) % 37); ox = int((t * 91.7) % 37)
    g = g[oy:oy + h, ox:ox + w]
    if mono:
        rgb += g[..., None] * amount
    else:
        rgb += np.stack([g, np.roll(g, 5, 0), np.roll(g, 9, 1)], -1) * amount
    return rgb

def tonemap(rgb, exposure=1.0, knee=0.92):
    x = rgb * exposure
    # filmic-ish soft clip
    return np.clip(x * (1 + x / (knee * knee)) / (1 + x), 0, 1).astype(np.float32)

def to_uint8(rgb):
    a = np.clip(rgb, 0, 1)
    a = (a ** (1 / 1.03))
    return (np.clip(a, 0, 1) * 255.0 + 0.5).astype(np.uint8)

def slice_glitch(rgb, t, amount=1.0, seed=0):
    if amount <= 0.001: return rgb
    r = np.random.default_rng(int(t * 60) + seed)
    out = rgb
    n = int(r.integers(3, 9) * amount)
    for _ in range(n):
        y0 = int(r.integers(0, H - 12))
        hh = int(r.integers(4, 60) * amount)
        sh = int(r.integers(-140, 140) * amount)
        band = np.roll(out[y0:y0 + hh], sh, axis=1)
        out = out.copy()
        out[y0:y0 + hh] = band
        if r.random() < 0.45:
            out[y0:y0 + hh, :, 0] *= 1.25
            out[y0:y0 + hh, :, 2] *= 0.8
    return out

def rgb_shift(rgb, dx, dy=0.0):
    if abs(dx) < 0.3 and abs(dy) < 0.3: return rgb
    out = rgb.copy()
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    out[:, :, 0] = cv2.warpAffine(rgb[:, :, 0], M, (W, H), borderMode=cv2.BORDER_REPLICATE)
    M2 = np.float32([[1, 0, -dx * 0.7], [0, 1, -dy * 0.7]])
    out[:, :, 2] = cv2.warpAffine(rgb[:, :, 2], M2, (W, H), borderMode=cv2.BORDER_REPLICATE)
    return out

def shake_frame(rgb, dx, dy, rot=0.0, scale=1.0):
    if abs(dx) < 0.2 and abs(dy) < 0.2 and abs(rot) < 1e-4 and abs(scale-1) < 1e-4: return rgb
    M = cv2.getRotationMatrix2D((CX, CY), math.degrees(rot), scale)
    M[0, 2] += dx; M[1, 2] += dy
    return cv2.warpAffine(rgb, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

def wipe_mask(t, direction="l2r", softness=0.06, ease=e_inout_expo):
    """returns full-frame alpha mask revealing by progress t"""
    p = ease(clamp(t))
    m = np.zeros((H, W), np.float32)
    band = max(softness, 0.001)
    if direction == "l2r":
        x = np.linspace(0, 1, W, dtype=np.float32)[None, :]
        m[:] = np.clip((p * (1 + band) - x) / band, 0, 1)
    elif direction == "r2l":
        x = np.linspace(1, 0, W, dtype=np.float32)[None, :]
        m[:] = np.clip((p * (1 + band) - x) / band, 0, 1)
    elif direction == "t2b":
        y = np.linspace(0, 1, H, dtype=np.float32)[:, None]
        m[:] = np.clip((p * (1 + band) - y) / band, 0, 1)
    else:
        y = np.linspace(1, 0, H, dtype=np.float32)[:, None]
        m[:] = np.clip((p * (1 + band) - y) / band, 0, 1)
    return m

def halftone(mask, cell=9, t=0.0):
    """convert a mask into a halftone dot field"""
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    gx = (x % cell) - cell / 2.0
    gy = (y % cell) - cell / 2.0
    d = np.sqrt(gx * gx + gy * gy) / (cell * 0.62)
    lvl = cv2.resize(mask, None, fx=1.0, fy=1.0)
    return np.clip(1.0 - d + (lvl - 0.5) * 1.6, 0, 1)

def rounded_rect_mask(x, y, w, h, r):
    v = Vec(ss=2)
    v.rect(x, y, w, h, fill=255, radius=r)
    return v.result()

_rr_cache = {}
def rrect(w, h, r):
    key = (int(w), int(h), int(r))
    if key not in _rr_cache:
        img = Image.new("L", (int(w) + 8, int(h) + 8), 0)
        ImageDraw.Draw(img).rounded_rectangle([4, 4, 4 + int(w), 4 + int(h)], radius=int(r), fill=255)
        _rr_cache[key] = np.asarray(img, np.float32) / 255.0
    return _rr_cache[key]
