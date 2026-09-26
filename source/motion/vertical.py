"""vertical.py -- 9:16 (1080x1920) recomposition of the 16:9 stage.
Dynamic centre-crop at 1.1667x from the full-res stage + designed type bands top/bottom
carrying a vertical-native HUD. Not a letterbox: the crop pans per scene focus."""
import math, numpy as np, cv2
from .engine import (TY, INK, CREAM, LIME, STEEL, MAG, clamp, seg, e_out_expo, e_inout_expo,
                     e_in_cubic, e_out_cubic, grain, DUR, FPS, CX)

VW, VH = 1080, 1920
TOP, BOT = 330, 330
MID_H = VH - TOP - BOT            # 1260
COVER = MID_H / 1080.0            # 1.1667 -> fills the window height

WORDS = ["IGNITION", "SYSTEMS", "KINETIC", "FLOW", "PRODUCT", "MONTAGE", "KEN-ANDRE"]
# per-scene framing: (scale, focus_x, focus_y).  scale>=COVER = full-bleed cover;
# smaller = contained float on the same INK ground (seamless).
FR = [
    (0.78, 960, 600),     # 1 title lockup must fit whole
    (0.70, 960, 530),     # 2 type stack + cube + counter
    (0.86, 960, 540),     # 3 type scene, whole words fit; full-bleed colour edge-extended
    (COVER, 960, 540),    # 4 particles
    (0.80, 700, 520),     # 5 cards + PRODUCT MOTION title
    (COVER, 960, 540),    # 6 montage
    (1.05, 960, 560),     # 7 outro lockup
]

def _put(c, m, x, y, color, op=1.0, mode="add"):
    mh, mw = m.shape
    x0, y0 = int(round(x)), int(round(y))
    sx0, sy0 = max(0, -x0), max(0, -y0)
    dx0, dy0 = max(0, x0), max(0, y0)
    dx1, dy1 = min(c.shape[1], x0 + mw), min(c.shape[0], y0 + mh)
    if dx1 <= dx0 or dy1 <= dy0: return
    a = m[sy0:sy0 + (dy1 - dy0), sx0:sx0 + (dx1 - dx0)].astype(np.float32) * op
    reg = c[dy0:dy1, dx0:dx1]
    col = np.asarray(color, np.float32)
    if mode == "add": reg += a[..., None] * col
    else:
        aa = a[..., None]; reg *= (1 - aa); reg += aa * col

def _hline(c, y, x0, x1, color, th=2, op=1.0):
    c[y:y + th, int(x0):int(x1)] += np.asarray(color, np.float32) * op

def _vline(c, x, y0, y1, color, th=2, op=1.0):
    c[int(y0):int(y1), x:x + th] += np.asarray(color, np.float32) * op

_vigv = None
def _vignette(c, amount=0.34, power=2.2):
    global _vigv
    if _vigv is None or _vigv.shape != (VH, VW):
        y, x = np.mgrid[0:VH, 0:VW].astype(np.float32)
        d = np.sqrt(((x - VW / 2) / (VW * 0.72)) ** 2 + ((y - VH / 2) / (VH * 0.66)) ** 2)
        _vigv = np.clip(d, 0, 1.6) ** power
    c *= (1 - _vigv * amount)[..., None]
    return c

def _put3(c, rgb, x, y):
    h, w = rgb.shape[:2]
    x0, y0 = int(round(x)), int(round(y))
    sx0, sy0 = max(0, -x0), max(0, -y0)
    dx0, dy0 = max(0, x0), max(0, y0)
    dx1, dy1 = min(c.shape[1], x0 + w), min(c.shape[0], y0 + h)
    if dx1 <= dx0 or dy1 <= dy0: return
    c[dy0:dy1, dx0:dx1] = rgb[sy0:sy0 + (dy1 - dy0), sx0:sx0 + (dx1 - dx0)]

def build(stage, t, scene_i, lt):
    c = np.zeros((VH, VW, 3), np.float32)
    c[:] = INK
    sc, fx, fy = FR[scene_i]
    sw, sh = int(round(1920 * sc)), int(round(1080 * sc))
    big = cv2.resize(stage, (sw, sh), interpolation=cv2.INTER_LANCZOS4)
    ox = 540 - fx * sc
    oy = TOP + MID_H / 2.0 - fy * sc
    _put3(c, big, ox, oy)
    # edge-extend so full-bleed colour stays full-bleed inside the window
    y_top, y_bot = int(round(oy)), int(round(oy)) + sh
    if y_top > TOP:
        c[TOP:y_top, :] = c[y_top:y_top + 6, :].mean(0, keepdims=True)
    if y_bot < TOP + MID_H:
        c[y_bot:TOP + MID_H, :] = c[y_bot - 6:y_bot, :].mean(0, keepdims=True)

    # ---- top band
    word = WORDS[scene_i]
    rev = e_inout_expo(seg(lt, 0.0, 0.38))
    m, w, h = TY.string_mask("anton", 158, word)
    cut = int(m.shape[1] * rev)
    clip = m.copy(); clip[:, cut:] = 0
    _put(c, clip, 60, 84, CREAM, 1.0)
    TY_ = f"SCENE {scene_i + 1:02d} / 07"
    _put(c, TY.string_mask("mono", 24, TY_)[0], 62, 84 + 168, STEEL, 0.85 * rev)
    rl = 320 * e_out_expo(seg(lt, 0.05, 0.5))
    _hline(c, 84 + 168 + 44, 62, 62 + rl, LIME, 3, 0.9)
    # corner monogram top-right
    _hline(c, 96, VW - 96, VW - 60, LIME, 2, 0.8)
    _hline(c, 130, VW - 96, VW - 60, LIME, 2, 0.8)
    _vline(c, VW - 96, 96, 132, LIME, 2, 0.8)
    _vline(c, VW - 62, 96, 132, LIME, 2, 0.8)
    cv2.circle(c, (VW - 79, 113), 5, tuple(float(v) for v in LIME), -1, lineType=cv2.LINE_AA)
    _hline(c, TOP - 3, 0, VW, LIME, 2, 0.55)

    # ---- bottom band
    y0b = TOP + MID_H
    _hline(c, y0b + 1, 0, VW, LIME, 2, 0.55)
    nm, nw, nh = TY.string_mask("archivo", 92, "KEN-ANDRE", axes=(760, 104))
    _put(c, nm, 60, y0b + 44, CREAM, 1.0)
    rm, rw, rh = TY.string_mask("mono", 26, "MOTION DESIGNER · AVAILABLE FOR WORK")
    _put(c, rm, 62, y0b + 44 + 108, STEEL, 0.9)
    cv2.circle(c, (62 + int(rw) + 26, y0b + 44 + 108 + 14), 6,
               tuple(float(v) for v in LIME), -1, lineType=cv2.LINE_AA)
    # rail
    ry = VH - 52
    _hline(c, ry, 60, VW - 60, STEEL, 2, 0.25)
    fw = (VW - 120) * (t / DUR)
    _hline(c, ry - 1, 60, 60 + fw, LIME, 4, 0.9)
    from . import scenes as S
    for ct in S.CUTS[1:-1]:
        _vline(c, int(60 + (VW - 120) * ct / DUR), ry - 6, ry + 8, LIME, 2, 0.8)
    cv2.circle(c, (int(60 + fw), ry), 6, (1.0, 1.0, 1.0), -1, lineType=cv2.LINE_AA)
    ff = int((t % 1.0) * FPS); ss = int(t)
    _put(c, TY.string_mask("dmmono", 22, f"00:00:{ss:02d}:{ff:02d}")[0], 60, ry - 44, STEEL, 0.8)
    fm = TY.string_mask("dmmono", 22, "1080×1920 · 60 FPS")[0]
    _put(c, fm, VW - 60 - fm.shape[1], ry - 44, STEEL, 0.8)

    # ---- montage treatment + close
    if 11.5 <= t < 13.5:
        from .engine import slice_glitch, scanlines
        c = slice_glitch(c, t, 1.1, seed=21)
        c = scanlines(c, 0.10, 3.0, t * 40)
    fade = 1.0 - e_in_cubic(seg(t, 14.80, 15.0))
    c *= fade
    c = _vignette(c)
    c = grain(c, t, 0.020)
    return c
