"""scenes.py -- the showreel. 15.0s @60fps, 7 scenes, beat-locked to a 120 BPM grid from t=1.0s."""
import math, os, numpy as np, cv2
from .engine import *
from . import px as PX

NAME = os.environ.get("REEL_NAME", "YOUR NAME")
ROLE = os.environ.get("REEL_ROLE", "MOTION DESIGNER")

# scene boundaries (global seconds)
CUTS = [0.0, 2.0, 4.0, 6.5, 9.0, 11.5, 13.5, 15.0]
SCENE_OF = lambda t: min(i for i in range(len(CUTS) - 1) if t < CUTS[i + 1]) if t < CUTS[-1] else len(CUTS) - 2

# ------------------------------------------------------------------ precompute
_B = cubic_bezier(0.72, 0.0, 0.18, 1.0)
_WHIP = cubic_bezier(0.55, 0.0, 0.25, 1.0)
_PUNCH = cubic_bezier(0.2, 0.9, 0.1, 1.0)

BEZ = cubic_samples((-80, H * 0.66), (W * 0.30, H * 0.10), (W * 0.60, H * 1.00), (W + 80, H * 0.34), 260)
BEZ_CUM = arclen(BEZ)

GLYPHSET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789#%&/\\<>*"

def _cube_edges():
    c = [(-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1),
         (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)]
    p = np.array(c, np.float32) * 0.5
    e = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
    return p, e
CUBE_P, CUBE_E = _cube_edges()

def _grid3d(n=5, s=0.62):
    pts = []
    edges = []
    for i in range(n):
        for j in range(n):
            pts.append(((-0.5 + i / (n - 1)) * s * 2, 0.62, (-0.5 + j / (n - 1)) * s * 2))
    idx = lambda i, j: i * n + j
    for i in range(n):
        for j in range(n):
            if i + 1 < n: edges.append((idx(i, j), idx(i + 1, j)))
            if j + 1 < n: edges.append((idx(i, j), idx(i, j + 1)))
    return np.array(pts, np.float32), edges
GRID_P, GRID_E = _grid3d()

def ring_targets_mask(text="FLUID", size=330):
    m, w, h = TY.string_mask("anton", size, text)
    x = CX - w / 2; y = CY - h / 2 - 20
    full = np.zeros((H, W), np.float32)
    blit_a = full
    mh, mw = m.shape
    full[int(y):int(y) + mh, int(x):int(x) + mw] = m
    return full

PRE = {}
def precompute():
    PRE["flow_mask"] = ring_targets_mask("FLUID", 340)
    PRE["grid_dots"] = _dots()
def _dots():
    cols, rows = 24, 13
    xs = np.linspace(W * 0.06, W * 0.94, cols)
    ys = np.linspace(H * 0.14, H * 0.88, rows)
    gx, gy = np.meshgrid(xs, ys)
    return gx.ravel(), gy.ravel(), cols, rows

# ------------------------------------------------------------------- utils
def impact(t, at, dur=0.34, power=1.0):
    d = (t - at) / dur
    if d < 0 or d > 1: return 0.0, 0.0
    e = (1 - d) ** 2.2
    shake = e * power
    flash = max(0.0, 1 - d * 3.2) * power
    return shake, flash

def bg_base(tint=INK, lift=0.0):
    c = new_rgb()
    c[:] = tint
    if lift > 0:
        y, x = np.mgrid[0:H, 0:W].astype(np.float32)
        d = np.sqrt(((x - CX) / (W * 0.75)) ** 2 + ((y - CY) / (H * 0.75)) ** 2)
        c += np.asarray(CREAM, np.float32) * (np.clip(1 - d, 0, 1) ** 2 * lift)[..., None]
    return c

# ================================================================ SCENE 1
def s1(lt):
    c = bg_base(INK, 0.035)
    t = lt

    # --- curve draw 0.12 -> 0.86
    dp = seg(t, 0.10, 0.86)
    s = _WHIP(dp) if dp > 0 else 0.0
    morph = seg(t, 0.86, 1.10)
    head = at_arclen(BEZ, BEZ_CUM, s) if s > 0 else BEZ[0]

    if s > 0.001:
        n = int(s * (len(BEZ) - 1))
        pts = BEZ[:max(n, 2)].copy()
        # morph into straight rule under the title
        if morph > 0:
            m = e_inout_cubic(morph)
            tx = np.linspace(CX - 300, CX + 300, len(pts))
            ty = np.full(len(pts), CY + 158.0)
            pts[:, 0] = lerp(pts[:, 0], tx, m)
            pts[:, 1] = lerp(pts[:, 1], ty, m)
        v = Vec(ss=2)
        v.poly(pts, width=3.4, fill=210)
        tail = pts[max(0, len(pts) - 26):]
        if len(tail) > 1: v.poly(tail, width=6.5, fill=255)
        mask = v.result()
        blit(c, mask, 0, 0, CREAM, "add", 1.0 - 0.85 * seg(t, 1.02, 1.34))
        # comet head
        if morph < 0.9:
            hm = np.zeros((H, W), np.float32)
            cv2.circle(hm, (int(head[0]), int(head[1])), 9, 1.0, -1, lineType=cv2.LINE_AA)
            hm = blur_mask(hm, 2.0)
            blit(c, hm, 0, 0, CREAM, "add", 1.6)
            hg = bloom_mask(hm, 2.4, 0.6)
            blit(c, hg, 0, 0, LIME, "add", 0.9)

    # --- impact + shake
    shk, flash = impact(t, 0.98, 0.42, 1.0)

    # --- hero title
    tp = seg(t, 0.96, 2.4)
    if tp > 0:
        wdth = lerp(66, 112, e_out_expo(seg(t, 0.96, 1.55)))
        wght = lerp(180, 900, e_out_expo(seg(t, 0.96, 1.45)))
        opac = 1.0 - e_in_cubic(seg(t, 1.86, 2.05))
        def gf(i, n, ch):
            d = i * 0.052
            p = spring(clamp((t - 0.96 - d) * 2.4), damp=7.0, freq=10.5)
            ent = e_out_expo(clamp((t - 0.96 - d) * 2.0))
            return dict(dy=(1 - p) * 300, op=clamp(ent * 1.6) * opac,
                        scale=0.80 + 0.20 * p, rot=(1 - p) * (0.18 if i % 2 else -0.18))
        TY.draw_glyphs(c, "archivo", 268, "MOTION", CX, CY - 46, CREAM,
                       axes=(wght, wdth), tracking=0.004, glyph_fn=gf)
        # rule under
        rl = e_out_expo(seg(t, 1.05, 1.5))
        if rl > 0.01:
            v = Vec(ss=2)
            w2 = 640 * rl
            v.rect(CX - w2 / 2, CY + 152, w2, 6, fill=255)
            blit(c, v.result(), 0, 0, LIME, "add", 1.0)
        # subtitle tracking-in
        sub_p = seg(t, 1.25, 1.9)
        if sub_p > 0:
            tr = lerp(0.62, 0.20, e_out_expo(sub_p))
            TY.draw_string(c, "mono", 27, "MOTION  DESIGNER  ·  SHOWREEL  ’26", CX, CY + 214,
                           STEEL, tracking=tr, mode="add", opacity=e_out_cubic(sub_p) * 0.95)
        # crop marks
        cp = e_out_expo(seg(t, 1.2, 1.7))
        if cp > 0.02:
            v = Vec(ss=2)
            L = 46 * cp
            for (mx, my, sx, sy) in [(96, 96, 1, 1), (W-96, 96, -1, 1), (96, H-96, 1, -1), (W-96, H-96, -1, -1)]:
                v.line((mx, my), (mx + L * sx, my), 2.0, 200)
                v.line((mx, my), (mx, my + L * sy), 2.0, 200)
            blit(c, v.result(), 0, 0, STEEL, "add", 0.55)

    if flash > 0.01:
        c += flash * 0.85 * CREAM
    if shk > 0.01:
        c = shake_frame(c, math.sin(t * 91) * 26 * shk, math.cos(t * 77) * 18 * shk,
                        rot=math.sin(t * 60) * 0.010 * shk, scale=1 + 0.012 * shk)
    # exit whip
    ex = seg(t, 1.86, 2.0)
    if ex > 0:
        c = zoom_blur(c, e_in_cubic(ex) * 0.10)
    return c

# ================================================================ SCENE 2
def s2(lt):
    t = lt
    c = bg_base(INK, 0.02)

    gx, gy, cols, rows = PRE["grid_dots"]
    # reveal wipe
    wp = e_inout_expo(seg(t, 0.0, 0.30))
    # dot grid with diagonal wave
    dm = np.zeros((H, W), np.float32)
    n = len(gx)
    dly = ((gx / W) * 0.5 + (gy / H) * 0.5)
    app = np.clip((t - 0.12 - dly * 0.42) * 6.0, 0, 1)
    wave = np.sin(t * 5.0 - (gx * 0.006 + gy * 0.008)) * 6.0 * np.clip((t - 0.7) * 2, 0, 1)
    rad = (1.4 + 2.1 * e_out_back(app)) * app
    # some lime accents
    accent = ((np.arange(n) * 7919) % 23 == 0)
    xs = (gx).astype(np.int32); ys = (gy + wave).astype(np.int32)
    ok = (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H) & (app > 0.01)
    tmp = np.zeros((H, W), np.float32)
    rr = np.clip(rad, 0.5, 5.0).astype(np.int32)
    for i in np.nonzero(ok)[0]:
        r = rr[i]
        x0, y0 = xs[i], ys[i]
        a = app[i]
        cv2.circle(tmp, (x0, y0), r, a, -1, lineType=cv2.LINE_AA)
    blit(c, tmp, 0, 0, STEEL, "add", 0.62)
    acc = tmp * 0
    for i in np.nonzero(ok & accent)[0]:
        cv2.circle(acc, (xs[i], ys[i]), rr[i] + 1, app[i], -1, lineType=cv2.LINE_AA)
    blit(c, acc, 0, 0, LIME, "add", 0.9)

    # 3D wireframe cube
    cp = seg(t, 0.30, 1.0)
    if cp > 0:
        R = rot3(ax=0.42 + t * 0.5, ay=t * 0.85, az=0.12)
        size = 1.15
        P, f, z = project(CUBE_P * size, R, cx=W * 0.665, cy=H * 0.46, fov=1250, dist=3.2)
        v = Vec(ss=2)
        drawn = e_out_expo(cp)
        for k, (a, b) in enumerate(CUBE_E):
            pr = clamp(drawn * 1.6 - k * 0.05)
            if pr <= 0: continue
            p0 = P[a]; p1 = lerp(P[a], P[b], pr)
            v.line(tuple(p0), tuple(p1), 2.2, 235)
        mask = v.result()
        blit(c, mask, 0, 0, CREAM, "add", 0.9)
        # vertices
        vm = np.zeros((H, W), np.float32)
        for p in P: cv2.circle(vm, (int(p[0]), int(p[1])), 5, 1.0, -1, lineType=cv2.LINE_AA)
        blit(c, vm, 0, 0, LIME, "add", 1.1)
        blit(c, bloom_mask(vm, 2.0, 0.5), 0, 0, LIME, "add", 0.8)

    # left type stack
    hp = seg(t, 0.34, 0.86)
    if hp > 0:
        m = TY.string_mask("anton", 150, "SYSTEMS")[0]
        rev = e_inout_expo(hp)
        clip = np.zeros_like(m)
        cut = int(m.shape[0] * rev)
        clip[:cut] = m[:cut]
        blit(c, clip, W * 0.075, CY - 170, CREAM, "add", 1.0)
    rows_txt = ["GRID LOGIC", "HIERARCHY", "CONSTRAINT"]
    for i, s in enumerate(rows_txt):
        rp = seg(t, 0.62 + i * 0.13, 1.05 + i * 0.13)
        if rp > 0:
            e = e_out_expo(rp)
            x = W * 0.078 + (1 - e) * -60
            TY.draw_string(c, "mono", 26, s, x, CY + 40 + i * 46, STEEL, align="left",
                           mode="add", opacity=e * 0.95, tracking=0.10)
            v = Vec(ss=2)
            v.rect(x - 26, CY + 40 + i * 46 + 4, 12 * e, 12, fill=255)
            blit(c, v.result(), 0, 0, LIME, "add", e)
    # leader lines to cube
    lp = seg(t, 0.9, 1.4)
    if lp > 0:
        v = Vec(ss=2)
        e = e_out_expo(lp)
        v.poly([(W*0.078, CY+52), (W*0.30, CY+52), (W*0.30 + (W*0.60-W*0.30)*e*0.55, CY-40)], 1.6, 170)
        blit(c, v.result(), 0, 0, STEEL, "add", 0.5)
    # counters
    cnt = int(seg(t, 0.5, 1.7) * 100)
    TY.draw_string(c, "mono", 40, f"{cnt:03d}%", W * 0.90, H * 0.86, CREAM, align="right",
                   mode="add", opacity=0.9)
    TY.draw_string(c, "mono", 20, "12 COL  /  8 PT", W * 0.90, H * 0.915, STEEL, align="right",
                   mode="add", opacity=0.8, tracking=0.12)
    # entry bar
    if t < 0.34:
        bw = 130 * (1 - seg(t, 0.20, 0.34))
        v = Vec(ss=2)
        v.rect(wp * (W + 200) - 100 - bw, 0, bw + 6, H, fill=255)
        blit(c, v.result(), 0, 0, LIME, "add", 1.0)
    # exit collapse
    ex = seg(t, 1.84, 2.0)
    if ex > 0:
        c = zoom_blur(c, e_in_cubic(ex) * 0.16)
        c = shake_frame(c, e_in_cubic(ex) * 40, 0)
    return c

# ================================================================ SCENE 3
def s3(lt):
    t = lt
    c = bg_base(INK, 0.02)
    # three words
    A, B, Cc = 0.0, 0.80, 1.62

    if t < A + 0.80:
        # KINETIC -- magenta block wipe, letters revealed by sweeping bar
        bp = e_inout_expo(seg(t, 0.0, 0.20))
        blk = np.zeros((H, W), np.float32)
        blk[:, :int(W * bp)] = 1.0
        blit(c, blk, 0, 0, MAG, "norm", 1.0)
        sweep = e_inout_cubic(seg(t, 0.10, 0.62))
        barx = -300 + (W + 600) * sweep
        def gf(i, n, ch):
            gx0 = CX - 700 + i * (1400 / 6)
            reveal = clamp((barx - gx0) / 220.0)
            return dict(op=reveal, dx=(1 - e_out_expo(seg(t, 0.06 + i*0.03, 0.5 + i*0.03))) * -180,
                        rot=(1 - reveal) * 0.30)
        TY.draw_glyphs(c, "ablack", 240, "KINETIC", CX, CY - 20, CREAM, tracking=0.0, glyph_fn=gf)
        # sweeping bar
        v = Vec(ss=2)
        v.rect(barx, 0, 26, H, fill=255)
        blit(c, v.result(), 0, 0, CREAM, "add", 0.9 * (1 - seg(t, 0.60, 0.72)))
        TY.draw_string(c, "mono", 24, "01 / TYPE IN MOTION", W * 0.07, H * 0.90, CREAM,
                       align="left", mode="add", opacity=0.75 * bp, tracking=0.16)

    elif t < B + 0.82:
        lt2 = t - B
        # RHYTHM -- scramble decode, beat pulse
        _, ph = beat_phase(t + 0.0, 120, 1.0)
        puls = max(0.0, 1 - ph * 3.0)
        # halftone bg
        hm = np.zeros((H, W), np.float32)
        yy, xx = np.mgrid[0:H:13, 0:W:13].astype(np.float32)
        d = np.sqrt((xx - CX) ** 2 + (yy - CY) ** 2)
        lvl = np.clip(1.05 - d / 950 + puls * 0.30, 0, 1)
        rr = (lvl * 2.4).astype(np.int32)
        for (yi, xi, r) in zip(yy.ravel().astype(int), xx.ravel().astype(int), rr.ravel()):
            if r > 0: cv2.circle(hm, (xi, yi), r, 0.55, -1)
        blit(c, hm, 0, 0, INK2, "add", 0.55)
        rng = np.random.default_rng(4)
        word = "RHYTHM"
        def gf(i, n, ch):
            res = seg(lt2, 0.10 + i * 0.075, 0.34 + i * 0.075)
            done = res >= 1
            ch_ = ch if done else GLYPHSET[(i * 7 + int(t * 22) * (i + 3)) % len(GLYPHSET)]
            j = math.sin(i * 13.7 + t * 40) * 26 * (1 - res)
            return dict(op=0.35 + 0.65 * res + (0.0 if done else 0.25), dx=0.0, rot=0.0,
                        dy=j, scale=1.0 + puls * 0.06 * (1 if i % 2 else -1), ch=ch_)
        # draw_glyphs doesn't accept ch override -> do manually
        glyphs, tw = TY.layout("anton", 250, word)
        ax = CX - tw / 2
        tops = [g["tb"] for g in glyphs]; bots = [g["tb"] + g["h"] for g in glyphs]
        base = CY - 20 - (min(tops) + max(bots)) / 2
        for i, g in enumerate(glyphs):
            p = gf(i, len(glyphs), g["ch"])
            ch = p.pop("ch")
            m, adv, lb, tb, gw, gh = TY.glyph("anton", 250, ch)
            gcx = ax + g["pen"] + lb + gw / 2
            gcy = base + tb + gh / 2
            warp_center(c, m, gcx + p["dx"], gcy + p["dy"], LIME if i % 2 == 0 else CREAM,
                        "add", p["scale"], p["rot"], opacity=min(1, p["op"]))
        TY.draw_string(c, "mono", 24, "02 / DECODE + BEAT-SYNC", W * 0.07, H * 0.90, STEEL,
                       align="left", mode="add", opacity=0.8, tracking=0.16)

    else:
        lt3 = t - Cc
        # CRAFT -- width/weight axis slam
        wp = seg(lt3, 0.04, 0.34)
        wdth = lerp(62, 125, e_out_elastic(wp))
        wght = lerp(100, 900, e_out_expo(seg(lt3, 0.04, 0.30)))
        exitp = seg(lt3, 0.62, 0.88)
        # lime disc behind
        rr0 = int(340 * e_out_back(seg(lt3, 0.0, 0.4)))
        if rr0 > 2:
            dm = np.zeros((H, W), np.float32)
            cv2.circle(dm, (int(CX), int(CY)), rr0, 1.0, -1, lineType=cv2.LINE_AA)
            v = Vec(ss=2)
            v.circle((CX, CY), rr0, 2.5, 255)
            ring = v.result()
            blit(c, ring, 0, 0, LIME, "add", 0.75)
            blit(c, bloom_mask(ring, 0.85, 0.8), 0, 0, LIME, "add", 0.55)
        def gf(i, n, ch):
            ang = (i - (n - 1) / 2) * 0.7
            ex = exitp ** 1.6
            return dict(dx=math.cos(ang) * ex * 900, dy=math.sin(ang) * ex * 500 - ex * 120,
                        op=1 - exitp, rot=ex * ang * 0.5,
                        scale=1 + ex * 0.6)
        TY.draw_glyphs(c, "archivo", 260, "CRAFT", CX, CY - 10, CREAM,
                       axes=(wght, wdth), tracking=lerp(-0.02, 0.06, exitp), glyph_fn=gf)
        TY.draw_string(c, "mono", 24, "03 / VARIABLE AXES · Wght 100→900 · Wdth 62→125",
                       W * 0.07, H * 0.90, STEEL, align="left", mode="add",
                       opacity=0.85 * seg(lt3, 0.2, 0.4), tracking=0.10)
    # cut flashes
    for ct in (B, Cc):
        f = max(0.0, 1 - abs(t - ct) * 26)
        if f > 0: c += f * 0.7 * CREAM
    return c

# ================================================================ SCENE 4
PART = None
def _part():
    global PART
    if PART is None:
        PART = PX.Particles(n=4200, seed=21)
    return PART

def s4_physics(t, dt=1.0/60.0):
    """advance the simulation only (lets stills/scrubs match the real timeline)"""
    P = _part()
    if t < 0.92:
        P.step_flow(t * 0.9, dt, damp=0.86, speed=1.0)
    elif t < 1.90:
        if P.target is None:
            P.set_targets_from_mask(PRE["flow_mask"], rng=9)
        k = lerp(0.004, 0.16, e_out_cubic(seg(t, 0.92, 1.45)))
        P.step_attract_px(dt, k, damp=0.80, jitter=0.5 * (1 - seg(t, 1.2, 1.7)))
    elif t < 1.98:
        if t - dt < 1.90:
            P.shock(CX, CY, power=34.0, falloff=700.0)
        P.step_attract_px(dt, 0.0, damp=0.965, jitter=0.0)
    else:
        if getattr(P, "_ring", False) is False:
            P.set_targets_ring(R=345.0)
            P._ring = True
        k = lerp(0.0, 0.10, e_out_cubic(seg(t, 1.98, 2.3)))
        P.step_attract_px(dt, k, damp=0.84, jitter=0.4)
        P.swirl(1.4)
    P.splat(decay=0.80 if t < 1.9 else 0.72, gain=1.0)


def s4(lt):
    t = lt
    P = _part()
    s4_physics(t)
    c = bg_base(INK, 0.015)
    trail = P.trail
    dens = cv2.resize(trail, (W, H), interpolation=cv2.INTER_LINEAR)
    dens = np.clip(dens * 0.5, 0, 3)
    m0 = P.mask_by_tint(0); m1 = P.mask_by_tint(1); m2 = P.mask_by_tint(2)
    blit(c, m0, 0, 0, CYAN, "add", 1.05)
    blit(c, m1, 0, 0, LIME, "add", 1.15)
    blit(c, m2, 0, 0, CREAM, "add", 0.55)
    g = bloom_mask(np.clip(dens, 0, 1.6), 1.1, 0.75)
    blit(c, g, 0, 0, CYAN, "add", 0.42)
    blit(c, g, 0, 0, LIME, "add", 0.20)

    # streamlines: integrate the field from seed points so the flow reads as a field
    sp = seg(t, 0.02, 0.95)
    if sp > 0 and t < 1.35:
        nL, nS = 16, 46
        r = np.random.default_rng(31)
        pts = np.stack([r.uniform(W * 0.05, W * 0.95, nL), r.uniform(H * 0.08, H * 0.92, nL)], 1)
        v = Vec(ss=2)
        fade = 1.0 - seg(t, 1.05, 1.35)
        for li in range(nL):
            p = pts[li].copy()
            path = [p.copy()]
            for s_i in range(nS):
                vx, vy = P.flow.vel(p[:1], p[1:2], t * 0.9)
                p = p + np.array([vx[0], vy[0]]) * 0.016
                path.append(p.copy())
            path = np.array(path)
            keep = int(nS * min(1.0, sp * 1.25))
            if keep < 3: continue
            v.poly([tuple(q) for q in path[:keep]], 1.4, 150)
            hd = path[keep - 1] * 2
            v.d.ellipse([hd[0] - 5, hd[1] - 5, hd[0] + 5, hd[1] + 5], fill=255)
        blit(c, v.result(), 0, 0, CYAN, "add", 0.30 * fade)

    lab = "SIMULATION · FLOW · FORCE · FORM"
    nshow = int(seg(t, 0.25, 1.2) * len(lab))
    if nshow > 0:
        TY.draw_string(c, "mono", 24, lab[:nshow], W * 0.07, H * 0.90, STEEL, align="left",
                       mode="add", opacity=0.85, tracking=0.14)
    TY.draw_string(c, "mono", 22, f"n=4200   t={t:04.2f}", W * 0.93, H * 0.10, STEEL,
                   align="right", mode="add", opacity=0.6, tracking=0.10)
    return c

# ================================================================ SCENE 5
def _card_plate(variant, w=470, h=310, t=0.0):
    """returns rgb(h,w,3), alpha(h,w) of a mini-UI card"""
    rgb = np.zeros((h, w, 3), np.float32)
    a = np.zeros((h, w), np.float32)
    a[:] = 1.0
    rgb[:] = INK2 * 1.9 + 0.035
    v_img = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(v_img)
    d.rounded_rectangle([0, 0, w - 1, h - 1], radius=18, fill=255)
    a = np.asarray(v_img, np.float32) / 255.0
    # header
    rgb[16:52, 20:w-20] = rgb[16:52, 20:w-20] * 0.4 + CREAM * 0.10
    # title dots
    dot = tuple(float(x * 0.5) for x in CREAM)
    for i in range(3):
        cv2.circle(rgb, (34 + i * 18, 34), 5, dot, -1, lineType=cv2.LINE_AA)
    # text lines
    for i, lw in enumerate((0.72, 0.55, 0.63)):
        y = 78 + i * 26
        cv2.rectangle(rgb, (24, y), (int(24 + (w - 48) * lw), y + 8), tuple(float(x*0.35) for x in CREAM), -1)
    # toggle
    tx, ty = w - 96, 96
    on = t > 0.9
    cv2.rectangle(rgb, (tx, ty), (tx + 66, ty + 32), (0.10, 0.11, 0.14), -1)
    knob = 1.0 if on else 0.0
    kx = tx + 4 + knob * 34
    col = LIME if on else STEEL
    cv2.rectangle(rgb, (tx, ty), (tx + 66, ty + 32), tuple(float(x) for x in (col * (0.5 if on else 0.35))), 2, lineType=cv2.LINE_AA)
    cv2.circle(rgb, (int(kx + 14), ty + 16), 12, tuple(float(x) for x in col), -1, lineType=cv2.LINE_AA)
    # progress bar
    pr = e_out_expo(seg(t, 1.0, 1.7))
    cv2.rectangle(rgb, (24, h - 92), (w - 24, h - 80), (0.10, 0.11, 0.14), -1)
    cv2.rectangle(rgb, (24, h - 92), (int(24 + (w - 48) * pr), h - 80), tuple(float(x) for x in LIME), -1)
    # bar chart
    for i in range(7):
        bh = (0.25 + 0.75 * abs(math.sin(i * 1.7 + variant))) * e_out_back(seg(t, 1.05 + i * 0.05, 1.6 + i * 0.05))
        x0 = 24 + i * ((w - 48) // 7) + 6
        bw = (w - 48) // 7 - 12
        y1 = h - 26
        y0 = int(y1 - 44 * bh)
        cc = CYAN if i % 2 else CREAM * 0.75
        cv2.rectangle(rgb, (x0, y0), (x0 + bw, y1), tuple(float(x) for x in cc), -1)
    return rgb, a

def s5(lt):
    t = lt
    c = bg_base(INK, 0.03)
    R = 620.0
    ncards = 5
    base_ang = math.pi - 0.16 + t * 0.10
    cards = []
    for i in range(ncards):
        th = base_ang + i * (2 * math.pi / ncards)
        cen = np.array([math.sin(th) * R, 0.0, math.cos(th) * R - 260.0], np.float32)
        Ry = rot3(ay=th * 0.85)
        cw, ch = 470, 310
        loc = np.array([(-cw/2, -ch/2, 0), (cw/2, -ch/2, 0), (cw/2, ch/2, 0), (-cw/2, ch/2, 0)], np.float32)
        world = cen[None, :] + loc @ Ry.T
        P, f, z = project(world, np.eye(3, dtype=np.float32), cx=CX, cy=CY + 20, fov=1700, dist=2000)
        cards.append((cen[2], i, P, th))
    cards.sort(key=lambda x: -x[0])          # farthest first (largest z) -> draw to nearest
    active_i = min(cards, key=lambda x: x[0])[1]
    for depth, i, P, th in cards:
        act = (i == active_i)
        rgb_p, a_p = _card_plate(i, t=t if act else 0.0)
        znorm = 1.0 - (depth + R + 260) / (2 * R)     # 0 far .. 1 near
        if znorm < 0.8:
            rgb_p = cv2.GaussianBlur(rgb_p, (0, 0), 3.2 * (0.8 - znorm) + 0.6)
        quad = P.astype(np.float32)
        src = np.array([[0, 0], [470, 0], [470, 310], [0, 310]], np.float32)
        M = cv2.getPerspectiveTransform(src, quad)
        wrgb = cv2.warpPerspective(rgb_p, M, (W, H), flags=cv2.INTER_LINEAR)
        wa = cv2.warpPerspective(a_p, M, (W, H), flags=cv2.INTER_LINEAR)
        dim = 0.24 + 0.76 * (znorm ** 1.4)
        reg_a = (wa * dim)[..., None]
        c[:] = c * (1 - reg_a) + wrgb * reg_a
        if act:
            # cursor + ripple on toggle
            tp = np.array([[[470 - 96 + 33, 96 + 16]]], np.float32)
            cur = cv2.perspectiveTransform(tp, M)[0][0]
            # cursor path
            cp = e_inout_cubic(seg(t, 0.35, 0.85))
            start = np.array([W * 0.16, H * 0.86])
            pos = start * (1 - cp) + cur * cp
            if cp > 0.001:
                cm = np.zeros((H, W), np.float32)
                pts = np.array([[0, 0], [0, 26], [7, 20], [12, 30], [16, 28], [11, 18], [20, 18]], np.float32)
                pts = pts * 1.4 + pos
                cv2.fillPoly(cm, [pts.astype(np.int32)], 1.0, lineType=cv2.LINE_AA)
                blit(c, cm, 0, 0, CREAM, "add", 1.0)
            rp = seg(t, 0.88, 1.3)
            if 0 < rp < 1:
                rm = np.zeros((H, W), np.float32)
                cv2.circle(rm, (int(cur[0]), int(cur[1])), int(10 + rp * 130), 1.0, 5, lineType=cv2.LINE_AA)
                blit(c, rm, 0, 0, LIME, "add", (1 - rp) * 1.2)
    # type
    hp = e_out_expo(seg(t, 0.15, 0.6))
    if hp > 0:
        m = TY.string_mask("anton", 120, "PRODUCT MOTION")[0]
        cut = int(m.shape[0] * hp)
        clip = np.zeros_like(m); clip[:cut] = m[:cut]
        blit(c, clip, W * 0.06, H * 0.10, CREAM, "add", 1.0)
    TY.draw_string(c, "mono", 24, "MICRO-INTERACTIONS / UI SYSTEMS / PROTOTYPING",
                   W * 0.94, H * 0.90, STEEL, align="right", mode="add",
                   opacity=0.85 * seg(t, 0.4, 0.8), tracking=0.14)
    ex = seg(t, 2.34, 2.5)
    if ex > 0:
        c = slice_glitch(c, t, e_in_cubic(ex) * 1.4, seed=3)
    return c

# ================================================================ SCENE 6
def s6(lt):
    t = lt
    shot_len = 2.0 / 12.0
    i = min(11, int(t / shot_len))
    st = t - i * shot_len
    c = bg_base(INK, 0.02)
    k = st / shot_len

    if i == 0:
        c[:] = MAG
        TY.draw_string(c, "ablack", 460, "01", CX, CY, CREAM, mode="norm", scale=1 + k * 0.25)
    elif i == 1:
        TY.draw_glyphs(c, "archivo", 240, "MOTION", CX, CY, CREAM, axes=(900, 112),
                       glyph_fn=lambda i2, n, ch: dict(dy=math.sin(i2 * 2 + t * 30) * 40))
        c = rgb_shift(c, 22, 0)
    elif i == 2:
        v = Vec(ss=2)
        for r0 in range(6):
            rr = 90 + r0 * 130 - k * 200
            v.circle((CX, CY), max(rr, 4), 3.0, 230)
        blit(c, v.result(), 0, 0, CYAN, "add", 1.0)
    elif i == 3:
        c[:] = INK
        v = Vec(ss=2)
        for s0 in range(0, H, 26):
            v.line((0, s0 + k * 26), (W, s0 + k * 26 - 90), 6, 60)
        blit(c, v.result(), 0, 0, LIME, "add", 0.5)
        TY.draw_string(c, "ablack", 460, "02", CX, CY, LIME, mode="add")
    elif i == 4:
        pm = np.zeros((H, W), np.float32)
        rng = np.random.default_rng(i)
        for _ in range(900):
            a = rng.uniform(0, 6.28); rr = rng.uniform(0, 1) ** 0.5 * 500
            cv2.circle(pm, (int(CX + math.cos(a) * rr), int(CY + math.sin(a) * rr * 0.8)),
                       int(rng.uniform(1, 4)), float(rng.uniform(0.3, 1)), -1)
        blit(c, pm, 0, 0, CREAM, "add", 1.0)
    elif i == 5:
        v = Vec(ss=2)
        off = -k * 900
        for r0 in range(6):
            v.text((off + r0 * 640, CY - 60 + (r0 % 2) * 130), "MOTION·DESIGN·MOTION·DESIGN·",
                   "anton", 96, fill=230)
        blit(c, v.result(), 0, 0, CREAM, "add", 0.9)
        c = rgb_shift(c, -16, 0)
    elif i == 6:
        c[:] = ORANGE
        TY.draw_string(c, "ablack", 460, "03", CX, CY, INK, mode="norm", rot=-0.06)
    elif i == 7:
        R = rot3(ax=t * 6, ay=t * 9)
        P, f, z = project(CUBE_P * 1.5, R, fov=1250, dist=3.0)
        v = Vec(ss=2)
        for a, b in CUBE_E: v.line(tuple(P[a]), tuple(P[b]), 3.0, 240)
        blit(c, v.result(), 0, 0, LIME, "add", 1.0)
    elif i == 8:
        hm = np.zeros((H, W), np.float32)
        yy, xx = np.mgrid[0:H:10, 0:W:10].astype(np.float32)
        d = np.sqrt((xx - CX) ** 2 + (yy - CY) ** 2)
        lvl = np.clip(1.25 - d / 620 + math.sin(t * 20) * 0.1, 0, 1)
        rr = (lvl * 4.6).astype(np.int32)
        for (yi, xi, r) in zip(yy.ravel().astype(int), xx.ravel().astype(int), rr.ravel()):
            if r > 0: cv2.circle(hm, (xi, yi), r, 0.8, -1)
        blit(c, hm, 0, 0, MAG, "add", 1.0)
    elif i == 9:
        v = Vec(ss=2)
        spikes = 14
        for s0 in range(spikes):
            a = s0 / spikes * 6.28 + t * 5
            v.line((CX, CY), (CX + math.cos(a) * 760, CY + math.sin(a) * 760), 5, 240)
        blit(c, v.result(), 0, 0, CREAM, "add", 1.0)
    elif i == 10:
        w_ = "HELLO"
        def gf(i2, n, ch):
            ch2 = ch if seg(st, 0.02 + i2 * 0.02, 0.05 + i2 * 0.02) >= 1 else GLYPHSET[(i2 * 11 + int(t * 30)) % len(GLYPHSET)]
            return dict(op=1.0, ch=ch2, rot=math.sin(i2 * 5 + t * 25) * 0.1)
        glyphs, tw = TY.layout("anton", 260, w_)
        ax = CX - tw / 2
        tops = [g["tb"] for g in glyphs]; bots = [g["tb"] + g["h"] for g in glyphs]
        base = CY - (min(tops) + max(bots)) / 2
        for ii, g in enumerate(glyphs):
            p = gf(ii, len(glyphs), g["ch"]); ch = p.pop("ch")
            m, adv, lb, tb, gw, gh = TY.glyph("anton", 260, ch)
            warp_center(c, m, ax + g["pen"] + lb + gw / 2, base + tb + gh / 2,
                        CREAM, "add", 1.0, p["rot"], opacity=1.0)
    else:
        cols = [MAG, LIME, CYAN, CREAM, ORANGE, VIOLET]
        bw = W / 6
        for j, col in enumerate(cols):
            blit(c, np.ones((H, int(bw) + 2), np.float32), int(j * bw), 0, col, "norm", 1.0)
        c = rgb_shift(c, 10 * math.sin(t * 60), 0)

    # global montage treatment
    zoom = 1.0 + t * 0.06
    c = shake_frame(c, math.sin(t * 130) * 6, math.cos(t * 97) * 5, scale=zoom)
    c = slice_glitch(c, t, 1.0 + t * 0.7, seed=11)
    if i % 2 == 1: c = scanlines(c, 0.16, 3.0, t * 40)
    fl = max(0.0, 1 - (st / shot_len) * 5) if i % 2 == 0 else 0.0
    if fl > 0: c += fl * 0.5
    # hard black at the very end
    if t > 1.86:
        c *= max(0.0, 1 - (t - 1.86) / 0.10)
    return c

# ================================================================ SCENE 7
def s7(lt):
    t = lt
    c = bg_base(INK, 0.02)
    # monogram: open arc + slash + orbiting dot, stroke-on
    a0 = -math.pi / 2
    p1 = e_out_expo(seg(t, 0.06, 0.55))
    p2 = e_out_expo(seg(t, 0.20, 0.62))
    v = Vec(ss=2)
    if p1 > 0:
        v.circle((CX, CY - 70), 118, 8.0, 255, a0=a0, a1=a0 + 2 * math.pi * p1 * 0.78)
    if p2 > 0:
        v.line((CX - 74, CY + 42), (CX - 74 + 148 * p2, CY + 42 - 158 * p2), 8.0, 255)
    blit(c, v.result(), 0, 0, CREAM, "add", 1.0)
    p3 = e_out_back(seg(t, 0.34, 0.70))
    if p3 > 0:
        dm = np.zeros((H, W), np.float32)
        cv2.circle(dm, (int(CX + 60), int(CY - 130)), max(1, int(15 * p3)), 1.0, -1, lineType=cv2.LINE_AA)
        blit(c, dm, 0, 0, LIME, "add", 1.2)
        blit(c, bloom_mask(dm, 2.4, 0.6), 0, 0, LIME, "add", 0.9)

    # name + role
    np_ = seg(t, 0.42, 0.92)
    if np_ > 0:
        m, w, h = TY.string_mask("archivo", 96, NAME, axes=(760, 104))
        cut = int(m.shape[0] * (1 - e_inout_expo(np_)))
        clip = m.copy(); clip[:cut] = 0
        tr = lerp(0.30, 0.06, e_out_expo(np_))
        # tracking via glyph draw instead: approximate by scale
        blit(c, clip, CX - w / 2, CY + 96, CREAM, "add", 1.0)
    rp = seg(t, 0.72, 1.1)
    if rp > 0:
        blink = 1.0 if (t * 2) % 1 < 0.6 else 0.25
        TY.draw_string(c, "mono", 26, f"{ROLE}   ·   AVAILABLE FOR WORK", CX, CY + 196,
                       STEEL, tracking=lerp(0.4, 0.16, e_out_expo(rp)), mode="add", opacity=e_out_cubic(rp) * 0.95)
        tr_ = lerp(0.4, 0.16, e_out_expo(rp))
        tw_ = TY.layout("mono", 26, f"{ROLE}   ·   AVAILABLE FOR WORK", None, tr_)[1]
        dm = np.zeros((H, W), np.float32)
        cv2.circle(dm, (int(CX - tw_ / 2 - 30), int(CY + 210)), 7, 1.0, -1, lineType=cv2.LINE_AA)
        blit(c, dm, 0, 0, LIME, "add", blink)

    # shine sweep
    sh = seg(t, 0.95, 1.30)
    if 0 < sh < 1:
        band = np.zeros((H, W), np.float32)
        x0 = -400 + (W + 800) * e_inout_cubic(sh)
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        d = (xx + yy * 0.35) - x0
        band = np.clip(1 - np.abs(d) / 130.0, 0, 1) ** 1.6
        lum = c.max(axis=2)
        blit(c, band * np.clip(lum, 0, 1), 0, 0, CREAM, "add", 0.55)

    # close down
    fade = 1.0 - e_in_cubic(seg(t, 1.30, 1.50))
    c *= fade
    vg = lerp(0.42, 0.72, seg(t, 0.8, 1.5))
    c = vignette(c, vg)
    return c

SCENES = [s1, s2, s3, s4, s5, s6, s7]
