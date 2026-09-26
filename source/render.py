"""render.py -- timeline composer + HUD + global grade + ffmpeg pipe."""
import os, sys, time, math, subprocess, numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from motion.engine import *
from motion import scenes as S

S.precompute()

def _hud_mask_alpha(t):
    a = e_out_cubic(seg(t, 0.45, 1.0)) * (1 - e_in_cubic(seg(t, 14.55, 14.95)))
    if 11.5 <= t < 13.5: a *= 0.0
    return a

def hud(t, c):
    a = _hud_mask_alpha(t)
    if a <= 0.002: return c
    v = Vec(ss=2)
    # bottom progress rail
    ty = H - 26
    v.rect(0, ty, W, 2.0, 255)
    rail = v.result()
    blit(c, rail, 0, 0, STEEL, "add", 0.22 * a)
    v = Vec(ss=2)
    fw = W * (t / DUR)
    v.rect(0, ty, fw, 3.0, 255)
    # scene ticks
    for ct in S.CUTS[1:-1]:
        v.rect(ct / DUR * W - 1, ty - 5, 2, 13, 255)
    m = v.result()
    blit(c, m, 0, 0, LIME, "add", 0.85 * a)
    # playhead dot
    pm = np.zeros((H, W), np.float32)
    cv2.circle(pm, (int(fw), ty + 1), 5, 1.0, -1, lineType=cv2.LINE_AA)
    blit(c, pm, 0, 0, CREAM, "add", a)
    # timecode
    ff = int((t % 1.0) * FPS)
    ss = int(t)
    TY.draw_string(c, "dmmono", 21, f"00:00:{ss:02d}:{ff:02d}", 40, H - 52, STEEL,
                   align="left", mode="add", opacity=0.75 * a, tracking=0.06)
    TY.draw_string(c, "dmmono", 21, "1920×1080  ·  60 FPS  ·  H.264", W - 40, H - 52, STEEL,
                   align="right", mode="add", opacity=0.75 * a, tracking=0.06)
    # top-left mark
    v = Vec(ss=2)
    v.rect(40, 40, 26, 26, outline=255, width=2.0)
    v.circle((53, 53), 5, 0, outline=False)
    m = v.result()
    blit(c, m, 0, 0, LIME, "add", 0.8 * a)
    TY.draw_string(c, "dmmono", 20, "REEL ’26", 78, 42, STEEL, align="left",
                   mode="add", opacity=0.7 * a, tracking=0.14)
    TY.draw_string(c, "dmmono", 20, S.NAME.upper(), W - 40, 42, STEEL, align="right",
                   mode="add", opacity=0.6 * a, tracking=0.14)
    return c

def cut_fx(t, c):
    """entry treatment at each scene boundary"""
    for k, ct in enumerate(S.CUTS[1:-1], start=1):
        d = t - ct
        if -0.06 < d < 0.30:
            if k == 1:   # 2.0 -> systems: bar handled in-scene
                pass
            elif k == 2:  # 4.0 -> type
                f = max(0.0, 1 - abs(d) * 30)
                if f > 0: c = c + f * 0.55 * CREAM
                if 0 <= d < 0.12: c = rgb_shift(c, 14 * (1 - d / 0.12), 0)
            elif k == 3:  # 6.5 -> flow
                if 0 <= d < 0.22:
                    c = zoom_blur(c, (1 - d / 0.22) * 0.20)
            elif k == 4:  # 9.0 -> interface
                if 0 <= d < 0.16:
                    c = slice_glitch(c, t, (1 - d / 0.16) * 2.2, seed=5)
            elif k == 5:  # 11.5 -> montage
                f = max(0.0, 1 - abs(d) * 26)
                if f > 0: c = c + f * 0.8 * CREAM
            elif k == 6:  # 13.5 -> outro
                f = max(0.0, 1 - abs(d) * 20)
                if f > 0: c = c + f * 0.25 * CREAM
    return c

def ca_amount(t):
    a = 0.0016
    for ct in S.CUTS[1:]:
        d = abs(t - ct)
        if d < 0.45: a += 0.010 * (1 - d / 0.45) ** 2
    if 0.9 < t < 1.5: a += 0.006 * (1 - abs(t - 1.0) / 0.5)
    return a

VERTICAL = False

def composite(t, with_hud=True):
    i = S.SCENE_OF(t)
    lt = t - S.CUTS[i]
    c = S.SCENES[i](lt)
    c = cut_fx(t, c)
    if with_hud:
        c = hud(t, c)
    c = chromatic(c, ca_amount(t))
    c = bloom(c, 0.78, 0.42, 0.40)
    c = vignette(c, 0.40)
    c = grain(c, t, 0.030)
    c = tonemap(c, 1.06, 0.95)
    return c

def frame(t):
    """final canvas for the current output format"""
    if VERTICAL:
        from motion import vertical as V
        i = S.SCENE_OF(t)
        lt = t - S.CUTS[i]
        stage = composite(t, with_hud=False)
        return V.build(stage, t, i, lt)
    return composite(t)

# ------------------------------------------------------------------ driver
def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=str, default="")     # comma list of times -> PNG stills
    ap.add_argument("--out", type=str, default="/home/user/showreel/out/showreel_silent.mp4")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=NF)
    ap.add_argument("--audio", type=str, default="")
    ap.add_argument("--final", type=str, default="")
    ap.add_argument("--vertical", action="store_true")
    args = ap.parse_args()
    global VERTICAL, W, H
    if args.vertical:
        VERTICAL = True
        from motion import vertical as V
        W, H = V.VW, V.VH

    if args.frames:
        for ts in args.frames.split(","):
            t = float(ts)
            t0 = time.time()
            c = frame(t)
            u8 = to_uint8(c)
            p = f"/home/user/showreel/prev/f_{t:05.2f}.png"
            cv2.imwrite(p, u8[..., ::-1])
            print(f"{t:6.2f}s  {time.time()-t0:5.2f}s  -> {p}")
        return

    ff = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bin", "ffmpeg")
    if not os.path.exists(ff):
        import imageio_ffmpeg
        ff = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [ff, "-y", "-loglevel", "error",
           "-f", "rawvideo", "-vcodec", "rawvideo", "-s", f"{W}x{H}", "-pix_fmt", "rgb24",
           "-r", str(FPS), "-i", "-"]
    if args.audio:
        cmd += ["-i", args.audio]
    cmd += ["-c:v", "libx264", "-preset", "medium", "-crf", "17", "-pix_fmt", "yuv420p",
            "-profile:v", "high", "-level", "4.2"]
    if args.audio:
        cmd += ["-c:a", "aac", "-b:a", "224k", "-ar", "48000", "-shortest"]
    cmd += ["-movflags", "+faststart", args.final or args.out]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

    t0 = time.time()
    for f in range(args.start, args.end):
        t = f / FPS
        c = frame(t)
        proc.stdin.write(to_uint8(c).tobytes())
        if f % 60 == 0:
            el = time.time() - t0
            eta = el / max(f - args.start, 1) * (args.end - f)
            print(f"frame {f:4d}/{args.end}  t={t:5.2f}  {el:6.1f}s elapsed  eta {eta:6.1f}s", flush=True)
    proc.stdin.close()
    err = proc.stderr.read().decode()[-2000:]
    proc.wait()
    print("ffmpeg done rc=", proc.returncode)
    if err.strip(): print(err)
    print("total %.1fs" % (time.time() - t0))

if __name__ == "__main__":
    main()
