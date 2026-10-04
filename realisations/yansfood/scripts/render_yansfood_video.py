#!/usr/bin/env python3
"""
render_yansfood_video.py -- 1440x1440 @ 60 FPS motion graphics renderer for YansFood.
Outputs:
  1. realisations/yansfood/videos/yansfood_master_1440x1440_60fps.mp4
  2. realisations/yansfood/videos/yansfood_social_1080x1080_30fps.mp4
  3. realisations/yansfood/videos/yansfood_vertical_1080x1920.mp4
  4. realisations/yansfood/docs/preview.gif
  5. realisations/yansfood/posters/poster_master.png
"""

import os, sys, time, math, subprocess
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

W, H = 1440, 1440
FPS = 60
DUR = 15.00
TOTAL_FRAMES = int(FPS * DUR) # 900 frames

# Brand Colors (RGB)
CREAM = (245, 242, 237)
DARK_UI = (10, 10, 10)
ORANGE = (255, 107, 53)
WHITE = (255, 255, 255)
MUTED = (142, 149, 165)

# Load fonts
FONT_BOLD_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT_MONO_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"

def get_font(path, size):
    try:
        return ImageFont.truetype(path, size)
    except:
        return ImageFont.load_default()

font_wordmark = get_font(FONT_BOLD_PATH, 160)
font_large = get_font(FONT_BOLD_PATH, 54)
font_medium = get_font(FONT_BOLD_PATH, 36)
font_mono = get_font(FONT_MONO_PATH, 28)
font_mono_small = get_font(FONT_MONO_PATH, 22)
font_tag = get_font(FONT_MONO_PATH, 18)

# Load Images
def load_img(path, target_size=(1440, 1440)):
    if os.path.exists(path):
        im = Image.open(path).convert('RGB')
        return im.resize(target_size, Image.Resampling.LANCZOS)
    return None

img_poulet = load_img("realisations/yansfood/assets/images/food_poulet_dg.jpg", (1440, 1440))
img_jollof = load_img("realisations/yansfood/assets/images/food_jollof.jpg", (1440, 1440))
img_wall = load_img("realisations/yansfood/assets/images/wall_plant_shadows.jpg", (1440, 1440))
still_glass = load_img("realisations/yansfood/stills/still_glass.png", (1440, 1440))
still_stage = load_img("realisations/yansfood/stills/still_stage.png", (1440, 1440))

# Analytical Closed-form Spring Solver
def spring(t, t0, omega=18.0, zeta=0.55, target=1.0, initial=0.0):
    if t < t0:
        return initial
    tau = t - t0
    delta = target - initial
    omega_d = omega * math.sqrt(max(0.0001, 1.0 - zeta * zeta))
    env = math.exp(-zeta * omega * tau)
    return target - env * (delta * math.cos(omega_d * tau) + ((zeta * omega * delta) / omega_d) * math.sin(omega_d * tau))

def clamp(v, vmin=0.0, vmax=1.0):
    return max(vmin, min(vmax, v))

def lerp(a, b, t):
    return a + (b - a) * t

def draw_iris(draw, cx, cy, radius, progress):
    """Draws 6-blade camera iris aperture."""
    if progress >= 0.999:
        return
    if progress <= 0.001:
        draw.rectangle([0, 0, W, H], fill=DARK_UI)
        return
    
    num_blades = 6
    r_blade = radius * (1.0 - progress * 0.96)
    rot = progress * 1.2
    
    for i in range(num_blades):
        a1 = (i / num_blades) * 2 * math.pi + rot
        a2 = ((i + 1) / num_blades) * 2 * math.pi + rot
        
        x1 = cx + math.cos(a1) * r_blade
        y1 = cy + math.sin(a1) * r_blade
        x2 = cx + math.cos(a2) * r_blade
        y2 = cy + math.sin(a2) * r_blade
        
        ext_x1 = cx + math.cos(a1 + 0.6) * 2000
        ext_y1 = cy + math.sin(a1 + 0.6) * 2000
        ext_x2 = cx + math.cos(a2 + 0.6) * 2000
        ext_y2 = cy + math.sin(a2 + 0.6) * 2000
        
        poly = [(x1, y1), (x2, y2), (ext_x2, ext_y2), (ext_x1, ext_y1)]
        draw.polygon(poly, fill=DARK_UI, outline=(30, 35, 45))

def render_frame(t):
    """Renders single frame at time t (0.00 to 15.00s) as RGB numpy array."""
    im = Image.new('RGB', (W, H), CREAM)
    draw = ImageDraw.Draw(im, 'RGBA')
    cx, cy = W // 2, H // 2

    # ----------------------------------------------------
    # SCENE 01: OPEN WORDMARK (0.00s - 1.50s)
    # ----------------------------------------------------
    if t < 1.50:
        if t < 0.50:
            # Wordmark full screen
            draw.text((cx - 30, cy), "YANS", font=font_wordmark, fill=DARK_UI, anchor="rm")
            draw.text((cx + 30, cy), "FOOD", font=font_wordmark, fill=ORANGE, anchor="lm")
        elif t < 1.05:
            # Squeeze accordion towards center
            sq = clamp((t - 0.50) / 0.50)
            sq_e = (1 - math.cos(sq * math.pi)) / 2 # easeInOut
            shift = lerp(0, 340, sq_e)
            scale = lerp(1.0, 0.4, sq_e)
            f_size = max(20, int(160 * scale))
            f_curr = get_font(FONT_BOLD_PATH, f_size)
            draw.text((cx - 30 + shift, cy), "YANS", font=f_curr, fill=DARK_UI, anchor="rm")
            draw.text((cx + 30 - shift, cy), "FOOD", font=f_curr, fill=ORANGE, anchor="lm")
        else:
            # Squeezed two-tone meal delivery pill
            pill_w = spring(t, 1.05, 20, 0.6, 320, 20)
            pill_h = spring(t, 1.05, 20, 0.6, 90, 20)
            # Left black pill
            draw.rounded_rectangle([cx - pill_w/2, cy - pill_h/2, cx, cy + pill_h/2], radius=45, fill=DARK_UI)
            # Right orange pill
            draw.rounded_rectangle([cx, cy - pill_h/2, cx + pill_w/2, cy + pill_h/2], radius=45, fill=ORANGE)
            draw.text((cx, cy), "YANSFOOD · 237", font=font_mono, fill=WHITE, anchor="mm")
            
        # Iris closure at 1.20s - 1.50s
        if t >= 1.20:
            iris_p = clamp(1.0 - (t - 1.20) / 0.30)
            draw_iris(draw, cx, cy, 900, iris_p)

    # ----------------------------------------------------
    # SCENE 02: BENTO DISCOVERY & ZOOM DROP (1.50s - 4.00s)
    # ----------------------------------------------------
    elif t < 4.00:
        iris_p = spring(t, 1.50, 22, 0.6, 1.0, 0.0)
        
        if t < 2.50:
            # Poulet DG reveal center
            scale = spring(t, 1.60, 16, 0.55, 1.0, 0.2)
            d_size = int(880 * scale)
            if img_poulet:
                sub = img_poulet.resize((d_size, d_size), Image.Resampling.LANCZOS)
                # Circular mask
                mask = Image.new('L', (d_size, d_size), 0)
                ImageDraw.Draw(mask).ellipse([0, 0, d_size, d_size], fill=255)
                im.paste(sub, (cx - d_size//2, cy - d_size//2), mask)
            # Dish tags
            draw.text((cx, cy + 500), "POULET DG · 3 500 FCFA", font=font_large, fill=DARK_UI, anchor="mm")
            draw.text((cx, cy + 560), "DOUALA · SPECIALITÉ MAISON", font=font_mono, fill=ORANGE, anchor="mm")
        else:
            # Bento Grid unroll and zoom at 3.50s
            zoom = spring(t, 3.50, 22, 0.5, 3.8, 1.0) if t >= 3.50 else 1.0
            
            # Base bento canvas
            bento_im = Image.new('RGB', (W, H), CREAM)
            b_draw = ImageDraw.Draw(bento_im)
            
            # 3x3 tiles
            dishes = [
                ("Ndolè Royal", "4 000 F"), ("Poisson Braisé", "4 500 F"), ("Koki Douala", "2 000 F"),
                ("Brochettes Bœuf", "1 500 F"), ("Poulet DG", "3 500 F"), ("Riz Jollof", "2 500 F"),
                ("Alloco Spécial", "1 000 F"), ("Taro Sauce Jaune", "3 500 F"), ("Bissap Glacé", "1 000 F")
            ]
            for idx, (name, price) in enumerate(dishes):
                row, col = idx // 3, idx % 3
                px = 240 + col * 340
                py = 240 + row * 340
                is_center = (idx == 4)
                
                b_draw.rounded_rectangle([px, py, px + 300, py + 300], radius=24,
                                         fill=(255, 248, 242) if is_center else (238, 234, 226),
                                         outline=ORANGE if is_center else (210, 205, 195),
                                         width=5 if is_center else 2)
                if is_center and img_poulet:
                    sub_thumb = img_poulet.resize((240, 180), Image.Resampling.LANCZOS)
                    bento_im.paste(sub_thumb, (px + 30, py + 25))
                b_draw.text((px + 150, py + 235), name, font=font_mono, fill=ORANGE if is_center else DARK_UI, anchor="mm")
                b_draw.text((px + 150, py + 270), price, font=font_mono_small, fill=(100, 100, 100), anchor="mm")
            
            # Zoom crop
            if zoom > 1.01:
                zw, zh = int(W / zoom), int(H / zoom)
                zx1, zy1 = cx - zw // 2, cy - zh // 2
                cropped = bento_im.crop((zx1, zy1, zx1 + zw, zy1 + zh)).resize((W, H), Image.Resampling.BILINEAR)
                im.paste(cropped, (0, 0))
            else:
                im.paste(bento_im, (0, 0))

        if iris_p < 0.999:
            draw_iris(draw, cx, cy, 900, iris_p)

    # ----------------------------------------------------
    # SCENE 03: LIQUID GLASS & RELIGHT (4.00s - 6.50s)
    # ----------------------------------------------------
    elif t < 6.50:
        if still_glass:
            im.paste(still_glass, (0, 0))
        else:
            if img_poulet: im.paste(img_poulet, (0, 0))
        # Overlay dynamic UI slider movement
        slide = spring(t, 5.50, 12, 0.6, 1.0, 0.0)
        draw.rounded_rectangle([cx - 400, cy + 320, cx + 400, cy + 460], radius=70, fill=(255, 255, 255, 140), outline=ORANGE, width=3)
        draw.text((cx - 320, cy + 370), "RELIGHT · GOLDEN HOUR", font=font_mono, fill=DARK_UI, anchor="lm")
        draw.text((cx + 320, cy + 370), f"{int(slide*100)}%", font=font_mono, fill=ORANGE, anchor="rm")
        # Rail
        draw.rounded_rectangle([cx - 320, cy + 410, cx + 320, cy + 422], radius=6, fill=(50, 50, 50, 80))
        draw.rounded_rectangle([cx - 320, cy + 410, cx - 320 + int(640 * slide), cy + 422], radius=6, fill=ORANGE)
        draw.ellipse([cx - 320 + int(640 * slide) - 18, cy + 416 - 18, cx - 320 + int(640 * slide) + 18, cy + 416 + 18], fill=WHITE, outline=ORANGE, width=4)

    # ----------------------------------------------------
    # SCENE 04: GLASS ORB & LOCKSCREEN (6.50s - 9.00s)
    # ----------------------------------------------------
    elif t < 9.00:
        if t < 8.00:
            # Dish 1 background softly dimmed
            if img_poulet:
                dark_p = Image.blend(img_poulet, Image.new('RGB', (W, H), (20, 20, 20)), 0.4)
                im.paste(dark_p, (0, 0))
            # Floating 3D Orb with Jollof Rice
            orb_s = spring(t, 6.70, 18, 0.55, 1.0, 0.1)
            orb_r = int(380 * orb_s)
            if img_jollof:
                sub_j = img_jollof.resize((orb_r * 2, orb_r * 2), Image.Resampling.LANCZOS)
                mask_j = Image.new('L', (orb_r * 2, orb_r * 2), 0)
                ImageDraw.Draw(mask_j).ellipse([0, 0, orb_r * 2, orb_r * 2], fill=255)
                im.paste(sub_j, (cx - orb_r, cy - orb_r), mask_j)
            draw.ellipse([cx - orb_r, cy - orb_r, cx + orb_r, cy + orb_r], outline=ORANGE, width=6)
            draw.text((cx, cy + orb_r + 60), "RIZ JOLLOF ROYAL · 2 500 FCFA", font=font_large, fill=WHITE, anchor="mm")
        else:
            # iOS Lockscreen UI
            im.paste(Image.new('RGB', (W, H), (15, 18, 24)), (0, 0))
            if img_jollof:
                sub_j = img_jollof.resize((1200, 900), Image.Resampling.LANCZOS)
                im.paste(Image.blend(sub_j, Image.new('RGB', (1200, 900), (0,0,0)), 0.35), (120, 240))
            draw.text((cx, 340), "09:41", font=get_font(FONT_BOLD_PATH, 160), fill=WHITE, anchor="mm")
            draw.text((cx, 440), "SAMEDI · TON REPAS EN COURS", font=font_mono, fill=ORANGE, anchor="mm")
            # Bottom widget
            draw.rounded_rectangle([cx - 420, cy + 420, cx + 420, cy + 540], radius=60, fill=(255, 255, 255, 180), outline=ORANGE, width=3)
            draw.text((cx, cy + 480), "YansFood ✦ Poulet DG + Jollof", font=font_mono, fill=DARK_UI, anchor="mm")

    # ----------------------------------------------------
    # SCENE 05: STAGE KEYNOTE & MAC BLINDS (9.00s - 11.50s)
    # ----------------------------------------------------
    elif t < 11.50:
        if still_stage:
            im.paste(still_stage, (0, 0))
        else:
            im.paste(Image.new('RGB', (W, H), CREAM), (0, 0))
            # Mac Window
            draw.rounded_rectangle([cx - 500, cy - 400, cx + 500, cy + 400], radius=24, fill=WHITE, outline=DARK_UI, width=4)
            draw.text((cx, cy), "YansFood Keynote Presentation", font=font_large, fill=DARK_UI, anchor="mm")

    # ----------------------------------------------------
    # SCENE 06: BREAKDOWN & COMMANDER BUTTON (11.50s - 12.75s)
    # ----------------------------------------------------
    elif t < 12.75:
        # Gallery card
        im.paste(Image.new('RGB', (W, H), CREAM), (0, 0))
        card_w, card_h = 920, 920
        draw.rounded_rectangle([cx - card_w//2, cy - card_h//2, cx + card_w//2, cy + card_h//2], radius=28, fill=WHITE, outline=DARK_UI, width=3)
        # Inner photo
        if img_poulet:
            p_sub = img_poulet.resize((card_w - 60, card_h - 260), Image.Resampling.LANCZOS)
            im.paste(p_sub, (cx - card_w//2 + 30, cy - card_h//2 + 30))
        # Orange lacquer sweep
        sw = clamp((t - 12.00) / 0.35)
        if sw > 0.01:
            draw.rectangle([cx - card_w//2 + 20, cy - card_h//2 + 20, cx - card_w//2 + int((card_w - 40)*sw), cy + card_h//2 - 20], outline=ORANGE, width=5)
        # Commander CTA
        btn_y = cy + card_h//2 - 120
        draw.rounded_rectangle([cx - 320, btn_y, cx + 320, btn_y + 90], radius=45, fill=ORANGE)
        draw.text((cx, btn_y + 45), "COMMANDER · 3 500 FCFA", font=font_large, fill=WHITE, anchor="mm")

    # ----------------------------------------------------
    # SCENE 07: BLACK FLOOD & STATUS CONFIRMATION (12.75s - 14.25s)
    # ----------------------------------------------------
    elif t < 14.25:
        # Black flood expanding
        flood_r = spring(t, 12.75, 14, 0.7, 1800, 50)
        draw.ellipse([cx - flood_r, cy - flood_r, cx + flood_r, cy + flood_r], fill=DARK_UI)
        
        if t < 13.25:
            # Commande confirmée
            draw.rounded_rectangle([cx - 400, cy - 60, cx + 400, cy + 60], radius=60, fill=WHITE)
            draw.text((cx, cy), "COMMANDE CONFIRMÉE ✓", font=font_large, fill=DARK_UI, anchor="mm")
        elif t < 13.50:
            # En préparation
            prep = clamp((t - 13.25) / 0.25)
            draw.text((cx, cy - 50), "EN PRÉPARATION...", font=font_large, fill=WHITE, anchor="mm")
            draw.rounded_rectangle([cx - 360, cy + 20, cx + 360, cy + 60], radius=20, fill=(40, 40, 40))
            draw.rounded_rectangle([cx - 360, cy + 20, cx - 360 + int(720 * prep), cy + 60], radius=20, fill=ORANGE)
        elif t < 13.75:
            # En route
            van_p = clamp((t - 13.50) / 0.25)
            draw.text((cx, cy - 100), "EN ROUTE / LIVRAISON RAPIDE", font=font_large, fill=WHITE, anchor="mm")
            draw.line([cx - 400, cy + 40, cx + 400, cy + 40], fill=(80, 80, 80), width=8)
            vx = lerp(cx - 360, cx + 360, van_p)
            draw.rounded_rectangle([vx - 60, cy + 10, vx + 60, cy + 70], radius=16, fill=ORANGE)
            draw.text((vx, cy + 40), "YANS", font=font_mono_small, fill=WHITE, anchor="mm")
        else:
            # Commande prête
            chk_s = spring(t, 13.75, 20, 0.55, 1.0, 0.0)
            chk_r = int(150 * chk_s)
            draw.ellipse([cx - chk_r, cy - chk_r, cx + chk_r, cy + chk_r], fill=ORANGE)
            draw.text((cx, cy), "✓", font=get_font(FONT_BOLD_PATH, int(160 * chk_s)), fill=WHITE, anchor="mm")
            draw.text((cx, cy + 220), "COMMANDE PRÊTE !", font=font_large, fill=WHITE, anchor="mm")

    # ----------------------------------------------------
    # SCENE 08: WALL & FINAL WORDMARK SLAM (14.25s - 15.00s)
    # ----------------------------------------------------
    else:
        if t < 14.75:
            if img_wall:
                im.paste(img_wall, (0, 0))
            else:
                im.paste(Image.new('RGB', (W, H), (235, 232, 224)), (0, 0))
            # Framed art print with 2px orange border
            pw, ph = 640, 640
            draw.rectangle([cx - pw//2, cy - ph//2, cx + pw//2, cy + ph//2], fill=WHITE, outline=DARK_UI, width=4)
            draw.rectangle([cx - pw//2 + 10, cy - ph//2 + 10, cx + pw//2 - 10, cy + ph//2 - 10], outline=ORANGE, width=4)
            if img_poulet:
                sub_w = img_poulet.resize((pw - 30, ph - 30), Image.Resampling.LANCZOS)
                im.paste(sub_w, (cx - pw//2 + 15, cy - ph//2 + 15))
            # Final Iris Close
            if t >= 14.50:
                iris_p = clamp((t - 14.50) / 0.25)
                draw_iris(draw, cx, cy, 900, 1.0 - iris_p)
        else:
            # 14.75s - 15.00s : FINAL SLAM & VALUE PROPOSITION
            im.paste(Image.new('RGB', (W, H), CREAM), (0, 0))
            
            # Springs for slam
            slam_yans = spring(t, 14.75, 22, 0.55, 0, -800)
            slam_food = spring(t, 14.75, 22, 0.55, 0, 800)
            
            draw.text((cx - 30 + slam_yans, cy - 90), "YANS", font=font_wordmark, fill=DARK_UI, anchor="rm")
            draw.text((cx + 30 + slam_food, cy - 90), "FOOD", font=font_wordmark, fill=ORANGE, anchor="lm")
            
            # Approved Value Proposition & Cities
            draw.text((cx, cy + 90), "TON REPAS, SANS LA FILE.", font=font_large, fill=DARK_UI, anchor="mm")
            draw.text((cx, cy + 160), "DOUALA · YAOUNDÉ · CAMEROUN", font=font_mono, fill=(100, 100, 100), anchor="mm")
            
            # App Store & Google Play Badges
            bw, bh = 240, 70
            # App Store
            draw.rounded_rectangle([cx - bw - 20, cy + 220, cx - 20, cy + 220 + bh], radius=16, fill=DARK_UI)
            draw.text((cx - bw//2 - 20, cy + 242), "Télécharger sur", font=font_tag, fill=(180, 180, 180), anchor="mm")
            draw.text((cx - bw//2 - 20, cy + 266), "App Store", font=font_mono, fill=WHITE, anchor="mm")
            # Google Play
            draw.rounded_rectangle([cx + 20, cy + 220, cx + bw + 20, cy + 220 + bh], radius=16, fill=DARK_UI)
            draw.text((cx + bw//2 + 20, cy + 242), "Disponible sur", font=font_tag, fill=(180, 180, 180), anchor="mm")
            draw.text((cx + bw//2 + 20, cy + 266), "Google Play", font=font_mono, fill=WHITE, anchor="mm")

    # Convert PIL Image to BGR for OpenCV / FFmpeg
    rgb_arr = np.array(im)
    bgr_arr = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR)
    return bgr_arr

def main():
    print(f"🎬 Starting YansFood Video Master Render: {W}x{H} @ {FPS} fps ({TOTAL_FRAMES} frames)...")
    
    audio_path = "realisations/yansfood/audio/yansfood_score.wav"
    master_path = "realisations/yansfood/videos/yansfood_master_1440x1440_60fps.mp4"
    social_path = "realisations/yansfood/videos/yansfood_social_1080x1080_30fps.mp4"
    vertical_path = "realisations/yansfood/videos/yansfood_vertical_1080x1920.mp4"
    gif_path = "realisations/yansfood/docs/preview.gif"
    poster_path = "realisations/yansfood/posters/poster_master.png"

    # Save first frame as poster
    f0 = render_frame(0.0)
    cv2.imwrite(poster_path, f0)
    print(f"📸 Poster saved to {poster_path}")

    # FFmpeg pipe for 1440x1440 60fps Master
    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-s", f"{W}x{H}",
        "-pix_fmt", "bgr24",
        "-r", str(FPS),
        "-i", "-",
        "-i", audio_path,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-profile:v", "high",
        "-level", "4.2",
        "-crf", "17",
        "-c:a", "aac",
        "-b:a", "320k",
        "-shortest",
        master_path
    ]

    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

    t_start = time.time()
    for f in range(TOTAL_FRAMES):
        t = f / FPS
        frame_bgr = render_frame(t)
        proc.stdin.write(frame_bgr.tobytes())
        if f % 150 == 0 or f == TOTAL_FRAMES - 1:
            elapsed = time.time() - t_start
            print(f"  Frame {f:3d}/{TOTAL_FRAMES} ({t:5.2f}s) -- {elapsed:5.1f}s elapsed", flush=True)

    proc.stdin.close()
    err = proc.stderr.read().decode()
    proc.wait()

    if proc.returncode != 0:
        print("❌ FFmpeg error:", err)
        sys.exit(1)

    print(f"✅ Master 1440x1440 @ 60fps created: {master_path} ({os.path.getsize(master_path)/1024/1024:.2f} MB)")

    # Generate 1080x1080 @ 30fps Social cut
    print("🔄 Exporting Social cut 1080x1080 @ 30fps...")
    subprocess.run([
        "ffmpeg", "-y",
        "-i", master_path,
        "-vf", "scale=1080:1080,fps=30",
        "-c:v", "libx264", "-crf", "19",
        "-c:a", "aac", "-b:a", "256k",
        social_path
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"✅ Social 1080x1080 cut created: {social_path} ({os.path.getsize(social_path)/1024/1024:.2f} MB)")

    # Generate 1080x1920 (9:16 Vertical Story / Reels)
    print("🔄 Exporting 9:16 Vertical cut 1080x1920...")
    subprocess.run([
        "ffmpeg", "-y",
        "-i", master_path,
        "-vf", "scale=1080:1080,pad=1080:1920:0:420:color=0x0A0A0A",
        "-c:v", "libx264", "-crf", "18",
        "-c:a", "copy",
        vertical_path
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"✅ Vertical 1080x1920 cut created: {vertical_path} ({os.path.getsize(vertical_path)/1024/1024:.2f} MB)")

    # Generate autoplay preview GIF (640x640 @ 15fps)
    print("🔄 Exporting preview GIF for GitHub / Web...")
    subprocess.run([
        "ffmpeg", "-y",
        "-ss", "0", "-t", "15",
        "-i", social_path,
        "-vf", "fps=15,scale=540:540:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=128[p];[s1][p]paletteuse=dither=bayer",
        gif_path
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"✅ Preview GIF created: {gif_path} ({os.path.getsize(gif_path)/1024/1024:.2f} MB)")

if __name__ == "__main__":
    main()
