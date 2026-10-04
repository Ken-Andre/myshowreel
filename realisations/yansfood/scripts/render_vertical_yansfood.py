#!/usr/bin/env python3
"""
render_vertical_yansfood.py -- Generates true 1080x1920 vertical recomposition
with vertical-native HUD top & bottom.
"""
import os, sys, time, subprocess
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont

VW, VH = 1080, 1920
FPS = 60
DUR = 15.00
TOTAL_FRAMES = int(FPS * DUR)

FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT_MONO = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"

def get_font(path, size):
    try: return ImageFont.truetype(path, size)
    except: return ImageFont.load_default()

f_title = get_font(FONT_BOLD, 42)
f_scene = get_font(FONT_BOLD, 26)
f_mono = get_font(FONT_MONO, 20)
f_mono_small = get_font(FONT_MONO, 16)
f_badge = get_font(FONT_MONO, 15)

master_video_path = "realisations/yansfood/videos/yansfood_master_1440x1440_60fps.mp4"
vertical_out_path = "realisations/yansfood/videos/yansfood_vertical_1080x1920.mp4"
audio_path = "realisations/yansfood/audio/yansfood_score.wav"

cap = cv2.VideoCapture(master_video_path)

cmd = [
    "ffmpeg", "-y",
    "-f", "rawvideo",
    "-vcodec", "rawvideo",
    "-s", f"{VW}x{VH}",
    "-pix_fmt", "bgr24",
    "-r", str(FPS),
    "-i", "-",
    "-i", audio_path,
    "-c:v", "libx264",
    "-pix_fmt", "yuv420p",
    "-profile:v", "high",
    "-level", "4.2",
    "-crf", "18",
    "-c:a", "aac",
    "-b:a", "256k",
    "-shortest",
    vertical_out_path
]

proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

print("📱 Rendering Native 9:16 Vertical Video (1080x1920 @ 60fps)...")

for f in range(TOTAL_FRAMES):
    ret, frame_bgr = cap.read()
    if not ret: break
    t = f / FPS
    
    # 1. Base Vertical Canvas (1080x1920) in Deep UI Black
    v_im = Image.new('RGB', (VW, VH), (10, 10, 12))
    v_draw = ImageDraw.Draw(v_im)
    
    # 2. Scale square frame to 1080x1080 at center (Y: 420 to 1500)
    center_1080 = cv2.resize(frame_bgr, (1080, 1080), interpolation=cv2.INTER_LINEAR)
    center_rgb = cv2.cvtColor(center_1080, cv2.COLOR_BGR2RGB)
    pil_center = Image.fromarray(center_rgb)
    v_im.paste(pil_center, (0, 420))
    
    # 3. Top HUD (0 to 420 px)
    # Header brand mark
    v_draw.text((60, 90), "YANS", font=f_title, fill=(255, 255, 255), anchor="lm")
    v_draw.text((180, 90), "FOOD", font=f_title, fill=(255, 107, 53), anchor="lm")
    
    # Live badge
    v_draw.rounded_rectangle([780, 68, 1020, 112], radius=22, fill=(25, 28, 36), outline=(255, 107, 53), width=2)
    v_draw.text((900, 90), "SHOWREEL '26", font=f_badge, fill=(255, 107, 53), anchor="mm")
    
    # Timecode & Frame Readout
    ss = int(t)
    ff = int((t % 1) * 60)
    tc_str = f"00:00:{ss:02d}:{ff:02d} · 120 BPM · 60 FPS"
    v_draw.text((60, 170), tc_str, font=f_mono, fill=(160, 168, 185), anchor="lm")
    
    # Scene Title
    scene_str = "01 · OPEN WORDMARK"
    if t >= 1.50 and t < 4.00: scene_str = "02 · BENTO DISCOVERY · POULET DG"
    elif t >= 4.00 and t < 6.50: scene_str = "03 · LIQUID GLASS · RELIGHT"
    elif t >= 6.50 and t < 9.00: scene_str = "04 · GLASS ORB · RIZ JOLLOF"
    elif t >= 9.00 and t < 11.50: scene_str = "05 · KEYNOTE STAGE · MAC EXPANSION"
    elif t >= 11.50 and t < 12.75: scene_str = "06 · BREAKDOWN · COMMANDER"
    elif t >= 12.75 and t < 14.25: scene_str = "07 · BLACK FLOOD · EN ROUTE"
    elif t >= 14.25 and t < 14.75: scene_str = "08 · MUR EN VÉRITÉ · LISIÈRE ORANGE"
    elif t >= 14.75: scene_str = "09 · SLAM FINAL · YANSFOOD"
    
    v_draw.text((60, 220), scene_str, font=f_scene, fill=(255, 255, 255), anchor="lm")
    
    # Progress rail line
    rail_w = 960
    rail_x = 60
    rail_y = 360
    v_draw.rectangle([rail_x, rail_y, rail_x + rail_w, rail_y + 4], fill=(35, 40, 52))
    v_draw.rectangle([rail_x, rail_y, rail_x + int(rail_w * (t / DUR)), rail_y + 4], fill=(255, 107, 53))
    
    # Dividing borders
    v_draw.line([0, 418, 1080, 418], fill=(45, 50, 65), width=2)
    v_draw.line([0, 1502, 1080, 1502], fill=(45, 50, 65), width=2)
    
    # 4. Bottom HUD (1500 to 1920 px)
    # Status Tag
    v_draw.rounded_rectangle([60, 1540, 460, 1590], radius=14, fill=(25, 28, 36), outline=(60, 65, 80), width=1)
    v_draw.text((80, 1565), "DOUALA & YAOUNDÉ · CAMEROUN", font=f_badge, fill=(255, 107, 53), anchor="lm")
    
    # Brand Promise
    v_draw.text((60, 1650), "TON REPAS, SANS LA FILE.", font=f_title, fill=(255, 255, 255), anchor="lm")
    v_draw.text((60, 1710), "Commandez en quelques clics via l'application.", font=f_mono, fill=(160, 168, 185), anchor="lm")
    
    # Store Badges
    v_draw.rounded_rectangle([60, 1760, 320, 1840], radius=16, fill=(20, 22, 28), outline=(60, 65, 80), width=2)
    v_draw.text((190, 1785), "Télécharger sur", font=f_mono_small, fill=(160, 168, 185), anchor="mm")
    v_draw.text((190, 1815), "App Store", font=f_mono, fill=(255, 255, 255), anchor="mm")
    
    v_draw.rounded_rectangle([360, 1760, 620, 1840], radius=16, fill=(20, 22, 28), outline=(60, 65, 80), width=2)
    v_draw.text((490, 1785), "Disponible sur", font=f_mono_small, fill=(160, 168, 185), anchor="mm")
    v_draw.text((490, 1815), "Google Play", font=f_mono, fill=(255, 255, 255), anchor="mm")
    
    # Write to FFmpeg
    out_bgr = cv2.cvtColor(np.array(v_im), cv2.COLOR_RGB2BGR)
    proc.stdin.write(out_bgr.tobytes())

cap.release()
proc.stdin.close()
proc.wait()

print(f"✅ Vertical 1080x1920 video updated: {vertical_out_path} ({os.path.getsize(vertical_out_path)/1024/1024:.2f} MB)")
