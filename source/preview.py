import sys, os, time, numpy as np, cv2, math
sys.path.insert(0, "/home/user/showreel")
import motion.scenes as SC
from render import frame, to_uint8
from motion.engine import W, H

def warm(t):
    """deterministically advance stateful sims so a still matches the real timeline"""
    if t < 6.5: 
        SC.PART = None
        return
    SC.PART = None
    lt_end = t - 6.5
    lt = 0.0
    while lt < lt_end - 1e-6:
        SC.s4_physics(min(lt, lt_end))
        lt += 1.0 / 60.0

TIMES = [float(x) for x in sys.argv[1].split(",")]
SHEET = sys.argv[2] if len(sys.argv) > 2 else "sheet"
tiles = []
for t in TIMES:
    warm(t)
    t0 = time.time()
    u8 = to_uint8(frame(t))
    print(f"t={t:5.2f} {time.time()-t0:.2f}s", flush=True)
    small = cv2.resize(u8, (960, 540), interpolation=cv2.INTER_AREA)
    tiles.append(small)
while len(tiles) % 2: tiles.append(np.zeros((540, 960, 3), np.uint8))
rows = []
for i in range(0, len(tiles), 2):
    rows.append(np.concatenate([tiles[i], tiles[i + 1]], 1))
sheet = np.concatenate(rows, 0)
cv2.imwrite(f"/home/user/showreel/prev/{SHEET}.png", sheet[..., ::-1])
print("sheet ->", SHEET, sheet.shape)
