# SHOWREEL '26 — 15.0 s motion graphics reel
1920×1080 · 60 fps · H.264 (high profile) · AAC 224 k stereo · 120 BPM original score

Everything is generated procedurally — no stock footage, no templates, no keyframe software.
A custom numpy/OpenCV compositor (`motion/engine.py`) rasterises variable-font type, supersampled
vector layers, 3D projections and a 4 200-particle curl-noise simulation, then grades every frame
(bloom, radial chromatic aberration, film grain, vignette, filmic tonemap) and pipes raw RGB
straight into x264.

## Timeline (beat-locked: downbeat = 1.000 s, 120 BPM, cuts on bars)
| t | scene | what's happening |
|---|-------|------------------|
| 0.00–2.00 | IGNITION | bezier comet draws on at constant arc-speed, morphs into the title rule; "MOTION" slams in per-glyph on a damped spring while Archivo's **weight 180→900 and width 66→112 axes animate**; impact shake + chromatic punch at 0.98 |
| 2.00–4.00 | SYSTEMS | lime bar wipe; 24×13 dot grid reveals on a diagonal wave then breathes; wireframe cube line-draws in true 3D perspective; type stack, leader lines, live counters |
| 4.00–6.50 | KINETIC TYPE | three treatments on the beat: magenta block + sweeping-bar letter reveal ("KINETIC"), scramble-decode with beat-pulse ("RHYTHM"), variable-axis slam width 62→125 / weight 100→900 with elastic settle ("CRAFT") |
| 6.50–9.00 | FLOW | curl-noise flow field with velocity streaks + streamlines → 4 200 particles attract into the word "FLUID" → shockwave → reform as a rotating orbit ring |
| 9.00–11.50 | PRODUCT MOTION | 5-card 3D carousel (real perspective transforms, depth blur + dim); cursor eases along a path, clicks the toggle → ripple, lime progress, staggered bar chart |
| 11.50–13.50 | MONTAGE | 12 hard cuts on 8th-notes: RGB-split type, scan sweeps, starburst, halftone, glitched colour bars; slice-displacement glitch + scanlines, riser underneath, hard cut to black |
| 13.50–15.00 | OUTRO | monogram stroke-on (arc + slash + orbit dot), name mask-reveal with tracking ease, blinking availability dot, specular shine sweep, vignette close, fade |

Persistent HUD: bottom progress rail with scene ticks + playhead, SMPTE-style timecode,
format readout, corner mark — all suppressed during the montage so the glitch section owns the frame.

## Score (`audio.py`, fully synthesised)
Kick / hats / clap on the grid, sub-bass following a 7-bar Am–F–C–G progression, filtered pluck arp,
sidechain duck on every kick, noise risers into cuts, sub impacts on every scene change,
bit-crush stutters in the montage, boom + shimmer for the outro. Master: glue comp → soft clip → −1 dBFS.

## Swap in your name
    REEL_NAME="ADA OKAFOR" REEL_ROLE="MOTION DESIGNER" python3 render.py --audio out/audio.wav --final out/showreel_1080p60.mp4
(name appears on the outro card and the HUD top-right corner.)

## Layout
    motion/engine.py   compositor: easings, bezier/3D, variable-font type engine, bloom/CA/grain/glitch
    motion/px.py       curl-noise flow field, particle attractors/shockwaves, streak splatting
    motion/scenes.py   the seven scenes + precompute
    render.py          timeline, HUD, grade, ffmpeg pipe
    audio.py           the score
    preview.py         contact-sheet stills for art direction
