# Leak Fix

Light-leak removal for film scans that subtracts the leak from the original pixels instead of regenerating them. The user (or, later, a trained detector) draws boxes; the physics decides how much to remove. Nothing outside a box is changed.

## Install and run

    pip install -r requirements.txt
    python -m leakfix.app          # local web app at http://127.0.0.1:7860

Or `pip install -e .` then `leakfix`.

## Using the app

1. **Load a scan.** It is shown at up to 1600 px; everything is processed and saved at full resolution.
2. **Keep the step on *Recipe v2 (main path)*** (selected at start). Drag one box around each light leak (see *Drawing boxes*), then press **Apply**. The recipe runs at fixed gains, so the strength slider and the leftover-pass checkbox do nothing in this mode; the tile dropdown still applies (*auto* by default). The status line reports the tile, the blue-tint mode and share of cyan tiles, the exposure side and lift per box, the black point, and the change inside and outside the padded boxes (outside is always 0).
3. **Check by eye**, using *Show original | current*. If something is left, pick one of the manual steps 1–5 and fix it on top: they work on the current image.
4. **Undo** steps back one Apply: the image, and the stage in `<scan>.boxes.json`, so the log always replays to what is on screen. If the file was edited elsewhere in between, the image is still restored and the file is left alone (the status says so).
5. **Save full-resolution JPEG** writes `<scan>_fixed.jpg` next to the scan.

**Proposals.** Instead of drawing, ask for boxes: *Propose (YOLO)* uses the trained detector (`pip install -e ".[detect]"` and weights at `work/detect/weights/best.pt`), *Propose boxes (Claude)* the Claude model (only shown when the `anthropic` package and a key are present). Proposals are drawn dashed; pick one and press **Accept**, **Adjust** or **Dismiss**. Nothing is applied until you press Apply, and every outcome is logged in `boxes.json` (dismissals are hard negatives for training).

The same recipe from the command line: `python -m leakfix.cli scan.jpg --recipe v2 --leak x0,x1,y0,y1 -o fixed.jpg` (it also writes `fixed.boxes.json`). `notebooks/6_recipe_v2.ipynb` walks through the recipe one sub-step at a time.

## The five steps

| step | what it does | notebook |
|---|---|---|
| **Recipe v2 (main path)** | Two deleak passes (1.3, 1.3) in boxes padded by 2 tiles; blue tint at 1.0 only on tiles bluer than the same-row neighbours outside the box (gentle 0.5 when under 10 % of tiles); exposure only against a side strip whose rows match the box edge; HSV 0.45; a black-point floor so nothing gets darker than the scan's own blacks; then a per-tile leak-mask fade so there is no box seam. Gains are fixed. CLI: `--recipe v2 --leak x0,x1,y0,y1`; box file `"recipe": "v2"`. Writes a replayable `recipe_v2` stage. Steps 1–4 below remain the manual path. | `notebooks/6_recipe_v2.ipynb` |
| 1 · Remove leak | Per tile (auto: 64 px on large scans, 32 or 16 px on small ones; override with the tile dropdown or `--tile`), the darkest `R−B` / `G−B` (chromatic dark channel) measures the orange fog; the clean level is interpolated from the tiles just outside the box; the excess is subtracted in gamma space with film toe/shoulder weighting, inside the box only with an 8 px ramp at its edge. | `notebooks/1_tile_method.ipynb` |
| 2 · Remove blue tint | Same method and tile rule with the channels swapped (anchor on red) for cyan patches left after step 1. | `notebooks/2_blue_tint_correction.ipynb` |
| 3 · Match exposure | Draw one box per dull patch, press *Next drag = reference*, then drag a REFERENCE box of the same surface that looks right (same rows). All patches are lifted to the reference's brightness, row by row, in linear light. Never darkens, capped at 1 stop. | `notebooks/3_exposure_matching.ipynb` |
| 4 · Neutralize warm (HSV) | Finishing pass, automatic. Masks bright warm pixels (HSV hue 0–25, sat ≥ 40, value ≥ 180) inside the boxes applied in steps 1–3, each padded by one tile (64 px on large scans, 32 or 16 px on small ones), softens the mask, pulls red toward green by the strength (default 0.45). Tick *whole frame* to run everywhere instead; draw boxes to override. | `notebooks/4_hsv_neutralize.ipynb` |
| 5 · Inpaint residue (opt-in) | Off the main path. For clipped cores only: where the original has red ≥ 0.97 and blue < 0.97 inside a box, fills with LaMa (`big-lama.pt`, `pip install -e ".[inpaint]"`) or OpenCV Telea, matches Lab a/b to a 61 px ring, blends the scan's own grain back (soft light 0.35), feathers the seam σ = 6. Refuses when the mask exceeds 15 % of the frame. CLI: `--inpaint x0,x1,y0,y1 --inpaint-method auto\|telea\|lama`. | `notebooks/5_inpaint_residue.ipynb` |

Steps chain: each works on the previous result. **Undo** steps back one apply, including the boxes it contributed to step 4. Step 4 needs no boxes of its own: it reuses the leak boxes from steps 1–3. If none have been applied yet it falls back to the whole frame, and in whole-frame mode it can touch warm objects anywhere (skin, orange seats), so check the result.

## Drawing boxes

Press on the image and drag in any direction; release to finish the box. A box only needs to *contain* the leak and *exclude* any warm object that isn't one (a red car, a yellow wall). Orientation is inferred from the box shape: taller than wide = vertical band (reference from left/right), wider than tall = horizontal (reference from above/below).

For step 3, draw every dull patch first (any number), press *Next drag = reference*, then drag one reference box of the same surface spanning the same rows. The status line always says whether the next drag will be a patch or the reference. Changing step clears a pending reference.

Step 4 needs no boxes. It runs automatically inside the boxes already applied in steps 1–3 (each padded by one tile). Tick the *whole frame* checkbox to run it everywhere, or draw boxes to confine it to exactly those.

## Output

* `<scan>_fixed.jpg` — full resolution.
* `<scan>.boxes.json` — every applied box with its stage, strength and (steps 1–2) tile size. This is the label set for training a model to propose the boxes. Replay it with

      python -m leakfix.cli scan.jpg --replay scan.boxes.json

  From the command line, `--hsv 0.45` adds the step-4 pass after the other corrections (`--hsv-box x0,x1,y0,y1` confines it).

## Audit

Every apply prints the mean per-channel change inside the boxes and outside them. Outside is exactly 0: every correction is applied inside its box only. That number is the method's guarantee.

## Known limits

* A clipped core (red at 255) has no scene under it; subtraction leaves a pale bar. Inpainting or blue-channel reconstruction is needed there.
* Bright content (white sky) under a leak is under-corrected (film shoulder); a leftover warm haze may remain.
* A warm object inside a box is treated as leak. Keep boxes off red cars and yellow walls.
