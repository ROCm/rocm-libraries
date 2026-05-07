#!/usr/bin/env python3
"""Generate RFC 0010 diagrams using Pillow.

Rules:
  - All arrows perfectly straight (horizontal or vertical only)
  - Arrowhead tips touch the target edge, never overlap boxes
  - Generous spacing between elements
"""

from PIL import Image, ImageDraw, ImageFont
import os

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Shared constants & helpers
# ---------------------------------------------------------------------------
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (232, 232, 232)
LINE_W = 2
ARROW_SZ = 10
CORNER_R = 10
PAD_X = 20
PAD_Y = 14


def _load_font(size):
    for name in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    ]:
        if os.path.exists(name):
            return ImageFont.truetype(name, size)
    return ImageFont.load_default()


def _load_bold(size):
    for name in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
    ]:
        if os.path.exists(name):
            return ImageFont.truetype(name, size)
    return _load_font(size)


FONT = _load_font(16)
FONT_SM = _load_font(14)
FONT_BOLD = _load_bold(16)


def _tsize(draw, text, font):
    bb = draw.textbbox((0, 0), text, font=font)
    return bb[2] - bb[0], bb[3] - bb[1]


def _block_size(draw, lines, font):
    ws = [_tsize(draw, l, font)[0] for l in lines]
    hs = [_tsize(draw, l, font)[1] for l in lines]
    return max(ws), sum(hs) + 4 * (len(lines) - 1)


def _draw_text(draw, cx, cy, lines, font, color=BLACK):
    """Draw multi-line text centred at (cx, cy)."""
    hs = [_tsize(draw, l, font)[1] for l in lines]
    total_h = sum(hs) + 4 * (len(lines) - 1)
    y = cy - total_h / 2
    for i, line in enumerate(lines):
        w = _tsize(draw, line, font)[0]
        draw.text((cx - w / 2, y), line, fill=color, font=font)
        y += hs[i] + 4


def box(draw, cx, cy, lines, font=FONT, fill=WHITE, min_w=0, min_h=0):
    """Rounded-rect box centred at (cx,cy). Returns (x0,y0,x1,y1)."""
    tw, th = _block_size(draw, lines, font)
    w = max(tw + PAD_X * 2, min_w)
    h = max(th + PAD_Y * 2, min_h)
    x0, y0 = cx - w / 2, cy - h / 2
    x1, y1 = cx + w / 2, cy + h / 2
    draw.rounded_rectangle([x0, y0, x1, y1], radius=CORNER_R,
                           fill=fill, outline=BLACK, width=LINE_W)
    _draw_text(draw, cx, cy, lines, font)
    return (x0, y0, x1, y1)


def diamond(draw, cx, cy, lines, font=FONT_SM, fill=WHITE, pad=28):
    """Diamond centred at (cx,cy). Returns (x0,y0,x1,y1, half_w, half_h)."""
    tw, th = _block_size(draw, lines, font)
    hw = tw / 2 + pad
    hh = th / 2 + pad
    pts = [(cx, cy - hh), (cx + hw, cy), (cx, cy + hh), (cx - hw, cy)]
    draw.polygon(pts, fill=fill, outline=BLACK, width=LINE_W)
    _draw_text(draw, cx, cy, lines, font)
    return (cx - hw, cy - hh, cx + hw, cy + hh, hw, hh)


def arr_right(draw, x0, y, x1):
    """Horizontal arrow from x0 to x1 at y. Tip touches x1."""
    draw.line([(x0, y), (x1 - ARROW_SZ, y)], fill=BLACK, width=LINE_W)
    draw.polygon([(x1, y),
                  (x1 - ARROW_SZ, y - ARROW_SZ // 2),
                  (x1 - ARROW_SZ, y + ARROW_SZ // 2)], fill=BLACK)


def arr_left(draw, x0, y, x1):
    """Horizontal arrow from x0 leftward to x1 at y. Tip touches x1."""
    draw.line([(x0, y), (x1 + ARROW_SZ, y)], fill=BLACK, width=LINE_W)
    draw.polygon([(x1, y),
                  (x1 + ARROW_SZ, y - ARROW_SZ // 2),
                  (x1 + ARROW_SZ, y + ARROW_SZ // 2)], fill=BLACK)


def arr_down(draw, x, y0, y1):
    """Vertical arrow from y0 to y1 at x. Tip touches y1."""
    draw.line([(x, y0), (x, y1 - ARROW_SZ)], fill=BLACK, width=LINE_W)
    draw.polygon([(x, y1),
                  (x - ARROW_SZ // 2, y1 - ARROW_SZ),
                  (x + ARROW_SZ // 2, y1 - ARROW_SZ)], fill=BLACK)


# ===========================================================================
# Diagram 1: Generation Pipeline (horizontal, 3 boxes)
# ===========================================================================
def make_generation():
    W, H = 1500, 200
    img = Image.new("RGB", (W, H), WHITE)
    d = ImageDraw.Draw(img)

    cy = H / 2
    bw = 360
    gap = 100
    total = 3 * bw + 2 * gap
    sx = (W - total) / 2 + bw / 2

    texts = [
        ["Step 1: construct", "buildGraph()"],
        ["Step 2: execute-reference", "Run trusted reference", "(CPU / GPU / PyTorch)"],
        ["Step 3: serialize", "Save inputs + outputs", "(includes graph.bin)"],
    ]
    rects = []
    for i, lines in enumerate(texts):
        r = box(d, sx + i * (bw + gap), cy, lines, min_w=bw, min_h=90)
        rects.append(r)

    for i in range(2):
        arr_right(d, rects[i][2], cy, rects[i + 1][0])

    img.save(os.path.join(OUTPUT_DIR, "generation_pipeline.png"))
    print(f"  generation_pipeline.png  {W}x{H}")


# ===========================================================================
# Diagram 2: Validation Pipeline (vertical flowchart)
# ===========================================================================
def make_validation():
    W, H = 750, 1000
    img = Image.new("RGB", (W, H), WHITE)
    d = ImageDraw.Draw(img)

    cx = 290
    gap = 50  # vertical gap between elements

    # Step 1
    y = 45
    b1 = box(d, cx, y, ["Step 1: construct", "buildGraph()"], min_w=260)

    # Diamond
    dia_cy = b1[3] + gap + 45
    dia = diamond(d, cx, dia_cy, ["Graph fingerprint", "match?"])
    arr_down(d, cx, b1[3], dia_cy - dia[5])

    # Mismatch → FAIL box to the right
    fail_cx = cx + dia[4] + 140
    fail_b = box(d, fail_cx, dia_cy, ["FAIL", "Graph changed.", "Regenerate."],
                 fill=GRAY, font=FONT_BOLD, min_w=180)
    arr_right(d, cx + dia[4], dia_cy, fail_b[0])
    d.text((cx + dia[4] + 8, dia_cy - 22), "Mismatch", fill=BLACK, font=FONT_SM)

    # Match → down
    d.text((cx + 8, dia_cy + dia[5] + 4), "Match", fill=BLACK, font=FONT_SM)

    # Steps 4-7
    steps = [
        ["Step 4: deserialize", "Load saved inputs"],
        ["Step 5: execute-engine", "Run MIOpen GPU"],
        ["Step 6: deserialize", "Load saved outputs"],
        ["Step 7: validate", "Compare engine output", "to golden output"],
    ]
    prev_bot = dia_cy + dia[5]
    step_rects = []
    for lines in steps:
        cy_box = prev_bot + gap + 25
        r = box(d, cx, cy_box, lines, min_w=260)
        arr_down(d, cx, prev_bot, r[1])
        step_rects.append(r)
        prev_bot = r[3]

    # Pass / Fail result boxes
    b7 = step_rects[-1]
    result_cy = b7[3] + gap + 30
    pass_cx = cx - 100
    fail2_cx = cx + 100

    # Horizontal bar from step 7 bottom
    fork_y = b7[3] + 10
    d.line([(pass_cx, fork_y), (fail2_cx, fork_y)], fill=BLACK, width=LINE_W)
    d.line([(cx, b7[3]), (cx, fork_y)], fill=BLACK, width=LINE_W)

    bp = box(d, pass_cx, result_cy, ["PASS"], font=FONT_BOLD, fill=GRAY, min_w=110, min_h=48)
    bf = box(d, fail2_cx, result_cy, ["FAIL", "mismatch"], font=FONT_BOLD, fill=GRAY, min_w=110, min_h=48)

    arr_down(d, pass_cx, fork_y, bp[1])
    arr_down(d, fail2_cx, fork_y, bf[1])

    d.text((pass_cx - 14, fork_y + 3), "Pass", fill=BLACK, font=FONT_SM)
    d.text((fail2_cx - 10, fork_y + 3), "Fail", fill=BLACK, font=FONT_SM)

    # Crop to actual content
    crop_h = int(bf[3] + 30)
    img = img.crop((0, 0, W, crop_h))
    img.save(os.path.join(OUTPUT_DIR, "validation_pipeline.png"))
    print(f"  validation_pipeline.png  {W}x{crop_h}")


# ===========================================================================
# Diagram 3: Graph-Level Correctness (vertical with 4-column branching)
# ===========================================================================
def make_graph_correctness():
    W, H = 1400, 1100
    img = Image.new("RGB", (W, H), WHITE)
    d = ImageDraw.Draw(img)

    top_cx = W // 2
    gap = 50

    # Row 0: buildGraph()
    r0 = box(d, top_cx, 45, ["buildGraph() in C++", "(single source of truth)"],
             fill=GRAY, min_w=300)

    # Row 1: graph.to_binary()
    r1_cy = r0[3] + gap + 20
    r1 = box(d, top_cx, r1_cy, ["graph.to_binary()"], min_w=220)
    arr_down(d, top_cx, r0[3], r1[1])

    # Row 2: graph.bin
    r2_cy = r1[3] + gap + 20
    r2 = box(d, top_cx, r2_cy, ["graph.bin"], fill=GRAY, min_w=180)
    arr_down(d, top_cx, r1[3], r2[1])

    # 4 columns: well spaced
    cols = [145, 385, 640, 1050]
    branch_y = r2[3] + 25  # horizontal bar
    col_top = branch_y + gap  # where column boxes start

    # Horizontal bar from graph.bin
    d.line([(top_cx, r2[3]), (top_cx, branch_y)], fill=BLACK, width=LINE_W)
    d.line([(cols[0], branch_y), (cols[-1], branch_y)], fill=BLACK, width=LINE_W)
    for cx in cols:
        arr_down(d, cx, branch_y, col_top)

    # Column 1: CPU ref
    c1a = box(d, cols[0], col_top + 28, ["CPU ref executor", "(C++ runs graph)"],
              font=FONT_SM, min_w=170)
    c1b_cy = c1a[3] + gap
    c1b = box(d, cols[0], c1b_cy, ["golden outputs"], font=FONT_SM, fill=GRAY, min_w=150)
    arr_down(d, cols[0], c1a[3], c1b[1])

    # Column 2: GPU ref
    c2a = box(d, cols[1], col_top + 28, ["GPU ref executor", "(C++ runs graph)"],
              font=FONT_SM, min_w=170)
    c2b_cy = c2a[3] + gap
    c2b = box(d, cols[1], c2b_cy, ["golden outputs"], font=FONT_SM, fill=GRAY, min_w=150)
    arr_down(d, cols[1], c2a[3], c2b[1])

    # Column 3: Python
    c3a = box(d, cols[2], col_top + 28, ["Python reads", "graph.bin (Graph Export)", "extracts params"],
              font=FONT_SM, min_w=180)
    c3b_cy = c3a[3] + gap
    c3b = box(d, cols[2], c3b_cy, ["Runs PyTorch"], font=FONT_SM, min_w=150)
    arr_down(d, cols[2], c3a[3], c3b[1])
    c3c_cy = c3b[3] + gap
    c3c = box(d, cols[2], c3c_cy, ["golden outputs"], font=FONT_SM, fill=GRAY, min_w=150)
    arr_down(d, cols[2], c3b[3], c3c[1])

    # Column 4: Validation
    c4a = box(d, cols[3], col_top + 28, ["VALIDATION (CI)", "current graph hash"],
              font=FONT_SM, min_w=180)

    # Diamond: compare hash
    c4_dia_cy = c4a[3] + gap + 45
    c4_dia = diamond(d, cols[3], c4_dia_cy, ["Compare hash", "to stored hash"],
                     font=FONT_SM, pad=25)
    arr_down(d, cols[3], c4a[3], c4_dia_cy - c4_dia[5])

    # Mismatch → FAIL box to the right of diamond
    c4_fail_cx = cols[3] + c4_dia[4] + 140
    c4_fail = box(d, c4_fail_cx, c4_dia_cy, ["FAIL", "Graph changed,", "regenerate"],
                  font=FONT_SM, fill=GRAY, min_w=160)
    arr_right(d, cols[3] + c4_dia[4], c4_dia_cy, c4_fail[0])
    d.text((cols[3] + c4_dia[4] + 8, c4_dia_cy - 22), "Mismatch", fill=BLACK, font=FONT_SM)

    # Match → down to "Proceed"
    c4_match_cy = c4_dia_cy + c4_dia[5] + gap + 20
    c4_match = box(d, cols[3], c4_match_cy, ["Proceed to", "value comparison"],
                   font=FONT_SM, min_w=160)
    arr_down(d, cols[3], c4_dia_cy + c4_dia[5], c4_match[1])
    d.text((cols[3] + 8, c4_dia_cy + c4_dia[5] + 3), "Match", fill=BLACK, font=FONT_SM)

    # Cross-validate diamond at bottom, fed by cols 1,2,3 golden outputs
    golden_bots = [c1b[3], c2b[3], c3c[3]]
    junction_y = max(golden_bots) + 35
    cross_cx = (cols[0] + cols[2]) / 2
    cross_cy = junction_y + gap + 45

    cross_dia = diamond(d, cross_cx, cross_cy,
                        ["Cross-validate:", "Do they agree?"],
                        font=FONT_SM, pad=25)

    # Vertical lines from each golden output down to junction, then horizontal bar
    for i in range(3):
        d.line([(cols[i], golden_bots[i]), (cols[i], junction_y)], fill=BLACK, width=LINE_W)
    d.line([(cols[0], junction_y), (cols[2], junction_y)], fill=BLACK, width=LINE_W)
    d.line([(cross_cx, junction_y), (cross_cx, cross_cy - cross_dia[5])], fill=BLACK, width=LINE_W)
    # Arrowhead into diamond top
    d.polygon([(cross_cx, cross_cy - cross_dia[5]),
               (cross_cx - ARROW_SZ // 2, cross_cy - cross_dia[5] - ARROW_SZ),
               (cross_cx + ARROW_SZ // 2, cross_cy - cross_dia[5] - ARROW_SZ)], fill=BLACK)

    # Cross-validate results: Yes → right, No → down
    # Yes → PASS box to the right
    cv_pass_cx = cross_cx + cross_dia[4] + 120
    cv_pass = box(d, cv_pass_cx, cross_cy, ["PASS", "References agree"],
                  font=FONT_SM, fill=GRAY, min_w=160)
    arr_right(d, cross_cx + cross_dia[4], cross_cy, cv_pass[0])
    d.text((cross_cx + cross_dia[4] + 8, cross_cy - 22), "Yes", fill=BLACK, font=FONT_SM)

    # No → FAIL box below
    cv_fail_cy = cross_cy + cross_dia[5] + gap + 20
    cv_fail = box(d, cross_cx, cv_fail_cy, ["FAIL", "Investigate"],
                  font=FONT_SM, fill=GRAY, min_w=140)
    arr_down(d, cross_cx, cross_cy + cross_dia[5], cv_fail[1])
    d.text((cross_cx + 8, cross_cy + cross_dia[5] + 3), "No", fill=BLACK, font=FONT_SM)

    # Crop to content
    crop_h = int(max(c4_match[3], cv_fail[3]) + 35)
    crop_w = int(max(W, c4_fail[2] + 30))
    img = img.crop((0, 0, min(crop_w, W), crop_h))
    img.save(os.path.join(OUTPUT_DIR, "graph_level_correctness.png"))
    print(f"  graph_level_correctness.png  {W}x{crop_h}")


# ===========================================================================
if __name__ == "__main__":
    print("Generating diagrams...")
    make_generation()
    make_validation()
    make_graph_correctness()
    print("Done.")
