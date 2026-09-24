"""Usage: pip install numpy pillow && python tools/generate_sky.py [wide] [tall]

Cumulus sky backgrounds, rendered as a lit height field.

Every puff is a dome; the cloud surface is the max of all domes. Lighting that surface
from the upper left gives bright tops and blue shadow in the folds between lobes.
"""
import math, os, random, sys
import numpy as np
from PIL import Image, ImageFilter



SITE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "images")


def billow(out, x, y, r, depth, rng, z=0.0):
    """Cauliflower recursion; each child dome sits on its parent's surface so it shows."""
    out.append((x, y, r, z))
    if depth == 0 or r < 6:
        return
    for _ in range(rng.randint(2, 4)):
        a = math.radians(rng.uniform(-195, 15))
        rr = r * rng.uniform(0.28, 0.58)
        d = r * rng.uniform(0.55, 1.0)
        surf = math.sqrt(max(r * r - d * d, 0))
        billow(out, x + math.cos(a) * d, y + math.sin(a) * d, rr, depth - 1, rng, z + surf * 0.3)


def fbm(rng, h, w, cell, octaves=4):
    out = np.zeros((h, w), np.float32)
    amp, tot = 1.0, 0.0
    for _ in range(octaves):
        gh, gw = int(h / cell) + 2, int(w / cell) + 2
        g = Image.fromarray((rng.random((gh, gw)) * 255).astype(np.uint8))
        img = g.resize((int(gw * cell), int(gh * cell)), Image.BICUBIC)
        out += amp * (np.asarray(img, np.float32)[:h, :w] / 255.0)
        tot += amp
        amp *= 0.5
        cell = max(cell / 2, 1.5)
    return out / tot


def mass(out, rng, x0, x1, base, bumps, r0, density=1.0):
    def top(x):
        return base - sum(hh * math.exp(-((x - c) / wd) ** 2) for c, wd, hh in bumps)
    x = x0
    while x < x1:
        billow(out, x, top(x) + rng.uniform(-r0 * 0.3, r0 * 0.5), r0 * rng.uniform(0.45, 1.3), 4, rng)
        x += r0 * rng.uniform(0.45, 1.0)
    area = sum(base - top(xx) for xx in range(int(x0), int(x1), 10)) * 10
    for _ in range(int(area / (r0 * r0) * density)):
        xx = rng.uniform(x0, x1)
        y = rng.uniform(top(xx) + r0 * 0.3, base + r0)
        billow(out, xx, y, r0 * rng.uniform(0.5, 1.7), 3, rng)


def cloudlet(out, rng, x, y, r):
    for _ in range(rng.randint(2, 4)):
        billow(out, x + rng.uniform(-r, r) * 1.4, y + rng.uniform(-r, r) * 0.4, r * rng.uniform(0.5, 1.1), 3, rng)


def height_field(h, w, puffs):
    H = np.full((h, w), -1e3, np.float32)
    for x, y, r, zb in puffs:
        x0, x1 = max(int(x - r), 0), min(int(x + r) + 1, w)
        y0, y1 = max(int(y - r), 0), min(int(y + r) + 1, h)
        if x0 >= x1 or y0 >= y1:
            continue
        yy, xx = np.mgrid[y0:y1, x0:x1]
        d2 = (xx - x) ** 2 + (yy - y) ** 2
        z = np.where(d2 < r * r, np.sqrt(np.maximum(r * r - d2, 0)) + zb, -1e3).astype(np.float32)
        np.maximum(H[y0:y1, x0:x1], z, out=H[y0:y1, x0:x1])
    return H


def warp(a, rng, amp, cell, fill):
    h, w = a.shape
    dx = (fbm(rng, h, w, cell, 3) - 0.5) * 2 * amp
    dy = (fbm(rng, h, w, cell, 3) - 0.5) * 2 * amp
    yy, xx = np.mgrid[0:h, 0:w]
    sx = np.clip((xx + dx).astype(np.int32), 0, w - 1)
    sy = np.clip((yy + dy).astype(np.int32), 0, h - 1)
    return a[sy, sx]


def _box(a, r, axis):
    r = max(int(r), 1)
    pad = [(0, 0)] * a.ndim
    pad[axis] = (r + 1, r)
    c = np.cumsum(np.pad(a, pad, mode="edge"), axis=axis, dtype=np.float64)
    n = a.shape[axis]
    hi = np.take(c, np.arange(2 * r + 1, 2 * r + 1 + n), axis=axis)
    lo = np.take(c, np.arange(0, n), axis=axis)
    return ((hi - lo) / (2 * r + 1)).astype(np.float32)


def blur(a, radius):
    """~Gaussian via three box passes."""
    r = max(radius * 0.87, 1)
    for _ in range(3):
        a = _box(_box(a, r, 0), r, 1)
    return a


def lerp3(t, a, b, c):
    t = t[..., None]
    a, b, c = (np.array(v, np.float32) for v in (a, b, c))
    return np.where(t < 0.5, a + (b - a) * (t / 0.5), b + (c - b) * ((t - 0.5) / 0.5))


def render(w, h, puffs, star_pts, seed, out_name):
    rng = np.random.default_rng(seed)
    H = height_field(h, w, puffs)
    inside = H > -1
    H = np.where(inside, H, -40)
    # fluffy, broken edges
    H = warp(H, rng, 22, 70, -40)
    H = warp(H, rng, 7, 14, -40)
    inside = H > 0

    # lighting from the upper left, slightly in front
    H = H + (fbm(rng, h, w, 9, 2) - 0.5) * 2.5 * (H > 0)
    Hs = blur(H, 3.5)
    gy, gx = np.gradient(Hs)
    n = np.stack([-gx, -gy, np.full_like(gx, 1.4)], -1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    L = np.array([-0.45, -0.75, 0.5], np.float32)
    L /= np.linalg.norm(L)
    lam = np.clip((n * L).sum(-1), 0, 1)
    ao = np.clip((blur(H, 18) - H) / 30, 0, 1)  # folds get darker
    t = np.clip(lam * 1.25 - ao * 0.9 + 0.12, 0, 1)
    t = t * t * (3 - 2 * t)
    cloud_rgb = lerp3(t, (74, 106, 222), (214, 216, 238), (255, 247, 230))

    # sky
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    g = np.clip(yy / h * 0.85 + xx / w * 0.15, 0, 1)[..., None]
    sky = np.array([15, 45, 192], np.float32) * (1 - g) + np.array([46, 92, 226], np.float32) * g
    for x, y, r in star_pts:
        d = np.sqrt((xx - x) ** 2 + (yy - y) ** 2)
        a = np.clip((r + 0.8 - d) / 1.6, 0, 1)[..., None]
        sky = sky * (1 - a) + 255 * a

    alpha = np.clip(H / 2.5, 0, 1)[..., None]
    im = sky * (1 - alpha) + cloud_rgb * alpha

    # print finish: stipple on the clouds, specks along cloud edges, grain everywhere
    c = alpha[..., 0]
    shade_amt = (1 - t) * c
    dots = (rng.random((h, w)) < 0.05 + 0.25 * shade_amt) * c
    im = im * (1 - dots[..., None] * 0.35) + np.array([110, 135, 225], np.float32) * dots[..., None] * 0.35
    near = np.clip(blur(c, 16) * 2.5 - c * 2.5, 0, 1)
    specks = (rng.random((h, w)) < 0.035 * near).astype(np.float32)[..., None]
    im = im + (255 - im) * specks * 0.8
    im = im + rng.normal(0, 6, (h, w, 1))

    path = os.path.join(SITE, out_name)
    Image.fromarray(np.clip(im, 0, 255).astype(np.uint8)).save(path, "JPEG", quality=80, optimize=True, progressive=True)
    print(path, os.path.getsize(path) // 1024, "KB")


def star_list(rng, w, h, n, region):
    pts = []
    while len(pts) < n:
        x, y = rng.uniform(0, w), rng.uniform(0, h)
        if region(x, y):
            pts.append((x, y, rng.choice([1.2, 1.6, 2.2, 3.2, 4.8])))
    return pts


def wide():
    w, h = 2400, 1500
    rng = random.Random(7)
    p = []
    mass(p, rng, 1450, 2550, 1600, [(2150, 260, 1050), (1900, 200, 620), (2420, 200, 900), (1650, 150, 420)], 105)
    mass(p, rng, 300, 1750, 1620, [(1100, 330, 470), (700, 220, 300), (1450, 200, 380)], 90)
    mass(p, rng, -150, 520, 1600, [(80, 220, 230), (380, 160, 120)], 65)
    for x, y, r in [(1760, 250, 24), (1340, 690, 20), (1190, 770, 15), (1580, 520, 17),
                    (960, 920, 18), (1880, 120, 15), (1460, 830, 13), (190, 1080, 18), (60, 1190, 16),
                    (2000, 60, 13), (1260, 960, 13)]:
        cloudlet(p, rng, x, y, r)
    return w, h, p, star_list(rng, w, h, 55, lambda x, y: y < 1000 and x < 1750)


def tall():
    w, h = 1200, 2400
    rng = random.Random(21)
    p = []
    mass(p, rng, 600, 1350, 2500, [(1060, 200, 1000), (820, 150, 600), (1250, 150, 800)], 85)
    mass(p, rng, -150, 900, 2520, [(330, 260, 560), (650, 160, 380), (60, 150, 420)], 80)
    for x, y, r in [(980, 1120, 18), (760, 1280, 15), (620, 1400, 13), (200, 1700, 16),
                    (1080, 930, 13), (420, 1560, 11), (880, 800, 11)]:
        cloudlet(p, rng, x, y, r)
    return w, h, p, star_list(rng, w, h, 40, lambda x, y: y < 1550)


if __name__ == "__main__":
    which = sys.argv[1:] or ["wide", "tall"]
    if "wide" in which:
        render(*wide(), 1, "sky-wide.jpg")
    if "tall" in which:
        render(*tall(), 2, "sky-tall.jpg")
