#!/usr/bin/env python3
"""
Rotating 3D Volumetric Heart in Terminal (shape - exact parametric
heart curve ♡, not an abstract surface, so the silhouette is always recognizable).

Requires: numpy, matplotlib (only for the contains_points from matplotlib.path)
  pip install numpy matplotlib --break-system-packages
Run:  python3 heart3d.py
Exit:   Ctrl+C
"""
import numpy as np
import time
import os
from matplotlib.path import Path


def clear():
    os.system('cls' if os.name == 'nt' else 'clear')


def build_heart(face_density=90, edge_layers=9, thickness=0.16, bulge=0.7):
    """
    Builds a cloud of points for a three-dimensional heart-"pillow":
      - contour — the exact classical curve of the heart (x=16 sin^3 t, ...),
        therefore, the silhouette always looks like a real heart,and not like a stone/potato;
      - the font and back edges are "bulge", like a soft toy;
      - the end (side edge) gives volume when turning.
    """
    t = np.linspace(0, 2 * np.pi, 400)
    hx = 16 * np.sin(t) ** 3
    hy = 13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t)

    scale = max(hx.max() - hx.min(), hy.max() - hy.min()) / 2
    hx = hx / scale
    hy = hy / scale
    hy = hy - (hy.max() + hy.min()) / 2
    hx = hx - (hx.max() + hx.min()) / 2

    poly = Path(np.column_stack([hx, hy]))

    grid = np.linspace(-1.1, 1.1, face_density)
    GX, GY = np.meshgrid(grid, grid)
    grid_pts = np.column_stack([GX.ravel(), GY.ravel()])
    inside = poly.contains_points(grid_pts)
    fx = grid_pts[inside][:, 0]
    fy = grid_pts[inside][:, 1]

    pts, norms = [], []

    for x, y in zip(fx, fy):
        pts.append((x, y, thickness))
        n = np.array([x * bulge, y * bulge, 1.0])
        norms.append(n / np.linalg.norm(n))
    for x, y in zip(fx, fy):
        pts.append((x, y, -thickness))
        n = np.array([x * bulge, y * bulge, -1.0])
        norms.append(n / np.linalg.norm(n))

    dt = t[1] - t[0]
    dhx = np.gradient(hx, dt)
    dhy = np.gradient(hy, dt)
    tl = np.sqrt(dhx ** 2 + dhy ** 2)
    tl[tl == 0] = 1
    nx, ny = dhy / tl, -dhx / tl
    cx, cy = hx.mean(), hy.mean()
    for i in range(len(t)):
        outward = np.array([hx[i] - cx, hy[i] - cy])
        cand = np.array([nx[i], ny[i]])
        if np.dot(outward, cand) < 0:
            cand = -cand
        for z in np.linspace(-thickness, thickness, edge_layers):
            pts.append((hx[i], hy[i], z))
            norms.append((cand[0], cand[1], 0))

    pts = np.array(pts)
    norms = np.array(norms)
    norms = norms / np.linalg.norm(norms, axis=1, keepdims=True)
    return pts, norms


def rotate_y(p, angle):
    """Rotation around a vertical axis - the heart spins like a pendant/medalion."""
    c, s = np.cos(angle), np.sin(angle)
    R = np.array([[c, 0, s],
                  [0, 1, 0],
                  [-s, 0, c]])
    return p @ R.T


SHADES = " .:-=+*#%@"
W, H = 72, 34
# Terminal characters are about 2 times higher than they are wider(typical monospaced
# font ~ 10x20px). Without this correction, the heart stretches verticaly in
# "egg", and the shallow upper notch visually disappears completely.
CHAR_ASPECT = 0.5   # char scale X / char scale Y
SCALE_X = 40
SCALE_Y = SCALE_X * CHAR_ASPECT


def render(pts, norms, angle, light):
    rp = rotate_y(pts, angle)
    rn = rotate_y(norms, angle)

    raw_x = rp[:, 0] * SCALE_X
    raw_y = rp[:, 1] * SCALE_Y
    cx = (raw_x.max() + raw_x.min()) / 2
    cy = (raw_y.max() + raw_y.min()) / 2

    screen = [[' '] * W for _ in range(H)]
    zbuf = [[-999.0] * W for _ in range(H)]

    for (x, y, z), n, rx, ry in zip(rp, rn, raw_x, raw_y):
        sx = int(W / 2 + (rx - cx))
        sy = int(H / 2 - (ry - cy))
        if 0 <= sx < W and 0 <= sy < H and z > zbuf[sy][sx]:
            zbuf[sy][sx] = z
            lum = max(0.0, float(np.dot(n, light)))
            idx = int(lum * (len(SHADES) - 1))
            screen[sy][sx] = SHADES[idx]
    return screen


def colorize(line, r, g, b):
    return f"\033[38;2;{r};{g};{b}m{line}\033[0m"


def main():
    print("Build heart...")
    pts, norms = build_heart()

    light = np.array([0.35, 0.5, 1.0])
    light = light / np.linalg.norm(light)

    angle = 0.0
    try:
        while True:
            frame = render(pts, norms, angle, light)
            clear()
            pulse = int(30 * np.sin(angle * 2))
            r, g, b = 255, 40 + max(0, pulse), 40 + max(0, pulse)
            out = "\n".join(colorize("".join(row), r, g, b) for row in frame)
            print(out)
            time.sleep(0.045)
            angle += 0.07
    except KeyboardInterrupt:
        clear()
        print(colorize("♡".center(W), 255, 60, 60))


if __name__ == "__main__":
    main()
