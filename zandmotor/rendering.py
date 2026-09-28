"""Raster-to-PNG rendering helpers: terrain colouring, data URIs, check images."""

from __future__ import annotations

import base64
import io
import math

import numpy as np
from PIL import Image
from pyproj import Transformer

from zandmotor.config import CFG, log


def terrain_rgb(z):
    """Colour the terrain: sea, wet sand, beach, dunes, with hillshade."""
    stops = [(-8, (52, 98, 128)), (-2, (86, 138, 166)), (-0.4, (140, 184, 200)),
             (0.3, (214, 203, 170)), (2.0, (233, 222, 190)), (4.0, (214, 208, 164)),
             (6.0, (166, 178, 132)), (12.0, (122, 142, 104))]
    zs = [a for a, _ in stops]
    rgb = np.stack([np.interp(z, zs, [c[i] for _, c in stops]) for i in range(3)], -1)
    gy, gx = np.gradient(z, CFG["grid_res_m"])
    # sun from the north-west, 45 degrees high
    slope = np.arctan(np.hypot(gx, gy) * 2.0)
    aspect = np.arctan2(-gx, gy)
    az, alt = math.radians(315), math.radians(45)
    shade = np.sin(alt) * np.cos(slope) + np.cos(alt) * np.sin(slope) * np.cos(az - aspect)
    shade = np.clip(0.55 + 0.6 * shade, 0.55, 1.15)
    land = z > 0.3
    rgb[land] *= shade[land, None]
    return np.clip(rgb, 0, 255).astype("uint8")


def png_uri(rgb_or_rgba):
    mode = "RGBA" if rgb_or_rgba.shape[-1] == 4 else "RGB"
    buf = io.BytesIO()
    Image.fromarray(rgb_or_rgba, mode).save(buf, "PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def outline_preview(rgb, ring, grid, title, path):
    """North-up check image: background, outline, north arrow, scale bar."""
    from PIL import ImageDraw, ImageFont
    scale = max(1, int(math.ceil(1100 / grid.width)))
    img = Image.fromarray(rgb, "RGB").resize((grid.width * scale, grid.height * scale),
                                             Image.NEAREST if scale > 1 else Image.BILINEAR)
    d = ImageDraw.Draw(img)
    to_merc = Transformer.from_crs(4326, 3857, always_xy=True)
    inv = ~grid.transform
    pts = []
    for lon, lat in ring:
        c, r = inv * to_merc.transform(lon, lat)
        pts.append((c * scale, r * scale))
    d.line(pts, fill=(240, 127, 30), width=4, joint="curve")
    try:
        font = ImageFont.load_default(size=22)
        small = ImageFont.load_default(size=16)
    except TypeError:
        font = small = ImageFont.load_default()
    W, H = img.size
    # north arrow
    ax, ay = W - 50, 30
    d.polygon([(ax, ay), (ax - 14, ay + 40), (ax, ay + 30), (ax + 14, ay + 40)],
              fill=(16, 41, 58), outline=(255, 255, 255))
    d.text((ax - 7, ay + 44), "N", fill=(16, 41, 58), font=font,
           stroke_width=3, stroke_fill=(255, 255, 255))
    # scale bar 500 m
    px = 500 / CFG["grid_res_m"] * scale
    x0, y0 = 30, H - 40
    d.rectangle([x0 - 6, y0 - 26, x0 + px + 50, y0 + 14], fill=(255, 255, 255))
    d.rectangle([x0, y0, x0 + px, y0 + 6], fill=(16, 41, 58))
    d.text((x0, y0 - 22), "500 m", fill=(16, 41, 58), font=small)
    d.rectangle([0, 0, W, 34], fill=(255, 255, 255))
    d.text((12, 6), title, fill=(16, 41, 58), font=small)
    img.save(path)
    log.info("Written %s", path.name)
