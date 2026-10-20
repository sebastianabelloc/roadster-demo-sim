"""Put the minimal HUD on the frames rendered by Blender and encode the final video.

Usage (from the repo root, after rendering the frames):
    python scripts/compose_video.py renders/trajectory.npz renders/frames renders/roadster_hop.mp4

Everything the HUD shows comes from the same trajectory file Blender animated (and the roll reference
from roadster.guidance): altitude, roll angle vs. reference, which thrusters fired, lift.
Needs: pillow, imageio-ffmpeg (pip install pillow imageio-ffmpeg).
"""
import argparse
import glob
import os
import re
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from roadster.guidance import roll_reference  # noqa: E402
from roadster.thrusters import THRUSTER_POSITIONS, THRUSTER_DIRECTIONS  # noqa: E402



M_TO_FT = 3.28084


# ----------------------------------------------------------------------------
# data for each video frame
# ----------------------------------------------------------------------------

def roll_deg(q):
    """Roll angle (rotation about the body x axis) from the quaternion (w, x, y, z)."""
    w, x, y, z = q
    return np.degrees(np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)))


def frame_data(traj, t_frames):
    """Interpolate the trajectory at each frame time. Thrusters count as fired if they were on at any
    controller sample during the frame (same rule as the Blender scene)."""
    t, states = traj["t"], traj["states"]
    on, lifts = traj["thrusters"] > 0.10, traj["lifts"]     # throttles: drawn as firing above 10 %
    rows = []
    for f, tf in enumerate(t_frames):
        s = np.array([np.interp(tf, t, r) for r in states])
        k = int(np.clip(np.searchsorted(t, tf, side="right") - 1, 0, len(t) - 1))
        t_prev = t_frames[f - 1] if f > 0 else tf
        k0 = int(np.clip(np.searchsorted(t, t_prev, side="right") - 1, 0, k))
        rows.append(dict(t=tf, z=s[2], vx=s[3], vz=s[5], q=s[6:10] / np.linalg.norm(s[6:10]),
                         fired=on[:, k0:k + 1].any(axis=1), lift=float(lifts[k])))
    # continuous roll angle (0 -> -360 instead of jumping at +-180)
    roll = np.degrees(np.unwrap(np.radians([roll_deg(r["q"]) for r in rows])))
    for r, phi in zip(rows, roll):
        r["roll"] = phi
        ref = roll_reference(r["t"])[0] if r["t"] >= 0 else 0.0
        r["roll_ref"] = np.degrees(ref)
    return rows


# ----------------------------------------------------------------------------
# look
# ----------------------------------------------------------------------------

def load_font(size, weight="Regular"):
    """Helvetica Neue in the given weight (macOS), else DejaVu (Linux), else PIL's default."""
    ttc = "/System/Library/Fonts/HelveticaNeue.ttc"
    if os.path.exists(ttc):
        for idx in range(16):
            try:
                font = ImageFont.truetype(ttc, size, index=idx)
            except OSError:
                break
            if font.getname()[1] == weight:
                return font
        return ImageFont.truetype(ttc, size, index=0)
    bold = weight in ("Medium", "Bold")
    dejavu = "/usr/share/fonts/truetype/dejavu/DejaVuSans" + ("-Bold" if bold else "") + ".ttf"
    if os.path.exists(dejavu):
        return ImageFont.truetype(dejavu, size)
    return ImageFont.load_default()


def post_fx(img, vignette=0.35, rng=np.random.default_rng(1)):
    """Light finish on top of the EEVEE render: a soft vignette."""
    x = img.astype(np.float32) / 255.0
    h, w = x.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.sqrt(((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2)
    x *= (1 - vignette * np.clip(r - 0.45, 0, None) ** 1.6)[..., None]
    # darker band at the bottom so the telemetry reads cleanly over the stage lights
    band = np.clip((yy[:, :1] - 0.72 * h) / (0.28 * h), 0, 1) ** 1.5
    x *= (1 - 0.55 * band)[..., None]
    return (np.clip(x, 0, 1) * 255).astype(np.uint8)


class Hud:
    """Minimal webcast-style overlay: speed and altitude on the left, mission clock and event timeline in
    the centre, thruster map and roll angle on the right. White only, no panels.
    Everything is drawn at SS x the video resolution and scaled down, so lines, circles and text are
    anti-aliased and placed with sub-pixel accuracy."""

    SS = 4
    WHITE, DIM, FAINT = (255, 255, 255, 245), (255, 255, 255, 150), (255, 255, 255, 80)

    def __init__(self, size, events):
        self.w, self.h = size
        self.u = self.h / 1080 * self.SS              # 1 unit = 1 px of a 1080p frame, in supersampled px
        u = self.u
        self.f_label = load_font(round(15 * u), "Medium")
        self.f_value = load_font(round(40 * u), "Light")
        self.f_unit = load_font(round(15 * u), "Regular")
        self.f_clock = load_font(round(30 * u), "Light")
        self.f_event = load_font(round(13 * u), "Medium")
        self.events = events                          # [(time, "NAME"), ...]

    # -- primitives -----------------------------------------------------------
    def text(self, d, xy, s, font, fill, tracking=0.0, anchor="l"):
        """Text with letter spacing (tracking in em); anchor l / c / r on x, y = top of the cap height."""
        sp = tracking * font.size
        width = sum(d.textlength(ch, font=font) for ch in s) + sp * max(len(s) - 1, 0)
        x, y = xy
        x = x - width / 2 if anchor == "c" else x - width if anchor == "r" else x
        for ch in s:
            d.text((x, y), ch, font=font, fill=fill, anchor="ls")
            x += d.textlength(ch, font=font) + sp
        return width

    @staticmethod
    def circle(d, c, r, **kw):
        d.ellipse([c[0] - r, c[1] - r, c[0] + r, c[1] + r], **kw)

    def readout(self, d, x, y, label, value, unit, anchor="l"):
        """Small spaced label with a big light value under it (y = label baseline)."""
        u = self.u
        self.text(d, (x, y), label, self.f_label, self.DIM, tracking=0.2, anchor=anchor)
        if anchor == "l":
            wv = self.text(d, (x - 2 * u, y + 44 * u), value, self.f_value, self.WHITE)
            if unit:
                self.text(d, (x + wv + 6 * u, y + 44 * u), unit, self.f_unit, self.DIM, tracking=0.15)
        else:
            wu = self.text(d, (x, y + 44 * u), unit, self.f_unit, self.DIM, tracking=0.15, anchor="r") if unit else 0
            self.text(d, (x - wu - (6 * u if unit else 0), y + 44 * u), value, self.f_value, self.WHITE, anchor="r")

    # -- thruster map -----------------------------------------------------------
    def thruster_map(self, d, glow, cx, cy, r):
        """The car seen from above (nose up) with every thruster at its mounting point from
        roadster/thrusters.py. Firing thrusters light up, showing where the gas goes:
          up / down (out of / into the screen): circle with a dot / a cross;
          sideways: a small jet leaving the hull on that side."""
        u = self.u
        s = 40 * u                                     # px per metre
        P = lambda x, y: (cx - y * s, cy - x * s)       # body (x fwd, y left) -> screen
        lw = 1.6 * u

        # hull: rounded top-view silhouette, slightly narrower at the nose
        xs = np.linspace(-2.1, 2.1, 121)
        half = 1.0 * (1 - np.abs(xs / 2.1) ** 5) ** (1 / 5) * (1 - 0.10 * np.clip(xs, 0, None) / 2.1)
        outline = [P(x, y) for x, y in zip(xs, half)] + [P(x, -y) for x, y in zip(xs[::-1], half[::-1])]
        d.line(outline + [outline[0]], fill=self.FAINT, width=round(lw), joint="curve")

        def updown(c, rad, exhaust):
            if exhaust is None:
                self.circle(d, c, rad, outline=self.FAINT, width=round(lw))
                return
            self.circle(glow, c, rad * 1.9, fill=(255, 255, 255, 110))
            self.circle(d, c, rad, outline=self.WHITE, width=round(lw * 1.2))
            if exhaust[2] > 0:                          # out of the screen: dot
                self.circle(d, c, rad * 0.34, fill=self.WHITE)
            else:                                       # into the screen: cross
                k = rad * 0.70 / np.sqrt(2)
                d.line([(c[0] - k, c[1] - k), (c[0] + k, c[1] + k)], fill=self.WHITE, width=round(lw * 1.2))
                d.line([(c[0] - k, c[1] + k), (c[0] + k, c[1] - k)], fill=self.WHITE, width=round(lw * 1.2))

        def side_jet(x, y_sign, firing):
            """Mount on the hull edge; when firing, a tapered jet leaves the hull sideways."""
            hw = np.interp(x, xs, half)
            base = np.array(P(x, y_sign * hw))
            out = np.array([-y_sign, 0.0])               # +y (left) is -x on screen
            along = np.array([0.0, 1.0])
            if not firing:
                self.circle(d, base, 2.2 * u, fill=self.FAINT)
                return
            L, W = 22 * u, 6 * u
            tip = base + out * L
            poly = [tuple(base + along * W), tuple(base - along * W), tuple(tip)]
            glow.polygon([tuple(base + along * W * 1.8), tuple(base - along * W * 1.8), tuple(tip + out * 6 * u)],
                         fill=(255, 255, 255, 110))
            d.polygon(poly, fill=self.WHITE)
            self.circle(d, base, 2.6 * u, fill=self.WHITE)

        # every nozzle at its position from roadster/thrusters.py: the 4 bottom ones blow down (cross),
        # the 2 side ones blow sideways (a jet leaving the hull)
        for i, (pos, f_dir) in enumerate(zip(THRUSTER_POSITIONS, THRUSTER_DIRECTIONS)):
            ex = -np.asarray(f_dir, float)
            fired = bool(r["fired"][i])
            if abs(ex[2]) > 0.5:
                # keep the symbol inside the hull outline
                x = np.sign(pos[0]) * min(abs(pos[0]), 1.72)
                y = np.sign(pos[1]) * min(abs(pos[1]), 0.55)
                updown(P(x, y), 10.5 * u, ex if fired else None)
            else:
                side_jet(float(pos[0]), int(np.sign(ex[1])), fired)

    # -- frame ------------------------------------------------------------------
    def draw(self, img, r):
        from PIL import ImageFilter
        SS, u = self.SS, self.u
        W, H = self.w * SS, self.h * SS
        layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        glow_l = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        d, g = ImageDraw.Draw(layer), ImageDraw.Draw(glow_l)
        m = 64 * u
        y0 = H - m - 46 * u                              # baseline of the lowest readout label

        # left: speed and altitude
        speed = np.hypot(r["vx"], r["vz"]) * 3.6
        self.readout(d, m, y0 - 84 * u, "SPEED", f"{speed:.0f}", "KM/H")
        self.readout(d, m, y0, "ALTITUDE", f"{max(r['z'], 0.0):.1f}", "M")

        # right: thruster map, roll angle to its left
        map_cx = W - m - 44 * u
        self.thruster_map(d, g, map_cx, y0 - 58 * u, r)
        self.readout(d, map_cx - 90 * u, y0, "ROLL", f"{r['roll']:.0f}°", "", anchor="r")

        # centre: mission clock and event timeline
        t = r["t"]
        sec = abs(t)
        clock = f"T{'+' if t >= 0 else '−'} 00:00:{int(sec):02d}.{int((sec % 1) * 10)}"
        self.text(d, (W / 2, y0 + 12 * u), clock, self.f_clock, self.WHITE, tracking=0.08, anchor="c")
        x0, x1, yl = W / 2 - 330 * u, W / 2 + 330 * u, y0 + 40 * u
        times = np.array([te for te, _ in self.events])
        slots = np.linspace(0.08, 0.92, len(times))
        X = lambda tt: x0 + (x1 - x0) * np.interp(tt, np.r_[times[0] - 1.5, times, times[-1] + 1.5],
                                                  np.r_[0.0, slots, 1.0])
        d.line([(x0, yl), (x1, yl)], fill=self.FAINT, width=round(1.5 * u))
        d.line([(x0, yl), (X(t), yl)], fill=self.WHITE, width=round(1.5 * u))
        for te, name in self.events:
            c = (X(te), yl)
            done = t >= te
            self.circle(d, c, 4.5 * u, fill=self.WHITE if done else (0, 0, 0, 255), outline=self.WHITE,
                        width=round(1.5 * u))
            self.text(d, (c[0], yl + 26 * u), name, self.f_event, self.WHITE if done else self.DIM,
                      tracking=0.18, anchor="c")

        glow_l = glow_l.filter(ImageFilter.GaussianBlur(7 * u))
        hud = Image.alpha_composite(glow_l, layer).resize((self.w, self.h), Image.LANCZOS)
        base = Image.fromarray(img).convert("RGBA")
        return np.array(Image.alpha_composite(base, hud).convert("RGB"))


# ----------------------------------------------------------------------------
# encode
# ----------------------------------------------------------------------------

def ffmpeg_exe():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return "ffmpeg"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("trajectory")
    p.add_argument("frames_dir")
    p.add_argument("out")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--crf", type=int, default=18)
    a = p.parse_args()

    traj = np.load(a.trajectory)
    t_frames = np.load(os.path.join(a.frames_dir, "frame_times.npy"))
    files = sorted(glob.glob(os.path.join(a.frames_dir, "frame_*.png")))
    if not files:
        sys.exit(f"no frame_*.png in {a.frames_dir}: render them with blender/render_hop.py first")
    numbers = [int(re.search(r"frame_(\d+)\.png", f).group(1)) for f in files]
    rows = frame_data(traj, t_frames)
    w, h = Image.open(files[0]).size
    t_land = float(traj["t_touchdown"])
    t, z = traj["t"], traj["states"][2]
    t_hover = float(t[np.argmax((t > 2.48) & (z <= 0.55))])
    events = [(0.0, "LIFTOFF"), (0.48, "ROLL"), (2.48, "BRAKE"), (t_hover, "HOVER"), (t_land, "TOUCHDOWN")]
    hud = Hud((w, h), events)

    cmd = [ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
           "-r", str(a.fps), "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", str(a.crf),
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", a.out]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    enc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for i, (f, n) in enumerate(zip(files, numbers)):
        img = np.array(Image.open(f).convert("RGB"))
        img = hud.draw(post_fx(img), rows[n - 1])
        enc.stdin.write(img.tobytes())
        if i % 30 == 0:
            print(f"  frame {n}/{len(t_frames)}")
    enc.stdin.close()
    enc.wait()
    print(f"saved {len(files)} frames ({len(files) / a.fps:.1f} s) -> {a.out}")


if __name__ == "__main__":
    main()
