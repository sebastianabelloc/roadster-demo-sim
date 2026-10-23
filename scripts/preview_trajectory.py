"""Quick look at a trajectory before rendering it in Blender (seconds instead of half an hour).

Usage (from the repo root):
    python scripts/preview_trajectory.py renders/trajectory.npz              # one image: renders/preview.png
    python scripts/preview_trajectory.py renders/trajectory.npz --anim       # also a 2D video: renders/preview.mp4

Side view (x, z) with the lineup to scale, front view (y, z) to see the roll and the lateral drift, and the
time histories of altitude, roll, lateral position and the 6 nozzles. Same file Blender reads, nothing added.
"""
import argparse
import ast
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from roadster.quaternions import rotate_vector  # noqa: E402
from roadster.thrusters import THRUSTER_POSITIONS, THRUSTER_DIRECTIONS  # noqa: E402

CAR = (4.2, 2.0, 1.2)          # box approximation of the Roadster (same as the inertia), m
COM_HEIGHT = 0.62              # CoM above the ground on the wheels (as in render_hop.py)
RUNWAY_HALF = 4.3              # red LED lines, m
HEIGHTS = {"cybercab": 1.45, "cybertruck": 1.9, "semi": 3.9, "modelx": 1.7, "models": 1.45}   # approx., m
NAMES = ["bottom FL", "bottom FR", "bottom RL", "bottom RR", "side R", "side L"]


def lineup_from_scene():
    """Reads LINEUP_START_X, GAP and LINEUP from blender/render_hop.py (without importing bpy)."""
    path = os.path.join(os.path.dirname(__file__), "..", "blender", "render_hop.py")
    values = {}
    for node in ast.parse(open(path).read()).body:
        if isinstance(node, ast.Assign) and node.targets[0].id in ("LINEUP_START_X", "GAP", "LINEUP"):
            values[node.targets[0].id] = ast.literal_eval(node.value)
    boxes, x = [], values["LINEUP_START_X"]
    for name, width in values["LINEUP"]:
        boxes.append((x, width, HEIGHTS.get(name, 1.5)))
        x += width + values["GAP"]
    return boxes


def car_outline(state, axes):
    """Outline of the car box seen along one axis: corners rotated with the quaternion, projected and hulled."""
    L, B, H = CAR
    corners = np.array([[sx * L / 2, sy * B / 2, sz * H / 2] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    world = np.array([rotate_vector(state[6:10], c) for c in corners]) + state[0:3] + [0, 0, COM_HEIGHT]
    pts = world[:, axes]
    # convex hull (gift wrapping, 8 points)
    hull, start = [], np.argmin(pts[:, 0])
    p = start
    while True:
        hull.append(pts[p])
        q = (p + 1) % len(pts)
        for r in range(len(pts)):
            u, v = pts[q] - pts[p], pts[r] - pts[p]
            cross = u[0] * v[1] - u[1] * v[0]
            if cross < 0:
                q = r
        p = q
        if p == start or len(hull) > 8:
            break
    return np.array(hull)


def jets(state, on, axes, length=1.5):
    """Segments for the firing nozzles: from the nozzle along the exhaust (opposite to the force)."""
    out = []
    for i in np.flatnonzero(on):
        start = rotate_vector(state[6:10], THRUSTER_POSITIONS[i]) + state[0:3] + [0, 0, COM_HEIGHT]
        end = start - length * rotate_vector(state[6:10], THRUSTER_DIRECTIONS[i])
        out.append((start[axes], end[axes]))
    return out


def roll_deg(states):
    """Roll angle (deg), unwrapped so the barrel roll reads 0 -> -360."""
    w, x, y, z = states[6:10]
    return np.degrees(np.unwrap(np.arctan2(2 * (w * x + y * z), 1 - 2 * (x**2 + y**2))))


def draw(fig, data, k_now=None):
    """Whole figure. k_now = sample shown as 'now' (animation); None = static summary with ghost cars."""
    t, s, th = data["t"], data["states"], data["thrusters"]
    on = th > 0.10
    fig.clf()
    gs = fig.add_gridspec(4, 3, width_ratios=[1.0, 1.4, 1.4], height_ratios=[1.7, 1, 1, 1],
                          hspace=0.6, wspace=0.25, left=0.05, right=0.98, top=0.95, bottom=0.06)
    side, front = fig.add_subplot(gs[0, :]), fig.add_subplot(gs[1:4, 0])
    plots = [fig.add_subplot(gs[1, 1]), fig.add_subplot(gs[1, 2]), fig.add_subplot(gs[2, 1]), fig.add_subplot(gs[2, 2])]
    raster = fig.add_subplot(gs[3, 1:])

    # side view: ground, lineup, path, cars
    side.axhline(0, color="k", lw=1)
    for x0, w, h in lineup_from_scene():
        side.add_patch(plt.Rectangle((x0, 0), w, h, color="0.75", zorder=1))
    side.plot(s[0], s[2] + COM_HEIGHT, color="tab:blue", lw=1, zorder=2)
    if k_now is None:
        ks = np.searchsorted(t, np.arange(t[0], t[-1], 0.25))
    else:
        ks = [k_now]
    for k in ks:
        alpha = 0.35 if k_now is None else 1.0
        side.fill(*car_outline(s[:, k], [0, 2]).T, color="0.15", alpha=alpha, zorder=3)
        for a, b in jets(s[:, k], on[:, k], [0, 2]):
            side.plot([a[0], b[0]], [a[1], b[1]], color="tab:orange", lw=2, alpha=alpha, zorder=4)
    side.set_aspect("equal")
    side.set_xlim(s[0].min() - 3, s[0].max() + 3)
    side.set_ylim(-0.5, max(12.0, s[2].max() + 3))
    side.set_xlabel("x [m]"); side.set_ylabel("z [m]")
    side.set_title(f"side view   apex {s[2].max():.1f} m   touchdown x {s[0][t <= float(data['t_touchdown'])][-1]:.1f} m")

    # front view: runway lines and the car seen from behind
    k = len(t) - 1 if k_now is None else k_now
    for y in (-RUNWAY_HALF, RUNWAY_HALF):
        front.axvline(y, color="tab:red", lw=1)
    front.axhline(0, color="k", lw=1)
    front.plot(s[1], s[2] + COM_HEIGHT, color="tab:blue", lw=1)
    front.fill(*car_outline(s[:, k], [1, 2]).T, color="0.15")
    for a, b in jets(s[:, k], on[:, k], [1, 2]):
        front.plot([a[0], b[0]], [a[1], b[1]], color="tab:orange", lw=2)
    front.set_aspect("equal")
    front.set_xlim(-6, 6); front.set_ylim(-0.5, max(12.0, s[2].max() + 3))
    front.set_xlabel("y [m]"); front.set_title(f"from behind   max |y| {np.abs(s[1]).max():.2f} m")

    # time histories
    series = [(s[2], "altitude z [m]", "z"), (s[5], "vertical speed vz [m/s]", "vz"),
              (roll_deg(s), "roll angle [deg]", "roll"), (s[1], "lateral offset y [m] (+ = left of the centre line)", "y")]
    for ax, (v, title, label) in zip(plots, series):
        ax.plot(t, v, lw=1.2)
        ax.set_title(title, fontsize=9); ax.set_ylabel(label, fontsize=8)
        ax.tick_params(labelsize=7); ax.grid(alpha=0.3)
        if k_now is not None:
            ax.axvline(t[k_now], color="tab:red", lw=1)

    # nozzles: throttle of each one over time (white = off), resampled every 10 ms (solver steps are uneven)
    grid = np.arange(t[0], t[-1], 0.01)
    idx = np.clip(np.searchsorted(t, grid, side="right") - 1, 0, len(t) - 1)
    raster.imshow(th[:, idx], aspect="auto", cmap="Greys", vmin=0, vmax=1, interpolation="nearest",
                  extent=[t[0], t[-1], len(NAMES) - 0.5, -0.5])
    raster.set_yticks(range(len(NAMES))); raster.set_yticklabels(NAMES, fontsize=7)
    raster.set_xlabel("t [s]", fontsize=8); raster.set_title("nozzle throttle (black = full)", fontsize=9)
    if k_now is not None:
        raster.axvline(t[k_now], color="tab:red", lw=1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("trajectory")
    p.add_argument("--out", default=None, help="output image (default: preview.png next to the trajectory)")
    p.add_argument("--anim", action="store_true", help="also write a 2D preview video (preview.mp4)")
    p.add_argument("--fps", type=int, default=30)
    args = p.parse_args()
    data = np.load(args.trajectory)
    folder = os.path.dirname(args.trajectory) or "."

    fig = plt.figure(figsize=(16, 9))
    draw(fig, data)
    out = args.out or os.path.join(folder, "preview.png")
    fig.savefig(out, dpi=100)
    print("saved", out)

    if args.anim:
        import imageio_ffmpeg
        t = data["t"]
        frames = np.searchsorted(t, np.arange(t[0], t[-1], 1 / args.fps))   # real time
        fig.set_size_inches(12.8, 7.2)
        w, h = fig.canvas.get_width_height()
        video = os.path.join(folder, "preview.mp4")
        writer = imageio_ffmpeg.write_frames(video, (w, h), fps=args.fps, codec="libx264", pix_fmt_out="yuv420p",
                                             output_params=["-crf", "20"])
        writer.send(None)
        for n, k in enumerate(frames):
            draw(fig, data, k_now=k)
            fig.canvas.draw()
            writer.send(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].tobytes())
            print(f"\rframe {n + 1}/{len(frames)}", end="")
        writer.close()
        print("\nsaved", video)


if __name__ == "__main__":
    main()
