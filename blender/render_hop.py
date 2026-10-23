"""Blender scene for the Roadster hop. Reads the trajectory exported by the simulation and renders PNG frames.

Usage (from the repo root):
    blender -b -P blender/render_hop.py -- trajectory.npz renders/frames --res 1920 1080 --engine EEVEE

Everything the car does comes from the trajectory file: position, attitude and the throttle of each of the
6 nozzles (roadster/thrusters.py). Blender only draws it.
"""
import argparse
import re
import math
import os
import sys

import bpy
import numpy as np
from mathutils import Quaternion, Vector

HERE = os.path.dirname(os.path.abspath(__file__))
REAL = os.path.join(HERE, "assets_real")   # normalized downloaded models (normalize_model.py)

COM_HEIGHT = 0.62      # m, CoM above the ground when the car sits on its wheels (visual offset only)
LINEUP_START_X = 5.5   # m
PLATFORM_H = 0.15      # m, low stage the lineup stands on (like the demo)
GAP = 1.0              # m
LINEUP = [             # (model in assets_real/, width across the runway in m)
    ("cybercab", 1.97), ("cybertruck", 1.97), ("semi", 2.59), ("modelx", 2.26), ("models", 2.17),
]
# colours of the demo lineup; each model's own material names -> role (anything unmatched = dark trim)
LINEUP_STYLE = {
    # one mesh with a baked texture: light texels are bodywork (gold), dark texels windows and tyres (black)
    "cybercab": {"paint": ("#c8a165", 0.9, 0.28, 1.0), "mask": "CabColor"},
    "cybertruck": {"paint": ("#a9adb2", 1.0, 0.32, 0.0),
                   "map": {r"^Material$": "paint", r"^Material\.001$": "glass",
                           r"^Material\.00[45]$": "light", r"^Material\.007$": "rims"}},
    "semi": {"paint": ("#ecebe7", 0.0, 0.18, 1.0),
             "map": {r"^Material$": "paint", r"^Glass": "glass", r"^Disk": "rims", r"^Lights": "light"}},
    "modelx": {"paint": ("#9e0f18", 0.55, 0.22, 1.0),
               "map": {r"^(Paint_Color|metal_white)$": "paint"}, "default": "glass"},
    "models": {"paint": ("#14244f", 0.6, 0.22, 1.0),
               "map": {r"^bod$": "paint", r"^(Rims|chr)$": "rims", r"^(glass|glack)$": "glass",
                       r"^emit$": "light", r"^red$": "tail"}},
}
# Roadster wheels (x, radius, outer face y) measured on the model; the baked ones are covered by real wheels
ROADSTER_WHEELS = [(1.26, 0.31, 0.915), (-1.60, 0.345, 0.95)]
# thrusters: the same table the simulation uses (roadster/thrusters.py), exhaust = -force direction
sys.path.insert(0, os.path.join(HERE, ".."))
from roadster.thrusters import THRUSTER_POSITIONS as THR_POS, THRUSTER_DIRECTIONS as THR_DIR  # noqa: E402


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------

def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    p = argparse.ArgumentParser()
    p.add_argument("trajectory")
    p.add_argument("out_dir")
    p.add_argument("--res", nargs=2, type=int, default=[1920, 1080])
    p.add_argument("--engine", default="EEVEE", choices=["EEVEE", "CYCLES"])
    p.add_argument("--samples", type=int, default=64)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--frames", nargs=2, type=int, default=None, help="render only this frame range")
    p.add_argument("--every", type=int, default=1, help="render every n-th frame (previews)")
    p.add_argument("--slowmo", type=float, default=1.0,
                   help="playback speed in the air, e.g. 0.4 = slow motion (default 1 = real time)")
    p.add_argument("--save", default=None, help="also save the animated scene as a .blend to open in Blender")
    p.add_argument("--no-render", action="store_true", help="only build (and save) the scene, render nothing")
    p.add_argument("--fog", type=float, default=0.0, help="volumetric fog density (0 = off)")
    return p.parse_args(argv)


def speed_profile(t, t_brake=2.48, t_touchdown=6.15, slowmo=1.0):
    """Playback speed (1 = real time). With slowmo < 1 the flight plays in slow motion (and a bit faster during
    the hover). Only the video timing changes: the car's motion is the simulation's either way."""
    if slowmo >= 1.0:
        return np.ones_like(t)
    def smooth(a, b, x):
        u = np.clip((x - a) / (b - a), 0, 1)
        return u * u * (3 - 2 * u)
    air = smooth(-0.35, 0.05, t) * (1 - smooth(t_touchdown, t_touchdown + 0.4, t))
    hover = smooth(t_brake + 0.6, t_brake + 1.2, t)
    slow_hover = min(1.0, slowmo * 1.875)
    slow = slowmo + (slow_hover - slowmo) * hover
    return 1.0 + (slow - 1.0) * air


def frame_times(t_sim, fps, t_touchdown, slowmo=1.0):
    """Simulation time shown at each video frame, with the variable playback speed."""
    fine = np.linspace(t_sim[0], t_sim[-1], 20000)
    video = np.concatenate([[0.0], np.cumsum(np.diff(fine) / speed_profile(fine[:-1], t_touchdown=t_touchdown, slowmo=slowmo))])
    n_frames = int(video[-1] * fps) + 1
    return np.interp(np.arange(n_frames) / fps, video, fine)


def material(name, color, metallic=0.0, roughness=0.5, coat=0.0, emission=None, strength=0.0):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    bsdf = m.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (*color, 1)
    bsdf.inputs["Metallic"].default_value = metallic
    bsdf.inputs["Roughness"].default_value = roughness
    if coat:
        bsdf.inputs["Coat Weight"].default_value = coat
    if emission is not None:
        bsdf.inputs["Emission Color"].default_value = (*emission, 1)
        bsdf.inputs["Emission Strength"].default_value = strength
    return m


def hex_rgb(h):
    h = h.lstrip("#")
    srgb = [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    return tuple(c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in srgb)  # to linear


def append_asset(name):
    """Bring in every object of assets_real/<name>.blend; returns its ROOT empty (or None if missing)."""
    path = os.path.join(REAL, f"{name}.blend")
    if not os.path.exists(path):
        return None
    with bpy.data.libraries.load(path, link=False) as (src, dst):
        dst.objects = src.objects
    root = None
    for o in dst.objects:
        bpy.context.scene.collection.objects.link(o)
        if o.name.startswith("ROOT"):
            root = o
    return root


def style_vehicle(root, style, shared):
    """Repaint a lineup vehicle: its own materials are mapped to paint / glass / trim / rims / lights."""
    if style.get("keep"):
        return
    hexcol, metallic, rough, coat = style["paint"]
    paint = material(f"Paint {root.name}", hex_rgb(hexcol), metallic=metallic, roughness=rough, coat=coat)
    if style.get("mask"):
        image = next((im for im in bpy.data.images if im.name.startswith(style["mask"])), None)
        nt = paint.node_tree
        body = nt.nodes["Principled BSDF"]
        dark = nt.nodes.new("ShaderNodeBsdfPrincipled")
        dark.inputs["Base Color"].default_value = (0.008, 0.008, 0.01, 1)
        dark.inputs["Roughness"].default_value = 0.08
        dark.inputs["Coat Weight"].default_value = 1.0
        mix = nt.nodes.new("ShaderNodeMixShader")
        out = nt.nodes["Material Output"]
        if image is not None:
            tex = nt.nodes.new("ShaderNodeTexImage")
            tex.image = image
            bw = nt.nodes.new("ShaderNodeRGBToBW")
            ramp = nt.nodes.new("ShaderNodeMapRange")
            ramp.interpolation_type = "SMOOTHSTEP"
            ramp.inputs["From Min"].default_value = 0.30
            ramp.inputs["From Max"].default_value = 0.45
            nt.links.new(tex.outputs["Color"], bw.inputs[0])
            nt.links.new(bw.outputs[0], ramp.inputs["Value"])
            nt.links.new(ramp.outputs["Result"], mix.inputs["Fac"])
        nt.links.new(dark.outputs[0], mix.inputs[1])
        nt.links.new(body.outputs[0], mix.inputs[2])
        nt.links.new(mix.outputs[0], out.inputs["Surface"])
        for o in root.children:
            for i in range(len(o.data.materials)):
                o.data.materials[i] = paint
        return
    roles = dict(shared, paint=paint)
    for o in root.children:
        for i, m in enumerate(o.data.materials):
            name = m.name if m else ""
            role = None
            # exact name first; then without the ".001" suffix Blender adds when names collide on append
            for candidate in (name, re.sub(r"\.\d{3}$", "", name)):
                for pattern, r in style.get("map", {}).items():
                    if re.search(pattern, candidate):
                        role = r
                        break
                if role:
                    break
            role = role or style.get("default", "trim")
            o.data.materials[i] = roles[role]


def smoke_jet_material(name, opacity=0.8, glow=0.35):
    """White cold-gas jet with a smoky, turbulent look. Surface + transparency (not a volume) so every jet
    is crisp and the same size in EEVEE and Cycles. Turbulence = 4D noise whose 4th coordinate follows the frame."""
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    if hasattr(m, "surface_render_method"):
        m.surface_render_method = "BLENDED"
    m.use_backface_culling = False
    nt = m.node_tree
    nt.nodes.clear()
    N, L = nt.nodes.new, nt.links.new

    def math(op, a, b=None, clamp=False):
        n = N("ShaderNodeMath"); n.operation = op; n.use_clamp = clamp
        for i, v in enumerate((a, b)):
            if v is None:
                continue
            if isinstance(v, (int, float)):
                n.inputs[i].default_value = v
            else:
                L(v, n.inputs[i])
        return n.outputs[0]

    out = N("ShaderNodeOutputMaterial")
    tc = N("ShaderNodeTexCoord")
    sep = N("ShaderNodeSeparateXYZ")
    L(tc.outputs["Generated"], sep.inputs[0])
    # along the jet: solid for most of its length, fading over the last 40 %
    axial = N("ShaderNodeMapRange")
    axial.interpolation_type = "SMOOTHERSTEP"
    axial.inputs["From Min"].default_value = 0.3
    axial.inputs["From Max"].default_value = 1.0
    axial.inputs["To Min"].default_value = 1.0
    axial.inputs["To Max"].default_value = 0.0
    L(sep.outputs["Z"], axial.inputs["Value"])
    # soft silhouette
    lw = N("ShaderNodeLayerWeight"); lw.inputs["Blend"].default_value = 0.5
    edge = math("POWER", math("SUBTRACT", 1.0, lw.outputs["Facing"]), 1.6)
    # turbulence
    noise = N("ShaderNodeTexNoise")
    noise.noise_dimensions = "4D"
    noise.inputs["Scale"].default_value = 3.0
    noise.inputs["Detail"].default_value = 8.0
    noise.inputs["Roughness"].default_value = 0.6
    noise.inputs["Distortion"].default_value = 1.2
    L(tc.outputs["Object"], noise.inputs["Vector"])
    drv = noise.inputs["W"].driver_add("default_value").driver
    drv.expression = "frame / 6"
    wisp = N("ShaderNodeMapRange")
    wisp.inputs["From Min"].default_value = 0.35
    wisp.inputs["From Max"].default_value = 0.65
    wisp.inputs["To Min"].default_value = 0.12
    wisp.inputs["To Max"].default_value = 1.0
    L(noise.outputs["Fac"], wisp.inputs["Value"])
    alpha = math("MULTIPLY", math("MULTIPLY", axial.outputs["Result"], edge), wisp.outputs["Result"])
    alpha = math("MULTIPLY", alpha, opacity, clamp=True)
    smoke = N("ShaderNodeBsdfPrincipled")
    smoke.inputs["Base Color"].default_value = (0.92, 0.94, 0.97, 1)
    smoke.inputs["Roughness"].default_value = 1.0
    smoke.inputs["Emission Color"].default_value = (0.9, 0.94, 1.0, 1)
    smoke.inputs["Emission Strength"].default_value = glow
    transp = N("ShaderNodeBsdfTransparent")
    mix = N("ShaderNodeMixShader")
    L(alpha, mix.inputs["Fac"])
    L(transp.outputs[0], mix.inputs[1]); L(smoke.outputs[0], mix.inputs[2])
    L(mix.outputs[0], out.inputs["Surface"])
    return m


def make_plume(name, mat, length, radius, parent, position, direction):
    """Cone with its base at the nozzle, pointing along `direction` (body frame), parented to the car."""
    bpy.ops.mesh.primitive_cone_add(vertices=48, radius1=radius * 0.15, radius2=radius, depth=length, end_fill_type="NOTHING",
                                    location=(0, 0, 0))
    cone = bpy.context.active_object
    cone.name = name
    cone.data.materials.append(mat)
    # move geometry so the narrow end (radius1, at -depth/2) sits at the origin and the cone grows along +z
    for v in cone.data.vertices:
        v.co.z += length / 2
    cone.parent = parent
    d = Vector(direction).normalized()
    cone.rotation_mode = "QUATERNION"
    cone.rotation_quaternion = Vector((0, 0, 1)).rotation_difference(d)
    cone.location = Vector(position) + d * 0.05
    return cone


def add_wheels(parent, tyre, rim, spoke):
    """Tyre + dark aero rim + 5 spokes on each wheel of the Roadster. Returns the wheel empties and radii."""
    wheels = []
    for x, r, y_out in ROADSTER_WHEELS:
        for side in (1, -1):
            hub = bpy.data.objects.new("Wheel", None)
            bpy.context.scene.collection.objects.link(hub)
            hub.parent = parent
            hub.location = (x, side * (y_out - 0.13), r)
            hub.rotation_mode = "XYZ"
            parts = []
            bpy.ops.mesh.primitive_cylinder_add(vertices=64, radius=r + 0.012, depth=0.26)
            parts.append((bpy.context.active_object, tyre, 0.0))
            bpy.ops.mesh.primitive_cylinder_add(vertices=64, radius=0.74 * r, depth=0.012)
            parts.append((bpy.context.active_object, rim, side * 0.133))
            for k in range(5):
                bpy.ops.mesh.primitive_cube_add(size=1)
                sp = bpy.context.active_object
                sp.scale = (0.68 * r, 0.10 * r, 0.01)
                bpy.ops.object.transform_apply(scale=True)
                for v in sp.data.vertices:          # from the hub outwards, then turned into place
                    v.co.x += 0.34 * r
                sp.rotation_euler = (0, 0, 2 * math.pi * k / 5)
                bpy.ops.object.transform_apply(rotation=True)
                parts.append((sp, spoke, side * 0.141))
            for obj, mat, dy in parts:
                obj.data.materials.append(mat)
                for poly in obj.data.polygons:
                    poly.use_smooth = True
                obj.parent = hub
                obj.rotation_euler = (math.pi / 2, 0, 0)     # cylinder axis (z) -> axle (y)
                obj.location = (0, dy, 0)
            wheels.append((hub, r))
    return wheels


def gloss_up(root):
    """The Roadster keeps its own colours (baked texture); only the finish changes: clear-coated paint."""
    for o in root.children:
        for m in o.data.materials:
            if m and m.node_tree and "Principled BSDF" in m.node_tree.nodes:
                b = m.node_tree.nodes["Principled BSDF"]
                b.inputs["Roughness"].default_value = 0.22
                b.inputs["Coat Weight"].default_value = 1.0
                b.inputs["Coat Roughness"].default_value = 0.04


# ----------------------------------------------------------------------------
# scene
# ----------------------------------------------------------------------------

def build_stage(x0, x1, led):
    """Low dark stage under the lineup, outlined by a red LED frame, and red LED lines along the runway."""
    half_y = 3.4
    stage = material("Stage", hex_rgb("#141519"), metallic=0.3, roughness=0.3, coat=0.5)
    bpy.ops.mesh.primitive_cube_add(size=1, location=((x0 + x1) / 2, 0, PLATFORM_H / 2))
    deck = bpy.context.active_object
    deck.name = "Stage"
    deck.scale = (x1 - x0, 2 * half_y, PLATFORM_H)
    deck.data.materials.append(stage)
    w = 0.06   # LED bar thickness
    bars = [((x0 + x1) / 2, half_y + w / 2, x1 - x0 + 2 * w, w), ((x0 + x1) / 2, -half_y - w / 2, x1 - x0 + 2 * w, w),
            (x0 - w / 2, 0, w, 2 * half_y), (x1 + w / 2, 0, w, 2 * half_y)]
    for i, (cx, cy, sx, sy) in enumerate(bars):
        for z, h in ((PLATFORM_H - 0.02, 0.04), (0.02, 0.04)):        # top edge and foot of the stage
            bpy.ops.mesh.primitive_cube_add(size=1, location=(cx, cy, z))
            bar = bpy.context.active_object
            bar.name = f"LED frame {i}"
            bar.scale = (sx, sy, h)
            bar.data.materials.append(led)
    # thin red lines on the floor along both sides of the runway
    floor_led = material("LED floor", (0.2, 0, 0), emission=hex_rgb("#ff1e2d"), strength=8)
    for y in (half_y + 0.9, -half_y - 0.9):
        bpy.ops.mesh.primitive_cube_add(size=1, location=(10, y, 0.005))
        line = bpy.context.active_object
        line.name = "LED runway"
        line.scale = (90, 0.035, 0.01)
        line.data.materials.append(floor_led)


def build_backdrop():
    """Low haze on the floor and a thin volume behind the stage so the blue beams show."""
    # haze: a flat volume on the floor, densest at the bottom, wispy
    m = bpy.data.materials.new("Haze")
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    vol = nt.nodes.new("ShaderNodeVolumePrincipled")
    gen = nt.nodes.new("ShaderNodeTexCoord")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(gen.outputs["Generated"], sep.inputs[0])
    fall = nt.nodes.new("ShaderNodeMath"); fall.operation = "POWER"
    inv = nt.nodes.new("ShaderNodeMath"); inv.operation = "SUBTRACT"; inv.inputs[0].default_value = 1.0
    nt.links.new(sep.outputs["Z"], inv.inputs[1])
    nt.links.new(inv.outputs[0], fall.inputs[0]); fall.inputs[1].default_value = 3.0
    noise = nt.nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 0.35
    noise.inputs["Detail"].default_value = 6
    nt.links.new(nt.nodes.new("ShaderNodeTexCoord").outputs["Object"], noise.inputs["Vector"])
    ramp = nt.nodes.new("ShaderNodeMapRange")
    ramp.inputs["From Min"].default_value = 0.4
    ramp.inputs["From Max"].default_value = 0.75
    nt.links.new(noise.outputs["Fac"], ramp.inputs["Value"])
    mul = nt.nodes.new("ShaderNodeMath"); mul.operation = "MULTIPLY"
    nt.links.new(fall.outputs[0], mul.inputs[0]); nt.links.new(ramp.outputs["Result"], mul.inputs[1])
    dens = nt.nodes.new("ShaderNodeMath"); dens.operation = "MULTIPLY"; dens.inputs[1].default_value = 0.08
    nt.links.new(mul.outputs[0], dens.inputs[0])
    nt.links.new(dens.outputs[0], vol.inputs["Density"])
    vol.inputs["Color"].default_value = (0.9, 0.92, 1.0, 1)
    nt.links.new(vol.outputs[0], out.inputs["Volume"])
    bpy.ops.mesh.primitive_cube_add(size=1, location=(24, -11, 0.6))
    haze = bpy.context.active_object
    haze.name = "Haze"
    haze.scale = (30, 10, 1.2)
    haze.data.materials.append(m)
    beam = bpy.data.materials.new("Beam air")
    beam.use_nodes = True
    bn = beam.node_tree
    bn.nodes.clear()
    bv = bn.nodes.new("ShaderNodeVolumePrincipled")
    bv.inputs["Density"].default_value = 0.008
    bn.links.new(bv.outputs[0], bn.nodes.new("ShaderNodeOutputMaterial").inputs["Volume"])
    bpy.ops.mesh.primitive_cube_add(size=1, location=(20, -14, 15))   # wider than the camera ever sees: no edges
    air = bpy.context.active_object
    air.name = "Beam air"
    air.scale = (160, 24, 30)
    air.data.materials.append(beam)


def build_scene(args):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    sc.unit_settings.system = "METRIC"
    sc.render.resolution_x, sc.render.resolution_y = args.res
    sc.render.fps = args.fps
    sc.render.image_settings.file_format = "PNG"
    sc.view_settings.view_transform = "AgX"
    sc.view_settings.look = "AgX - Medium High Contrast"
    sc.render.use_motion_blur = True             # real cameras smear fast motion (wheels, jets)
    sc.render.motion_blur_shutter = 0.5
    if args.engine == "CYCLES":
        sc.render.engine = "CYCLES"
        sc.cycles.samples = args.samples
        sc.cycles.use_denoising = True
        sc.cycles.device = "GPU" if sys.platform == "darwin" else "CPU"
    else:
        sc.render.engine = "BLENDER_EEVEE"
        sc.eevee.taa_render_samples = args.samples
        sc.eevee.use_shadows = True
        sc.eevee.volumetric_tile_size = "2"
        sc.eevee.volumetric_samples = 96
        sc.eevee.volumetric_end = 150

    # world: night sky + thin fog for light shafts
    world = bpy.data.worlds.new("Night")
    sc.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes["Background"]
    bg.inputs["Color"].default_value = (*hex_rgb("#000000"), 1)
    bg.inputs["Strength"].default_value = 0.5
    if args.fog > 0:
        fog = world.node_tree.nodes.new("ShaderNodeVolumePrincipled")
        fog.inputs["Density"].default_value = args.fog
        fog.inputs["Anisotropy"].default_value = 0.4
        world.node_tree.links.new(fog.outputs[0], world.node_tree.nodes["World Output"].inputs["Volume"])

    # ground: dark asphalt, slightly wet (roughness varies)
    bpy.ops.mesh.primitive_plane_add(size=400, location=(15, 0, 0))
    ground = bpy.context.active_object
    gm = material("Asphalt", hex_rgb("#0b0c0e"), roughness=0.6)
    nt = gm.node_tree
    noise = nt.nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 0.6
    ramp = nt.nodes.new("ShaderNodeMapRange")
    ramp.inputs["To Min"].default_value = 0.45
    ramp.inputs["To Max"].default_value = 0.8
    nt.links.new(noise.outputs["Fac"], ramp.inputs["Value"])
    nt.links.new(ramp.outputs["Result"], nt.nodes["Principled BSDF"].inputs["Roughness"])
    ground.data.materials.append(gm)

    led = material("LED red", (0.3, 0, 0), emission=hex_rgb("#ff1e2d"), strength=25)

    # lineup of vehicles the car jumps over
    glass = material("Glass", (0.01, 0.01, 0.012), metallic=0.2, roughness=0.05, coat=1.0)
    shared = {"glass": glass, "trim": material("Trim", (0.012, 0.012, 0.013), roughness=0.5),
              "rims": material("Rims lineup", (0.55, 0.57, 0.6), metallic=1.0, roughness=0.28),
              "light": material("Lamp", (1, 1, 1), emission=(1, 1, 1), strength=6),
              "tail": led}
    x = LINEUP_START_X
    for real, width in LINEUP:
        cx = x + width / 2
        root = append_asset(real)
        # parked broadside to the runway, nose towards the camera side (+y), painted like the demo
        root.location = (cx, 0, PLATFORM_H)
        root.rotation_euler = (0, 0, math.pi / 2)
        style_vehicle(root, LINEUP_STYLE.get(real, {}), shared)
        x += width + GAP
    build_stage(LINEUP_START_X - 0.8, x - GAP + 0.8, led)

    # the car: one empty moves everything
    car = bpy.data.objects.new("Car", None)
    sc.collection.objects.link(car)
    car.rotation_mode = "QUATERNION"
    hero = append_asset("roadster")       # downloaded model keeps its own textures; wheels on the ground at rest
    hero.parent = car
    hero.location = (0, 0, -COM_HEIGHT)
    gloss_up(hero)
    wheels = add_wheels(hero, material("Tyre", (0.012, 0.012, 0.013), roughness=0.75),
                        material("Rim", (0.03, 0.032, 0.036), metallic=1.0, roughness=0.32),
                        material("Spoke", (0.16, 0.17, 0.18), metallic=1.0, roughness=0.25))

    # one gas jet per nozzle (4 bottom + 2 side), pointing where the gas goes
    gas = smoke_jet_material("Cold gas")
    jets = [make_plume(f"jet_{i}", gas, 3.2, 0.75, car, p, -d) for i, (p, d) in enumerate(zip(THR_POS, THR_DIR))]
    # gas hitting the floor near the ground (take-off and landing): a low cloud that spreads under the car
    bpy.ops.mesh.primitive_uv_sphere_add(segments=48, ring_count=24, radius=1.0)
    splash = bpy.context.active_object
    splash.name = "Ground splash"
    splash.scale = (3.4, 2.8, 1.1)
    splash.data.materials.append(smoke_jet_material("Ground gas", opacity=0.85, glow=0.45))

    # lights: two stadium spots, a cold fill and a rim light behind the lineup
    def add_light(kind, name, loc, energy, color, target, size=None, spot=None):
        data = bpy.data.lights.new(name, kind)
        data.energy = energy
        data.color = color
        if size is not None and kind == "AREA":
            data.size = size
        if spot is not None:
            data.spot_size = math.radians(spot)
            data.spot_blend = 1.0
        obj = bpy.data.objects.new(name, data)
        sc.collection.objects.link(obj)
        obj.location = loc
        direction = Vector(target) - Vector(loc)
        obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
        return obj
    add_light("SPOT", "Stadium A", (2, 16, 24), 110000, (1.0, 0.95, 0.88), (7, 0, 4), spot=60)
    add_light("SPOT", "Stadium B", (22, 12, 24), 110000, (1.0, 0.95, 0.88), (24, 0, 4), spot=60)
    add_light("SPOT", "Stadium C", (42, 14, 22), 110000, (1.0, 0.95, 0.88), (42, 0, 0), spot=60)
    add_light("SPOT", "Lineup key", (11, 14, 9), 90000, (1.0, 0.97, 0.92), (11, 0, 1), spot=50)
    add_light("AREA", "Fill", (-15, 35, 20), 12000, (0.6, 0.7, 1.0), (10, 0, 0), size=25)
    add_light("AREA", "Rim", (12, -30, 12), 6000, (0.7, 0.8, 1.0), (12, 0, 3), size=15)
    # blue stage lights behind the lineup, beams visible in the haze (as in the demo photo)
    for i, bx in enumerate((14.0, 22.0, 30.0)):
        add_light("SPOT", f"Stage blue {i}", (bx, -20, 16), 70000, (0.32, 0.5, 1.0), (bx - 2, -6, 0), spot=28)
    build_backdrop()

    cam_data = bpy.data.cameras.new("Camera")
    cam_data.lens = 40
    cam = bpy.data.objects.new("Camera", cam_data)
    sc.collection.objects.link(cam)
    sc.camera = cam
    target = bpy.data.objects.new("CamTarget", None)
    sc.collection.objects.link(target)
    track = cam.constraints.new("TRACK_TO")
    track.target = target
    track.track_axis = "TRACK_NEGATIVE_Z"
    track.up_axis = "UP_Y"
    return car, jets, wheels, splash, cam, target


# ----------------------------------------------------------------------------
# camera path (in simulation time)
# ----------------------------------------------------------------------------

CAM_Y, CAM_Z = 26.0, 1.7    # m: camera on a tripod beside the runway (+y side), at eye height
SCENE_CENTER = np.array([16.0, 0.0, 3.0])


def camera_at(t, car_pos):
    """Tripod beside the runway: it only slides slowly along x and pans to keep the car in the frame."""
    cam_x = 0.7 * car_pos[0] + 0.3 * SCENE_CENTER[0] - 2.0
    focal = 0.8 * car_pos + 0.2 * SCENE_CENTER
    focal[2] = 1.8 + 0.45 * car_pos[2]      # tilt up only half way: the lineup stays in the frame
    return np.array([cam_x, CAM_Y, CAM_Z]), focal


# ----------------------------------------------------------------------------
# animation
# ----------------------------------------------------------------------------

def animate(data, car, jets, wheels, splash, cam, target, fps, slowmo=1.0):
    t_sim, states = data["t"], data["states"]
    on = data["thrusters"] > 0.10       # throttles (0..1): a jet is drawn when its nozzle is above 10 %
    t_frames = frame_times(t_sim, fps, float(data["t_touchdown"]), slowmo)
    sc = bpy.context.scene
    sc.frame_start, sc.frame_end = 1, len(t_frames)
    q_prev = None
    for f, t in enumerate(t_frames, start=1):
        s = np.array([np.interp(t, t_sim, row) for row in states])
        k = int(np.clip(np.searchsorted(t_sim, t, side="right") - 1, 0, len(t_sim) - 1))
        # a thruster is drawn as firing if the controller had it on at any sample during this video frame
        t_prev = t_frames[f - 2] if f > 1 else t
        k0 = int(np.clip(np.searchsorted(t_sim, t_prev, side="right") - 1, 0, k))
        fired = on[:, k0:k + 1].any(axis=1)
        throttle = data["thrusters"][:, k0:k + 1].max(axis=1)    # strongest push of each nozzle in this frame
        pos = s[0:3] + np.array([0, 0, COM_HEIGHT])
        q = s[6:10] / np.linalg.norm(s[6:10])
        if q_prev is not None and np.dot(q, q_prev) < 0:
            q = -q                         # q and -q are the same attitude: keep the keys continuous
        q_prev = q
        car.location = pos
        car.rotation_quaternion = Quaternion(q)
        car.keyframe_insert("location", frame=f)
        car.keyframe_insert("rotation_quaternion", frame=f)
        for hub, r in wheels:                # rolling without slipping: angle = distance / radius
            hub.rotation_euler.y = s[0] / r
            hub.keyframe_insert("rotation_euler", index=1, frame=f)
        # the bottom nozzles are throttled (0..1): the jet grows with the throttle, from 40 % of its length
        # (never a tiny jet) to full size; below 10 % it is not drawn. The side nozzles are on/off: full size.
        for obj, active, thr in zip(jets, fired, throttle):
            obj.hide_render = not bool(active)
            obj.hide_viewport = not bool(active)
            size = 0.4 + 0.6 * min(max((thr - 0.10) / 0.90, 0.0), 1.0)
            obj.scale = (0.6 + 0.4 * size, 0.6 + 0.4 * size, size)
            obj.keyframe_insert("hide_render", frame=f)
            obj.keyframe_insert("hide_viewport", frame=f)
            obj.keyframe_insert("scale", frame=f)
        # ground cloud: only while a bottom nozzle fires within 2.5 m of the floor
        near = bool(fired[:4].any()) and s[2] < 2.5
        splash.location = (s[0], s[1], 0.05)
        splash.hide_render = splash.hide_viewport = not near
        splash.keyframe_insert("location", frame=f)
        splash.keyframe_insert("hide_render", frame=f)
        splash.keyframe_insert("hide_viewport", frame=f)
        cam_pos, focal = camera_at(t, pos)
        cam.location = cam_pos
        target.location = focal
        cam.keyframe_insert("location", frame=f)
        target.keyframe_insert("location", frame=f)
    # step interpolation for on/off thrusters, linear for the rest
    for obj in [car, cam, target, splash] + jets + [hub for hub, _ in wheels]:
        if obj.animation_data and obj.animation_data.action:
            for fc in _fcurves(obj.animation_data.action):
                interp = "CONSTANT" if fc.data_path in ("hide_render", "hide_viewport") else "LINEAR"
                for kp in fc.keyframe_points:
                    kp.interpolation = interp
    return t_frames


def _fcurves(action):
    """F-curves of an action (Blender 4.4+ stores them in layered channelbags)."""
    if hasattr(action, "fcurves") and len(action.fcurves):
        return list(action.fcurves)
    curves = []
    for layer in getattr(action, "layers", []):
        for strip in layer.strips:
            for bag in strip.channelbags:
                curves.extend(bag.fcurves)
    return curves


def main():
    args = parse_args()
    data = np.load(args.trajectory)
    car, jets, wheels, splash, cam, target = build_scene(args)
    t_frames = animate(data, car, jets, wheels, splash, cam, target, args.fps, args.slowmo)
    os.makedirs(args.out_dir, exist_ok=True)
    np.save(os.path.join(args.out_dir, "frame_times.npy"), t_frames)
    sc = bpy.context.scene
    if args.save:
        bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(args.save), compress=True)
        print("saved scene to", args.save)
    if args.no_render:
        return
    first, last = args.frames if args.frames else (1, len(t_frames))
    for f in range(first, min(last, len(t_frames)) + 1, args.every):
        sc.frame_set(f)
        sc.render.filepath = os.path.join(args.out_dir, f"frame_{f:04d}.png")
        bpy.ops.render.render(write_still=True)
    print(f"rendered frames {first}-{last} of {len(t_frames)} into {args.out_dir}")


if __name__ == "__main__":
    main()
