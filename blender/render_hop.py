"""Blender scene for the Roadster hop. Reads the trajectory exported by the simulation and renders PNG frames.

Usage (from the repo root):
    blender -b -P blender/render_hop.py -- trajectory.npz renders/frames --res 1920 1080 --engine EEVEE

Everything the car does comes from the trajectory file: position, attitude, which attitude thrusters
are on and how much lift the lift thrusters give. Blender only draws it.
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
ASSETS = os.path.join(HERE, "assets")
REAL = os.path.join(HERE, "assets_real")   # normalized downloaded models (normalize_model.py)

COM_HEIGHT = 0.62      # m, CoM above the ground when the car sits on its wheels (visual offset only)
LINEUP_START_X = 3.1   # m
PLATFORM_H = 0.15      # m, low stage the lineup stands on (like the demo)
GAP = 1.0              # m
LINEUP = [             # (real model, provisional silhouette, width across the runway in m)
    ("cybercab", "coupe", 1.97), ("cybertruck", "pickup", 1.97), ("semi", "truck", 2.59),
    ("modelx", "suv", 2.26), ("models", "sedan", 2.17),
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
HALF_L, HALF_W = 2.1, 1.0
# attitude thrusters, body frame (same table as roadster/thrusters.py), exhaust = -force direction
THR_POS = np.array([[0, HALF_W, 0], [0, -HALF_W, 0], [0, HALF_W, 0], [0, -HALF_W, 0],
                    [HALF_L, 0, 0], [-HALF_L, 0, 0], [HALF_L, 0, 0], [-HALF_L, 0, 0],
                    [HALF_L, 0, 0], [-HALF_L, 0, 0], [HALF_L, 0, 0], [-HALF_L, 0, 0]], float)
THR_DIR = np.array([[0, 0, 1], [0, 0, -1], [0, 0, -1], [0, 0, 1],
                    [0, 0, -1], [0, 0, 1], [0, 0, 1], [0, 0, -1],
                    [0, 1, 0], [0, -1, 0], [0, -1, 0], [0, 1, 0]], float)
LIFT_NOZZLES = np.array([[1.35, 0.75, -0.38], [1.35, -0.75, -0.38],
                         [-1.35, 0.75, -0.38], [-1.35, -0.75, -0.38]])


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


def import_ply(path, mat, parent=None, smooth=True, location=(0, 0, 0)):
    bpy.ops.wm.ply_import(filepath=path)
    obj = bpy.context.selected_objects[0]
    obj.data.materials.append(mat)
    if smooth:
        obj.data.shade_smooth()
    obj.location = location
    if parent is not None:
        obj.parent = parent
    return obj


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


def gas_material(name, strength):
    """Volumetric gas: emissive, wispy (noise), dense at the nozzle and fading towards the tip and the edges.
    Generated coords on the cone: z goes 0 (nozzle) .. 1 (tip), x/y 0..1 across the bounding box."""
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    N = nt.nodes.new
    L = nt.links.new

    def math(op, a=None, b=None):
        n = N("ShaderNodeMath")
        n.operation = op
        for i, v in enumerate((a, b)):
            if v is None:
                continue
            if isinstance(v, (int, float)):
                n.inputs[i].default_value = v
            else:
                L(v, n.inputs[i])
        return n.outputs[0]

    out = N("ShaderNodeOutputMaterial")
    vol = N("ShaderNodeVolumePrincipled")
    gen = N("ShaderNodeTexCoord").outputs["Generated"]
    sep = N("ShaderNodeSeparateXYZ")
    L(gen, sep.inputs[0])
    z = sep.outputs["Z"]
    # axial fade: 1 at the nozzle -> 0 at the tip, squared for a quick decay
    axial = math("POWER", math("SUBTRACT", 1.0, z), 2.0)
    # radial fade: distance from the axis relative to the local cone radius (0.25 at nozzle .. 1 at tip)
    dx = math("SUBTRACT", sep.outputs["X"], 0.5)
    dy = math("SUBTRACT", sep.outputs["Y"], 0.5)
    r = math("MULTIPLY", math("SQRT", math("ADD", math("MULTIPLY", dx, dx), math("MULTIPLY", dy, dy))), 2.0)
    r_rel = math("DIVIDE", r, math("ADD", 0.25, math("MULTIPLY", z, 0.75)))
    radial = math("POWER", math("MAXIMUM", math("SUBTRACT", 1.0, r_rel), 0.0), 1.5)
    noise = N("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 5.0
    noise.inputs["Detail"].default_value = 8.0
    noise.inputs["Roughness"].default_value = 0.6
    L(N("ShaderNodeTexCoord").outputs["Object"], noise.inputs["Vector"])
    wisp = math("ADD", 0.35, math("MULTIPLY", noise.outputs["Fac"], 0.9))
    shape = math("MULTIPLY", math("MULTIPLY", axial, radial), wisp)
    L(math("MULTIPLY", shape, 4.0), vol.inputs["Density"])
    L(math("MULTIPLY", shape, strength), vol.inputs["Emission Strength"])
    vol.inputs["Color"].default_value = (0.9, 0.93, 1.0, 1)
    vol.inputs["Emission Color"].default_value = (0.7, 0.82, 1.0, 1)
    L(vol.outputs[0], out.inputs["Volume"])
    return m


def jet_material(name, strength):
    """Glowing, semi-transparent surface for a jet cone: bright at the nozzle, fading to the tip and to the
    silhouette edges. A surface (not a volume) so EEVEE draws thin jets crisply."""
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    if hasattr(m, "surface_render_method"):
        m.surface_render_method = "BLENDED"
    m.use_backface_culling = False
    nt = m.node_tree
    nt.nodes.clear()
    N, L = nt.nodes.new, nt.links.new
    out = N("ShaderNodeOutputMaterial")
    gen = N("ShaderNodeTexCoord")
    sep = N("ShaderNodeSeparateXYZ")
    L(gen.outputs["Generated"], sep.inputs[0])
    inv = N("ShaderNodeMath"); inv.operation = "SUBTRACT"; inv.inputs[0].default_value = 1.0
    L(sep.outputs["Z"], inv.inputs[1])
    axial = N("ShaderNodeMath"); axial.operation = "POWER"; axial.inputs[1].default_value = 1.6
    L(inv.outputs[0], axial.inputs[0])
    lw = N("ShaderNodeLayerWeight"); lw.inputs["Blend"].default_value = 0.35
    edge = N("ShaderNodeMath"); edge.operation = "SUBTRACT"; edge.inputs[0].default_value = 1.0
    L(lw.outputs["Facing"], edge.inputs[1])
    alpha = N("ShaderNodeMath"); alpha.operation = "MULTIPLY"
    L(axial.outputs[0], alpha.inputs[0]); L(edge.outputs[0], alpha.inputs[1])
    fac = N("ShaderNodeMath"); fac.operation = "MULTIPLY"; fac.inputs[1].default_value = 0.9
    fac.use_clamp = True
    L(alpha.outputs[0], fac.inputs[0])
    emis = N("ShaderNodeEmission")
    emis.inputs["Color"].default_value = (0.78, 0.88, 1.0, 1)
    emis.inputs["Strength"].default_value = strength
    transp = N("ShaderNodeBsdfTransparent")
    mix = N("ShaderNodeMixShader")
    L(fac.outputs[0], mix.inputs["Fac"])
    L(transp.outputs[0], mix.inputs[1]); L(emis.outputs[0], mix.inputs[2])
    L(mix.outputs[0], out.inputs["Surface"])
    return m


def smoke_jet_material(name, opacity=0.95, glow=0.7):
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
    axial.inputs["From Min"].default_value = 0.6
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
    bpy.ops.mesh.primitive_cone_add(vertices=48, radius1=radius * 0.25, radius2=radius, depth=length, end_fill_type="NOTHING",
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
    """Big blue-lit arch behind the stage and low haze on the floor."""
    arch_mat = material("Arch", hex_rgb("#0b1530"), metallic=0.6, roughness=0.35,
                        emission=hex_rgb("#2d5bff"), strength=0.6)
    if False:  # arch removed: read as a giant horseshoe; the blue beams are enough
      bpy.ops.mesh.primitive_torus_add(major_radius=15, minor_radius=0.6, major_segments=128, minor_segments=16,
                                     location=(34, -34, -3.0), rotation=(math.pi / 2, 0, math.radians(-25)))
      arch = bpy.context.active_object
      arch.name = "Arch"
      arch.data.materials.append(arch_mat)
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
    bpy.ops.mesh.primitive_cube_add(size=1, location=(22, -16, 9))
    air = bpy.context.active_object
    air.name = "Beam air"
    air.scale = (44, 18, 18)
    air.data.materials.append(beam)


# ----------------------------------------------------------------------------
# cold-gas puffs
# ----------------------------------------------------------------------------
# A cold-gas jet is almost invisible; what the demo shows is the condensing cloud. Each puff is released at a
# nozzle while the simulation says that thruster is on, leaves with the car's velocity plus the exhaust velocity,
# is stopped by the air (exponential drag), grows and fades. This is a visual effect: it reads the simulation
# (position, attitude, velocity, thrusters, lift) but does not feed anything back into it.

PUFF = {"att": dict(dt=0.025, v_gas=16.0, tau=0.15, life=0.5, r0=0.10, r1=0.40),
        "lift": dict(dt=0.02, v_gas=12.0, tau=0.25, life=1.4, r0=0.18, r1=1.0)}
# the four lift nozzles are drawn as two emitters (front and rear) to keep the number of puffs manageable
PUFFS = False        # loose puffs read as small blobs; the jets alone show clearly which thrusters fire
LIFT_PUFFS = False   # the lift exhaust cloud hid which attitude thrusters were firing; cones only
LIFT_EMITTERS = np.array([[1.35, 0.0, -0.38], [-1.35, 0.0, -0.38]])


def quat_rotate(q, v):
    """Rotate v (body frame) to the world frame with q = (w, x, y, z): q (x) (0, v) (x) q*."""
    w, u = q[0], q[1:4]
    return v + 2.0 * np.cross(u, np.cross(u, v) + w * v)


def puff_material():
    m = bpy.data.materials.new("Puff")
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    N, L = nt.nodes.new, nt.links.new
    out = N("ShaderNodeOutputMaterial")
    vol = N("ShaderNodeVolumePrincipled")
    info = N("ShaderNodeObjectInfo")             # per-puff opacity is keyframed in the object's colour alpha
    gen = N("ShaderNodeTexCoord")
    dist = N("ShaderNodeVectorMath"); dist.operation = "DISTANCE"
    dist.inputs[1].default_value = (0.5, 0.5, 0.5)
    L(gen.outputs["Generated"], dist.inputs[0])
    edge = N("ShaderNodeMapRange")               # 1 in the centre -> 0 at the surface of the sphere
    edge.interpolation_type = "SMOOTHERSTEP"
    edge.inputs["From Min"].default_value = 0.5
    edge.inputs["From Max"].default_value = 0.1
    L(dist.outputs["Value"], edge.inputs["Value"])
    noise = N("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 1.6
    noise.inputs["Detail"].default_value = 3.0
    noise.noise_dimensions = "4D"
    L(gen.outputs["Object"], noise.inputs["Vector"])
    L(info.outputs["Random"], noise.inputs["W"])
    wisp = N("ShaderNodeMapRange")
    wisp.inputs["From Min"].default_value = 0.2
    wisp.inputs["From Max"].default_value = 0.8
    wisp.inputs["To Min"].default_value = 0.35
    L(noise.outputs["Fac"], wisp.inputs["Value"])
    a = N("ShaderNodeMath"); a.operation = "MULTIPLY"
    L(edge.outputs["Result"], a.inputs[0]); L(wisp.outputs["Result"], a.inputs[1])
    b = N("ShaderNodeMath"); b.operation = "MULTIPLY"
    L(a.outputs[0], b.inputs[0]); L(info.outputs["Alpha"], b.inputs[1])
    d = N("ShaderNodeMath"); d.operation = "MULTIPLY"; d.inputs[1].default_value = 2.0
    L(b.outputs[0], d.inputs[0])
    e = N("ShaderNodeMath"); e.operation = "MULTIPLY"; e.inputs[1].default_value = 0.25
    L(b.outputs[0], e.inputs[0])
    L(d.outputs[0], vol.inputs["Density"])
    L(e.outputs[0], vol.inputs["Emission Strength"])
    vol.inputs["Color"].default_value = (0.95, 0.96, 1.0, 1)
    vol.inputs["Emission Color"].default_value = (0.8, 0.88, 1.0, 1)
    L(vol.outputs[0], out.inputs["Volume"])
    return m


def set_keys(obj, data_path, index, frames, values, constant=False):
    """Write many keyframes at once (much faster than keyframe_insert)."""
    if obj.animation_data is None:
        obj.animation_data_create()
    if obj.animation_data.action is None:
        obj.animation_data.action = bpy.data.actions.new(obj.name + "Action")
    fc = obj.animation_data.action.fcurve_ensure_for_datablock(obj, data_path, index=index)
    fc.keyframe_points.add(len(frames))
    co = np.empty(2 * len(frames)); co[0::2] = frames; co[1::2] = values
    fc.keyframe_points.foreach_set("co", co)
    fc.keyframe_points.foreach_set("interpolation", [0 if constant else 1] * len(frames))  # CONSTANT / LINEAR
    fc.update()


def add_puffs(data, t_frames):
    t_sim, states = data["t"], data["states"]
    on, lifts = data["thrusters"].astype(bool), data["lifts"]
    t_end = float(data["t_touchdown"]) + 0.1
    sc = bpy.context.scene
    bpy.ops.mesh.primitive_uv_sphere_add(segments=16, ring_count=8, radius=0.5)
    proto = bpy.context.active_object
    mesh = proto.data
    mesh.materials.append(puff_material())
    bpy.data.objects.remove(proto, do_unlink=True)
    rng = np.random.default_rng(7)
    frames = np.arange(1, len(t_frames) + 1, dtype=float)

    def state_at(t):
        return np.array([np.interp(t, t_sim, row) for row in states])

    def emit(kind, t0, nozzle_body, exhaust_body, strength):
        P = PUFF[kind]
        s = state_at(t0)
        q = s[6:10] / np.linalg.norm(s[6:10])
        origin = s[0:3] + np.array([0, 0, COM_HEIGHT]) + quat_rotate(q, nozzle_body)
        d = quat_rotate(q, exhaust_body)
        jitter = rng.normal(0, 0.12, 3)
        v0 = 0.9 * s[3:6] + (d + jitter) * P["v_gas"] * strength
        alive = (t_frames >= t0) & (t_frames <= t0 + P["life"])
        if not alive.any():
            return
        idx = np.flatnonzero(alive)
        lo, hi = max(idx[0] - 1, 0), min(idx[-1] + 1, len(t_frames) - 1)
        span = np.arange(lo, hi + 1)
        tau = np.clip(t_frames[span] - t0, 0, None)
        u = np.clip(tau / P["life"], 0, 1)
        pos = origin + np.outer(P["tau"] * (1 - np.exp(-tau / P["tau"])), v0) + np.outer(0.25 * tau, [0, 0, 1])
        pos[:, 2] = np.maximum(pos[:, 2], 0.2)                 # the floor stops the gas
        r = (P["r0"] + (P["r1"] - P["r0"]) * np.sqrt(u)) * (0.6 + 0.4 * strength)
        alpha = np.clip(tau / 0.04, 0, 1) * (1 - u) ** 1.6 * (0.5 + 0.5 * strength)
        obj = bpy.data.objects.new(f"puff_{kind}", mesh)
        sc.collection.objects.link(obj)
        obj.color = (1, 1, 1, 0)
        f = frames[span]
        for i in range(3):
            set_keys(obj, "location", i, f, pos[:, i])
            set_keys(obj, "scale", i, f, 2 * r)
        set_keys(obj, "color", 3, f, alpha)
        vis_f = [1.0, frames[span[0]], frames[span[-1]] + 1]
        set_keys(obj, "hide_render", 0, vis_f, [1, 0, 1], constant=True)
        set_keys(obj, "hide_viewport", 0, vis_f, [1, 0, 1], constant=True)

    for t0 in np.arange(0.0, t_end, PUFF["att"]["dt"]):
        k = int(np.clip(np.searchsorted(t_sim, t0, side="right") - 1, 0, len(t_sim) - 1))
        for i in np.flatnonzero(on[:, k]):
            emit("att", t0, THR_POS[i], -THR_DIR[i], 1.0)
    for t0 in (np.arange(0.0, t_end, PUFF["lift"]["dt"]) if LIFT_PUFFS else []):
        k = int(np.clip(np.searchsorted(t_sim, t0, side="right") - 1, 0, len(t_sim) - 1))
        lf = float(lifts[k])
        if lf > 0.02:
            for p in LIFT_EMITTERS:
                emit("lift", t0 + rng.uniform(0, PUFF["lift"]["dt"]), p, np.array([0, 0, -1.0]), min(1.0, lf))


def build_scene(args):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    sc.unit_settings.system = "METRIC"
    sc.render.resolution_x, sc.render.resolution_y = args.res
    sc.render.fps = args.fps
    sc.render.image_settings.file_format = "PNG"
    sc.view_settings.view_transform = "AgX"
    sc.view_settings.look = "AgX - Medium High Contrast"
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
    gm = material("Asphalt", hex_rgb("#0d0e11"), roughness=0.35)
    nt = gm.node_tree
    noise = nt.nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 0.6
    ramp = nt.nodes.new("ShaderNodeMapRange")
    ramp.inputs["To Min"].default_value = 0.12
    ramp.inputs["To Max"].default_value = 0.55
    nt.links.new(noise.outputs["Fac"], ramp.inputs["Value"])
    nt.links.new(ramp.outputs["Result"], nt.nodes["Principled BSDF"].inputs["Roughness"])
    ground.data.materials.append(gm)

    led = material("LED red", (0.3, 0, 0), emission=hex_rgb("#ff1e2d"), strength=25)
    led_white = material("LED white", (1, 1, 1), emission=(1, 1, 1), strength=30)

    # lineup of vehicles the car jumps over
    paint = material("Lineup paint", hex_rgb("#5c626c"), metallic=0.85, roughness=0.25, coat=0.8)
    glass = material("Glass", (0.01, 0.01, 0.012), metallic=0.2, roughness=0.05, coat=1.0)
    rubber = material("Rubber", (0.01, 0.01, 0.01), roughness=0.9)
    shared = {"glass": glass, "trim": material("Trim", (0.012, 0.012, 0.013), roughness=0.5),
              "rims": material("Rims lineup", (0.55, 0.57, 0.6), metallic=1.0, roughness=0.28),
              "light": material("Lamp", (1, 1, 1), emission=(1, 1, 1), strength=6),
              "tail": led}
    x = LINEUP_START_X
    for real, kind, width in LINEUP:
        cx = x + width / 2
        root = append_asset(real)
        if root is not None:
            # parked broadside to the runway, one uniform paint for the whole lineup
            root.location = (cx, 0, PLATFORM_H)
            root.rotation_euler = (0, 0, math.pi / 2)   # nose towards the camera side (+y)
            style_vehicle(root, LINEUP_STYLE.get(real, {}), shared)
        else:
            for part in ("body", "wheels", "glass"):
                path = os.path.join(ASSETS, f"veh_{kind}_{part}.ply")
                if os.path.exists(path):
                    import_ply(path, paint, location=(cx, 0, PLATFORM_H), smooth=(part != "glass"))
        x += width + GAP
    build_stage(LINEUP_START_X - 0.8, x - GAP + 0.8, led)

    # the car: one empty moves everything
    car = bpy.data.objects.new("Car", None)
    sc.collection.objects.link(car)
    car.rotation_mode = "QUATERNION"
    car_paint = material("Car paint", hex_rgb("#b3121f"), metallic=0.6, roughness=0.22, coat=1.0)
    rims = material("Rims", (0.6, 0.62, 0.65), metallic=1.0, roughness=0.25)
    hero = append_asset("roadster")
    if hero is not None:   # downloaded model keeps its own textures; wheels on the ground at rest
        hero.parent = car
        hero.location = (0, 0, -COM_HEIGHT)
    else:
        for part, mat in (("body", car_paint), ("cabin", glass), ("tires", rubber), ("rims", rims),
                          ("head", led_white), ("tail", led)):
            import_ply(os.path.join(ASSETS, f"car_{part}.ply"), mat, parent=car)

    # gas plumes: 12 attitude thrusters + 4 lift nozzles
    att_gas = smoke_jet_material("Jet attitude")
    lift_gas = smoke_jet_material("Jet lift", opacity=0.8, glow=0.5)
    att = [make_plume(f"jet_att_{i}", att_gas, 2.2, 0.45, car, p, -d) for i, (p, d) in enumerate(zip(THR_POS, THR_DIR))]
    lift = [make_plume(f"jet_lift_{i}", lift_gas, 2.0, 0.40, car, p, (0, 0, -1)) for i, p in enumerate(LIFT_NOZZLES)]
    # the model assumes lift nozzles on both faces: the ones facing the ground fire (see README)
    lift += [make_plume(f"jet_lift_top_{i}", lift_gas, 2.0, 0.40, car, p * [1, 1, -1.3], (0, 0, 1))
             for i, p in enumerate(LIFT_NOZZLES)]

    # lights: two stadium spots, a cold fill and a rim light behind the lineup
    def add_light(kind, name, loc, energy, color, target, size=None, spot=None):
        data = bpy.data.lights.new(name, kind)
        data.energy = energy
        data.color = color
        if size is not None and kind == "AREA":
            data.size = size
        if spot is not None:
            data.spot_size = math.radians(spot)
            data.spot_blend = 0.6
        obj = bpy.data.objects.new(name, data)
        sc.collection.objects.link(obj)
        obj.location = loc
        direction = Vector(target) - Vector(loc)
        obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
        return obj
    add_light("SPOT", "Stadium A", (2, 16, 24), 200000, (1.0, 0.95, 0.88), (7, 0, 0), spot=50)
    add_light("SPOT", "Stadium B", (22, 12, 24), 200000, (1.0, 0.95, 0.88), (24, 0, 0), spot=50)
    add_light("SPOT", "Lineup key", (11, 14, 9), 90000, (1.0, 0.97, 0.92), (11, 0, 1), spot=50)
    add_light("AREA", "Fill", (-15, 35, 20), 12000, (0.6, 0.7, 1.0), (10, 0, 0), size=25)
    add_light("AREA", "Rim", (12, -30, 12), 6000, (0.7, 0.8, 1.0), (12, 0, 3), size=15)
    # blue stage lights behind the lineup, beams visible in the haze (as in the demo photo)
    for i, bx in enumerate((14.0, 22.0, 30.0)):
        add_light("SPOT", f"Stage blue {i}", (bx, -20, 16), 70000, (0.32, 0.5, 1.0), (bx - 2, -6, 0), spot=28)
    build_backdrop()

    cam_data = bpy.data.cameras.new("Camera")
    cam_data.lens = 28
    cam = bpy.data.objects.new("Camera", cam_data)
    sc.collection.objects.link(cam)
    sc.camera = cam
    target = bpy.data.objects.new("CamTarget", None)
    sc.collection.objects.link(target)
    track = cam.constraints.new("TRACK_TO")
    track.target = target
    track.track_axis = "TRACK_NEGATIVE_Z"
    track.up_axis = "UP_Y"
    return car, att, lift, cam, target


# ----------------------------------------------------------------------------
# camera path (in simulation time)
# ----------------------------------------------------------------------------

CAM_KEYS = [  # time, offset from the car, how much the camera looks at the car (vs. the lineup)
    (-1.5, (-11.0, 6.0, 1.3), 0.95),   # chasing the car on the runway
    (0.0, (-9.0, 12.0, 1.0), 0.85),    # take-off
    (0.9, (-2.0, 21.0, 0.0), 0.65),    # side view, a bit below: the roll against the sky
    (2.2, (4.0, 20.0, -3.0), 0.70),
    (3.4, (7.0, 12.0, -1.0), 0.90),    # front-left, low: braking and hover
    (6.2, (9.0, 8.0, -0.2), 0.95),     # touchdown
    (8.0, (14.0, 6.0, 0.2), 0.95),     # driving away
]
SCENE_CENTER = np.array([11.0, 0.0, 2.0])


def camera_at(t, car_pos):
    times = np.array([k[0] for k in CAM_KEYS])
    i = int(np.clip(np.searchsorted(times, t) - 1, 0, len(CAM_KEYS) - 2))
    u = np.clip((t - times[i]) / (times[i + 1] - times[i]), 0, 1)
    u = u * u * (3 - 2 * u)
    off = (1 - u) * np.array(CAM_KEYS[i][1]) + u * np.array(CAM_KEYS[i + 1][1])
    blend = (1 - u) * CAM_KEYS[i][2] + u * CAM_KEYS[i + 1][2]
    pos = car_pos + off
    pos[2] = max(pos[2], 0.6)
    return pos, blend * car_pos + (1 - blend) * SCENE_CENTER


# ----------------------------------------------------------------------------
# animation
# ----------------------------------------------------------------------------

def animate(data, car, att, lift, cam, target, fps, slowmo=1.0):
    t_sim, states = data["t"], data["states"]
    on, lifts = data["thrusters"].astype(bool), data["lifts"]
    t_frames = frame_times(t_sim, fps, float(data["t_touchdown"]), slowmo)
    sc = bpy.context.scene
    sc.frame_start, sc.frame_end = 1, len(t_frames)
    q_prev = None
    rng = np.random.default_rng(3)
    for f, t in enumerate(t_frames, start=1):
        s = np.array([np.interp(t, t_sim, row) for row in states])
        k = int(np.clip(np.searchsorted(t_sim, t, side="right") - 1, 0, len(t_sim) - 1))
        # a thruster is drawn as firing if the controller had it on at any sample during this video frame
        t_prev = t_frames[f - 2] if f > 1 else t
        k0 = int(np.clip(np.searchsorted(t_sim, t_prev, side="right") - 1, 0, k))
        fired = on[:, k0:k + 1].any(axis=1)
        pos = s[0:3] + np.array([0, 0, COM_HEIGHT])
        q = s[6:10] / np.linalg.norm(s[6:10])
        if q_prev is not None and np.dot(q, q_prev) < 0:
            q = -q                         # q and -q are the same attitude: keep the keys continuous
        q_prev = q
        car.location = pos
        car.rotation_quaternion = Quaternion(q)
        car.keyframe_insert("location", frame=f)
        car.keyframe_insert("rotation_quaternion", frame=f)
        # all or nothing: a jet is either drawn at full size or not drawn at all (render and viewport)
        lf = float(lifts[k])
        upright = (1 - 2 * (q[1] ** 2 + q[2] ** 2)) >= 0      # body z axis pointing up?
        n_bottom = len(lift) // 2
        lift_on = [lf > 0.02 and (upright if j < n_bottom else not upright) for j in range(len(lift))]
        for obj, active in [(o, bool(fired[i])) for i, o in enumerate(att)] + list(zip(lift, lift_on)):
            obj.hide_render = not active
            obj.hide_viewport = not active
            obj.keyframe_insert("hide_render", frame=f)
            obj.keyframe_insert("hide_viewport", frame=f)
        cam_pos, focal = camera_at(t, pos)
        cam.location = cam_pos
        target.location = focal
        cam.keyframe_insert("location", frame=f)
        target.keyframe_insert("location", frame=f)
    # step interpolation for on/off thrusters, linear for the rest
    for obj in [car, cam, target] + att + lift:
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
    car, att, lift, cam, target = build_scene(args)
    t_frames = animate(data, car, att, lift, cam, target, args.fps, args.slowmo)
    if PUFFS:
        add_puffs(data, t_frames)
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
