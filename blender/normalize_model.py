"""Normalize a downloaded car model into a clean .blend asset for render_hop.py.

Result: every mesh parented to an empty called ROOT; front of the car along +x, up +z, wheels on z = 0,
centered in x/y, scaled so the length along x is `--length` metres. Also renders a side + top preview
(red marker = front) to check the orientation.

usage: python normalize_model.py <model> <out.blend> --length 4.2 [--yaw 180] [--decimate 0.2]
                                 [--skip REGEX] [--preview out.png]
"""
import sys, re, math, argparse, json
import bpy
from mathutils import Vector, Matrix

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
p = argparse.ArgumentParser()
p.add_argument("model"); p.add_argument("out")
p.add_argument("--length", type=float, required=True)
p.add_argument("--yaw", type=float, default=0.0, help="extra rotation about z (deg) applied before scaling")
p.add_argument("--axis", default="x", choices=["x", "y"], help="model's length axis before yaw")
p.add_argument("--decimate", type=float, default=1.0)
p.add_argument("--skip", default=None, help="regex of object names to delete")
p.add_argument("--preview", default=None)
p.add_argument("--clay", action="store_true", help="preview with one grey material")
p.add_argument("--coll", default=None, help="keep only objects of this collection")
p.add_argument("--keep", default=None, help="keep only this object and its descendants")
a = p.parse_args(argv)

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.context.preferences.filepaths.use_scripts_auto_execute = False
sc = bpy.context.scene
path = a.model
if path.endswith(".obj"):
    bpy.ops.wm.obj_import(filepath=path)
elif path.endswith(".fbx"):
    bpy.ops.import_scene.fbx(filepath=path)
elif path.endswith((".glb", ".gltf")):
    bpy.ops.import_scene.gltf(filepath=path)
elif path.endswith(".blend"):
    bpy.ops.wm.open_mainfile(filepath=path, load_ui=False)
    sc = bpy.context.scene

if a.coll:
    for o in list(sc.objects):
        if a.coll not in [c.name for c in o.users_collection]:
            o["_drop"] = 1
if a.keep:
    top = sc.objects[a.keep]
    keep = {top.name} | {c.name for c in top.children_recursive}
    bpy.context.view_layer.update()
    for o in list(sc.objects):
        if o.name not in keep:
            o.hide_render = True
            o["_drop"] = 1
# apply modifiers (mirror, subdivision, ...) and bake parent transforms into the meshes
bpy.context.view_layer.update()
dg = bpy.context.evaluated_depsgraph_get()
for o in list(sc.objects):
    if o.type == "MESH" and "_drop" not in o and not o.hide_render:
        if o.modifiers:
            me = bpy.data.meshes.new_from_object(o.evaluated_get(dg))
            o.modifiers.clear()
            o.data = me
    elif o.type == "MESH" and o.hide_render:
        o["_drop"] = 1
# then drop everything that is not a mesh
for o in list(sc.objects):
    if o.type == "MESH":
        mw = o.matrix_world.copy()
        o.parent = None
        o.matrix_world = mw
for o in list(sc.objects):
    if o.type != "MESH" or (a.skip and re.search(a.skip, o.name)) or "_drop" in o:
        bpy.data.objects.remove(o, do_unlink=True)

meshes = [o for o in sc.objects if o.type == "MESH"]
# make mesh data single-user and apply transforms
for o in meshes:
    if o.data.users > 1:
        o.data = o.data.copy()
    o.data.transform(o.matrix_world)
    o.matrix_world = Matrix.Identity(4)

# orientation: length axis -> x, then yaw
R = Matrix.Rotation(math.radians(a.yaw + (-90 if a.axis == "y" else 0)), 4, "Z")
for o in meshes:
    o.data.transform(R)

def bbox(objs):
    lo = Vector((1e9,) * 3); hi = Vector((-1e9,) * 3)
    for o in objs:
        for v in o.data.vertices:
            lo = Vector(map(min, lo, v.co)); hi = Vector(map(max, hi, v.co))
    return lo, hi

lo, hi = bbox(meshes)
s = a.length / (hi.x - lo.x)
T = Matrix.Scale(s, 4) @ Matrix.Translation(Vector((-(lo.x + hi.x) / 2, -(lo.y + hi.y) / 2, -lo.z)))
for o in meshes:
    o.data.transform(T)
    o.data.update()

if a.decimate < 1.0:
    for o in meshes:
        if len(o.data.polygons) > 20000:
            m = o.modifiers.new("dec", "DECIMATE")
            m.ratio = a.decimate
            bpy.context.view_layer.objects.active = o
            with bpy.context.temp_override(object=o, active_object=o):
                bpy.ops.object.modifier_apply(modifier=m.name)

for o in meshes:
    for poly in o.data.polygons:
        poly.use_smooth = True

root = bpy.data.objects.new("ROOT", None)
sc.collection.objects.link(root)
for o in meshes:
    o.parent = root

lo, hi = bbox(meshes)
tris = sum(len(pl.vertices) - 2 for o in meshes for pl in o.data.polygons)
info = {"length": hi.x - lo.x, "width": hi.y - lo.y, "height": hi.z - lo.z, "tris": tris, "scale": s,
        "objects": len(meshes)}
print("INFO", json.dumps(info))
for img in bpy.data.images:
    try:
        img.pack()
    except Exception as e:
        print("skip texture", img.name, e)
if bpy.data.use_autopack:
    bpy.ops.file.autopack_toggle()
bpy.ops.wm.save_as_mainfile(filepath=a.out, compress=True)

if a.preview:
    # side view from -y and top view, red marker at the front (+x)
    sc.render.engine = "CYCLES"; sc.cycles.samples = 8; sc.cycles.use_denoising = True
    sc.render.resolution_x, sc.render.resolution_y = 800, 300
    w = bpy.data.worlds.new("w"); sc.world = w; w.use_nodes = True
    w.node_tree.nodes["Background"].inputs["Strength"].default_value = 1.2
    sun = bpy.data.objects.new("sun", bpy.data.lights.new("sun", "SUN")); sc.collection.objects.link(sun)
    sun.data.energy = 3; sun.rotation_euler = (0.6, 0.3, 0.4)
    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.15, location=(hi.x + 0.3, 0, 0.5))
    mk = bpy.context.active_object
    mm = bpy.data.materials.new("mk"); mm.use_nodes = True
    mm.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (1, 0, 0, 1)
    mk.data.materials.append(mm)
    if a.clay:
        clay = bpy.data.materials.new("clay"); clay.use_nodes = True
        for o in meshes:
            o.data.materials.clear(); o.data.materials.append(clay)
    cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam")); sc.collection.objects.link(cam)
    cam.data.type = "ORTHO"; cam.data.ortho_scale = a.length * 1.35
    sc.camera = cam
    outs = []
    for name, loc, rot in (("side", (0, -20, info["height"] / 2), (math.pi / 2, 0, 0)),
                           ("top", (0, 0, 20), (0, 0, 0))):
        cam.location = loc; cam.rotation_euler = rot
        f = a.preview.replace(".png", f"_{name}.png")
        sc.render.filepath = f
        bpy.ops.render.render(write_still=True)
        outs.append(f)
