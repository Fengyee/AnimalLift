"""blender -b --python render_glb.py -- <glb> <out_prefix> <hdr> [res] [samples] [clay|tex|flat] [views,comma-separated|all] [head]
Orbit renders of a GLB: 8 azimuths at 10 deg elevation plus a top view.
flat = uniform white light and an orthographic camera at 0 deg elevation (views az000.., top, bottom),
for like-for-like comparison with reference photos."""
import bpy, sys, math
from mathutils import Vector
a = sys.argv[sys.argv.index("--") + 1:]
glb, out, hdr = a[0], a[1], a[2]
res = int(a[3]) if len(a) > 3 else 512
spp = int(a[4]) if len(a) > 4 else 48
clay = len(a) > 5 and a[5] == "clay"
flat = len(a) > 5 and a[5] == "flat"
only = a[6].split(",") if len(a) > 6 and a[6] != "all" else None
head = len(a) > 7 and a[7] == "head"   # frame the head only (the animal faces -Y after glTF import)
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=glb)
objs = [o for o in bpy.context.scene.objects if o.type == "MESH"]
pts = [o.matrix_world @ Vector(c) for o in objs for c in o.bound_box]
lo = Vector([min(p[i] for p in pts) for i in range(3)]); hi = Vector([max(p[i] for p in pts) for i in range(3)])
ctr, size = (lo + hi) / 2, max(hi - lo)
for o in objs:
    for poly in o.data.polygons: poly.use_smooth = True
if head:
    vs = [o.matrix_world @ v.co for o in objs for v in o.data.vertices]
    hv = [v for v in vs if v.y < lo.y + 0.28 * (hi.y - lo.y) and v.z > lo.z + 0.62 * (hi.z - lo.z)]
    hlo = Vector([min(v[i] for v in hv) for i in range(3)]); hhi = Vector([max(v[i] for v in hv) for i in range(3)])
    ctr, size = (hlo + hhi) / 2, max(hhi - hlo) * 1.25
if clay:
    m = bpy.data.materials.new("clay"); m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]; b.inputs["Base Color"].default_value = (0.6, 0.6, 0.6, 1); b.inputs["Roughness"].default_value = 0.6
    for o in objs:
        o.data.materials.clear(); o.data.materials.append(m)
sc = bpy.context.scene
sc.render.engine = "CYCLES"
p = bpy.context.preferences.addons["cycles"].preferences
p.compute_device_type = "OPTIX"; p.get_devices()
for d in p.devices: d.use = d.type == "OPTIX"
sc.cycles.device = "GPU"; sc.cycles.samples = spp; sc.cycles.use_denoising = True
sc.render.resolution_x = sc.render.resolution_y = res
sc.render.film_transparent = True
w = bpy.data.worlds.new("w"); sc.world = w; w.use_nodes = True
if not flat:
    env = w.node_tree.nodes.new("ShaderNodeTexEnvironment"); env.image = bpy.data.images.load(hdr)
    w.node_tree.links.new(env.outputs["Color"], w.node_tree.nodes["Background"].inputs["Color"])
else:
    w.node_tree.nodes["Background"].inputs["Color"].default_value = (1, 1, 1, 1)
    sc.view_settings.view_transform = "Standard"
cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam")); sc.collection.objects.link(cam); sc.camera = cam
cam.data.lens = 70
if flat:
    cam.data.type = "ORTHO"; cam.data.ortho_scale = size * 1.15
tgt = bpy.data.objects.new("t", None); sc.collection.objects.link(tgt); tgt.location = ctr
c = cam.constraints.new("TRACK_TO"); c.target = tgt
views = [(f"az{az:03d}", az, 0 if flat else 10) for az in range(0, 360, 45)] + [("top", 0, 89.9 if flat else 85), ("bottom", 0, -89.9)]
for name, az, el in views:
    if (only and name not in only) or (name == "bottom" and not only): continue
    r, e, z = size * 2.3, math.radians(el), math.radians(az)
    cam.location = ctr + Vector((r * math.cos(e) * math.sin(z), -r * math.cos(e) * math.cos(z), r * math.sin(e)))
    sc.render.filepath = f"{out}_{name}.png"
    bpy.ops.render.render(write_still=True)
