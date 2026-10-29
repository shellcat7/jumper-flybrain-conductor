"""Scanned props for the back alley (10/28, the user: the ground shots need more detail -- objects).

CC0 models from Poly Haven (most from its "hidden alley" collection), fetched as glTF with their
textures: outdoor air-conditioner units, a trash can, rubbish bags, beer crates, cardboard, wall
utility boxes, propane cylinders (as behind any izakaya), a drum, a tyre, a manhole cover and small
litter. The alley's own grey box stand-ins for the air-conditioners and crates are swapped for the
scans in the same places; the rest go against the walls where the floor is free, never in the
robot's lane or where the drones fly.

Each model is imported once into an off-stage collection and placed as collection instances, so
twenty air-conditioners cost one model. Front faces -Y in Poly Haven's glTF as Blender imports it;
an instance's origin is the model's back-bottom-centre, so it stands against the wall it backs on.

    import props
    lib = props.fetch(Path("/content/props_cache"))
    props.dress(lib, seed=5)
"""

from __future__ import annotations

import json
import math
import random
import urllib.request
from pathlib import Path

import bpy
from mathutils import Vector

API = "https://api.polyhaven.com/files/"
UA = {"User-Agent": "jumper-film/1.0 (CC0 assets, credited)"}
MODELS = ("exterior_aircon_unit", "metal_trash_can", "trashbag", "plastic_crate_01", "plastic_crate_02",
          "plastic_crate_03", "cardboard_box_01", "utility_box_01", "utility_box_02", "propane_tank",
          "Barrel_02", "water_manhole_cover", "old_tyre", "can_rusted", "plastic_bottle_gallon",
          "spray_paint_bottles_02", "plastic_container")


def _get(url: str, path: Path) -> None:
    if path.exists() and path.stat().st_size > 0:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA)) as r, open(path, "wb") as f:
        f.write(r.read())


def fetch(cache: Path, res: str = "2k") -> dict:
    """Download each model's glTF, buffer and textures into `cache/<id>/`; returns id -> .gltf path and
    writes the credits beside them."""
    cache.mkdir(parents=True, exist_ok=True)
    out, credits = {}, []
    for mid in MODELS:
        try:
            files = json.loads(urllib.request.urlopen(urllib.request.Request(API + mid, headers=UA)).read())
            g = files.get("gltf", {})
            entry = (g.get(res) or g.get("1k") or next(iter(g.values())))["gltf"]
            main = cache / mid / entry["url"].rsplit("/", 1)[-1]
            _get(entry["url"], main)
            for rel, inc in entry.get("include", {}).items():
                _get(inc["url"], cache / mid / rel)
            out[mid] = main
            credits.append({"model": mid, "url": f"https://polyhaven.com/a/{mid}", "source": "Poly Haven",
                            "license": "CC0"})
        except Exception as e:  # a missing model leaves a gap, not a failed render
            print(f"[props] {mid}: {e}", flush=True)
    (cache / "credits.json").write_text(json.dumps(credits, indent=1))
    print(f"[props] {len(out)} models in {cache}", flush=True)
    return out


def _library(mid: str, path: Path):
    """Import a glTF into its own collection, off stage; the collection's instance offset is the
    model's back-bottom-centre. Returns (collection, size xyz)."""
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=str(path))
    objs = [o for o in bpy.data.objects if o not in before]
    col = bpy.data.collections.new(f"lib_{mid}")
    bpy.context.scene.collection.children.link(col)
    for o in objs:
        for c in list(o.users_collection):
            c.objects.unlink(o)
        col.objects.link(o)
    bpy.context.view_layer.update()
    pts = [o.matrix_world @ Vector(c) for o in objs if o.type == "MESH" for c in o.bound_box]
    lo = Vector((min(p.x for p in pts), min(p.y for p in pts), min(p.z for p in pts)))
    hi = Vector((max(p.x for p in pts), max(p.y for p in pts), max(p.z for p in pts)))
    col.instance_offset = ((lo.x + hi.x) / 2, hi.y, lo.z)
    lc = bpy.context.view_layer.layer_collection.children.get(col.name)
    if lc is not None:
        lc.exclude = True  # only the instances render
    return col, hi - lo


def _place(col, root, u, v, z, yaw, scale=1.0, name="prop"):
    o = bpy.data.objects.new(name, None)
    o.instance_type = "COLLECTION"
    o.instance_collection = col
    bpy.context.scene.collection.objects.link(o)
    o.parent = root
    o.location = (u, v, z)
    o.rotation_euler = (0.0, 0.0, yaw)
    o.scale = (scale, scale, scale)
    return o


def _dumpster(name: str):
    """A big steel dumpster as in the back alleys of American films, built here: a tapered bin on four
    castors, two lids (one propped open), side ribs and lifting pockets, painted dark green and worn."""
    import bmesh
    from mathutils import Matrix

    paint = bpy.data.materials.new(f"{name}_paint")
    paint.use_nodes = True
    nt = paint.node_tree
    b = next(n for n in nt.nodes if n.type == "BSDF_PRINCIPLED")
    noise = nt.nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 3.0
    noise.inputs["Detail"].default_value = 8.0
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].position, ramp.color_ramp.elements[0].color = 0.45, (0.05, 0.12, 0.07, 1.0)
    ramp.color_ramp.elements[1].position, ramp.color_ramp.elements[1].color = 0.75, (0.16, 0.08, 0.04, 1.0)
    nt.links.new(noise.outputs["Fac"], ramp.inputs["Fac"])
    nt.links.new(ramp.outputs["Color"], b.inputs["Base Color"])
    b.inputs["Roughness"].default_value = 0.6
    b.inputs["Metallic"].default_value = 0.5
    dark = bpy.data.materials.new(f"{name}_black")
    dark.use_nodes = True
    db = next(n for n in dark.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
    db.inputs["Base Color"].default_value = (0.02, 0.02, 0.02, 1.0)
    db.inputs["Roughness"].default_value = 0.7
    bm = bmesh.new()
    L, D, H, z0 = 1.8, 1.1, 1.0, 0.14
    # the bin: a box narrower at the bottom
    verts = [bm.verts.new(v) for v in (
        (-L / 2 + 0.06, -D / 2 + 0.08, z0), (L / 2 - 0.06, -D / 2 + 0.08, z0), (L / 2 - 0.06, D / 2, z0), (-L / 2 + 0.06, D / 2, z0),
        (-L / 2, -D / 2, z0 + H), (L / 2, -D / 2, z0 + H), (L / 2, D / 2, z0 + H), (-L / 2, D / 2, z0 + H))]
    for f in ((0, 1, 2, 3), (4, 7, 6, 5), (0, 4, 5, 1), (1, 5, 6, 2), (2, 6, 7, 3), (3, 7, 4, 0)):
        bm.faces.new([verts[i] for i in f])
    for k in range(5):  # vertical ribs on the long sides
        x = -L / 2 + 0.2 + k * (L - 0.4) / 4
        for y in (-D / 2 - 0.02, D / 2 + 0.02):
            bmesh.ops.create_cube(bm, size=1.0, matrix=Matrix.Translation((x, y, z0 + H / 2)) @ Matrix.Diagonal((0.06, 0.05, H * 0.95, 1)))
    for x in (-L / 2 - 0.08, L / 2 + 0.08):  # lifting pockets
        bmesh.ops.create_cube(bm, size=1.0, matrix=Matrix.Translation((x, 0, z0 + H * 0.7)) @ Matrix.Diagonal((0.16, 0.5, 0.12, 1)))
    n_body = len(bm.faces)
    # lids: one shut, one propped open
    bmesh.ops.create_cube(bm, size=1.0, matrix=Matrix.Translation((-L / 4, 0.02, z0 + H + 0.03)) @ Matrix.Diagonal((L / 2 - 0.02, D + 0.06, 0.04, 1)))
    lid = Matrix.Translation((L / 4, D / 2 + 0.03, z0 + H + 0.03)) @ Matrix.Rotation(1.1, 4, "X") @ Matrix.Translation((0, -(D + 0.06) / 2, 0))
    bmesh.ops.create_cube(bm, size=1.0, matrix=lid @ Matrix.Diagonal((L / 2 - 0.02, D + 0.06, 0.04, 1)))
    for x in (-L / 2 + 0.15, L / 2 - 0.15):  # castors
        for y in (-D / 2 + 0.2, D / 2 - 0.2):
            bmesh.ops.create_cone(bm, cap_ends=True, segments=12, radius1=0.07, radius2=0.07, depth=0.06,
                                  matrix=Matrix.Translation((x, y, 0.07)) @ Matrix.Rotation(1.5708, 4, "X"))
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    me.materials.append(paint)
    me.materials.append(dark)
    for i, poly in enumerate(me.polygons):
        poly.material_index = 0 if i < n_body + 12 else 1
    return me


def obstacles(path_uv, half_width: float, seed: int = 31, clear: float = 0.42) -> int:
    """(10/28 evening, the user: weave round bins and big dumpsters on the run) Where the robot's
    path (u along the alley, v across it, in the set's frame) swings to one side, an obstacle stands on
    the other, just clear of it -- a steel dumpster, or a trash can with bags -- so the weave reads as
    dodging them. The weave itself was commanded in the simulation; the obstacles are set dressing."""
    import numpy as np

    root = bpy.data.objects.get("set")
    if root is None or len(path_uv) < 10:
        return 0
    rnd = random.Random(seed)
    # where it swerves hardest: peaks of its heading off the alley's line (200 Hz path, smoothed a
    # quarter second), at least 0.6 s apart -- the obstacle stands on the side it swerved away from
    sm = np.stack([np.convolve(path_uv[:, k], np.ones(51) / 51, mode="same") for k in (0, 1)], axis=1)
    d = np.gradient(sm, axis=0)
    head = np.arctan2(d[:, 1], np.maximum(d[:, 0], 1e-6))
    picks, last = [], -10**9
    for i in np.argsort(-np.abs(head)):
        if i < 60 or i > len(head) - 60 or abs(head[i]) < np.radians(12):
            continue
        if all(abs(i - j) > 120 for j in picks):
            picks.append(int(i))
    picks.sort()
    v = sm[:, 1]
    n = 0
    dump = _dumpster("dumpster")
    boxes = []  # (u0, u1, v0, v1) already standing
    for k, i in enumerate(picks):
        u, vv = float(sm[i, 0]) + 0.15, float(v[i])
        side = -1.0 if head[i] > 0 else 1.0  # swerving toward +v: the obstacle on the -v side
        for big in (True, False):  # a dumpster if it fits, else a trash can and bags
            hd, hl = (0.55, 0.95) if big else (0.32, 0.4)  # half depth across, half length along the alley
            c = vv + side * (clear + hd)
            near = np.abs(sm[:, 0] - u) < hl + 0.2  # the whole run must pass clear of it, not just here
            if near.any():
                gap = np.abs(v[near] - c).min()
                c += side * max(0.0, hd + 0.22 - gap)
            box = (u - hl, u + hl, c - hd, c + hd)
            fits = abs(c) <= half_width - hd - 0.05 and all(
                box[1] < b[0] - 0.1 or box[0] > b[1] + 0.1 or box[3] < b[2] - 0.1 or box[2] > b[3] + 0.1 for b in boxes)
            if fits:
                break
        if not fits:
            continue
        boxes.append(box)
        ob = bpy.data.objects.new(f"obstacle{k}", dump if big else None)
        bpy.context.scene.collection.objects.link(ob)
        ob.parent = root
        ob.location = (u + rnd.uniform(-0.1, 0.1), c, 0.0)
        ob.rotation_euler = (0.0, 0.0, rnd.uniform(-0.12, 0.12))
        if not big:
            # a scanned prop's origin is its back: turn it to reach away from the robot's path
            ob.rotation_euler = (0.0, 0.0, (0.0 if side < 0 else math.pi) + rnd.uniform(-0.3, 0.3))  # a trash can and a heap of bags, from the scanned props if they are loaded
            col = bpy.data.collections.get("lib_metal_trash_can")
            if col is not None:
                ob.instance_type, ob.instance_collection = "COLLECTION", col
            bags = bpy.data.collections.get("lib_trashbag")
            for j in range(3 if bags is not None else 0):
                bo = bpy.data.objects.new(f"obstacle{k}_bag{j}", None)
                bo.instance_type, bo.instance_collection = "COLLECTION", bags
                bpy.context.scene.collection.objects.link(bo)
                bo.parent = root
                bo.location = (u + rnd.uniform(-0.5, 0.5), c + side * rnd.uniform(0.1, 0.3), 0.0)
                bo.rotation_euler = (0.0, 0.0, rnd.uniform(0, 6.28))
        n += 1
    print(f"[props] {n} obstacles where the run swerved ({len(picks)} swerves)", flush=True)
    return n


def dress(lib: dict, seed: int = 5, lane: tuple = (-1.0, 1.0)) -> int:
    """Swap the back alley's box stand-ins for scanned models and fill the walls with more.
    `lane`: the band across the alley (v, m, in the set's frame) kept clear for the robot and the
    drones. Returns the number of props placed."""
    root = bpy.data.objects.get("set")
    floor = bpy.data.objects.get("alley_floor")
    if root is None or floor is None or not lib:
        return 0
    rnd = random.Random(seed)
    half = (floor.dimensions.y - 1.0) / 2
    u_lo = floor.location.x - floor.dimensions.x / 2 + 2.5
    u_hi = floor.location.x + floor.dimensions.x / 2 - 2.5
    libs = {mid: _library(mid, p) for mid, p in lib.items()}
    inv = root.matrix_world.inverted()
    taken = []  # (u0, u1, side) spans already used along each wall
    n = 0

    def free(u0, u1, side):
        return all(side != s or u1 < a - 0.05 or u0 > b + 0.05 for a, b, s in taken)

    def wall_yaw(side):  # the model's back (+Y) to the wall it stands against
        return 0.0 if side > 0 else math.pi

    # 1. the stand-ins: air-conditioners where the grey boxes stood, crates where the crate boxes stood
    for o in list(root.children):
        nm = o.name
        if nm.startswith("ac") and "aircon" not in nm and "exterior_aircon_unit" in libs:
            p = inv @ o.matrix_world.translation
            side = 1 if p.y > 0 else -1
            col, size = libs["exterior_aircon_unit"]
            z = max(0.0, p.z - o.dimensions.z / 2)
            s = 0.8 / max(size.x, 1e-3)  # as wide as the box it replaces
            _place(col, root, p.x, side * half, z, wall_yaw(side), s, f"aircon_{nm}")
            if z < 0.5:
                taken.append((p.x - 0.5, p.x + 0.5, side))
            n += 1
            for kid in [o] + [k for k in root.children if k.name == nm.replace("ac", "fan", 1)]:
                kid.hide_render = kid.hide_viewport = True
        elif nm.startswith("crate") and any(m in libs for m in ("plastic_crate_01", "plastic_crate_02")):
            p = inv @ o.matrix_world.translation
            side = 1 if p.y > 0 else -1
            mid = rnd.choice([m for m in ("plastic_crate_01", "plastic_crate_02") if m in libs])
            col, size = libs[mid]
            for k in range(rnd.randint(2, 4)):  # a stack
                _place(col, root, p.x + rnd.uniform(-0.03, 0.03), side * (half - 0.02), k * size.z * 0.98,
                       wall_yaw(side) + rnd.uniform(-0.08, 0.08), 1.0, f"{mid}_{nm}_{k}")
                n += 1
            taken.append((p.x - 0.4, p.x + 0.4, side))
            o.hide_render = o.hide_viewport = True
    # what else stands on the floor against the walls already (stairs, skip, steps, bags)
    for o in root.children:
        if o.type != "MESH" or o.hide_render or o.name.startswith(("wall", "alley_floor", "gutter", "grate", "win", "bar",
                                                                   "pipe", "conduit", "meter", "notice", "wire", "cable")):
            continue
        pts = [inv @ (o.matrix_world @ Vector(c)) for c in o.bound_box]
        if min(p.z for p in pts) > 0.6:
            continue
        v0, v1 = min(p.y for p in pts), max(p.y for p in pts)
        side = 1 if (v0 + v1) > 0 else -1
        taken.append((min(p.x for p in pts), max(p.x for p in pts), side))

    # 2. clusters along the walls: a bin with bags, crates, propane by a door, boxes, a drum, a tyre
    groups = [("metal_trash_can", 1, 1.0), ("trashbag", 4, 0.0), ("propane_tank", 2, 0.0), ("cardboard_box_01", 3, 0.0),
              ("Barrel_02", 1, 0.0), ("old_tyre", 1, 0.0), ("plastic_container", 2, 0.0), ("plastic_crate_03", 3, 0.0)]
    for rep in range(3):
        for mid, count, _ in groups:
            if mid not in libs:
                continue
            col, size = libs[mid]
            width = max(size.x, 0.3) * count * 0.8 + 0.2
            for _try in range(30):
                side = rnd.choice((-1, 1))
                u0 = rnd.uniform(u_lo, u_hi - width)
                if not free(u0, u0 + width, side):
                    continue
                for k in range(count):
                    uu = u0 + 0.1 + (k + 0.5) * width / max(count, 1) * 0.9
                    depth = min(size.y, 0.9)
                    if side * half - side * depth < lane[1] and side * half - side * depth > lane[0]:
                        continue
                    yaw = wall_yaw(side) + rnd.uniform(-0.35, 0.35)
                    lift = 0.0
                    if mid == "old_tyre" and rnd.random() < 0.5:  # leaning, or lying flat
                        yaw += rnd.uniform(-1, 1)
                    _place(col, root, uu, side * (half - 0.03 - rnd.uniform(0.0, 0.12)), lift, yaw, rnd.uniform(0.92, 1.08),
                           f"{mid}_{rep}_{k}")
                    n += 1
                taken.append((u0, u0 + width, side))
                break
    # 3. on the walls: utility boxes at chest height
    for k in range(6):
        mid = rnd.choice([m for m in ("utility_box_01", "utility_box_02") if m in libs] or [None])
        if mid is None:
            break
        col, size = libs[mid]
        side = rnd.choice((-1, 1))
        _place(col, root, rnd.uniform(u_lo, u_hi), side * half, rnd.uniform(1.0, 1.6), wall_yaw(side), 1.0, f"{mid}_w{k}")
        n += 1
    # 4. litter in the gutter line and by the walls; a manhole cover off the robot's lane
    for k in range(40):
        mid = rnd.choice([m for m in ("can_rusted", "plastic_bottle_gallon", "spray_paint_bottles_02") if m in libs] or [None])
        if mid is None:
            break
        col, size = libs[mid]
        side = rnd.choice((-1, 1))
        v = side * rnd.uniform(half - 0.5, half - 0.05)
        _place(col, root, rnd.uniform(u_lo, u_hi), v, 0.0, rnd.uniform(0, 2 * math.pi),
               rnd.uniform(0.8, 1.0) if mid != "plastic_bottle_gallon" else 0.6, f"{mid}_l{k}")
        n += 1
    if "water_manhole_cover" in libs:
        col, size = libs["water_manhole_cover"]
        for k, uu in enumerate((7.5, 17.0)):
            v = lane[1] + 0.5 if lane[1] + 0.5 + size.x / 2 < half else lane[0] - 0.5
            o = _place(col, root, uu, v + size.y / 2, 0.003, 0.0, 1.0, f"manhole{k}")
            n += 1
    print(f"[props] {n} placed ({len(libs)} models)", flush=True)
    return n
