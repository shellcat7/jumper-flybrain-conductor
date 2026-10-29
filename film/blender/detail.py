"""Geometry detail for the back alley at the robot's eye height (10/28). The user's note: shiny
materials are not enough -- model the fine detail.

- **True displacement** on the floor, walls and steps: Cycles' adaptive subdivision dices them to
  about a pixel and the scanned height maps (`photoreal.pbr`) push real relief -- asphalt grit,
  cracks, pitted concrete -- that a camera 5-10 cm off the ground sees in silhouette.
- **Debris** on the ground: a few thousand pebbles and grit instanced along the walls and gutter,
  cigarette ends, leaves and bottle caps.
- **Hardware** on the set's boxes: air-conditioner grilles and fan guards, pipe clamps and wall
  brackets, door frames, handles and hinges, slotted drain grates, window frames and sills.

Run after `photoreal.upgrade_set` (it reads the materials that names). The set's frame: +x down the
alley, +y to its left, z up; everything is parented to the `set` empty.
"""

from __future__ import annotations

import math
import random

import bmesh
import bpy
from mathutils import Vector


def _link(obj, root):
    if obj.name not in bpy.context.scene.collection.objects:
        bpy.context.scene.collection.objects.link(obj)
    obj.parent = root
    return obj


def _box_mesh(name, size):
    me = bpy.data.meshes.new(name)
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    bmesh.ops.scale(bm, vec=Vector(size), verts=bm.verts)
    bm.to_mesh(me)
    bm.free()
    return me


def _box(name, size, loc, mat, root, rot=(0.0, 0.0, 0.0), bevel=0.0):
    obj = bpy.data.objects.new(name, _box_mesh(name, size))
    obj.location, obj.rotation_euler = loc, rot
    obj.data.materials.append(mat)
    if bevel:
        m = obj.modifiers.new("bevel", "BEVEL")
        m.width, m.segments = bevel, 2
    return _link(obj, root)


def _cyl(name, r, depth, loc, mat, root, rot=(0.0, 0.0, 0.0), verts=20):
    me = bpy.data.meshes.new(name)
    bm = bmesh.new()
    bmesh.ops.create_cone(bm, cap_ends=True, segments=verts, radius1=r, radius2=r, depth=depth)
    bm.to_mesh(me)
    bm.free()
    obj = bpy.data.objects.new(name, me)
    obj.location, obj.rotation_euler = loc, rot
    obj.data.materials.append(mat)
    for p in obj.data.polygons:
        p.use_smooth = True
    return _link(obj, root)


def _mat(name, rgb, rough, metallic=0.0):
    m = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    m.use_nodes = True
    b = next(n for n in m.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
    b.inputs["Base Color"].default_value = (*rgb, 1.0)
    b.inputs["Roughness"].default_value = rough
    b.inputs["Metallic"].default_value = metallic
    return m


def true_displacement(scales=None, dicing: float = 1.0) -> None:
    """Adaptive subdivision and the height maps as real displacement on the scanned surfaces."""
    scene = bpy.context.scene
    if hasattr(scene.cycles, "feature_set"):  # Blender 4: adaptive subdivision is experimental
        scene.cycles.feature_set = "EXPERIMENTAL"
    scene.cycles.dicing_rate = dicing
    scene.cycles.offscreen_dicing_scale = 8.0
    scales = scales or {"pr_floor": 0.012, "pr_steps": 0.01, "pr_wall_a": 0.02, "pr_wall_b": 0.025, "pr_wall_c": 0.02}
    for mat_name, scale in scales.items():
        mat = bpy.data.materials.get(mat_name)
        if mat is None:
            continue
        nt = mat.node_tree
        height = next((n for n in nt.nodes if n.type == "TEX_IMAGE" and n.image and "_disp_" in n.image.name), None)
        out = next(n for n in nt.nodes if n.type == "OUTPUT_MATERIAL")
        if height is None:
            continue
        disp = nt.nodes.new("ShaderNodeDisplacement")
        disp.inputs["Scale"].default_value = scale
        disp.inputs["Midlevel"].default_value = 0.5
        # the floor's puddles: displace the flattened height the material already mixes for them
        flat = next((n for n in nt.nodes if n.type == "MIX" and n.data_type == "FLOAT"
                     and n.inputs["B"].default_value == 0.2), None)
        nt.links.new((flat.outputs["Result"] if flat else height.outputs["Color"]), disp.inputs["Height"])
        nt.links.new(disp.outputs["Displacement"], out.inputs["Displacement"])
        if hasattr(mat, "displacement_method"):
            mat.displacement_method = "BOTH"
        else:
            mat.cycles.displacement_method = "BOTH"
        if hasattr(mat, "max_vertex_displacement"):  # Blender 5 bounds the displacement by this
            mat.max_vertex_displacement = scale
    n = 0
    for obj in bpy.data.objects:
        if obj.type != "MESH" or not obj.data.materials or obj.data.materials[0] is None:
            continue
        if obj.data.materials[0].name in scales:
            sub = obj.modifiers.new("adaptive", "SUBSURF")
            sub.subdivision_type = "SIMPLE"
            sub.levels, sub.render_levels = 0, 1
            if hasattr(sub, "use_adaptive_subdivision"):  # Blender 5: on the modifier
                sub.use_adaptive_subdivision = True
                if hasattr(sub, "adaptive_pixel_size"):
                    sub.adaptive_pixel_size = dicing
            elif hasattr(obj, "cycles") and hasattr(obj.cycles, "use_adaptive_subdivision"):
                obj.cycles.use_adaptive_subdivision = True
            n += 1
    print(f"[detail] true displacement on {n} objects (dicing {dicing} px)")


def _rock(name, rnd, mat):
    me = bpy.data.meshes.new(name)
    bm = bmesh.new()
    bmesh.ops.create_icosphere(bm, subdivisions=2, radius=1.0)
    for v in bm.verts:
        n = v.co.normalized()
        v.co = n * (1.0 + 0.25 * math.sin(4.1 * n.x + rnd.random() * 6) * math.cos(3.3 * n.y + rnd.random() * 6))
        v.co.z *= rnd.uniform(0.45, 0.8)
    bm.to_mesh(me)
    bm.free()
    obj = bpy.data.objects.new(name, me)
    obj.data.materials.append(mat)
    for p in obj.data.polygons:
        p.use_smooth = True
    return obj


def scatter_debris(seed: int = 3, pebbles: int = 6000) -> None:
    """Pebbles, grit, cigarette ends, leaves and caps instanced on the ground, thickest along the walls."""
    rnd = random.Random(seed)
    root = bpy.data.objects.get("set")
    floor = bpy.data.objects.get("alley_floor")
    if root is None or floor is None:
        return
    length = floor.dimensions.x - 4.0
    half = (floor.dimensions.y - 1.0) / 2
    x0 = floor.location.x - floor.dimensions.x / 2 + 2.0
    lib = bpy.data.collections.new("debris_lib")
    bpy.context.scene.collection.children.link(lib)
    lib.hide_render = False
    stone = _mat("pebble", (0.32, 0.31, 0.3), 0.85)
    stone_dark = _mat("pebble_dark", (0.12, 0.12, 0.12), 0.7)
    for k in range(10):
        r = _rock(f"pebble_src{k}", rnd, stone if k % 3 else stone_dark)
        lib.objects.link(r)
        r.location = (-100.0, -100.0 - k, -10.0)  # off stage: only the instances render
    butt = _mat("butt", (0.85, 0.82, 0.75), 0.8)
    filt = _mat("butt_filter", (0.75, 0.45, 0.2), 0.8)
    leaf = _mat("leaf", (0.25, 0.17, 0.06), 0.6)
    cap = _mat("cap", (0.6, 0.6, 0.62), 0.35, metallic=0.8)

    def instance(src, loc, scale, rot):
        o = bpy.data.objects.new(f"{src.name}_i", src.data)
        o.location, o.scale, o.rotation_euler = loc, scale, rot
        return _link(o, root)

    # pebbles: a few big ones and many small, in a band along each wall and the gutter
    srcs = list(lib.objects)
    for k in range(pebbles):
        band = rnd.random()
        if band < 0.75:  # along a wall
            side = rnd.choice((-1, 1))
            y = side * (half - abs(rnd.gauss(0.0, 0.18)))
        else:  # anywhere
            y = rnd.uniform(-half + 0.1, half - 0.1)
        x = x0 + rnd.uniform(0.0, length)
        s = rnd.choice((0.004, 0.006, 0.008, 0.012, 0.02)) * rnd.uniform(0.7, 1.3)
        instance(rnd.choice(srcs), (x, y, s * 0.3), (s, s, s), (0.0, 0.0, rnd.uniform(0, 6.3)))
    for k in range(80):  # cigarette ends
        x, y = x0 + rnd.uniform(0, length), rnd.uniform(-half + 0.1, half - 0.1)
        a = rnd.uniform(0, 6.3)
        _cyl(f"butt{k}", 0.0037, 0.018, (x, y, 0.0037), butt, root, rot=(math.pi / 2, 0.0, a), verts=10)
        _cyl(f"filter{k}", 0.0038, 0.012, (x + 0.015 * math.cos(a + math.pi / 2), y + 0.015 * math.sin(a + math.pi / 2), 0.0038),
             filt, root, rot=(math.pi / 2, 0.0, a), verts=10)
    for k in range(160):  # dead leaves, curled
        me = bpy.data.meshes.new(f"leaf{k}")
        bm = bmesh.new()
        s = rnd.uniform(0.02, 0.05)
        pts = [Vector((0, -s / 2, 0)), Vector((s * 0.35, 0, s * 0.12)), Vector((0, s / 2, 0)), Vector((-s * 0.35, 0, s * 0.1))]
        f = bm.faces.new([bm.verts.new(p) for p in pts])
        bmesh.ops.subdivide_edges(bm, edges=list(f.edges), cuts=2, use_grid_fill=True)
        bm.to_mesh(me)
        bm.free()
        side = rnd.choice((-1, 1))
        o = bpy.data.objects.new(f"leaf{k}", me)
        o.location = (x0 + rnd.uniform(0, length), side * (half - abs(rnd.gauss(0, 0.4))), 0.002)
        o.rotation_euler = (rnd.uniform(-0.3, 0.3), rnd.uniform(-0.3, 0.3), rnd.uniform(0, 6.3))
        o.data.materials.append(leaf)
        _link(o, root)
    for k in range(25):  # bottle caps
        _cyl(f"cap{k}", 0.014, 0.006, (x0 + rnd.uniform(0, length), rnd.uniform(-half + 0.1, half - 0.1), 0.003),
             cap, root, verts=21)
    print(f"[detail] debris: {pebbles} pebbles, 80 cigarette ends, 160 leaves, 25 caps")


def hardware() -> None:
    """Grilles, guards, clamps, frames, handles and slotted grates on the set's boxes."""
    root = bpy.data.objects.get("set")
    if root is None:
        return
    steel = _mat("hw_steel", (0.35, 0.36, 0.37), 0.35, metallic=0.9)
    dark = _mat("hw_dark", (0.03, 0.03, 0.03), 0.5, metallic=0.6)
    white = _mat("hw_white", (0.75, 0.74, 0.7), 0.5)
    frame = _mat("hw_frame", (0.18, 0.19, 0.2), 0.45, metallic=0.7)
    counts = {"ac": 0, "pipe": 0, "door": 0, "grate": 0, "win": 0}
    for obj in list(bpy.data.objects):
        if obj.type != "MESH" or obj.parent != root:
            continue
        n = obj.name
        loc, dim = obj.location, obj.dimensions
        if n.startswith("ac") and not n.startswith("ac_"):  # an air-conditioner: slats, a fan guard, feet, pipes
            side = 1 if loc.y > 0 else -1
            face_y = loc.y - side * dim.y / 2
            for k in range(9):
                _box(f"{n}_slat{k}", (dim.x * 0.42, 0.012, 0.006), (loc.x + dim.x * 0.22, face_y - side * 0.004,
                     loc.z - dim.z * 0.38 + k * dim.z * 0.085), dark, root, rot=(math.radians(-25 * side), 0, 0))
            for k, r in enumerate((0.2, 0.14, 0.08)):
                bpy.ops.mesh.primitive_torus_add(major_radius=r, minor_radius=0.004, location=(loc.x - dim.x * 0.18, face_y - side * 0.006, loc.z))
                t = bpy.context.active_object
                t.name = f"{n}_guard{k}"
                t.rotation_euler = (math.pi / 2, 0, 0)
                t.data.materials.append(dark)
                _link(t, root)
            for k in range(6):
                a = k * math.pi / 3
                _box(f"{n}_spoke{k}", (0.4, 0.006, 0.006), (loc.x - dim.x * 0.18, face_y - side * 0.006, loc.z), dark, root,
                     rot=(0, a, 0))
            for k in (-1, 1):
                _box(f"{n}_foot{k}", (0.06, dim.y * 1.1, 0.04), (loc.x + k * dim.x * 0.35, loc.y, loc.z - dim.z / 2 - 0.02), steel, root)
            _cyl(f"{n}_pipe_a", 0.009, 0.5, (loc.x + dim.x * 0.42, loc.y + side * 0.05, loc.z + 0.25), white, root, verts=14)
            _cyl(f"{n}_pipe_b", 0.006, 0.5, (loc.x + dim.x * 0.45, loc.y + side * 0.05, loc.z + 0.25), white, root, verts=14)
            counts["ac"] += 1
        elif n.startswith("pipe"):  # clamps every metre and a wall bracket with bolts
            r = max(dim.x, dim.y) / 2
            for k in range(int(dim.z)):
                z = loc.z - dim.z / 2 + 0.6 + k
                bpy.ops.mesh.primitive_torus_add(major_radius=r + 0.006, minor_radius=0.008, location=(loc.x, loc.y, z))
                c = bpy.context.active_object
                c.name = f"{n}_clamp{k}"
                c.data.materials.append(steel)
                _link(c, root)
                side = 1 if loc.y > 0 else -1
                _box(f"{n}_bracket{k}", (0.03, 0.08, 0.02), (loc.x, loc.y + side * 0.05, z), steel, root)
            counts["pipe"] += 1
        elif n.startswith("door"):  # frame, handle, hinges, kick plate
            side = 1 if loc.y > 0 else -1
            fy = loc.y - side * 0.035
            for k, (sx, sz, dx, dz) in enumerate(((dim.x + 0.12, 0.06, 0, dim.z / 2 + 0.03), (0.06, dim.z + 0.06, -dim.x / 2 - 0.03, 0),
                                                  (0.06, dim.z + 0.06, dim.x / 2 + 0.03, 0))):
                _box(f"{n}_frame{k}", (sx, 0.05, sz), (loc.x + dx, fy, loc.z + dz), frame, root, bevel=0.004)
            _cyl(f"{n}_knob", 0.022, 0.06, (loc.x + dim.x * 0.36, fy - side * 0.03, loc.z), steel, root, rot=(math.pi / 2, 0, 0))
            _box(f"{n}_lever", (0.12, 0.018, 0.02), (loc.x + dim.x * 0.3, fy - side * 0.06, loc.z), steel, root, bevel=0.004)
            for k in (-1, 1):
                _cyl(f"{n}_hinge{k}", 0.012, 0.11, (loc.x - dim.x / 2 + 0.01, fy - side * 0.01, loc.z + k * dim.z * 0.35), steel, root)
            _box(f"{n}_kick", (dim.x * 0.9, 0.006, 0.22), (loc.x, fy - side * 0.004, loc.z - dim.z / 2 + 0.13), steel, root)
            counts["door"] += 1
        elif n.startswith("grate"):  # slots
            obj.hide_render = True
            for k in range(12):
                _box(f"{n}_bar{k}", (0.018, dim.y, 0.012), (loc.x - dim.x / 2 + 0.02 + k * dim.x / 12, loc.y, 0.005), dark, root)
            for k in (-1, 1):
                _box(f"{n}_rim{k}", (dim.x, 0.02, 0.014), (loc.x, loc.y + k * dim.y / 2, 0.006), dark, root)
            counts["grate"] += 1
        elif n.startswith("win") and not n.startswith("win_"):  # frame and sill
            side = 1 if loc.y > 0 else -1
            _box(f"{n}_sill", (dim.x + 0.1, 0.12, 0.04), (loc.x, loc.y - side * 0.05, loc.z - dim.z / 2 - 0.03), white, root, bevel=0.005)
            for k, (sx, sz, dx, dz) in enumerate(((dim.x + 0.06, 0.04, 0, dim.z / 2), (0.04, dim.z, -dim.x / 2, 0), (0.04, dim.z, dim.x / 2, 0))):
                _box(f"{n}_wf{k}", (sx, 0.05, sz), (loc.x + dx, loc.y - side * 0.02, loc.z + dz), frame, root)
            counts["win"] += 1
    print(f"[detail] hardware: {counts}")
