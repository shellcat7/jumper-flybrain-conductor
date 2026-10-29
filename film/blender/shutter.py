"""The ending (10/28, Claude's proposal the user took): a roller shutter closing across the back
alley; the robot, on five legs, gets under it with a few centimetres to spare, and the drone that
dives after it hits the closing curtain and falls.

The shutter is set dressing keyed by hand on the take's clock: its curtain's bottom edge comes down
to a height chosen from where the robot is in the take (it does not touch the robot, and the
robot's motion is the recorded take, untouched). The drone's dive and crash are keyframed too, not
simulated -- the film's caption says so.

Built in the alley's own frame (`alley.build_backalley`'s "set" empty: +u along the alley, +v to its
left), so it spans wall to wall: a steel box header with the drum inside, two guide rails, a curtain
of corrugated galvanised slats (clipped where they run up into the box) with a bottom bar, grime
toward the floor and rain streaks. The curtain shudders when the drone hits.
"""

from __future__ import annotations

import math

import bmesh
import bpy
from mathutils import Matrix, Vector


def _mat(name, rgb, metallic, rough, clip_z=None):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    b = next(n for n in nt.nodes if n.type == "BSDF_PRINCIPLED")
    b.inputs["Metallic"].default_value = metallic
    # grime: darker and rougher toward the floor and in streaks running down from the slats
    geo = nt.nodes.new("ShaderNodeNewGeometry")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(geo.outputs["Position"], sep.inputs[0])
    wave = nt.nodes.new("ShaderNodeTexWave")
    wave.wave_type, wave.bands_direction = "BANDS", "X"
    wave.inputs["Scale"].default_value = 9.0
    wave.inputs["Distortion"].default_value = 6.0
    wave.inputs["Detail"].default_value = 4.0
    noise = nt.nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 22.0
    noise.inputs["Detail"].default_value = 8.0
    low = nt.nodes.new("ShaderNodeMapRange")
    nt.links.new(sep.outputs["Z"], low.inputs["Value"])
    low.inputs["From Min"].default_value, low.inputs["From Max"].default_value = 0.0, 0.9
    low.inputs["To Min"].default_value, low.inputs["To Max"].default_value = 0.45, 1.0
    dirt = nt.nodes.new("ShaderNodeMath")
    dirt.operation = "MULTIPLY"
    nt.links.new(low.outputs[0], dirt.inputs[0])
    streak = nt.nodes.new("ShaderNodeMapRange")
    nt.links.new(wave.outputs["Fac"], streak.inputs["Value"])
    streak.inputs["To Min"].default_value, streak.inputs["To Max"].default_value = 0.8, 1.0
    nt.links.new(streak.outputs[0], dirt.inputs[1])
    col = nt.nodes.new("ShaderNodeMix")
    col.data_type = "RGBA"
    col.inputs["A"].default_value = (rgb[0] * 0.35, rgb[1] * 0.33, rgb[2] * 0.3, 1)
    col.inputs["B"].default_value = (*rgb, 1)
    nt.links.new(dirt.outputs[0], col.inputs["Factor"])
    nt.links.new(col.outputs["Result"], b.inputs["Base Color"])
    r = nt.nodes.new("ShaderNodeMapRange")
    nt.links.new(noise.outputs["Fac"], r.inputs["Value"])
    r.inputs["To Min"].default_value, r.inputs["To Max"].default_value = rough * 0.7, rough * 1.4
    nt.links.new(r.outputs[0], b.inputs["Roughness"])
    if clip_z is not None:  # the curtain runs up into the header box: nothing above its bottom edge
        out = next(n for n in nt.nodes if n.type == "OUTPUT_MATERIAL")
        cut = nt.nodes.new("ShaderNodeMath")
        cut.operation = "LESS_THAN"
        nt.links.new(sep.outputs["Z"], cut.inputs[0])
        cut.inputs[1].default_value = clip_z
        hole = nt.nodes.new("ShaderNodeBsdfTransparent")
        mix = nt.nodes.new("ShaderNodeMixShader")
        nt.links.new(cut.outputs[0], mix.inputs["Fac"])
        nt.links.new(hole.outputs[0], mix.inputs[1])
        nt.links.new(b.outputs[0], mix.inputs[2])
        nt.links.new(mix.outputs[0], out.inputs["Surface"])
    return m


def _box(name, size, center, mat, parent):
    me = bpy.data.meshes.new(name)
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    bmesh.ops.scale(bm, vec=size, verts=bm.verts)
    bmesh.ops.translate(bm, vec=center, verts=bm.verts)
    bm.to_mesh(me)
    bm.free()
    me.materials.append(mat)
    o = bpy.data.objects.new(name, me)
    bpy.context.collection.objects.link(o)
    o.parent = parent
    bevel = o.modifiers.new("bevel", "BEVEL")
    bevel.width, bevel.segments = min(min(size) * 0.3, 0.004), 2
    return o


def _curtain(name, width, height, pitch, mat, parent):
    """Corrugated slats: each a shallow S in (u, z), swept across v; the bottom of slat 0 at z = 0."""
    me = bpy.data.meshes.new(name)
    bm = bmesh.new()
    half = width / 2
    prof = [(0.0, 0.0), (0.006, 0.008), (0.011, 0.02), (0.012, 0.035), (0.011, 0.05), (0.006, 0.062), (0.0, pitch * 0.92),
            (-0.002, pitch)]
    rows = int(height / pitch)
    pts = []
    for k in range(rows):
        for du, dz in prof:
            pts.append((du * (pitch / 0.075), k * pitch + dz * (pitch / 0.075)))
    left = [bm.verts.new((u, -half, z)) for u, z in pts]
    right = [bm.verts.new((u, half, z)) for u, z in pts]
    for i in range(len(pts) - 1):
        bm.faces.new((left[i], left[i + 1], right[i + 1], right[i]))
    bm.to_mesh(me)
    bm.free()
    for p in me.polygons:
        p.use_smooth = True
    me.materials.append(mat)
    o = bpy.data.objects.new(name, me)
    bpy.context.collection.objects.link(o)
    o.parent = parent
    sol = o.modifiers.new("thickness", "SOLIDIFY")
    sol.thickness = 0.0012
    return o


def add(set_root, u_s: float, width: float, frame_of, schedule, crash, drone_style="hostile", drone_scale=2.0):
    """The shutter across the alley at `u_s` (metres along it), the curtain's bottom keyed through
    `schedule` [(take seconds, height m)]; `crash`: the drone's dive {"start": (t, u, v, z),
    "hit": (t, v, z)} or None. `frame_of(t)` maps take seconds to scene frames."""
    head_z = 2.75
    paint = _mat("shutter_slat", (0.52, 0.54, 0.55), 0.55, 0.42, clip_z=head_z + 0.02)
    steel = _mat("shutter_steel", (0.32, 0.33, 0.34), 0.7, 0.5)
    alu = _mat("shutter_bar", (0.6, 0.6, 0.6), 0.9, 0.3)
    half = width / 2
    frame = bpy.data.objects.new("shutter", None)
    bpy.context.collection.objects.link(frame)
    frame.parent = set_root
    frame.location = (u_s, 0.0, 0.0)
    _box("shutter_header", (0.34, width + 0.1, 0.5), (0.0, 0.0, head_z + 0.25), steel, frame)
    for side in (-1, 1):  # C-section guide rails on the walls
        v = side * (half - 0.03)
        _box(f"shutter_rail_back{side}", (0.012, 0.07, head_z), (0.0, v + side * 0.03, head_z / 2), steel, frame)
        for du in (-0.02, 0.02):
            _box(f"shutter_rail_{side}_{du}", (0.004, 0.06, head_z), (du, v, head_z / 2), steel, frame)
    hinge = bpy.data.objects.new("shutter_hinge", None)  # the curtain hangs from the drum: it shudders about it
    bpy.context.collection.objects.link(hinge)
    hinge.parent = frame
    hinge.location = (0.0, 0.0, head_z)
    curtain_root = bpy.data.objects.new("shutter_curtain_root", None)
    bpy.context.collection.objects.link(curtain_root)
    curtain_root.parent = hinge
    _curtain("shutter_curtain", width - 0.06, 3.4, 0.075, paint, curtain_root).location = (0.0, 0.0, 0.05)
    _box("shutter_bottom_bar", (0.045, width - 0.06, 0.05), (0.0, 0.0, 0.025), alu, curtain_root)
    _box("shutter_rubber", (0.03, width - 0.08, 0.012), (0.0, 0.0, 0.004), _mat("shutter_rubber", (0.02, 0.02, 0.02), 0.0, 0.8),
         curtain_root)
    for t, h in schedule:
        curtain_root.location = (0.0, 0.0, h - head_z)
        curtain_root.keyframe_insert("location", frame=frame_of(t))
    # a cold tube on the wall past the shutter, lighting the curtain's inside face and the way out
    tube = bpy.data.lights.new("shutter_tube", "AREA")
    tube.shape, tube.size, tube.size_y = "RECTANGLE", 0.9, 0.05
    tube.energy, tube.color = 55.0, (0.78, 0.88, 1.0)
    tl = bpy.data.objects.new("shutter_tube", tube)
    bpy.context.collection.objects.link(tl)
    tl.parent = frame
    tl.location = (1.6, half - 0.15, 2.3)
    tl.rotation_euler = (math.radians(-60), 0.0, 0.0)
    if crash is None:
        return frame
    import drone as drone_kit
    import fx

    root = drone_kit.add_drone("chaser", scale=drone_scale, style=drone_style)
    root.parent = set_root
    t0, u0, v0, z0 = crash["start"]
    t1, v1, z1 = crash["hit"]
    u1 = u_s - 0.03 - 0.06 * drone_scale
    keys = [(t0, (u0, v0, z0), (0.0, 0.08, 0.0))]
    n = 8
    for k in range(1, n + 1):  # the dive: dropping as it closes, nose down
        a = k / n
        keys.append((t0 + (t1 - t0) * a, (u0 + (u1 - u0) * a, v0 + (v1 - v0) * a, z0 + (z1 - z0) * (a ** 1.6)),
                     (0.05 * math.sin(7 * a), 0.25 + 0.1 * a, 0.0)))
    # the hit: thrown back nose-up, over onto its back, a bounce and a skid (a pitch past -pi/2 only,
    # so it ends upside down and not righted)
    keys += [(t1 + 0.08, (u1 - 0.12, v1 + 0.03, z1 + 0.05), (0.3, -1.3, 0.2)),
             (t1 + 0.2, (u1 - 0.25, v1 + 0.06, z1 * 0.6), (0.5, -2.5, 0.5)),
             (t1 + 0.34, (u1 - 0.33, v1 + 0.09, 0.035), (0.3, -3.0, 0.7)),
             (t1 + 0.42, (u1 - 0.36, v1 + 0.1, 0.06), (0.15, -3.05, 0.75)),
             (t1 + 0.55, (u1 - 0.42, v1 + 0.12, 0.03), (0.05, -3.14, 0.8)),
             (t1 + 0.9, (u1 - 0.47, v1 + 0.13, 0.03), (0.05, -3.14, 0.85))]
    for t, loc, rot in keys:
        root.location = loc
        root.rotation_euler = rot  # +X forward: flying toward +u it faces along the alley
        root.keyframe_insert("location", frame=frame_of(t))
        root.keyframe_insert("rotation_euler", frame=frame_of(t))
    f0, f1 = frame_of(t0), frame_of(t1)
    for k, name in enumerate(root.get("props", [])):  # props spinning until the hit, then still
        prop = bpy.data.objects[name]
        for f in range(f0, f1 + 1):
            prop.rotation_euler = (0, 0, (-1) ** k * f * (math.pi / 2 + 0.31))
            prop.keyframe_insert("rotation_euler", frame=f)
    hit_world = set_root.matrix_world @ Vector((u1 + 0.06, v1, z1))
    fps = bpy.context.scene.render.fps
    fx.add_impact("shutter_hit", frame_of(t1), hit_world, "robot", seed=77, fps=fps)
    fx.add_impact("shutter_hit_b", frame_of(t1) + 1, hit_world + Vector((0, 0, 0.05)), "robot", seed=78, fps=fps)
    # the curtain shudders: a decaying swing about the drum
    f_hit = frame_of(t1)
    for k in range(0, 24):
        t = k / fps
        hinge.rotation_euler = (0.0, 0.012 * math.exp(-t / 0.25) * math.sin(2 * math.pi * 6.0 * t) if k else 0.0, 0.0)
        hinge.keyframe_insert("rotation_euler", frame=f_hit + k)
    hinge.rotation_euler = (0.0, 0.0, 0.0)
    hinge.keyframe_insert("rotation_euler", frame=f_hit - 1)
    print(f"[shutter] at u = {u_s:.2f} m, {len(schedule)} keys; chaser hits at {t1:.2f} s (frame {f_hit})", flush=True)
    return frame
