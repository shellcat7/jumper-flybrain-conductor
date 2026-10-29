"""The film's effects (10/26): laser bolts, their impacts, and drones that stay on station.

After the reference frames for the assault: a bolt is a thin white-hot core inside a red sheath
and a faint glow, a streak about 70 ms of flight long; where it lands it throws sparks, chips of
asphalt and a puff of smoke, flashes, and leaves a glowing scorch. The take decides everything
that happens: where each bolt flew, which ones hit (the recorder's "struck"), which leg died
("wounded"); a bolt that missed is drawn on along its straight line to where it meets the ground.
"""

from __future__ import annotations

import math
import random

import bpy
import numpy as np
from mathutils import Vector


def _bsdf(mat):
    nt = mat.node_tree
    bsdf = next((n for n in nt.nodes if n.type == "BSDF_PRINCIPLED"), None)
    if bsdf is None:
        bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
        out = next((n for n in nt.nodes if n.type == "OUTPUT_MATERIAL"), None) or nt.nodes.new("ShaderNodeOutputMaterial")
        nt.links.new(bsdf.outputs[0], out.inputs["Surface"])
    return bsdf


def glow(name, rgb, strength, alpha=1.0):
    mat = bpy.data.materials.get(name)
    if mat is None:
        mat = bpy.data.materials.new(name)
        mat.use_nodes = True
        b = _bsdf(mat)
        b.inputs["Base Color"].default_value = (0.0, 0.0, 0.0, 1.0)
        b.inputs["Emission Color"].default_value = (*rgb, 1.0)
        b.inputs["Emission Strength"].default_value = strength
        if alpha < 1.0:
            b.inputs["Alpha"].default_value = alpha
            for attr, value in (("surface_render_method", "BLENDED"), ("blend_method", "BLEND")):
                if hasattr(mat, attr):
                    setattr(mat, attr, value)
        mat.diffuse_color = (*rgb, 1.0)
    return mat


def _paint(name, rgb, roughness=0.6):
    mat = bpy.data.materials.get(name)
    if mat is None:
        mat = bpy.data.materials.new(name)
        mat.use_nodes = True
        b = _bsdf(mat)
        b.inputs["Base Color"].default_value = (*rgb, 1.0)
        b.inputs["Roughness"].default_value = roughness
        mat.diffuse_color = (*rgb, 1.0)
    return mat


def _cylinder_x(name, radius, mat, verts=16):
    """A unit-length cylinder from x = -1 to x = 0 (a streak's tail to its head)."""
    bpy.ops.mesh.primitive_cylinder_add(radius=radius, depth=1.0, vertices=verts)
    obj = bpy.context.active_object
    obj.name = name
    obj.data.transform(np.array([[0, 0, 1, -0.5], [0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 1]]).tolist())
    obj.data.materials.append(mat)
    return obj


def _hide_key(objs, hidden, frame):
    for o in objs:
        o.hide_render = hidden
        o.hide_viewport = hidden
        o.keyframe_insert("hide_render", frame=frame)
        o.keyframe_insert("hide_viewport", frame=frame)


def flights(track, base, frame_dt, near=3.6):
    """The bolt flights in one ball's track (frames x 3): (first frame, last frame, start, velocity),
    a flight starting where the ball appears within `near` m of the robot's base."""
    close = np.linalg.norm(track[:, :2] - base[:, :2], axis=-1) < near
    # A throw teleports the ball to its start: a jump no flight makes in one frame (a spent bolt
    # can lie within `near` of the robot, so arriving there is not enough to mark a new throw).
    leap = np.r_[False, np.linalg.norm(np.diff(track, axis=0), axis=-1) > 0.6]
    out = []
    f = 1
    while f < len(track) - 1:
        if close[f] and (not close[f - 1] or leap[f]):
            g = f + 1
            while g < len(track) and close[g] and not leap[g]:
                g += 1
            v = (track[f + 1] - track[f]) / frame_dt
            out.append((f, g - 1, track[f].copy(), v))
            f = g
        else:
            f += 1
    return out


def add_laser(name, frames, flights_, struck_frames, frame_dt, length_s=0.07, light_w=0.6):
    """One ball's bolts as laser streaks. `flights_` from `flights`; `struck_frames` maps a
    flight's first frame to the frame it hit the robot (or None: it missed and runs on to the
    ground). Frames here are the take's indices, keyed one later (Blender counts from 1).
    Returns the impacts [(index, position, "robot" | "ground")]."""
    holder = bpy.data.objects.new(name, None)
    bpy.context.collection.objects.link(holder)
    holder.rotation_mode = "QUATERNION"
    core = _cylinder_x(f"{name}_core", 0.0018, glow("laser_core", (1.0, 0.8, 0.72), 30.0))
    sheath = _cylinder_x(f"{name}_sheath", 0.0045, glow("laser_sheath", (1.0, 0.06, 0.015), 10.0, alpha=0.45))
    halo = _cylinder_x(f"{name}_halo", 0.011, glow("laser_halo", (1.0, 0.03, 0.006), 2.0, alpha=0.06))
    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.0055, segments=16, ring_count=8)
    tip = bpy.context.active_object
    tip.name = f"{name}_tip"
    tip.data.materials.append(glow("laser_tip", (1.0, 0.75, 0.6), 30.0))
    light = bpy.data.objects.new(f"{name}_light", bpy.data.lights.new(f"{name}_light", type="POINT"))
    light.data.energy, light.data.color, light.data.shadow_soft_size = light_w, (1.0, 0.18, 0.05), 0.01
    bpy.context.collection.objects.link(light)
    parts = [core, sheath, halo, tip, light]
    for o in parts:
        o.parent = holder
    _hide_key(parts, True, 1)
    impacts = []
    for f0, f1, p0, v in flights_:
        speed = float(np.linalg.norm(v))
        if speed < 0.5:
            continue
        hit = struck_frames.get(f0)
        if hit is not None:
            end = hit
        else:  # a miss: on along the line until it meets the ground
            t_ground = (p0[2] - 0.004) / max(-v[2], 1e-3) if v[2] < 0 else 2.0
            end = min(frames, f0 + int(math.ceil(t_ground / frame_dt)))
        d = Vector(v.tolist()).normalized()
        holder.rotation_quaternion = d.to_track_quat("X", "Z")
        holder.keyframe_insert("rotation_quaternion", frame=f0 + 1)  # frame f + 1 shows index f, as the robot's
        for f in range(f0, end + 1):
            t = (f - f0) * frame_dt
            head = p0 + v * t
            if f == end:
                if hit is None:
                    head = p0 + v * ((p0[2] - 0.004) / max(-v[2], 1e-3))
                impacts.append((f, head.copy(), "robot" if hit is not None else "ground"))
            holder.location = head.tolist()
            holder.scale = (max(min(speed * length_s, speed * t), 0.01), 1.0, 1.0)
            holder.keyframe_insert("location", frame=f + 1)
            holder.keyframe_insert("scale", frame=f + 1)
        _hide_key(parts, False, f0 + 1)
        _hide_key(parts, True, end + 2)
    return impacts


def add_impact(name, frame, pos, kind, seed=0, fps=50.0):
    """Sparks, a flash and, on the ground, chips of asphalt, a smoke puff and a glowing scorch."""
    rnd = random.Random(seed)
    p = Vector(pos.tolist() if hasattr(pos, "tolist") else pos)
    if kind == "ground":
        p.z = 0.002
    made = []
    flash = bpy.data.objects.new(f"{name}_flash", bpy.data.lights.new(f"{name}_flash", type="POINT"))
    flash.data.color, flash.data.shadow_soft_size = (1.0, 0.42, 0.12), 0.02
    bpy.context.collection.objects.link(flash)
    flash.location = p + Vector((0.0, 0.0, 0.03))
    for f, e in ((frame - 1, 0.0), (frame, 60.0), (frame + 2, 22.0), (frame + 5, 4.0), (frame + 9, 0.0)):
        flash.data.energy = e
        flash.data.keyframe_insert("energy", frame=f)
    made.append(flash)
    # sparks: a particle burst of short glowing streaks, aligned with their velocity
    bpy.ops.mesh.primitive_ico_sphere_add(radius=0.004, subdivisions=1, location=p)
    emitter = bpy.context.active_object
    emitter.name = f"{name}_sparks"
    bpy.ops.object.particle_system_add()
    ps = emitter.particle_systems[0].settings
    ps.count = 140 if kind == "ground" else 90
    ps.frame_start, ps.frame_end = frame, frame + 1
    ps.lifetime, ps.lifetime_random = int(0.22 * fps), 0.6
    ps.emit_from = "VERT"
    ps.normal_factor = 0.0
    ps.object_align_factor = (0.0, 0.0, 1.6 if kind == "ground" else 0.4)
    ps.factor_random = 2.6
    ps.render_type = "OBJECT"
    spark = _cylinder_x(f"{name}_spark", 0.0011, glow("spark", (1.0, 0.5, 0.12), 30.0), verts=6)
    spark.scale = (0.022, 1.0, 1.0)
    spark.location = (0.0, -50.0, -50.0)  # the instanced shape, out of shot
    ps.instance_object = spark
    ps.particle_size, ps.size_random = 1.0, 0.6
    ps.use_rotations, ps.rotation_mode = True, "VEL"
    ps.effector_weights.gravity = 1.0
    emitter.show_instancer_for_render = False
    emitter.show_instancer_for_viewport = False
    made += [emitter, spark]
    if kind != "ground":
        return made
    chip_mat = _paint("asphalt_chip", (0.02, 0.02, 0.022), 0.7)
    for k in range(10):  # chips of asphalt on ballistic arcs, resting where they land
        bpy.ops.mesh.primitive_ico_sphere_add(radius=rnd.uniform(0.003, 0.008), subdivisions=1)
        chip = bpy.context.active_object
        chip.name = f"{name}_chip{k}"
        chip.scale = (rnd.uniform(0.6, 1.4), rnd.uniform(0.6, 1.4), rnd.uniform(0.5, 1.0))
        chip.data.materials.append(chip_mat)
        a, up, spd = rnd.uniform(0, 2 * math.pi), rnd.uniform(0.9, 1.4), rnd.uniform(1.0, 2.6)
        v = Vector((math.cos(a) * math.cos(up), math.sin(a) * math.cos(up), math.sin(up))) * spd
        spin = Vector((rnd.uniform(-20, 20), rnd.uniform(-20, 20), rnd.uniform(-20, 20)))
        _hide_key([chip], True, frame - 1)
        _hide_key([chip], False, frame)
        landed = None
        for i in range(int(0.9 * fps)):
            t = i / fps
            q = p + v * t + Vector((0.0, 0.0, -0.5 * 9.81 * t * t))
            if q.z < 0.003:
                landed = landed or Vector((q.x, q.y, 0.003))
                q = landed
            chip.location = q
            chip.rotation_euler = (spin * min(t, 0.5)).to_tuple()
            chip.keyframe_insert("location", frame=frame + i)
            chip.keyframe_insert("rotation_euler", frame=frame + i)
        made.append(chip)
    # the scorch: a dark disc with an ember ring that fades over a second
    bpy.ops.mesh.primitive_circle_add(radius=0.04, vertices=24, fill_type="NGON", location=(p.x, p.y, 0.0012))
    scorch = bpy.context.active_object
    scorch.name = f"{name}_scorch"
    scorch.data.materials.append(_paint("scorch", (0.004, 0.003, 0.003), 0.9))
    _hide_key([scorch], True, frame - 1)
    _hide_key([scorch], False, frame)
    bpy.ops.mesh.primitive_torus_add(major_radius=0.026, minor_radius=0.006, location=(p.x, p.y, 0.0015))
    ember = bpy.context.active_object
    ember.name = f"{name}_ember"
    ember.scale = (1.0, 1.0, 0.15)
    mat = glow(f"{name}_ember", (1.0, 0.3, 0.06), 12.0)
    ember.data.materials.append(mat)
    strength = _bsdf(mat).inputs["Emission Strength"]
    for f, s in ((frame - 1, 0.0), (frame, 16.0), (frame + int(0.3 * fps), 5.0), (frame + int(1.2 * fps), 0.0)):
        strength.default_value = s
        strength.keyframe_insert("default_value", frame=f)
    _hide_key([ember], True, frame - 1)
    _hide_key([ember], False, frame)
    _hide_key([ember], True, frame + int(1.2 * fps) + 1)  # spent, the ring goes; the scorch stays
    made += [scorch, ember]
    # a puff of smoke: a volume ball that swells and thins
    bpy.ops.mesh.primitive_ico_sphere_add(radius=1.0, subdivisions=3, location=(p.x, p.y, 0.05))
    puff = bpy.context.active_object
    puff.name = f"{name}_smoke"
    smoke = bpy.data.materials.new(f"{name}_smoke")
    smoke.use_nodes = True
    nt = smoke.node_tree
    for n in list(nt.nodes):
        if n.type == "BSDF_PRINCIPLED":
            nt.nodes.remove(n)
    out = next(n for n in nt.nodes if n.type == "OUTPUT_MATERIAL")
    vol = nt.nodes.new("ShaderNodeVolumePrincipled")
    vol.inputs["Color"].default_value = (0.55, 0.55, 0.6, 1.0)
    noise = nt.nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 3.0
    grad = nt.nodes.new("ShaderNodeTexGradient")  # thinner toward the ball's edge
    grad.gradient_type = "SPHERICAL"
    mul = nt.nodes.new("ShaderNodeMath")
    mul.operation = "MULTIPLY"
    nt.links.new(noise.outputs["Fac"], mul.inputs[0])
    nt.links.new(grad.outputs["Fac"], mul.inputs[1])
    dens = nt.nodes.new("ShaderNodeMath")
    dens.operation = "MULTIPLY"
    nt.links.new(mul.outputs[0], dens.inputs[0])
    nt.links.new(dens.outputs[0], vol.inputs["Density"])
    nt.links.new(vol.outputs[0], out.inputs["Volume"])
    puff.data.materials.append(smoke)
    for f, s, d in ((frame - 1, 0.02, 0.0), (frame + 2, 0.05, 30.0), (frame + int(0.6 * fps), 0.16, 10.0),
                    (frame + int(1.6 * fps), 0.3, 0.0)):
        puff.scale = (s, s, s * 0.8)
        puff.location = (p.x, p.y, 0.02 + s * 0.6)
        puff.keyframe_insert("scale", frame=f)
        puff.keyframe_insert("location", frame=f)
        dens.inputs[1].default_value = d
        dens.inputs[1].keyframe_insert("default_value", frame=f)
    made.append(puff)
    return made


def wound_sparks(name, holder, frame, last, fps=50.0):
    """Sparks spitting from a dead leg's joint now and then, from `frame` to `last`."""
    bpy.ops.mesh.primitive_ico_sphere_add(radius=0.004, subdivisions=1)
    emitter = bpy.context.active_object
    emitter.name = f"{name}_wound"
    emitter.parent = holder
    emitter.location = (0.0, 0.0, 0.0)
    bpy.ops.object.particle_system_add()
    ps = emitter.particle_systems[0].settings
    ps.count = int(30 * (last - frame) / fps)
    ps.frame_start, ps.frame_end = frame, last
    ps.lifetime, ps.lifetime_random = int(0.15 * fps), 0.7
    ps.emit_from = "VERT"
    ps.normal_factor = 0.0
    ps.object_align_factor = (0.0, 0.0, 0.5)
    ps.factor_random = 1.2
    ps.render_type = "OBJECT"
    spark = _cylinder_x(f"{name}_wspark", 0.0009, glow("spark", (1.0, 0.5, 0.12), 30.0), verts=6)
    spark.scale = (0.014, 1.0, 1.0)
    spark.location = (0.0, -50.0, -50.0)
    ps.instance_object = spark
    ps.use_rotations, ps.rotation_mode = True, "VEL"
    emitter.show_instancer_for_render = False
    emitter.show_instancer_for_viewport = False
    return emitter


def drone_pool(launches, base, quat, frames, fps, add_drone, muzzle, scale, style, key_fire, descend_s=1.4,
               chase=False, pull_up_s=None):
    """One drone per firing direction (the launch's bearing in the robot's frame, to 45 degrees),
    keyed through its launch points -- the emitter at each bolt's start on its frame -- gliding
    between them, dropping in from above at the start, always facing the robot.
    `launches`: [(frame, start xyz)].
    (10/28, the user: the enemy was never seen, and the robot seemed to walk and lie down on its own)
    `chase`: after its last shot each drone follows the robot -- behind it, off to one side, half a
    metre up -- so the run is a pursuit; `pull_up_s` (s): there they climb away, out of the alley."""
    def yaw_of(q):
        return math.atan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))

    groups = {}
    for f, start in sorted(launches, key=lambda x: x[0]):
        rel = start[:2] - base[f - 1, :2]
        bearing = math.atan2(rel[1], rel[0]) - yaw_of(quat[f - 1])
        key = int(round(math.degrees(bearing) / 45.0)) % 8
        groups.setdefault(key, []).append((f, np.array(start)))
    roots = []
    smooth = np.copy(base)  # the robot's track without the gait's sway, for the chasers to follow
    for f in range(1, frames):
        smooth[f] = 0.94 * smooth[f - 1] + 0.06 * base[f]
    for g, (key, shots) in enumerate(sorted(groups.items())):
        root = add_drone(f"drone{g}", location=(0.0, 0.0, 0.0), scale=scale, style=style)
        props = [bpy.data.objects[n] for n in root["props"]]
        first = shots[0][0]
        drop_from = max(1, first - int(descend_s * fps) - int(0.4 * fps))
        for f in range(1, frames + 1):
            # where the emitter should be: on a shot's frame its start point; between shots, eased
            prev = max([s for s in shots if s[0] <= f], key=lambda s: s[0], default=None)
            nxt = min([s for s in shots if s[0] > f], key=lambda s: s[0], default=None)
            if prev is None:
                target = nxt[1].copy()
                k = min(max((f - drop_from) / max(first - drop_from - int(0.4 * fps), 1), 0.0), 1.0)
                k = k * k * (3 - 2 * k)
                target[2] += (1.0 - k) * 2.5  # dropping in from above the street
            elif nxt is None and chase:
                yw = yaw_of(quat[f - 1])
                fwd = np.array([math.cos(yw), math.sin(yw), 0.0])
                side = np.array([-math.sin(yw), math.cos(yw), 0.0])
                tail = smooth[f - 1] - fwd * (1.9 + 0.5 * (g % 2)) + side * (0.55 if g % 2 else -0.55) * (1 + 0.3 * (g // 2))
                tail[2] = 1.25 + 0.15 * g  # (10/28 evening) above the dumpsters it weaves round
                k = min(max((f - prev[0] - 0.3 * fps) / (0.9 * fps), 0.0), 1.0)
                k = k * k * (3 - 2 * k)
                target = prev[1] * (1 - k) + tail * k
                if pull_up_s is not None and f / fps > pull_up_s:
                    c = min((f / fps - pull_up_s) / 1.2, 1.0)
                    target = target + np.array([0.0, 0.0, 4.0 * c * c * (3 - 2 * c)])
            elif nxt is None:
                target = prev[1]
            else:
                k = (f - prev[0]) / (nxt[0] - prev[0])
                k = k * k * (3 - 2 * k)
                target = prev[1] * (1 - k) + nxt[1] * k
            aim = base[f - 1, :2] - target[:2]
            yaw = math.atan2(aim[1], aim[0])
            m = Vector((math.cos(yaw) * muzzle.x - math.sin(yaw) * muzzle.y,
                        math.sin(yaw) * muzzle.x + math.cos(yaw) * muzzle.y, muzzle.z)) * scale
            t = f / fps
            root.location = (Vector(target.tolist()) - m + Vector((0.0, 0.0, 0.004 * math.sin(2 * math.pi * 0.7 * t + g)))).to_tuple()
            root.rotation_euler = (0.03 * math.sin(2 * math.pi * 0.5 * t + g), 0.04 * math.cos(2 * math.pi * 0.4 * t), yaw)
            root.keyframe_insert("location", frame=f)
            root.keyframe_insert("rotation_euler", frame=f)
            for k_, prop in enumerate(props):
                prop.rotation_euler = (0, 0, (-1) ** k_ * f * (math.pi / 2 + 0.31))
                prop.keyframe_insert("rotation_euler", frame=f)
        for f, _ in shots:
            key_fire(root, f)
        roots.append(root)
    return roots
