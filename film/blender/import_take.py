"""Blender: build an animated robot from a take recorded by experiments/vnc_v1/record_scene.py.

Each recorded body gets its visual mesh from the official MJCF (the STL a body's visual geom
names, its rgba as the material, the geom's pos/quat as the mesh's offset), and its world pose
is keyframed from the take, so the mesh moves exactly as the simulation did. Nothing here
changes the motion; the cinematography (city, lenses, light) comes later. A take with throws
also gets the ball, and the brain's state (the director's choice of motor program, the giant
fibre) is keyed as custom properties on `<name>:brain` for overlays to read.

    blender --background --python film/blender/import_take.py -- --scene <take.npz> --take 0 \
        --mjcf <jumper>/assets/jumper/jumper.xml --out <file.blend> [--every 4] [--render <png>]
"""

import argparse
import json
import math
import time
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import bpy
import numpy as np


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--scene", required=True)
    p.add_argument("--take", type=int, default=0)
    p.add_argument("--mjcf", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--every", type=int, default=4, help="control steps per frame (4 at 200 Hz: 50 fps)")
    p.add_argument("--render", default="", help="also render a frame to this PNG (Workbench)")
    p.add_argument("--frame", type=int, default=0, help="the frame to render (default: the middle one)")
    p.add_argument("--follow", action="store_true", help="key the camera to follow the base at a fixed offset")
    p.add_argument("--cam-offset", default="-0.6,-0.6,0.35", help="camera offset from the base, m")
    p.add_argument("--lens", type=float, default=50.0)
    p.add_argument("--compare", default="", help="a second take (e.g. the official on the same scenario) beside it")
    p.add_argument("--compare-take", type=int, default=0)
    p.add_argument("--compare-offset", default="0,1.0,0", help="m added to the second take's positions")
    p.add_argument("--movie", default="", help="render frames --range to this folder as PNGs (Workbench)")
    p.add_argument("--range", default="", help="first,last frame for --movie")
    p.add_argument("--name", default="jumper")
    p.add_argument("--bullet-time", default="",
                   help="FRAME,HOLD,DEG for --movie: stop the take at FRAME for HOLD images while the "
                        "camera swings DEG degrees round the robot")
    p.add_argument("--engine", default="workbench", choices=["workbench", "eevee", "cycles"],
                   help="for --render / --movie: Workbench (fast, flat), EEVEE (lit; the glows show) or "
                        "Cycles on the GPU (the final renders, on Colab)")
    p.add_argument("--shard", default="", help="i/n: render only output frames i, i+n, ... (several machines)")
    p.add_argument("--photoreal", default="", help="cache folder: scanned CC0 materials, HDRI, robot paint and eyes")
    p.add_argument("--haze-pr", type=float, default=0.006, help="with --photoreal: the world haze density")
    p.add_argument("--shutter", type=float, default=0.0,
                   help="the ending: a roller shutter closing across the back alley where the robot is at this "
                        "take time (s), and a chasing drone crashing into it (shutter.py)")
    p.add_argument("--follow-smooth", type=float, default=0.1,
                   help="with --follow: how fast the camera catches up with the robot each frame (1 = no lag)")
    p.add_argument("--cam-shutter", action="store_true", help="with --shutter: a still camera past it, looking back")
    p.add_argument("--dicing", type=float, default=1.0, help="with --photoreal: adaptive subdivision, pixels per micropolygon")
    p.add_argument("--drones", action="store_true",
                   help="a micro-drone (drone.py) at each throw's launch point, firing the shot")
    p.add_argument("--drone-scale", type=float, default=2.0,
                   help="the drones' size (1: a 94 mm micro-quad, about a third of the robot's width)")
    p.add_argument("--projectile", default="ball", choices=["ball", "energy", "laser"],
                   help="how a thrown ball looks: the ball itself, or an energy shot (a glowing core the "
                        "ball's size with a light trail behind it, aimed along its flight, and a light), or "
                        "a laser bolt (`fx.add_laser`: a red streak; a miss runs on to the ground)")
    p.add_argument("--impacts", action="store_true", help="with --projectile laser: sparks, chips, smoke, scorch")
    p.add_argument("--drone-style", default="cyan", choices=["cyan", "hostile"])
    p.add_argument("--set", default="none", choices=["none", "alley", "backalley"],
                   help="alley: the night street; backalley: a narrow concrete back alley (alley.py)")
    p.add_argument("--haze", type=float, default=0.012, help="the alley's volumetric haze density")
    p.add_argument("--cam-local", action="store_true",
                   help="--cam-offset in the robot's starting frame (x ahead, y left) rather than the world's")
    p.add_argument("--dof", type=float, default=0.0, help="depth of field: the lens's f-stop, focused on the robot")
    p.add_argument("--res", default="1280x720")
    # (10/28, the user: no enemy in sight, no idea of the layout, the ground blotchy, more objects)
    p.add_argument("--aim-offset", default="0,0,0",
                   help="m added to the point the camera looks at (robot frame with --cam-local): frame the drones too")
    p.add_argument("--pebbles", type=int, default=6000, help="with --photoreal: pebbles scattered on the alley floor")
    p.add_argument("--props", default="", help="with --photoreal: cache folder for the scanned CC0 props (props.py)")
    p.add_argument("--alley-ref", default="",
                   help="a take npz to lay the back alley out from (the re-recorded run keeps the first take's alley)")
    p.add_argument("--obstacles", default="",
                   help="T0,T1 (s): dumpsters and bins where the robot swerved between these times (props.py)")
    p.add_argument("--reverse", action="store_true",
                   help="with --movie: render from the last frame back (a machine picking up another's leftovers)")
    p.add_argument("--chase", action="store_true", help="with --drones: after their last shot the drones chase the robot")
    p.add_argument("--pull-up", type=float, default=None, help="with --chase: when (s) the chasers climb away")
    p.add_argument("--samples", type=int, default=32)
    return p.parse_args(argv)


def visual_geoms(mjcf: Path):
    """body name -> list of (mesh file, rgba, pos, quat) for its visual geoms."""
    root = ET.parse(mjcf).getroot()
    meshdir = mjcf.parent / (root.find("compiler").get("meshdir", "") if root.find("compiler") is not None else "")
    files = {m.get("name"): meshdir / m.get("file") for m in root.iter("mesh")}
    out = {}
    for body in root.iter("body"):
        geoms = []
        for g in body.findall("geom"):
            if g.get("class") != "visual" or g.get("type") != "mesh":
                continue
            rgba = [float(x) for x in (g.get("rgba") or "0.5 0.5 0.5 1").split()]
            pos = [float(x) for x in (g.get("pos") or "0 0 0").split()]
            quat = [float(x) for x in (g.get("quat") or "1 0 0 0").split()]
            geoms.append((files[g.get("mesh")], rgba, pos, quat))
        out[body.get("name")] = geoms
    return out


def _bsdf(mat):
    """The material's Principled BSDF, created and wired to the output if missing (Blender 5's
    new materials do not always come with one)."""
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = next((n for n in nt.nodes if n.type == "BSDF_PRINCIPLED"), None)
    if bsdf is None:
        bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
        out = next((n for n in nt.nodes if n.type == "OUTPUT_MATERIAL"), None) or nt.nodes.new("ShaderNodeOutputMaterial")
        nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    return bsdf


def glow_material(name, rgb, strength, alpha=1.0):
    """An emissive material (the energy shot's core and trail); `alpha` < 1 blends."""
    mat = bpy.data.materials.get(name)
    if mat is None:
        mat = bpy.data.materials.new(name)
        bsdf = _bsdf(mat)
        bsdf.inputs["Base Color"].default_value = (*rgb, 1.0)
        for key in ("Emission Color", "Emission"):  # Blender 4 / 3 names
            if key in bsdf.inputs:
                bsdf.inputs[key].default_value = (*rgb, 1.0)
                break
        bsdf.inputs["Emission Strength"].default_value = strength
        bsdf.inputs["Alpha"].default_value = alpha
        if alpha < 1.0:
            for attr, value in (("surface_render_method", "BLENDED"), ("blend_method", "BLEND")):
                if hasattr(mat, attr):
                    setattr(mat, attr, value)
        mat.diffuse_color = (*rgb, alpha)
    return mat


def material(rgba):
    key = "jumper_%.2f_%.2f_%.2f" % tuple(rgba[:3])
    mat = bpy.data.materials.get(key)
    if mat is None:
        mat = bpy.data.materials.new(key)
        mat.diffuse_color = rgba
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes.get("Principled BSDF")
        if bsdf is not None:
            bsdf.inputs["Base Color"].default_value = rgba
            bsdf.inputs["Roughness"].default_value = 0.45
    return mat


def main():
    args = parse_args()
    z = np.load(args.scene)
    meta = json.loads(str(z["meta"]))
    pos = z["body_pos"][args.take][:: args.every]  # [frames, bodies, 3]
    quat = z["body_quat"][args.take][:: args.every]  # [frames, bodies, 4] (w, x, y, z)
    frames = pos.shape[0]
    geoms = visual_geoms(Path(args.mjcf))

    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.fps = round(1.0 / (meta["control_dt"] * args.every))
    scene.frame_start, scene.frame_end = 1, frames

    def build_robot(prefix, pos, quat, tint=None):
        for b, name in enumerate(meta["body_names"]):
            holder = bpy.data.objects.new(f"{prefix}:{name}", None)
            scene.collection.objects.link(holder)
            holder.rotation_mode = "QUATERNION"
            for mesh_file, rgba, gpos, gquat in geoms.get(name, []):
                bpy.ops.wm.stl_import(filepath=str(mesh_file))
                part = bpy.context.selected_objects[0]
                part.name = f"{prefix}:{name}:mesh"
                part.data.materials.append(material(tint(rgba) if tint else rgba))
                part.parent = holder
                part.location = gpos
                part.rotation_mode = "QUATERNION"
                part.rotation_quaternion = gquat
            for f in range(min(frames, pos.shape[0])):
                holder.location = pos[f, b].tolist()
                holder.rotation_quaternion = quat[f, b].tolist()
                holder.keyframe_insert("location", frame=f + 1)
                holder.keyframe_insert("rotation_quaternion", frame=f + 1)

    build_robot(args.name, pos, quat)
    if args.compare:  # the second take, grey, beside the first (its world frame shifted to line up)
        z2 = np.load(args.compare)
        pos2 = z2["body_pos"][args.compare_take][:: args.every].copy()
        quat2 = z2["body_quat"][args.compare_take][:: args.every].copy()
        # The evaluation spawns each robot with its own heading: turn the second take about z so
        # that it starts facing the way the first does, then move its start onto the first's.
        yaw = lambda q: math.atan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))  # noqa: E731
        dyaw = yaw(quat[0, 0]) - yaw(quat2[0, 0])
        c, s = math.cos(dyaw), math.sin(dyaw)
        rel = pos2 - pos2[0, 0]
        pos2 = np.stack([c * rel[..., 0] - s * rel[..., 1], s * rel[..., 0] + c * rel[..., 1], rel[..., 2]], axis=-1)
        pos2 += pos[0, 0] * np.array([1.0, 1.0, 0.0]) + np.array([0.0, 0.0, z2["body_pos"][args.compare_take][0, 0, 2]])
        qz = np.array([math.cos(dyaw / 2), 0.0, 0.0, math.sin(dyaw / 2)])  # (w, x, y, z)
        w1, x1, y1, z1 = qz
        w2, x2, y2, z2_ = quat2[..., 0], quat2[..., 1], quat2[..., 2], quat2[..., 3]
        quat2 = np.stack([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2_, w1 * x2 + x1 * w2 + y1 * z2_ - z1 * y2,
                          w1 * y2 - x1 * z2_ + y1 * w2 + z1 * x2, w1 * z2_ + x1 * y2 - y1 * x2 + z1 * w2], axis=-1)
        # The offset in the first robot's frame: 'left' of it stays left of it.
        ox, oy, oz = (float(v) for v in args.compare_offset.split(","))
        y0 = yaw(quat[0, 0])
        pos2 += np.array([math.cos(y0) * ox - math.sin(y0) * oy, math.sin(y0) * ox + math.cos(y0) * oy, oz])
        grey = lambda rgba: [0.55 + 0.25 * (c - 0.5) for c in rgba[:3]] + [rgba[3]]  # noqa: E731
        build_robot("official", pos2, quat2, tint=grey)

    # The thrown balls (a scenario with throws; a volley has several, "balls_pos"): 25 mm spheres
    # keyed from the take, hidden while they rest parked far from the robot between throws.
    if "balls_pos" in z.files:
        tracks = z["balls_pos"][args.take][:: args.every]  # frames x balls x 3
    elif "ball_pos" in z.files:
        tracks = z["ball_pos"][args.take][:: args.every][:, None]
    else:
        tracks = None
    frame_dt = meta["control_dt"] * args.every
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    impacts, launches = [], []
    if args.projectile == "laser" and tracks is not None:
        import fx

        struck = z["struck"][args.take][:: args.every] if "struck" in z.files else None  # frames x balls
        for j in range(tracks.shape[1]):
            fl = fx.flights(tracks[:, j], pos[:, 0], frame_dt)
            hits = {}
            for f0, f1, _, _ in fl:
                if struck is not None:
                    later = np.nonzero(struck[f0:, j])[0]
                    if len(later) and f0 + later[0] <= f1 + 2:
                        hits[f0] = f0 + int(later[0])
            impacts += fx.add_laser(f"bolt{j}", frames, fl, hits, frame_dt)
            launches += [(f0 + 1, start) for f0, _, start, _ in fl]  # 1-based frames
        print(f"[import] {len(launches)} bolts, {sum(1 for i in impacts if i[2] == 'robot')} hit the robot")
        if args.impacts:
            for i, (f, at, kind) in enumerate(impacts):
                fx.add_impact(f"impact{i}", f + 1, at, kind, seed=i, fps=scene.render.fps)
        if args.drones:
            import drone as drone_kit

            fx.drone_pool(launches, pos[:, 0], quat[:, 0], frames, scene.render.fps, drone_kit.add_drone,
                          drone_kit.MUZZLE, args.drone_scale, args.drone_style, drone_kit.key_fire,
                          chase=args.chase, pull_up_s=args.pull_up)
            print(f"[import] drones on station for {len(launches)} shots")
        if "wounded" in z.files:  # the leg a hit killed spits sparks from its knee from then on
            w = z["wounded"][args.take][:: args.every]
            hurt = np.nonzero(w >= 0)[0]
            if len(hurt):
                legs = ("LF", "RF", "LM", "RM", "LR", "RR")
                leg = legs[int(w[hurt[0]])]
                joint = f"{args.name}:{leg}_forearm_link" if leg in ("LF", "RF") else f"{args.name}:{leg}_calf_link"
                fx.wound_sparks(f"wound_{leg}", bpy.data.objects[joint], int(hurt[0]) + 1, frames, scene.render.fps)
                print(f"[import] wounded leg {leg} from frame {int(hurt[0]) + 1}")
        tracks = None  # drawn
    for j in range(0 if tracks is None else tracks.shape[1]):
        bpy.ops.mesh.primitive_uv_sphere_add(radius=0.025, segments=32, ring_count=16)
        ball = bpy.context.active_object
        ball.name = "ball" if j == 0 else f"ball{j}"
        parts = [ball]
        if args.projectile == "energy":
            # The core is the simulated ball exactly (25 mm), so a hit or a miss reads as it
            # happened; the trail only reaches back along the flight, and the light rides with it.
            from mathutils import Vector

            ball.data.materials.append(glow_material("energy_core", (0.25, 0.85, 1.0), 9.0))
            bpy.ops.mesh.primitive_cone_add(radius1=0.014, radius2=0.0, depth=1.0, vertices=24)
            trail = bpy.context.active_object
            trail.name = f"{ball.name}_trail"
            trail.data.transform(np.array([[0, 0, -1, -0.5], [0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1]]).tolist())
            trail.data.materials.append(glow_material("energy_trail", (0.1, 0.75, 1.0), 4.0, alpha=0.22))
            trail.parent = ball
            light_data = bpy.data.lights.new(f"{ball.name}_light", type="POINT")
            light_data.energy = 4.0
            light_data.color = (0.3, 0.85, 1.0)
            light_data.shadow_soft_size = 0.02
            light = bpy.data.objects.new(f"{ball.name}_light", light_data)
            bpy.context.collection.objects.link(light)
            light.parent = ball
            parts += [trail, light]
            ball.rotation_mode = "QUATERNION"
            heading = Vector((1.0, 0.0, 0.0)).to_track_quat("X", "Z")
        for f in range(frames):
            ball.location = tracks[f, j].tolist()
            far = float(np.linalg.norm(tracks[f, j, :2] - pos[f, 0, :2])) > 3.0
            if args.projectile == "energy":
                a, b = tracks[max(f - 1, 0), j], tracks[min(f + 1, frames - 1), j]
                v = (b - a) / (frame_dt * max(min(f + 1, frames - 1) - max(f - 1, 0), 1))
                speed = float(np.linalg.norm(v))
                if speed > 0.05:
                    heading = Vector(v.tolist()).to_track_quat("X", "Z")
                ball.rotation_quaternion = heading
                ball.keyframe_insert("rotation_quaternion", frame=f + 1)
                trail.scale = (min(max(speed * 0.06, 0.03), 0.5), 1.0, 1.0)  # 60 ms of flight
                trail.keyframe_insert("scale", frame=f + 1)
            ball.keyframe_insert("location", frame=f + 1)
            for part in parts:
                part.hide_render = far
                part.hide_viewport = far
                part.keyframe_insert("hide_render", frame=f + 1)
                part.keyframe_insert("hide_viewport", frame=f + 1)

    # The shooters (`--drones`): a micro-drone (`drone.py`) hovering at each throw's launch point
    # -- where the ball appeared beside the robot, the barrel's tip there -- facing the robot, with
    # a muzzle flash on the frame of the shot. Each throw has its own drone.
    if args.drones and tracks is not None:
        import drone as drone_kit
        from mathutils import Matrix, Vector

        n_drone = 0
        for j in range(tracks.shape[1]):
            near = np.linalg.norm(tracks[:, j, :2] - pos[:, 0, :2], axis=-1) < 3.0
            launches = [f for f in range(1, frames) if near[f] and not near[f - 1]]
            for f0 in launches:
                start = Vector(tracks[f0, j].tolist())
                aim = Vector(pos[f0, 0].tolist()) - start
                aim.z = 0.0
                aim.normalize()
                yaw = math.atan2(aim.y, aim.x)
                muzzle = Matrix.Rotation(yaw, 3, "Z") @ (drone_kit.MUZZLE * args.drone_scale)
                root = drone_kit.add_drone(f"drone{n_drone}", location=start - muzzle, scale=args.drone_scale)
                root.rotation_euler = (0.0, 0.0, yaw)
                drone_kit.key_hover(root, 1, frames, scene.render.fps)
                drone_kit.key_fire(root, f0 + 1)
                n_drone += 1
        print(f"[import] {n_drone} drones at the launch points")

    # The brain's state as keyed custom properties on an empty, for overlays and lights to read
    # (drivers): the director's share for each motor program, and the giant fibre firing.
    brain = bpy.data.objects.new(f"{args.name}:brain", None)
    scene.collection.objects.link(brain)
    channels = {}
    if "gate" in z.files:
        gate = z["gate"][args.take][:: args.every].astype(np.float32)
        for k, prog in enumerate(meta.get("programs") or [f"program_{i}" for i in range(gate.shape[-1])]):
            channels[f"program:{prog}"] = gate[:, k]
    if "gf_fired" in z.files:
        channels["giant_fibre"] = z["gf_fired"][args.take][:: args.every].astype(np.float32)
    for key, values in channels.items():
        brain[key] = 0.0
        for f in range(min(frames, len(values))):
            brain[key] = float(values[f])
            brain.keyframe_insert(f'["{key}"]', frame=f + 1)

    yaw0 = math.atan2(2 * (quat[0, 0, 0] * quat[0, 0, 3] + quat[0, 0, 1] * quat[0, 0, 2]),
                      1 - 2 * (quat[0, 0, 2] ** 2 + quat[0, 0, 3] ** 2))
    if args.set in ("alley", "backalley"):
        import alley

        origin = (float(pos[0, 0, 0]), float(pos[0, 0, 1]))
        if args.set == "alley":
            alley.build(origin=origin, yaw=yaw0)
        else:
            # The back alley is narrow: centre it across the street on where the robot and the drones
            # actually went (the robot's dodges add up sideways), wide enough for them all.
            left = np.array([-math.sin(yaw0), math.cos(yaw0)])
            ref_pos, ref_launches = pos, launches
            if args.alley_ref:  # (10/28 evening) the re-recorded run: the same alley as the shots before it
                zr = np.load(args.alley_ref)
                ref_pos = zr["body_pos"][args.take][:: args.every]
                rt = zr["balls_pos"][args.take][:: args.every]
                ref_launches = [(f0 + 1, st) for j in range(rt.shape[1])
                                for f0, _, st, _ in fx.flights(rt[:, j], ref_pos[:, 0], frame_dt)]
            lat = list((ref_pos[:, 0, :2] - ref_pos[0, 0, :2]) @ left)
            lat += [float((np.asarray(st)[:2] - ref_pos[0, 0, :2]) @ left) for _, st in ref_launches]
            mid = 0.5 * (min(lat) + max(lat))
            width = max(4.4, max(lat) - min(lat) + 1.0)
            origin = (origin[0] + mid * left[0], origin[1] + mid * left[1])
            alley.build_backalley(origin=origin, yaw=yaw0, width=width)
            print(f"[import] back alley {width:.1f} m wide, centred {mid:+.2f} m across")
            alley_frame = (np.array(origin), yaw0, width)
        floor = None
        base_obj = bpy.data.objects[f"{args.name}:base_link"]
        for lname, loc, energy, rgb, size in (("hero_key", (0.5, -0.35, 0.55), 9.0, (1.0, 0.96, 0.92), 0.4),
                                              ("hero_rim", (-0.6, 0.3, 0.35), 14.0, (0.35, 0.6, 1.0), 0.3)):
            # key and rim riding with the robot (its position, not its heading), so the white shell
            # reads against the neon
            light = bpy.data.objects.new(lname, bpy.data.lights.new(lname, type="AREA"))
            light.data.energy, light.data.color, light.data.size = energy, rgb, size
            if hasattr(light.data, "specular_factor"):  # not mirrored as a white card in the wet ground
                light.data.specular_factor = 0.0
            scene.collection.objects.link(light)
            con = light.constraints.new("COPY_LOCATION")
            con.target, con.use_offset = base_obj, True
            light.location = loc
            track = light.constraints.new("TRACK_TO")
            track.target = base_obj
            track.track_axis, track.up_axis = "TRACK_NEGATIVE_Z", "UP_Y"
    else:
        bpy.ops.mesh.primitive_plane_add(size=20.0, location=(0, 0, 0))
        floor = bpy.context.active_object
        floor.name = "floor"
        floor.data.materials.append(material([0.55, 0.55, 0.55, 1.0]))

    base = pos[:, 0]
    mid = frames // 2
    cam_data = bpy.data.cameras.new("cam")
    cam_data.lens = args.lens
    cam = bpy.data.objects.new("cam", cam_data)
    scene.collection.objects.link(cam)
    offset = np.array([float(v) for v in args.cam_offset.split(",")])
    look_off = np.array([float(v) for v in args.aim_offset.split(",")])
    if args.cam_local:  # x ahead of the robot's start, y to its left
        offset = np.array([math.cos(yaw0) * offset[0] - math.sin(yaw0) * offset[1],
                           math.sin(yaw0) * offset[0] + math.cos(yaw0) * offset[1], offset[2]])
        look_off = np.array([math.cos(yaw0) * look_off[0] - math.sin(yaw0) * look_off[1],
                             math.sin(yaw0) * look_off[0] + math.cos(yaw0) * look_off[1], look_off[2]])
    cam_data.clip_start = 0.01
    if args.dof:
        cam_data.dof.use_dof = True
        cam_data.dof.focus_object = bpy.data.objects[f"{args.name}:base_link"]
        cam_data.dof.aperture_fstop = args.dof

    def aim(target):
        cam.location = tuple((np.array(target) + offset).tolist())
        if args.cam_local:  # a ground-level camera stays at its height and tilts to follow a jump
            cam.location = (cam.location[0], cam.location[1], float(offset[2]))
        direction = np.array(target) + look_off - np.array(cam.location)
        yaw = math.atan2(direction[1], direction[0])
        pitch = math.atan2(direction[2], math.hypot(direction[0], direction[1]))
        cam.rotation_euler = (math.pi / 2 + pitch, 0.0, yaw - math.pi / 2)

    if args.follow:  # smoothed, so the camera rides the walk without the gait's bob
        smooth = np.copy(base)
        for f in range(1, frames):
            smooth[f] = (1 - args.follow_smooth) * smooth[f - 1] + args.follow_smooth * base[f]
        for f in range(frames):
            aim(smooth[f])
            cam.keyframe_insert("location", frame=f + 1)
            cam.keyframe_insert("rotation_euler", frame=f + 1)
    else:
        aim(base[(args.frame or mid) - 1])
    scene.camera = cam
    sun = bpy.data.objects.new("sun", bpy.data.lights.new("sun", type="SUN"))
    sun.rotation_euler = (0.6, 0.2, 0.8)
    scene.collection.objects.link(sun)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(Path(args.out).resolve()))
    print(f"FILM_TAKE_BUILT {args.out} frames={frames} fps={scene.render.fps} bodies={len(meta['body_names'])}")
    if (args.render or args.movie) and args.engine in ("eevee", "cycles") and args.set in ("alley", "backalley"):
        import alley

        alley.setup_render(scene, samples=args.samples, haze=args.haze,
                           res=tuple(int(v) for v in args.res.split("x")), engine=args.engine)
        bpy.data.objects.remove(sun, do_unlink=True)
        if args.photoreal:
            import photoreal

            assets = photoreal.fetch(Path(args.photoreal))
            photoreal.upgrade_set(assets, haze=args.haze_pr)
            photoreal.upgrade_robot(args.name)
            photoreal.dress_set(font_path=alley.FONT)
            import detail

            detail.true_displacement(dicing=args.dicing)
            detail.scatter_debris(pebbles=args.pebbles)
            if args.props and args.set == "backalley":
                import props

                o_xy, a_yaw, _w = alley_frame
                across = np.array([-math.sin(a_yaw), math.cos(a_yaw)])
                lat = (pos[:, 0, :2] - o_xy) @ across
                props.dress(props.fetch(Path(args.props)), lane=(float(lat.min()) - 0.5, float(lat.max()) + 0.5))
                if args.obstacles:  # (10/28 evening, the user: weave round bins and dumpsters on the run)
                    t0, t1 = (float(v) for v in args.obstacles.split(","))
                    along = np.array([math.cos(a_yaw), math.sin(a_yaw)])
                    full = z["body_pos"][args.take][:, 0, :2]  # at 200 Hz: the same obstacles in every shot
                    k0, k1 = int(t0 / meta["control_dt"]), int(t1 / meta["control_dt"])
                    uv = np.stack([(full[k0:k1] - o_xy) @ along, (full[k0:k1] - o_xy) @ across], axis=1)
                    props.obstacles(uv, _w / 2)
            detail.hardware()
            for probe in [o for o in bpy.data.objects if o.type == "LIGHT_PROBE"]:
                bpy.data.objects.remove(probe, do_unlink=True)
        if args.shutter and args.set == "backalley":
            # (10/28) The ending: a shutter closing across the alley where the robot's base is at
            # --shutter seconds; it clears the robot's shell by a few centimetres, and a chasing drone
            # dives at the gap and hits the curtain. Keyed by hand on the take's clock.
            import shutter

            o_xy, a_yaw, a_width = alley_frame
            along, across = np.array([math.cos(a_yaw), math.sin(a_yaw)]), np.array([-math.sin(a_yaw), math.cos(a_yaw)])
            fi = min(int(round(args.shutter / frame_dt)), frames - 1)
            u_s = float((pos[fi, 0, :2] - o_xy) @ along)
            v_r = float((pos[fi, 0, :2] - o_xy) @ across)
            T = args.shutter
            frame_of = lambda t: int(round(t / frame_dt)) + 1  # noqa: E731
            sched = [(T - 9.0, 2.6), (T - 2.25, 0.95), (T - 0.75, 0.40), (T - 0.25, 0.28), (T, 0.245), (T + 0.25, 0.215),
                     (T + 0.55, 0.165), (T + 1.25, 0.05), (T + 2.0, 0.0)]
            crash = {"start": (T - 0.75, u_s - 2.0, v_r + 0.25, 0.75), "hit": (T + 0.5, v_r + 0.1, 0.26)}
            shutter.add(bpy.data.objects["set"], u_s, a_width, frame_of, sched, crash, args.drone_style, args.drone_scale)
            if args.cam_shutter:  # a still camera past the shutter, low, looking back at the gap
                if cam.animation_data:
                    cam.animation_data_clear()
                eye = o_xy + along * (u_s + 0.95) + across * (v_r + 0.32)
                look = o_xy + along * (u_s - 0.15) + across * v_r
                cam.location = (float(eye[0]), float(eye[1]), 0.075)
                d = np.array([look[0] - eye[0], look[1] - eye[1], 0.15 - 0.075])
                cam.rotation_euler = (math.pi / 2 + math.atan2(d[2], math.hypot(d[0], d[1])), 0.0,
                                      math.atan2(d[1], d[0]) - math.pi / 2)
                args.follow = False
    elif (args.render or args.movie) and args.engine == "eevee":
        # Lit render: the energy shots and LEDs glow. A near-black world, a dark floor, a sun.
        engines = [e.identifier for e in bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items]
        scene.render.engine = "BLENDER_EEVEE_NEXT" if "BLENDER_EEVEE_NEXT" in engines else "BLENDER_EEVEE"
        scene.render.resolution_x, scene.render.resolution_y = 1280, 720
        world = bpy.data.worlds.new("world")
        world.use_nodes = True
        nt = world.node_tree
        bg = next((n for n in nt.nodes if n.type == "BACKGROUND"), None) or nt.nodes.new("ShaderNodeBackground")
        out = next((n for n in nt.nodes if n.type == "OUTPUT_WORLD"), None) or nt.nodes.new("ShaderNodeOutputWorld")
        nt.links.new(bg.outputs["Background"], out.inputs["Surface"])
        bg.inputs[0].default_value = (0.006, 0.007, 0.011, 1.0)
        scene.world = world
        floor.data.materials[0].diffuse_color = (0.03, 0.03, 0.035, 1.0)
        bsdf = _bsdf(floor.data.materials[0])
        bsdf.inputs["Base Color"].default_value = (0.03, 0.03, 0.035, 1.0)
        bsdf.inputs["Roughness"].default_value = 0.4
        sun.data.energy = 2.5
        views = [v.identifier for v in bpy.types.ColorManagedViewSettings.bl_rna.properties["view_transform"].enum_items]
        if "AgX" in views:
            scene.view_settings.view_transform = "AgX"
    elif args.render or args.movie:
        scene.render.engine = "BLENDER_WORKBENCH"
        scene.render.resolution_x, scene.render.resolution_y = 960, 540
        # A dark world, a light floor and cast shadows, so the floor and the ball's path read.
        world = bpy.data.worlds.new("world")
        world.color = (0.05, 0.05, 0.06)
        scene.world = world
        floor.data.materials[0].diffuse_color = (0.75, 0.75, 0.72, 1.0)
        shading = scene.display.shading
        shading.light = "STUDIO"
        shading.show_shadows = True
        shading.show_cavity = True
    if args.render:
        scene.frame_set(args.frame or mid)
        scene.render.filepath = str(Path(args.render).resolve())
        bpy.ops.render.render(write_still=True)
        print(f"FILM_TAKE_RENDERED {args.render}")
    if args.movie:
        first, last = (int(v) for v in args.range.split(",")) if args.range else (1, frames)
        folder = Path(args.movie).resolve()
        folder.mkdir(parents=True, exist_ok=True)
        # Bullet time (--bullet-time FRAME,HOLD,DEG): the take stops at FRAME for HOLD output frames
        # while the camera swings DEG degrees round the robot, then runs on from the new angle.
        # Only the camera moves; the take's motion is untouched.
        # Several holds: "F,H,D;F,H,D" -- each orbit starts where the last one left the camera.
        shots, swing, f = [], 0.0, first
        for seg in sorted((tuple(float(v) for v in s_.split(",")) for s_ in args.bullet_time.split(";") if s_),
                          key=lambda x: x[0]):
            bf, hold, deg = int(seg[0]), int(seg[1]), seg[2]
            shots += [(g, swing) for g in range(f, bf)]
            shots += [(bf, swing + deg * (i / max(hold - 1, 1))) for i in range(hold)]
            swing += deg
            f = bf + 1
        shots += [(g, swing) for g in range(f, last + 1)]
        cam_action = cam.animation_data.action if cam.animation_data else None
        shard_i, shard_n = (int(v) for v in args.shard.split("/")) if args.shard else (0, 1)
        order = list(enumerate(shots))
        for out_i, (f, swing) in (reversed(order) if args.reverse else order):
            # (10/28) Several machines share a shot by output frame, and a frame already on disk is not
            # rendered again, so a stopped render picks up where it left off.
            path = folder / f"frame_{out_i + 1:04d}.png"
            if out_i % shard_n != shard_i or path.exists():
                continue
            # The follow camera is keyed, and a render re-evaluates its keys: an orbit placed by hand
            # was overwritten (10/26: every bullet-time hold until then rendered a still camera). So
            # the camera's keys are detached while it is swung, and put back after.
            if cam.animation_data:
                cam.animation_data.action = None if swing else cam_action
            scene.frame_set(f)
            if swing:  # orbit the follow camera's offset about the vertical through the robot
                # (10/28) the robot itself, not the lagging follow point: at a jump's apex the lag left it out of frame
                target = np.array(base[f - 1])
                a = math.radians(swing)
                rot = np.array([[math.cos(a), -math.sin(a), 0.0], [math.sin(a), math.cos(a), 0.0], [0.0, 0.0, 1.0]])
                saved = offset.copy()
                offset[:] = rot @ saved
                aim(target)
                offset[:] = saved
            scene.render.filepath = str(path)
            t_frame = time.time()
            bpy.ops.render.render(write_still=True)
            print(f"[render] {path.name} {time.time() - t_frame:.1f} s", flush=True)
        if cam.animation_data:
            cam.animation_data.action = cam_action
        print(f"FILM_TAKE_MOVIE {folder} frames {first}-{last} -> {len(shots)} images")


main()
