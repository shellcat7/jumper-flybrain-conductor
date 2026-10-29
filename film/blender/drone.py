"""Blender: the security micro-drone that fires the energy shots -- an original design, built in
code so it carries no third-party model, licence or brand.

Modelled as a real ducted micro-quad would be built, at the robot's own scale (motor to motor
94 mm diagonally, 40 mm props):

- a 1.5 mm carbon bottom plate (centre, arms and motor pads in one outline) with M2 holes, the
  20 x 20 mm stack pattern and two battery-strap slots;
- thin-walled ducts with a rolled bellmouth lip, bolted to the arms through two tabs each;
- 14-series motors (stator base, anodised bell with cooling slots, shaft, prop nut) and 40 mm
  three-blade props with a twisted, cambered section;
- a flight controller over an ESC on M2 standoffs (chips, MOSFETs, capacitor, connectors, gold
  pads) with three motor leads run along each arm;
- a 1S LiPo under a rubber strap, with its leads and connector;
- a nano FPV camera tilted up in a printed mount at the front, a whip antenna at the back;
- the emitter: a laser-module barrel under the nose on a printed bracket, lens glowing cyan;
- LED bars under the arms.

`add_drone()` builds one and returns its root (+X forward: the emitter fires along +X);
`key_hover()` keys a gentle hover and spinning props, `key_fire()` a muzzle flash. Run alone, it
builds one drone in a small studio and renders a still:

    blender --background --python film/blender/drone.py -- --render film/stills/drone.png
"""

from __future__ import annotations

import math
import sys

import bmesh
import bpy
from mathutils import Matrix, Vector

# Motor positions: on the diagonals, 47 mm from the centre (94 mm motor to motor).
ARM_R = 0.047
MOTORS = [Vector((ARM_R * math.cos(math.radians(a)), ARM_R * math.sin(math.radians(a)), 0.0))
          for a in (45, 135, 225, 315)]
PLATE_T = 0.0015  # bottom plate thickness; its top is at z = 0

# -- materials -----------------------------------------------------------------------------


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


def _paint(name, rgb, metallic=0.0, roughness=0.4, coat=0.0, alpha=1.0):
    mat = bpy.data.materials.get(name)
    if mat is None:
        mat = bpy.data.materials.new(name)
        bsdf = _bsdf(mat)
        bsdf.inputs["Base Color"].default_value = (*rgb, 1.0)
        bsdf.inputs["Metallic"].default_value = metallic
        bsdf.inputs["Roughness"].default_value = roughness
        if coat and "Coat Weight" in bsdf.inputs:
            bsdf.inputs["Coat Weight"].default_value = coat
        if alpha < 1.0:
            bsdf.inputs["Alpha"].default_value = alpha
            for attr, value in (("surface_render_method", "BLENDED"), ("blend_method", "BLEND")):
                if hasattr(mat, attr):
                    setattr(mat, attr, value)
        mat.diffuse_color = (*rgb, alpha)
    return mat


def _glow(name, rgb, strength):
    mat = bpy.data.materials.get(name)
    if mat is None:
        mat = bpy.data.materials.new(name)
        bsdf = _bsdf(mat)
        bsdf.inputs["Base Color"].default_value = (*rgb, 1.0)
        for key in ("Emission Color", "Emission"):  # Blender 4-5 / 3 names
            if key in bsdf.inputs:
                bsdf.inputs[key].default_value = (*rgb, 1.0)
                break
        bsdf.inputs["Emission Strength"].default_value = strength
        mat.diffuse_color = (*rgb, 1.0)
    return mat


def _carbon(name="drone_carbon"):
    """Twill carbon in object space (a 1.6 mm weave) under a clear coat."""
    mat = bpy.data.materials.get(name)
    if mat is None:
        mat = bpy.data.materials.new(name)
        bsdf = _bsdf(mat)
        nt = mat.node_tree
        coord = nt.nodes.new("ShaderNodeTexCoord")
        rot = nt.nodes.new("ShaderNodeMapping")
        rot.inputs["Rotation"].default_value = (0.0, 0.0, math.radians(45))
        tex = nt.nodes.new("ShaderNodeTexChecker")
        tex.inputs["Scale"].default_value = 620.0
        tex.inputs["Color1"].default_value = (0.010, 0.010, 0.012, 1.0)
        tex.inputs["Color2"].default_value = (0.035, 0.036, 0.040, 1.0)
        nt.links.new(coord.outputs["Object"], rot.inputs["Vector"])
        nt.links.new(rot.outputs["Vector"], tex.inputs["Vector"])
        nt.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
        bsdf.inputs["Roughness"].default_value = 0.42
        if "Coat Weight" in bsdf.inputs:
            bsdf.inputs["Coat Weight"].default_value = 0.6
            bsdf.inputs["Coat Roughness"].default_value = 0.08
        mat.diffuse_color = (0.025, 0.025, 0.028, 1.0)
    return mat


def _materials(style: str = "cyan"):
    """The airframe's materials; `style` "hostile" (10/26, after the film's reference sheet) swaps
    the cyan lights for red, darkens the motor bells and lights the camera's iris red."""
    if style == "hostile":
        mats = _materials("cyan")
        mats.update({
            "alu_bell": _paint("drone_alu_bell_dark", (0.045, 0.045, 0.05), metallic=1.0, roughness=0.3),
            "battery_band": _paint("drone_battery_band_black", (0.025, 0.025, 0.027), roughness=0.9),
            # Saturated and not too bright: AgX bleaches a strong emission toward white. (10/28 afternoon,
            # the user could not see the enemy: three times brighter, still red in AgX)
            "led": _glow("drone_led_red", (1.0, 0.012, 0.004), 6.0),
            "eye": _glow("drone_eye_red", (1.0, 0.01, 0.004), 6.0),
            "muzzle": _glow("drone_muzzle_red", (1.0, 0.05, 0.01), 1.5),
            "glass": _glow("drone_eye_glass", (1.0, 0.015, 0.006), 1.2),
        })
        return mats
    return {
        "carbon": _carbon(),
        "duct": _paint("drone_duct", (0.008, 0.0085, 0.01), roughness=0.55, coat=0.1),
        "tpu": _paint("drone_tpu", (0.014, 0.015, 0.017), roughness=0.8),
        "alu_dark": _paint("drone_alu_dark", (0.08, 0.085, 0.095), metallic=1.0, roughness=0.32),
        "alu_bell": _paint("drone_alu_bell", (0.02, 0.32, 0.42), metallic=1.0, roughness=0.28),
        "steel": _paint("drone_steel", (0.55, 0.56, 0.58), metallic=1.0, roughness=0.22),
        "black_screw": _paint("drone_screw", (0.02, 0.02, 0.022), metallic=1.0, roughness=0.3),
        "pcb": _paint("drone_pcb", (0.012, 0.014, 0.016), roughness=0.55),
        "chip": _paint("drone_chip", (0.008, 0.008, 0.009), roughness=0.5),
        "gold": _paint("drone_gold", (0.95, 0.72, 0.32), metallic=1.0, roughness=0.25),
        "ceramic": _paint("drone_ceramic", (0.45, 0.38, 0.28), roughness=0.5),
        "connector": _paint("drone_connector", (0.82, 0.8, 0.74), roughness=0.55),
        "copper": _paint("drone_copper", (0.85, 0.42, 0.22), metallic=1.0, roughness=0.3),
        "wire_black": _paint("drone_wire_black", (0.015, 0.015, 0.016), roughness=0.38),
        "wire_red": _paint("drone_wire_red", (0.5, 0.03, 0.02), roughness=0.38),
        "battery": _paint("drone_battery", (0.012, 0.013, 0.016), roughness=0.55),
        "battery_band": _paint("drone_battery_band", (0.02, 0.55, 0.7), roughness=0.3),
        "rubber": _paint("drone_rubber", (0.02, 0.02, 0.022), roughness=0.85),
        "prop": _paint("drone_prop", (0.012, 0.014, 0.017), roughness=0.32, coat=0.4),
        "glass": _paint("drone_glass", (0.01, 0.02, 0.03), roughness=0.03, coat=1.0),
        "lens_ring": _paint("drone_lens_ring", (0.03, 0.03, 0.033), metallic=0.8, roughness=0.3),
        "led": _glow("drone_led", (0.05, 0.75, 1.0), 12.0),
        "eye": _glow("drone_eye", (1.0, 0.04, 0.02), 6.0),
        "muzzle": _glow("drone_muzzle", (0.3, 0.9, 1.0), 3.0),
    }


# -- geometry helpers ------------------------------------------------------------------------


def _mesh_from(name, build):
    bm = bmesh.new()
    build(bm)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    mesh = bpy.data.meshes.new(name)
    bm.to_mesh(mesh)
    bm.free()
    return mesh


def _obj(name, mesh, parent, mat, location=(0, 0, 0), rotation=(0, 0, 0), smooth=False):
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    obj.location, obj.rotation_euler = location, rotation
    obj.parent = parent
    obj.data.materials.append(mat)
    if smooth:
        for poly in obj.data.polygons:
            poly.use_smooth = True
    return obj


def _bevel(obj, width=0.0002, segments=2):
    m = obj.modifiers.new("bevel", "BEVEL")
    m.width, m.segments, m.limit_method = width, segments, "ANGLE"
    return obj


def _extrude(name, outline, z0, z1):
    """A flat part: a closed (x, y) outline extruded from z0 to z1."""
    def build(bm):
        face = bm.faces.new([bm.verts.new((x, y, z0)) for x, y in outline])
        ext = bmesh.ops.extrude_face_region(bm, geom=[face])
        bmesh.ops.translate(bm, vec=(0, 0, z1 - z0),
                            verts=[v for v in ext["geom"] if isinstance(v, bmesh.types.BMVert)])
    return _mesh_from(name, build)


def _lathe(name, profile, steps=48):
    """A solid of revolution about Z from a closed (r, z) profile."""
    def build(bm):
        verts = [bm.verts.new((r, 0.0, z)) for r, z in profile]
        edges = [bm.edges.new((verts[i], verts[(i + 1) % len(verts)])) for i in range(len(verts))]
        bmesh.ops.spin(bm, geom=verts + edges, angle=2 * math.pi, steps=steps, axis=(0, 0, 1), cent=(0, 0, 0))
        bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-7)
    return _mesh_from(name, build)


def _box(name, sx, sy, sz, center=(0.0, 0.0, 0.0)):
    def build(bm):
        bmesh.ops.create_cube(bm, size=1.0)
        bmesh.ops.scale(bm, vec=(sx, sy, sz), verts=bm.verts)
        bmesh.ops.translate(bm, vec=center, verts=bm.verts)
    return _mesh_from(name, build)


def _cyl(name, r, h, steps=32):
    """A cylinder on Z from z = 0 to z = h."""
    return _lathe(name, [(0.0, 0.0), (r, 0.0), (r, h), (0.0, h)], steps)


def _rounded(cx, cy, w, h, r, seg=6):
    """A rounded rectangle outline (centre, size, corner radius), counter-clockwise."""
    pts = []
    for qx, qy, a0 in ((1, 1, 0), (-1, 1, 90), (-1, -1, 180), (1, -1, 270)):
        ox, oy = cx + qx * (w / 2 - r), cy + qy * (h / 2 - r)
        for i in range(seg + 1):
            a = math.radians(a0 + 90 * i / seg)
            pts.append((ox + r * math.cos(a), oy + r * math.sin(a)))
    return pts


def _circle(cx, cy, r, steps=24):
    return [(cx + r * math.cos(2 * math.pi * i / steps), cy + r * math.sin(2 * math.pi * i / steps))
            for i in range(steps)]


def _boolean(obj, cutters, operation="DIFFERENCE"):
    """Apply boolean modifiers (exact solver) with each cutter, then delete the cutters."""
    for c in cutters:
        m = obj.modifiers.new(f"bool_{c.name}", "BOOLEAN")
        m.operation, m.object, m.solver = operation, c, "EXACT"
    bpy.context.view_layer.objects.active = obj
    for m in list(obj.modifiers):
        if m.type == "BOOLEAN":
            bpy.ops.object.modifier_apply(modifier=m.name)
    for c in cutters:
        bpy.data.objects.remove(c, do_unlink=True)


def _cutter(name, mesh, location=(0, 0, 0), rotation=(0, 0, 0)):
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    obj.location, obj.rotation_euler = location, rotation
    return obj


def _wire(name, points, radius, parent, mat):
    """A cable: a smooth curve through `points` with a round section."""
    curve = bpy.data.curves.new(name, "CURVE")
    curve.dimensions = "3D"
    curve.bevel_depth = radius
    curve.bevel_resolution = 3
    spline = curve.splines.new("BEZIER")
    spline.bezier_points.add(len(points) - 1)
    for bp, p in zip(spline.bezier_points, points):
        bp.co = p
        bp.handle_left_type = bp.handle_right_type = "AUTO"
    obj = bpy.data.objects.new(name, curve)
    bpy.context.collection.objects.link(obj)
    obj.parent = parent
    curve.materials.append(mat)
    return obj


def _screw(name, parent, mats, location, axis="Z", flip=False):
    """An M2 socket-head cap screw head (3.8 mm across, 2 mm tall) with its hex recess."""
    head = _obj(name, _lathe(name, [(0.0, 0.0), (0.0019, 0.0), (0.0019, 0.0017), (0.0017, 0.0020),
                                    (0.0, 0.0020)], 24), parent, mats["black_screw"], location=location,
                smooth=True)
    hexa = _obj(f"{name}_hex", _extrude(f"{name}_hex", _circle(0.0, 0.0, 0.00085, 6), 0.0015, 0.00205),
                parent, mats["chip"], location=location)
    rot = {"Z": (0, 0, 0), "X": (0, math.radians(90), 0), "Y": (math.radians(-90), 0, 0)}[axis]
    if flip:
        rot = (rot[0] + math.pi, rot[1], rot[2])
    head.rotation_euler = hexa.rotation_euler = rot
    return head


# -- parts -------------------------------------------------------------------------------------


def _plate(root, mats, name):
    """The carbon bottom plate: centre, arms and motor pads unioned into one outline, then drilled."""
    t0, t1 = -PLATE_T, 0.0
    plate = _obj(f"{name}_plate", _extrude(f"{name}_plate", _rounded(0.0, 0.0, 0.040, 0.032, 0.006), t0, t1),
                 root, mats["carbon"])
    adds = []
    for k, m in enumerate(MOTORS):
        a = math.atan2(m.y, m.x)
        arm = _cutter(f"{name}_arm{k}", _extrude(f"{name}_arm{k}", _rounded(ARM_R / 2 + 0.002, 0.0, ARM_R - 0.004,
                                                                            0.0085, 0.0025), t0 - 1e-5, t1 + 1e-5),
                      rotation=(0, 0, a))
        pad = _cutter(f"{name}_pad{k}", _extrude(f"{name}_pad{k}", _circle(0.0, 0.0, 0.0105, 40), t0 - 2e-5, t1 + 2e-5),
                      location=(m.x, m.y, 0.0))
        adds += [arm, pad]
    _boolean(plate, adds, "UNION")
    holes = []
    for i, (x, y) in enumerate(((0.01, 0.01), (-0.01, 0.01), (-0.01, -0.01), (0.01, -0.01))):  # the stack
        holes.append(_cutter(f"{name}_h{i}", _cyl(f"{name}_h{i}", 0.0011, 0.004, 16), location=(x, y, -0.002)))
    for k, m in enumerate(MOTORS):  # motor bolts (9 x 9 mm) and the shaft clearance
        for j in range(4):
            b = math.radians(45 + 90 * j)
            holes.append(_cutter(f"{name}_m{k}{j}", _cyl(f"{name}_m{k}{j}", 0.0011, 0.004, 16),
                                 location=(m.x + 0.0045 * math.cos(b), m.y + 0.0045 * math.sin(b), -0.002)))
        holes.append(_cutter(f"{name}_s{k}", _cyl(f"{name}_s{k}", 0.0028, 0.004, 24), location=(m.x, m.y, -0.002)))
    for i, y in enumerate((0.0125, -0.0125)):  # battery-strap slots
        holes.append(_cutter(f"{name}_slot{i}", _extrude(f"{name}_slot{i}", _rounded(0.0, y, 0.011, 0.0022, 0.001),
                                                         -0.003, 0.002)))
    _boolean(plate, holes)
    _bevel(plate, 0.00015, 1)
    return plate


def _duct(root, mats, name, c, a):
    """A thin duct (0.8 mm wall, 10.5 mm tall) with a small rolled lip, standing on the motor pad,
    two bolted tabs."""
    profile = [(0.0205, 0.0), (0.0205, 0.0094), (0.0207, 0.0101), (0.0212, 0.0105), (0.0218, 0.0106),
               (0.0222, 0.0103), (0.0220, 0.0099), (0.0215, 0.0097), (0.0213, 0.0092), (0.0213, 0.0)]
    duct = _obj(f"{name}", _lathe(f"{name}", profile, 96), root, mats["duct"], location=(c.x, c.y, 0.0), smooth=True)
    for s in (-1, 1):  # tabs on the arm's sides, inside the duct, bolted to the plate
        b = a + math.pi + s * math.radians(28)
        p = Vector((c.x + 0.018 * math.cos(b), c.y + 0.018 * math.sin(b), 0.0))
        tab = _obj(f"{name}_tab{s}", _extrude(f"{name}_tab{s}", _rounded(0.0, 0.0, 0.006, 0.0055, 0.0015), 0.0, 0.0018),
                   root, mats["duct"], location=p, rotation=(0, 0, b))
        _bevel(tab, 0.0002)
        _screw(f"{name}_screw{s}", root, mats, location=(p.x, p.y, 0.0018))
    return duct


def _ring_guard(root, mats, name, c, a):
    """A prop guard as on the reference sheet: a thin open ring (0.8 mm wall, 5 mm tall) round the
    prop's plane, held off the motor mount by three thin carbon struts."""
    profile = [(0.0213, 0.0062), (0.0213, 0.0104), (0.0215, 0.0110), (0.0219, 0.0112), (0.0222, 0.0109),
               (0.0221, 0.0062)]
    ring = _obj(f"{name}", _lathe(f"{name}", profile, 96), root, mats["duct"], location=(c.x, c.y, 0.0), smooth=True)
    for k in range(3):  # struts, clear of the arm (which runs from the motor back toward the centre)
        b = a + math.pi + math.radians(60 + 120 * k)
        u = Vector((math.cos(b), math.sin(b), 0.0))
        _wire(f"{name}_strut{k}", [Vector((c.x, c.y, 0.0012)) + u * 0.0092, Vector((c.x, c.y, 0.0030)) + u * 0.016,
                                   Vector((c.x, c.y, 0.0064)) + u * 0.0216], 0.00055, root, mats["carbon"])
    return ring


def _motor(root, mats, name, c, spin):
    """An 11-series motor (low, so the prop sits inside the duct): stator base, anodised bell with
    cooling slots, shaft, prop nut."""
    base = _obj(f"{name}_base", _lathe(f"{name}_base", [(0.0, 0.0), (0.0088, 0.0), (0.0088, 0.0012),
                                                         (0.0060, 0.0016), (0.0060, 0.0022), (0.0, 0.0022)], 48),
                root, mats["alu_dark"], location=(c.x, c.y, 0.0), smooth=True)
    _bevel(base, 0.0001, 1)
    bell = _obj(f"{name}_bell", _lathe(f"{name}_bell", [(0.0, 0.0024), (0.0080, 0.0024), (0.0080, 0.0058),
                                                         (0.0076, 0.0065), (0.0062, 0.0068), (0.0028, 0.0069),
                                                         (0.0024, 0.0072), (0.0, 0.0072)], 64),
                root, mats["alu_bell"], location=(c.x, c.y, 0.0), smooth=True)
    slots = []
    for j in range(6):  # cooling windows in the bell's top, the copper windings showing through
        b = math.radians(60 * j)
        slots.append(_cutter(f"{name}_slot{j}", _box(f"{name}_slot{j}", 0.0028, 0.0016, 0.003),
                             location=(c.x + 0.0046 * math.cos(b), c.y + 0.0046 * math.sin(b), 0.0071),
                             rotation=(0, 0, b)))
    _boolean(bell, slots)
    _obj(f"{name}_windings", _cyl(f"{name}_windings", 0.0064, 0.0040, 36), root, mats["copper"],
         location=(c.x, c.y, 0.0026))
    _obj(f"{name}_shaft", _cyl(f"{name}_shaft", 0.00075, 0.0034, 16), root, mats["steel"],
         location=(c.x, c.y, 0.0072))
    hub = bpy.data.objects.new(f"{name}_prop", None)
    bpy.context.collection.objects.link(hub)
    hub.parent, hub.location = root, (c.x, c.y, 0.0078)
    hub.empty_display_size = 0.01
    _prop(hub, mats, f"{name}_prop", spin)
    nut = _obj(f"{name}_nut", _extrude(f"{name}_nut", _circle(0.0, 0.0, 0.0018, 6), 0.0, 0.0016), root,
               mats["steel"], location=(c.x, c.y, 0.0099))
    _bevel(nut, 0.0001, 1)
    return hub


def _prop(hub, mats, name, spin):
    """A 40 mm three-blade prop: hub, twisted cambered blades (root 30 degrees, tip 12)."""
    _obj(f"{name}_hub", _cyl(f"{name}_hub", 0.0028, 0.0022, 24), hub, mats["prop"], location=(0, 0, 0.0))

    def blade(bm):
        rings, prev = 12, None
        for i in range(rings + 1):
            t = i / rings
            r = 0.0026 + t * (0.0198 - 0.0026)
            chord = 0.0062 * (1 - t) + 0.0028 * t if t < 0.92 else 0.0028 * (1 - (t - 0.92) / 0.08 * 0.8)
            pitch = math.radians(30 * (1 - t) + 12 * t)
            thick = 0.0007 * (1 - 0.6 * t)
            section = [(-0.5, 0.0), (-0.3, 0.55), (0.05, 0.9), (0.35, 0.55), (0.5, 0.0), (0.05, -0.2)]
            ring = []
            for u, v in section:
                y, z = u * chord, v * thick
                ring.append(bm.verts.new((r, spin * (y * math.cos(pitch) - z * math.sin(pitch)),
                                          y * math.sin(pitch) + z * math.cos(pitch) + 0.0011)))
            if prev:
                n = len(ring)
                for j in range(n):
                    bm.faces.new((prev[j], prev[(j + 1) % n], ring[(j + 1) % n], ring[j]))
            else:
                bm.faces.new(ring[::-1])
            prev = ring
        bm.faces.new(prev)
    for i in range(3):
        _obj(f"{name}_blade{i}", _mesh_from(f"{name}_blade{i}", blade), hub, mats["prop"],
             rotation=(0, 0, i * 2 * math.pi / 3), smooth=True)


def _stack(root, mats, name):
    """An ESC under a flight controller on four M2 standoffs; components and pads."""
    for i, (x, y) in enumerate(((0.01, 0.01), (-0.01, 0.01), (-0.01, -0.01), (0.01, -0.01))):
        _obj(f"{name}_standoff{i}", _extrude(f"{name}_standoff{i}", _circle(0.0, 0.0, 0.0016, 6), 0.0, 0.0086),
             root, mats["alu_dark"], location=(x, y, 0.0))
        _screw(f"{name}_screw{i}", root, mats, location=(x, y, 0.0086))
    for level, z in (("esc", 0.0035), ("fc", 0.0075)):
        board = _obj(f"{name}_{level}", _extrude(f"{name}_{level}", _rounded(0.0, 0.0, 0.0255, 0.0255, 0.002),
                                                 z, z + 0.001), root, mats["pcb"])
        holes = [_cutter(f"{name}_{level}_h{i}", _cyl(f"{name}_{level}_h{i}", 0.00105, 0.004, 16),
                         location=(x, y, z - 0.001))
                 for i, (x, y) in enumerate(((0.01, 0.01), (-0.01, 0.01), (-0.01, -0.01), (0.01, -0.01)))]
        _boolean(board, holes)
        for i, (x, y) in enumerate(((0.01, 0.01), (-0.01, 0.01), (-0.01, -0.01), (0.01, -0.01))):  # plated rings
            _obj(f"{name}_{level}_ring{i}", _lathe(f"{name}_{level}_ring{i}", [(0.00105, 0.0), (0.0018, 0.0),
                                                                                (0.0018, 0.00005), (0.00105, 0.00005)],
                                                   20),
                 root, mats["gold"], location=(x, y, z + 0.001))
        top = z + 0.001
        if level == "esc":
            for k in range(4):  # a MOSFET pair at each corner
                a = math.radians(45 + 90 * k)
                for j in (-1, 1):
                    p = Vector((0.0068 * math.cos(a), 0.0068 * math.sin(a), 0.0)) + \
                        Vector((-math.sin(a), math.cos(a), 0.0)) * j * 0.0019
                    _obj(f"{name}_fet{k}{j}", _box(f"{name}_fet{k}{j}", 0.003, 0.003, 0.0009,
                                                   center=(p.x, p.y, top + 0.00045)), root, mats["chip"])
        else:
            _obj(f"{name}_mcu", _box(f"{name}_mcu", 0.007, 0.007, 0.0009, center=(0.0, 0.0, top + 0.00045)), root,
                 mats["chip"])
            for i in range(12):  # the MCU's pins along two sides
                for side in (-1, 1):
                    _obj(f"{name}_pin{i}{side}", _box(f"{name}_pin{i}{side}", 0.0002, 0.0007, 0.0002,
                                                      center=(-0.0033 + 0.0006 * i, side * 0.0039, top + 0.0001)),
                         root, mats["steel"])
            _obj(f"{name}_gyro", _box(f"{name}_gyro", 0.003, 0.003, 0.0009, center=(-0.0068, 0.0055, top + 0.00045)),
                 root, mats["chip"])
            for i, (x, y) in enumerate(((0.0062, -0.0062), (0.0075, -0.0045), (-0.0072, -0.006))):
                _obj(f"{name}_cap{i}", _box(f"{name}_cap{i}", 0.0016, 0.0008, 0.0008, center=(x, y, top + 0.0004)),
                     root, mats["ceramic"])
            _obj(f"{name}_conn", _box(f"{name}_conn", 0.006, 0.0032, 0.0028, center=(-0.0055, -0.0083, top + 0.0014)),
                 root, mats["connector"])
        for k, m in enumerate(MOTORS):  # motor pads on the ESC's edge facing each arm
            a = math.atan2(m.y, m.x)
            for j in (-1, 0, 1):
                p = Vector((0.0112 * math.cos(a), 0.0112 * math.sin(a), 0.0)) + \
                    Vector((-math.sin(a), math.cos(a), 0.0)) * j * 0.0019
                if level == "esc":
                    _obj(f"{name}_pad{k}{j}", _box(f"{name}_pad{k}{j}", 0.0016, 0.0012, 0.00008,
                                                   center=(p.x, p.y, top + 0.00004)), root, mats["gold"])
    # The ESC's bulk capacitor on its side behind the stack, with its leads.
    _obj(f"{name}_bulk", _cyl(f"{name}_bulk", 0.0031, 0.0085, 28), root, mats["battery_band"],
         location=(-0.0175, 0.0, 0.0031), rotation=(0, math.radians(-90), 0), smooth=True)


def _motor_leads(root, mats, name):
    """Three leads from each motor along its arm to the ESC's pads."""
    for k, m in enumerate(MOTORS):
        a = math.atan2(m.y, m.x)
        u, n = Vector((math.cos(a), math.sin(a), 0.0)), Vector((-math.sin(a), math.cos(a), 0.0))
        for j in (-1, 0, 1):
            start = Vector((m.x, m.y, 0.0012)) - u * 0.0082 + n * j * 0.0011
            mid = Vector((0.0, 0.0, 0.0006)) + u * 0.027 + n * j * 0.0009
            end = Vector((0.0, 0.0, 0.0046)) + u * 0.0112 + n * j * 0.0019
            _wire(f"{name}_lead{k}{j}", [start, mid, end + Vector((0, 0, -0.001)), end], 0.00035, root,
                  mats["wire_black"])


def _battery(root, mats, name):
    """A 1S LiPo (55 x 18 x 9 mm) on the flight controller, its band, the strap, leads, plug."""
    z0 = 0.0108
    pack = _obj(f"{name}", _box(f"{name}", 0.055, 0.018, 0.009, center=(-0.004, 0.0, z0 + 0.0045)), root,
                mats["battery"])
    _bevel(pack, 0.0016, 4)
    band = _obj(f"{name}_band", _box(f"{name}_band", 0.006, 0.0184, 0.0094, center=(0.012, 0.0, z0 + 0.0045)), root,
                mats["battery_band"])
    _bevel(band, 0.0017, 4)
    # The strap: a flat rubber band (10 mm wide, 0.8 mm thick) over the pack, down through the
    # plate's slots, under the plate.
    path = [(-0.0125, -0.0016), (-0.0125, 0.0), (-0.0112, z0 + 0.003), (-0.0098, z0 + 0.0088),
            (-0.006, z0 + 0.0097), (0.006, z0 + 0.0097), (0.0098, z0 + 0.0088), (0.0112, z0 + 0.003),
            (0.0125, 0.0), (0.0125, -0.0016), (0.006, -0.0024), (-0.006, -0.0024)]

    def band(bm):
        rows = []
        for y, z in path:
            rows.append((bm.verts.new((-0.005, y, z)), bm.verts.new((0.005, y, z))))
        n = len(rows)
        for i in range(n):
            a, b = rows[i], rows[(i + 1) % n]
            bm.faces.new((a[0], a[1], b[1], b[0]))
        bmesh.ops.solidify(bm, geom=bm.faces[:], thickness=0.0008)
    strap = _obj(f"{name}_strap", _mesh_from(f"{name}_strap", band), root, mats["rubber"], smooth=True)
    _bevel(strap, 0.0002, 1)
    for color, dy in (("wire_red", 0.0018), ("wire_black", -0.0018)):
        _wire(f"{name}_{color}", [Vector((-0.0315, dy, z0 + 0.0045)), Vector((-0.037, dy, z0 + 0.004)),
                                  Vector((-0.041, dy * 1.5, z0 + 0.0015)), Vector((-0.0435, dy * 1.5, z0 + 0.0005))],
              0.0006, root, mats[color])
    plug = _obj(f"{name}_plug", _box(f"{name}_plug", 0.006, 0.0075, 0.0045, center=(-0.0465, 0.0, z0 + 0.0005)), root,
                mats["connector"])
    _bevel(plug, 0.0005, 2)


def _camera(root, mats, name):
    """A nano FPV camera (14 mm) tilted up 20 degrees between two printed side plates."""
    tilt = math.radians(-20)
    cam = bpy.data.objects.new(f"{name}", None)
    bpy.context.collection.objects.link(cam)
    cam.parent, cam.location, cam.rotation_euler = root, (0.0215, 0.0, 0.0085), (0.0, tilt, 0.0)
    body = _obj(f"{name}_body", _box(f"{name}_body", 0.012, 0.014, 0.014), cam, mats["chip"])
    _bevel(body, 0.0008, 3)
    barrel = _obj(f"{name}_barrel", _cyl(f"{name}_barrel", 0.0046, 0.0055, 40), cam, mats["lens_ring"],
                  location=(0.006, 0.0, 0.0), rotation=(0, math.radians(90), 0), smooth=True)
    for i in range(4):  # the lens's focus thread
        _obj(f"{name}_thread{i}", _lathe(f"{name}_thread{i}", [(0.0046, -0.0002), (0.0049, 0.0), (0.0046, 0.0002)], 40),
             cam, mats["lens_ring"], location=(0.0068 + 0.0009 * i, 0.0, 0.0), rotation=(0, math.radians(90), 0),
             smooth=True)
    _obj(f"{name}_glass", _lathe(f"{name}_glass", [(0.0, 0.0), (0.0036, 0.0), (0.0034, 0.0006), (0.0, 0.0009)], 40),
         cam, mats["glass"], location=(0.0115, 0.0, 0.0), rotation=(0, math.radians(90), 0), smooth=True)
    if mats.get("iris"):  # the hostile style's red eye, glowing behind the glass
        _obj(f"{name}_iris", _cyl(f"{name}_iris", 0.0016, 0.0004, 24), cam, mats["eye"],
             location=(0.0112, 0.0, 0.0), rotation=(0, math.radians(90), 0), smooth=True)
    for s in (-1, 1):  # the printed mount's side plates and their screws
        side = _obj(f"{name}_side{s}",
                    _extrude(f"{name}_side{s}", [(-0.012, 0.0), (0.003, 0.0), (0.009, 0.007), (0.006, 0.019),
                                                 (-0.008, 0.019), (-0.012, 0.012)], -0.0008, 0.0008),
                    root, mats["tpu"], location=(0.0185, s * 0.0083, 0.0), rotation=(math.radians(90), 0, 0))
        _bevel(side, 0.0003)
        _screw(f"{name}_screw{s}", root, mats, location=(0.0215, s * 0.0092, 0.0085), axis="Y", flip=s < 0)


def _antenna(root, mats, name):
    """A whip antenna at the back: coax from the stack, a black sleeve, a red tip light."""
    base = Vector((-0.019, 0.0, 0.0045))
    tip_base = Vector((-0.026, 0.0, 0.016))
    _wire(f"{name}_coax", [base, Vector((-0.022, 0.0, 0.006)), tip_base], 0.0005, root, mats["wire_black"])
    sleeve = _obj(f"{name}_sleeve", _cyl(f"{name}_sleeve", 0.0011, 0.024, 16), root, mats["rubber"],
                  location=tip_base, rotation=(0, math.radians(-28), 0), smooth=True)
    tip = tip_base + Vector((-math.sin(math.radians(28)), 0.0, math.cos(math.radians(28)))) * 0.024
    _obj(f"{name}_tip", _lathe(f"{name}_tip", [(0.0, 0.0), (0.0011, 0.0), (0.0008, 0.0012), (0.0, 0.0015)], 16),
         root, mats["eye"], location=tip, rotation=(0, math.radians(-28), 0), smooth=True)
    return sleeve


def _emitter(root, mats, name):
    """The emitter: a laser-module barrel (8 mm, knurled) on a printed bracket under the nose."""
    bracket = _obj(f"{name}_bracket", _extrude(f"{name}_bracket", [(-0.008, -0.0055), (0.008, -0.0055),
                                                                   (0.008, 0.0055), (-0.008, 0.0055)],
                                               -0.0045, -PLATE_T), root, mats["tpu"], location=(0.014, 0.0, 0.0))
    _bevel(bracket, 0.0005, 2)
    for s in (-1, 1):
        _screw(f"{name}_screw{s}", root, mats, location=(0.014 + s * 0.005, 0.0, -PLATE_T - 0.0045), flip=True)
    axis = (0.0085, 0.0, -0.0085)
    body = _obj(f"{name}_body", _cyl(f"{name}_body", 0.004, 0.021, 48), root, mats["alu_dark"], location=axis,
                rotation=(0, math.radians(90), 0), smooth=True)
    _bevel(body, 0.0002, 2)
    for i in range(5):  # knurled grip rings
        _obj(f"{name}_knurl{i}", _lathe(f"{name}_knurl{i}", [(0.004, -0.0003), (0.0043, 0.0), (0.004, 0.0003)], 48),
             root, mats["alu_dark"], location=(axis[0] + 0.004 + 0.0012 * i, 0.0, axis[2]),
             rotation=(0, math.radians(90), 0), smooth=True)
    _obj(f"{name}_bezel", _lathe(f"{name}_bezel", [(0.0028, 0.0), (0.0042, 0.0), (0.0042, 0.0014), (0.0028, 0.0010)],
                                 48),
         root, mats["steel"], location=(axis[0] + 0.021, 0.0, axis[2]), rotation=(0, math.radians(90), 0), smooth=True)
    lens = _obj(f"{name}_lens", _lathe(f"{name}_lens", [(0.0, 0.0), (0.0028, 0.0), (0.0026, 0.0006), (0.0, 0.0009)], 40),
                root, mats["muzzle"], location=(axis[0] + 0.0212, 0.0, axis[2]), rotation=(0, math.radians(90), 0),
                smooth=True)
    _wire(f"{name}_lead", [Vector((axis[0], 0.0, axis[2] + 0.002)), Vector((0.004, 0.0, -0.004)),
                           Vector((0.0, 0.0, -0.0025))], 0.0004, root, mats["wire_red"])
    return lens


def _leds(root, mats, name):
    """LED bars under the arms: a strip with three lit dots each."""
    for k, m in enumerate(MOTORS):
        a = math.atan2(m.y, m.x)
        mid = Vector((math.cos(a), math.sin(a), 0.0)) * 0.026
        strip = _obj(f"{name}{k}", _box(f"{name}{k}", 0.014, 0.0045, 0.0007, center=(0.0, 0.0, -0.00035)), root,
                     mats["pcb"], location=(mid.x, mid.y, -PLATE_T), rotation=(0, 0, a))
        for j in (-1, 0, 1):
            p = mid + Vector((math.cos(a), math.sin(a), 0.0)) * j * 0.0042
            _obj(f"{name}{k}_dot{j}", _box(f"{name}{k}_dot{j}", 0.0018, 0.0018, 0.0006,
                                           center=(0.0, 0.0, -0.0003)), root, mats["led"],
                 location=(p.x, p.y, -PLATE_T - 0.0007), rotation=(0, 0, a))
    return strip


# -- the drone -------------------------------------------------------------------------------


#: Where the emitter's lens sits in the drone's own frame (m, at scale 1): a shot leaves from here.
MUZZLE = Vector((0.0297, 0.0, -0.0085))


def add_drone(name: str = "drone", location=(0.0, 0.0, 0.1), scale: float = 1.0, style: str = "cyan"):
    """Build one drone; its root is an empty at `location` (+X forward, the emitter along +X).
    `scale` enlarges the whole airframe (1: the 94 mm micro-quad; the film uses 2 beside the robot).
    `style` "hostile": red lights and eye, dark motors, open ring guards instead of ducts."""
    root = bpy.data.objects.new(name, None)
    root.empty_display_size = 0.05
    bpy.context.collection.objects.link(root)
    mats = _materials(style)
    mats["iris"] = style == "hostile"
    _plate(root, mats, name)
    props = []
    for k, m in enumerate(MOTORS):
        guard = _ring_guard if style == "hostile" else _duct
        guard(root, mats, f"{name}_duct{k}", m, math.atan2(m.y, m.x))
        props.append(_motor(root, mats, f"{name}_motor{k}", m, spin=1 if k % 2 == 0 else -1))
    _stack(root, mats, f"{name}_stack")
    _motor_leads(root, mats, f"{name}_leads")
    _battery(root, mats, f"{name}_battery")
    _camera(root, mats, f"{name}_camera")
    _antenna(root, mats, f"{name}_antenna")
    lens = _emitter(root, mats, f"{name}_emitter")
    _leds(root, mats, f"{name}_led")
    root.location = location
    root.scale = (scale, scale, scale)
    if style == "hostile":  # (10/28 afternoon, the user could not find the enemy) a red glow under it:
        # the wet ground round it lights up red, so where it hovers reads even against a dark wall
        lamp = bpy.data.lights.new(f"{name}_underglow", type="POINT")
        lamp.energy, lamp.color, lamp.shadow_soft_size = 1.2 * scale, (1.0, 0.06, 0.03), 0.02
        glow_obj = bpy.data.objects.new(f"{name}_underglow", lamp)
        bpy.context.collection.objects.link(glow_obj)
        glow_obj.parent = root
        glow_obj.location = (0.0, 0.0, -0.025)
    root["props"] = [p.name for p in props]
    root["muzzle"] = lens.name
    return root


def key_hover(root, first: int, last: int, fps: float, bob: float = 0.004) -> None:
    """Key a gentle hover (a few mm of bob and sway) and the props' spin (a quarter turn plus a
    little per frame, so they read as turning rather than strobing still)."""
    base = root.location.copy()
    yaw = root.rotation_euler.z
    for f in range(first, last + 1):
        t = f / fps
        root.location = (base.x, base.y, base.z + bob * math.sin(2 * math.pi * 0.7 * t))
        root.rotation_euler = (0.03 * math.sin(2 * math.pi * 0.5 * t), 0.03 * math.cos(2 * math.pi * 0.4 * t), yaw)
        root.keyframe_insert("location", frame=f)
        root.keyframe_insert("rotation_euler", frame=f)
        for k, name in enumerate(root["props"]):
            prop = bpy.data.objects[name]
            prop.rotation_euler = (0, 0, (-1) ** k * f * (math.pi / 2 + 0.31))
            prop.keyframe_insert("rotation_euler", frame=f)
    root.location = base


def key_fire(root, frame: int) -> None:
    """A muzzle flash on `frame`: the emitter's lens flares and fades over three frames."""
    flash = bpy.data.objects[root["muzzle"]]
    for f, s in ((frame - 1, 1.0), (frame, 3.0), (frame + 1, 2.2), (frame + 2, 1.5), (frame + 3, 1.0)):
        flash.scale = (s, s, s)
        flash.keyframe_insert("scale", frame=f)


# -- a studio still ----------------------------------------------------------------------------


def _studio(scene, target):
    world = bpy.data.worlds.new("studio")
    world.use_nodes = True
    nt = world.node_tree
    bg = next((n for n in nt.nodes if n.type == "BACKGROUND"), None) or nt.nodes.new("ShaderNodeBackground")
    out = next((n for n in nt.nodes if n.type == "OUTPUT_WORLD"), None) or nt.nodes.new("ShaderNodeOutputWorld")
    nt.links.new(bg.outputs["Background"], out.inputs["Surface"])
    bg.inputs[0].default_value = (0.004, 0.005, 0.008, 1.0)
    scene.world = world
    fl = bpy.data.objects.new("floor", _mesh_from("floor", lambda bm: bmesh.ops.create_grid(
        bm, x_segments=1, y_segments=1, size=1.5)))
    scene.collection.objects.link(fl)
    fl.data.materials.append(_paint("studio_floor", (0.006, 0.0065, 0.008), metallic=0.0, roughness=0.6))
    for name, loc, energy, color, size in (("key", (0.2, -0.24, 0.3), 7.0, (1.0, 0.97, 0.94), 0.08),
                                           ("rim", (-0.3, 0.22, 0.16), 26.0, (0.3, 0.7, 1.0), 0.06),
                                           ("top", (0.0, 0.05, 0.35), 0.7, (0.9, 0.92, 1.0), 0.3)):
        light = bpy.data.objects.new(name, bpy.data.lights.new(name, type="AREA"))
        light.data.energy, light.data.color, light.data.size = energy, color, size
        light.location = loc
        light.rotation_euler = (Vector(target) - Vector(loc)).to_track_quat("-Z", "Y").to_euler()
        scene.collection.objects.link(light)


def main() -> None:
    import argparse

    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--render", default="")
    p.add_argument("--yaw", type=float, default=-35.0, help="the drone's heading in the still, degrees")
    p.add_argument("--cam", default="0.17,-0.17,0.17", help="camera position, m")
    p.add_argument("--lens", type=float, default=55.0)
    p.add_argument("--blend", default="", help="also save the scene to this .blend")
    p.add_argument("--style", default="cyan", choices=["cyan", "hostile"])
    args = p.parse_args(argv)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    root = add_drone(location=(0.0, 0.0, 0.08), style=args.style)
    root.rotation_euler = (math.radians(3), math.radians(-5), math.radians(args.yaw))
    _studio(scene, root.location)
    cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam"))
    scene.collection.objects.link(cam)
    cam.location = Vector([float(v) for v in args.cam.split(",")])
    cam.data.lens = args.lens
    cam.data.clip_start = 0.005
    cam.rotation_euler = (root.location + Vector((0, 0, 0.004)) - cam.location).to_track_quat("-Z", "Y").to_euler()
    scene.camera = cam
    if args.blend:
        bpy.ops.wm.save_as_mainfile(filepath=args.blend)
    if args.render:
        engines = [e.identifier for e in bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items]
        scene.render.engine = "BLENDER_EEVEE_NEXT" if "BLENDER_EEVEE_NEXT" in engines else "BLENDER_EEVEE"
        scene.render.resolution_x, scene.render.resolution_y = 1600, 900
        views = [v.identifier for v in bpy.types.ColorManagedViewSettings.bl_rna.properties["view_transform"].enum_items]
        if "AgX" in views:
            scene.view_settings.view_transform = "AgX"
            looks = [v.identifier for v in bpy.types.ColorManagedViewSettings.bl_rna.properties["look"].enum_items]
            punchy = next((k for k in looks if "Punchy" in k), None)
            if args.style == "hostile" and punchy:  # keeps the red lights red rather than salmon
                scene.view_settings.look = punchy
        scene.render.filepath = args.render
        bpy.ops.render.render(write_still=True)
        print(f"DRONE_RENDERED {args.render}")


if __name__ == "__main__":
    main()
