"""The film's set (10/26): a rain-soaked back street at night, built procedurally around a take.

After the reference frames the user generated for the assault (night, wet asphalt mirroring neon,
a ground-level wide lens, haze): a street 8 m wide -- the drones fire from 3 m to the robot's
side, so the walls stand clear of them -- between rows of buildings with lit windows, shopfronts,
vertical neon signs, air-conditioner units, pipes, vending machines and cables, an overpass and
towers far down the street. Every sign carries only generic Japanese words (alley, night wind,
repairs, ...); no brand names, no logos.

`build(origin, yaw)` lays the street along `yaw` (radians) through `origin` (x, y) and returns
its objects; `setup_render(scene)` sets EEVEE for it (ray-traced reflections for the wet ground,
volumetric haze, AgX).
"""

from __future__ import annotations

import math
import random

import bpy
from mathutils import Matrix, Vector

#: Noto Sans JP (OFL): the PC's system copy, or the one Colab gets from Drive (`/content/up`).
FONT = next((f for f in ("C:/Windows/Fonts/NotoSansJP-VF.ttf", "/content/up/NotoSansJP-VF.ttf")
             if __import__("os").path.exists(f)), "/content/up/NotoSansJP-VF.ttf")
#: (text, rgb) for the vertical neon signs: generic words only.
SIGNS = [("路地", (0.1, 0.55, 1.0)), ("夜風", (0.05, 0.85, 1.0)), ("月下", (1.0, 0.1, 0.45)),
         ("修理", (1.0, 0.25, 0.05)), ("電脳", (0.75, 0.15, 1.0)), ("回路", (0.05, 1.0, 0.6)),
         ("飲食", (1.0, 0.08, 0.3)), ("第七区画", (0.1, 0.6, 1.0)), ("夜市", (1.0, 0.45, 0.05)),
         ("非常口", (0.1, 1.0, 0.3)), ("立入禁止", (1.0, 0.05, 0.05)), ("屋台", (1.0, 0.6, 0.1))]


def _bsdf(mat):
    nt = mat.node_tree
    bsdf = next((n for n in nt.nodes if n.type == "BSDF_PRINCIPLED"), None)
    if bsdf is None:
        bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
        out = next((n for n in nt.nodes if n.type == "OUTPUT_MATERIAL"), None) or nt.nodes.new("ShaderNodeOutputMaterial")
        nt.links.new(bsdf.outputs[0], out.inputs["Surface"])
    return bsdf


def _mat(name, rgb, roughness=0.6, metallic=0.0):
    mat = bpy.data.materials.get(name)
    if mat is None:
        mat = bpy.data.materials.new(name)
        mat.use_nodes = True
        b = _bsdf(mat)
        b.inputs["Base Color"].default_value = (*rgb, 1.0)
        b.inputs["Roughness"].default_value = roughness
        b.inputs["Metallic"].default_value = metallic
        mat.diffuse_color = (*rgb, 1.0)
    return mat


def _glow(name, rgb, strength):
    mat = bpy.data.materials.get(name)
    if mat is None:
        mat = bpy.data.materials.new(name)
        mat.use_nodes = True
        b = _bsdf(mat)
        b.inputs["Base Color"].default_value = (0.0, 0.0, 0.0, 1.0)
        b.inputs["Emission Color"].default_value = (*rgb, 1.0)
        b.inputs["Emission Strength"].default_value = strength
        mat.diffuse_color = (*rgb, 1.0)
    return mat


def _asphalt():
    """Wet asphalt: dark, fine-grained, with puddles (a large noise thresholded) nearly mirror-smooth."""
    mat = bpy.data.materials.new("wet_asphalt")
    mat.use_nodes = True
    nt = mat.node_tree
    b = _bsdf(mat)
    coord = nt.nodes.new("ShaderNodeTexCoord")
    big = nt.nodes.new("ShaderNodeTexNoise")
    big.inputs["Scale"].default_value = 0.35
    big.inputs["Detail"].default_value = 6.0
    big.inputs["Roughness"].default_value = 0.55
    nt.links.new(coord.outputs["Object"], big.inputs["Vector"])
    ramp = nt.nodes.new("ShaderNodeValToRGB")  # puddle mask: 1 in a puddle
    ramp.color_ramp.elements[0].position, ramp.color_ramp.elements[1].position = 0.42, 0.50
    nt.links.new(big.outputs["Fac"], ramp.inputs["Fac"])
    rough = nt.nodes.new("ShaderNodeMapRange")
    rough.inputs["To Min"].default_value, rough.inputs["To Max"].default_value = 0.08, 0.01
    nt.links.new(ramp.outputs["Color"], rough.inputs["Value"])
    nt.links.new(rough.outputs["Result"], b.inputs["Roughness"])
    grain = nt.nodes.new("ShaderNodeTexNoise")
    grain.inputs["Scale"].default_value = 60.0
    grain.inputs["Detail"].default_value = 2.0
    nt.links.new(coord.outputs["Object"], grain.inputs["Vector"])
    shade = nt.nodes.new("ShaderNodeMapRange")
    shade.inputs["To Min"].default_value, shade.inputs["To Max"].default_value = 0.010, 0.030
    nt.links.new(grain.outputs["Fac"], shade.inputs["Value"])
    mix = nt.nodes.new("ShaderNodeMix")  # puddles darker, asphalt grain lighter
    mix.data_type = "FLOAT"
    nt.links.new(ramp.outputs["Color"], mix.inputs["Factor"])
    nt.links.new(shade.outputs["Result"], mix.inputs["A"])
    mix.inputs["B"].default_value = 0.006
    nt.links.new(mix.outputs["Result"], b.inputs["Base Color"])
    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.03
    nt.links.new(grain.outputs["Fac"], bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], b.inputs["Normal"])
    mat.diffuse_color = (0.02, 0.02, 0.022, 1.0)
    return mat


def _facade(name, seed):
    """Dark concrete with a grid of windows, about one in three lit warm or cool."""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    b = _bsdf(mat)
    b.inputs["Base Color"].default_value = (0.03, 0.03, 0.034, 1.0)
    b.inputs["Roughness"].default_value = 0.8
    coord = nt.nodes.new("ShaderNodeTexCoord")
    brick = nt.nodes.new("ShaderNodeTexBrick")  # bricks: windows, mortar: wall
    brick.offset = 0.0
    brick.inputs["Scale"].default_value = 1.0
    brick.inputs["Brick Width"].default_value = 1.6
    brick.inputs["Row Height"].default_value = 3.0
    brick.inputs["Mortar Size"].default_value = 0.55
    brick.inputs["Color1"].default_value = (1.0, 1.0, 1.0, 1.0)
    brick.inputs["Color2"].default_value = (0.0, 0.0, 0.0, 1.0)
    brick.inputs["Mortar"].default_value = (0.0, 0.0, 0.0, 1.0)
    brick.inputs["Bias"].default_value = -0.35 + 0.1 * (seed % 3)
    face = nt.nodes.new("ShaderNodeMapping")  # the wall's plane (along the street, up) onto the brick's XY
    face.inputs["Rotation"].default_value = (math.pi / 2, 0.0, 0.0)
    nt.links.new(coord.outputs["Object"], face.inputs["Vector"])
    nt.links.new(face.outputs["Vector"], brick.inputs["Vector"])
    warm = nt.nodes.new("ShaderNodeMix")
    warm.data_type = "RGBA"
    noise = nt.nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 0.4
    nt.links.new(coord.outputs["Object"], noise.inputs["Vector"])
    nt.links.new(noise.outputs["Fac"], warm.inputs["Factor"])
    warm.inputs["A"].default_value = (1.0, 0.55, 0.22, 1.0)
    warm.inputs["B"].default_value = (0.35, 0.6, 1.0, 1.0)
    lit = nt.nodes.new("ShaderNodeMix")
    lit.data_type = "RGBA"
    nt.links.new(brick.outputs["Color"], lit.inputs["Factor"])
    lit.inputs["A"].default_value = (0.0, 0.0, 0.0, 1.0)
    nt.links.new(warm.outputs["Result"], lit.inputs["B"])
    nt.links.new(lit.outputs["Result"], b.inputs["Emission Color"])
    b.inputs["Emission Strength"].default_value = 3.0
    mat.diffuse_color = (0.03, 0.03, 0.034, 1.0)
    return mat


def _box(name, size, location, mat, rotation=(0.0, 0.0, 0.0), parent=None):
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=location, rotation=rotation)
    obj = bpy.context.active_object
    obj.name = name
    # the size baked into the mesh, so that object coordinates are metres (textures keep scale)
    obj.data.transform(Matrix.Diagonal((*size, 1.0)))
    obj.data.materials.append(mat)
    if parent is not None:
        obj.parent = parent
    return obj


def _cyl(name, radius, depth, location, mat, rotation=(0.0, 0.0, 0.0), verts=24):
    bpy.ops.mesh.primitive_cylinder_add(radius=radius, depth=depth, location=location, rotation=rotation,
                                        vertices=verts)
    obj = bpy.context.active_object
    obj.name = name
    obj.data.materials.append(mat)
    return obj


def _text(name, body, size, location, rotation, mat, font):
    curve = bpy.data.curves.new(name, "FONT")
    curve.body = body
    curve.font = font
    curve.size = size
    curve.align_x, curve.align_y = "CENTER", "CENTER"
    curve.extrude = size * 0.03
    obj = bpy.data.objects.new(name, curve)
    bpy.context.collection.objects.link(obj)
    obj.location, obj.rotation_euler = location, rotation
    curve.materials.append(mat)
    return obj


def _light(name, kind, location, energy, rgb, size=0.5, rotation=(0.0, 0.0, 0.0)):
    data = bpy.data.lights.new(name, type=kind)
    data.energy, data.color = energy, rgb
    if kind == "AREA":
        data.size = size
    else:
        data.shadow_soft_size = size
    obj = bpy.data.objects.new(name, data)
    bpy.context.collection.objects.link(obj)
    obj.location, obj.rotation_euler = location, rotation
    return obj


def build(origin=(0.0, 0.0), yaw: float = 0.0, length: float = 28.0, width: float = 8.0, behind: float = 6.0,
          seed: int = 7):
    """Lay the street; +u runs along `yaw` from `origin`, +v to its left. Returns the set's root."""
    rnd = random.Random(seed)
    root = bpy.data.objects.new("set", None)
    bpy.context.collection.objects.link(root)
    root.location = (origin[0], origin[1], 0.0)
    root.rotation_euler = (0.0, 0.0, yaw)
    font = bpy.data.fonts.load(FONT)
    made = []

    def keep(obj):
        obj.parent = root
        made.append(obj)
        return obj

    half = width / 2
    u0, u1 = -behind, length
    ground = keep(_box("street", (u1 - u0 + 4, width + 2, 0.02), ((u0 + u1) / 2, 0.0, -0.01), _asphalt()))
    ground.data.materials.clear()
    ground.data.materials.append(bpy.data.materials["wet_asphalt"])
    concrete = _mat("kerb", (0.08, 0.08, 0.085), 0.7)
    paint = _mat("lane_paint", (0.18, 0.18, 0.16), 0.4)
    for s in (-1, 1):  # kerbs and pavements
        keep(_box(f"pavement{s}", (u1 - u0, 0.9, 0.12), ((u0 + u1) / 2, s * (half - 0.45), 0.06), concrete))
    for k in range(int((u1 - u0) / 3)):  # a faded centre line
        keep(_box(f"line{k}", (1.2, 0.07, 0.003), (u0 + 1.0 + 3 * k, 0.0, 0.0015), paint))
    keep(_cyl("manhole", 0.32, 0.006, (3.5, -1.6, 0.003), _mat("manhole", (0.05, 0.05, 0.055), 0.35, 0.7)))

    steel = _mat("steel_dark", (0.12, 0.12, 0.13), 0.4, 0.8)
    ac_mat = _mat("ac_unit", (0.42, 0.42, 0.4), 0.55)
    grille = _mat("grille", (0.02, 0.02, 0.02), 0.6, 0.5)
    shutter = _mat("shutter", (0.16, 0.16, 0.17), 0.45, 0.6)
    bin_mat = _mat("bin", (0.04, 0.05, 0.045), 0.5)
    signs = list(SIGNS)
    rnd.shuffle(signs)
    sign_i = 0
    for s in (-1, 1):  # the two rows of buildings
        u = u0
        while u < u1:
            w = rnd.uniform(2.6, 5.0)
            h = rnd.uniform(7.0, 22.0)
            v = s * (half + 2.0)
            keep(_box(f"bldg{s}_{u:.0f}", (w - 0.08, 4.0, h), (u + w / 2, v, h / 2), _facade(f"facade{s}_{u:.0f}", int(u))))
            front = s * half
            # a shopfront: lit glass or a closed shutter, 2.6 m tall
            if rnd.random() < 0.55:
                rgb = rnd.choice([(1.0, 0.35, 0.08), (0.05, 0.35, 1.0), (1.0, 0.08, 0.3), (0.6, 0.1, 1.0)])
                keep(_box(f"shop{s}_{u:.0f}", (w * 0.7, 0.04, 2.4), (u + w / 2, front - s * 0.02, 1.32),
                          _glow(f"shopglow_{rgb}", rgb, 0.22)))
                for k in range(4):  # window mullions over the lit glass
                    keep(_box(f"mullion{s}_{u:.0f}_{k}", (0.06, 0.06, 2.4),
                              (u + w * 0.15 + w * 0.7 * k / 3, front - s * 0.05, 1.32), shutter))
            else:
                keep(_box(f"shutter{s}_{u:.0f}", (w * 0.7, 0.05, 2.4), (u + w / 2, front - s * 0.03, 1.32), shutter))
            # air-conditioner units on brackets
            for k in range(rnd.randint(1, 3)):
                cu = u + rnd.uniform(0.5, w - 0.5)
                cz = rnd.uniform(3.0, 6.5)
                keep(_box(f"ac{s}_{u:.0f}_{k}", (0.8, 0.3, 0.6), (cu, front - s * 0.17, cz), ac_mat))
                keep(_cyl(f"fan{s}_{u:.0f}_{k}", 0.22, 0.02, (cu - 0.12, front - s * 0.33, cz), grille,
                          rotation=(math.pi / 2, 0.0, 0.0)))
            # a drainpipe
            keep(_cyl(f"pipe{s}_{u:.0f}", 0.05, h, (u + 0.15, front - s * 0.06, h / 2), steel))
            # a vertical neon sign, sticking out from the wall
            if rnd.random() < 0.8 and sign_i < len(signs):
                word, rgb = signs[sign_i]
                sign_i += 1
                tall = 0.85 * len(word) + 0.4
                cz = rnd.uniform(2.4, 3.6) + tall / 2
                cu = u + rnd.uniform(0.6, w - 0.6)
                box = keep(_box(f"signbox_{word}", (0.18, 1.0, tall), (cu, front - s * 0.6, cz),
                                _mat("sign_back", (0.02, 0.02, 0.025), 0.4)))
                for side in (-1, 1):  # the word on both faces, top to bottom
                    t = keep(_text(f"sign_{word}{side}", "\n".join(word), 0.75,
                                   (cu + side * 0.095, front - s * 0.6, cz), (math.pi / 2, 0.0, side * math.pi / 2),
                                   _glow(f"neon_{word}", rgb, 9.0), font))
                    t.data.space_line = 0.9
                frame = _glow(f"neonframe_{word}", rgb, 6.0)
                for side in (-1, 1):
                    keep(_box(f"signedge_{word}{side}", (0.2, 0.025, tall), (cu, front - s * 0.6 + side * 0.5, cz), frame))
                keep(_light(f"signlight_{word}", "AREA", (cu, front - s * 1.0, cz - tall / 2), 60.0, rgb, size=0.8,
                            rotation=(0.0, 0.0, 0.0)))
            # a bin or a vending machine on the pavement
            r = rnd.random()
            if r < 0.3:
                keep(_cyl(f"bin{s}_{u:.0f}", 0.25, 0.75, (u + rnd.uniform(0.5, w - 0.5), front - s * 0.4, 0.495), bin_mat))
            elif r < 0.5:
                vu = u + rnd.uniform(0.6, w - 0.6)
                keep(_box(f"vend{s}_{u:.0f}", (0.9, 0.7, 1.8), (vu, front - s * 0.4, 1.02),
                          _mat("vend_body", (0.6, 0.05, 0.06), 0.35)))
                keep(_box(f"vendfront{s}_{u:.0f}", (0.75, 0.02, 1.2), (vu, front - s * 0.76, 1.2),
                          _glow("vend_panel", (0.85, 0.95, 1.0), 3.0)))
            u += w
        # overhead cables between the rows
    cable = _mat("cable", (0.01, 0.01, 0.01), 0.5)
    for k in range(6):
        cu = rnd.uniform(1.0, u1 - 2.0)
        z = rnd.uniform(5.0, 8.5)
        pts = [Vector((cu, -half, z)), Vector((cu + rnd.uniform(-1, 1), 0.0, z - rnd.uniform(0.4, 1.0))),
               Vector((cu + rnd.uniform(-2, 2), half, z + rnd.uniform(-0.5, 0.5)))]
        curve = bpy.data.curves.new(f"cable{k}", "CURVE")
        curve.dimensions, curve.bevel_depth = "3D", 0.015
        sp = curve.splines.new("BEZIER")
        sp.bezier_points.add(len(pts) - 1)
        for bp, p in zip(sp.bezier_points, pts):
            bp.co = p
            bp.handle_left_type = bp.handle_right_type = "AUTO"
        obj = bpy.data.objects.new(f"cable{k}", curve)
        bpy.context.collection.objects.link(obj)
        curve.materials.append(cable)
        keep(obj)
    # street lamps, warm
    for k, cu in enumerate((2.0, 9.0, 16.0)):
        s = 1 if k % 2 else -1
        keep(_cyl(f"lamppost{k}", 0.05, 4.2, (cu, s * (half - 0.2), 2.1), steel))
        keep(_box(f"lamphead{k}", (0.45, 0.18, 0.08), (cu, s * (half - 0.45), 4.15), _glow("lamp", (1.0, 0.62, 0.3), 25.0)))
        keep(_light(f"lamp{k}", "POINT", (cu, s * (half - 0.5), 4.0), 220.0, (1.0, 0.62, 0.32), size=0.3))
    # an overpass far down the street, and towers beyond it
    keep(_box("overpass", (2.0, width + 6, 1.0), (u1 - 3.0, 0.0, 11.0), _mat("overpass", (0.05, 0.05, 0.055), 0.7)))
    keep(_box("overpass_lights", (0.05, width + 6, 0.06), (u1 - 3.0 - 1.02, 0.0, 10.6), _glow("overpass_glow", (0.3, 0.7, 1.0), 6.0)))
    for k in range(7):
        h = rnd.uniform(40, 90)
        keep(_box(f"tower{k}", (6, 6, h), (u1 + 12 + rnd.uniform(0, 25), rnd.uniform(-14, 14), h / 2),
                  _facade(f"tower_facade{k}", k + 3)))
    keep(_light("backlight", "AREA", (u1 + 4, 0.0, 6.0), 4000.0, (0.45, 0.35, 1.0), size=10.0,
                rotation=(0.0, math.radians(-90), 0.0)))
    # A planar reflection probe over the street: the wet asphalt mirrors the signs even where
    # they are out of shot (screen-space tracing alone loses them).
    try:
        bpy.ops.object.lightprobe_add(type="PLANE", location=((u0 + u1) / 2, 0.0, 0.001))
        probe = bpy.context.active_object
        probe.name = "street_mirror"
        probe.scale = ((u1 - u0) / 2, half, 1.0)
        keep(probe)
    except Exception as err:  # noqa: BLE001 -- the probe is an embellishment
        print(f"[alley] no planar probe: {err}")
    return root


def _concrete(name, seed=0, tone=0.30):
    """Weathered concrete: cast panels with dark seams, rain streaks running down, fine grit."""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    b = _bsdf(mat)
    b.inputs["Roughness"].default_value = 0.88
    coord = nt.nodes.new("ShaderNodeTexCoord")
    face = nt.nodes.new("ShaderNodeMapping")  # the wall's plane (along the alley, up) onto XY
    face.inputs["Rotation"].default_value = (math.pi / 2, 0.0, 0.0)
    face.inputs["Location"].default_value = (seed * 3.1, 0.0, 0.0)
    nt.links.new(coord.outputs["Object"], face.inputs["Vector"])
    panels = nt.nodes.new("ShaderNodeTexBrick")
    panels.offset = 0.0
    panels.inputs["Scale"].default_value = 1.0
    panels.inputs["Brick Width"].default_value = 1.8
    panels.inputs["Row Height"].default_value = 0.9
    panels.inputs["Mortar Size"].default_value = 0.012
    panels.inputs["Color1"].default_value = (tone, tone, tone * 1.02, 1.0)
    panels.inputs["Color2"].default_value = (tone * 0.9, tone * 0.9, tone * 0.92, 1.0)
    panels.inputs["Mortar"].default_value = (tone * 0.35, tone * 0.35, tone * 0.36, 1.0)
    nt.links.new(face.outputs["Vector"], panels.inputs["Vector"])
    streak_map = nt.nodes.new("ShaderNodeMapping")  # stretched down the wall: rain streaks
    streak_map.inputs["Scale"].default_value = (6.0, 0.35, 1.0)
    nt.links.new(face.outputs["Vector"], streak_map.inputs["Vector"])
    streaks = nt.nodes.new("ShaderNodeTexNoise")
    streaks.inputs["Scale"].default_value = 1.5
    streaks.inputs["Detail"].default_value = 6.0
    nt.links.new(streak_map.outputs["Vector"], streaks.inputs["Vector"])
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].position, ramp.color_ramp.elements[1].position = 0.45, 0.75
    ramp.color_ramp.elements[0].color = (0.35, 0.35, 0.35, 1.0)
    ramp.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0)
    nt.links.new(streaks.outputs["Fac"], ramp.inputs["Fac"])
    dirt = nt.nodes.new("ShaderNodeMix")
    dirt.data_type = "RGBA"
    dirt.blend_type = "MULTIPLY"
    dirt.inputs["Factor"].default_value = 1.0
    nt.links.new(panels.outputs["Color"], dirt.inputs["A"])
    nt.links.new(ramp.outputs["Color"], dirt.inputs["B"])
    nt.links.new(dirt.outputs["Result"], b.inputs["Base Color"])
    grit = nt.nodes.new("ShaderNodeTexNoise")
    grit.inputs["Scale"].default_value = 40.0
    nt.links.new(face.outputs["Vector"], grit.inputs["Vector"])
    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.12
    nt.links.new(grit.outputs["Fac"], bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], b.inputs["Normal"])
    mat.diffuse_color = (tone, tone, tone, 1.0)
    return mat


def _bag(name, location, scale, rnd, mat):
    """A full rubbish bag: a lumpy blob of glossy black plastic, sagging on the ground."""
    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=3, radius=1.0, location=location)
    obj = bpy.context.active_object
    obj.name = name
    for v in obj.data.vertices:
        n = v.co.normalized()
        bulge = 1.0 + 0.18 * math.sin(3.1 * n.x + rnd.random() * 0.4) * math.cos(2.7 * n.y) + 0.06 * rnd.random()
        v.co = n * bulge
        if v.co.z < -0.35:  # sagging onto the ground
            v.co.z = -0.35 - (v.co.z + 0.35) * 0.25
    obj.data.transform(Matrix.Diagonal((*scale, 1.0)))
    obj.location.z = location[2] + 0.35 * scale[2]
    obj.rotation_euler = (0.0, 0.0, rnd.uniform(0, math.pi))
    bpy.ops.object.shade_smooth()
    obj.data.materials.append(mat)
    # the knot on top
    bpy.ops.mesh.primitive_cone_add(radius1=0.05 * scale[0] / 0.3, depth=0.12 * scale[2] / 0.28,
                                    location=(location[0], location[1], location[2] + 1.25 * scale[2]))
    knot = bpy.context.active_object
    knot.name = f"{name}_knot"
    knot.data.materials.append(mat)
    return obj, knot


def build_backalley(origin=(0.0, 0.0), yaw: float = 0.0, length: float = 26.0, width: float = 4.4,
                    behind: float = 5.0, seed: int = 11):
    """A narrow back alley (the user's note, 10/26: grey concrete, stairs, rubbish bags, narrow, not a
    shopping street): concrete walls close on both sides with few barred windows, steel doors, an
    outside concrete stair with a rail, pipes and meters, air-conditioner units, rubbish bags and a
    skip, cables across, a gutter down the middle of wet concrete; lit by a few bare bulbs and one
    cold tube, with a single small sign far down the alley. +u along `yaw`, +v to its left."""
    rnd = random.Random(seed)
    root = bpy.data.objects.new("set", None)
    bpy.context.collection.objects.link(root)
    root.location = (origin[0], origin[1], 0.0)
    root.rotation_euler = (0.0, 0.0, yaw)
    font = bpy.data.fonts.load(FONT)

    def keep(obj):
        obj.parent = root
        return obj

    half = width / 2
    u0, u1 = -behind, length
    ground = keep(_box("alley_floor", (u1 - u0 + 4, width + 1, 0.02), ((u0 + u1) / 2, 0.0, -0.01), _asphalt()))
    ground.data.materials.clear()
    ground.data.materials.append(bpy.data.materials["wet_asphalt"])
    grate = _mat("grate", (0.015, 0.015, 0.015), 0.7, 0.6)
    keep(_box("gutter", (u1 - u0, 0.22, 0.004), ((u0 + u1) / 2, 0.55, 0.002), _mat("gutter", (0.012, 0.012, 0.012), 0.3)))
    for k in range(int((u1 - u0) / 2.5)):
        keep(_box(f"grate{k}", (0.5, 0.24, 0.006), (u0 + 1.0 + 2.5 * k, 0.55, 0.003), grate))
    steel_door = _mat("steel_door", (0.13, 0.14, 0.14), 0.5, 0.7)
    pipe_mat = _mat("pipe_grey", (0.22, 0.22, 0.23), 0.55, 0.4)
    rust = _mat("rust", (0.16, 0.07, 0.035), 0.8)
    ac_mat = _mat("ac_dirty", (0.5, 0.49, 0.45), 0.7)
    grille = _mat("grille", (0.02, 0.02, 0.02), 0.6, 0.5)
    glass = _mat("dark_glass", (0.01, 0.012, 0.015), 0.08)
    bag_mat = _mat("bin_bag", (0.006, 0.006, 0.007), 0.22)
    for s in (-1, 1):  # the two walls, in sections of different heights and tones
        u = u0
        while u < u1:
            w = rnd.uniform(3.0, 6.0)
            h = rnd.uniform(6.0, 13.0)
            tone = rnd.uniform(0.22, 0.34)
            keep(_box(f"wall{s}_{u:.0f}", (w, 0.6, h), (u + w / 2, s * (half + 0.3), h / 2),
                      _concrete(f"concrete{s}_{u:.0f}", seed=int(u * 7 + s), tone=tone)))
            front = s * half
            for k in range(rnd.randint(0, 2)):  # small barred windows, a few dimly lit
                wu = u + rnd.uniform(0.6, w - 0.6)
                wz = rnd.uniform(2.6, h - 1.0)
                lit = rnd.random() < 0.3
                keep(_box(f"win{s}_{u:.0f}_{k}", (0.7, 0.05, 0.8), (wu, front - s * 0.01, wz),
                          _glow("win_lit", (1.0, 0.6, 0.3), 0.6) if lit else glass))
                for j in range(4):
                    keep(_cyl(f"bar{s}_{u:.0f}_{k}_{j}", 0.012, 0.8, (wu - 0.27 + 0.18 * j, front - s * 0.05, wz), steel_door))
            if rnd.random() < 0.5:  # a steel door with a step
                du = u + rnd.uniform(0.8, w - 0.8)
                keep(_box(f"door{s}_{u:.0f}", (0.9, 0.06, 2.0), (du, front - s * 0.03, 1.0), steel_door))
                keep(_box(f"step{s}_{u:.0f}", (1.1, 0.35, 0.12), (du, front - s * 0.17, 0.06),
                          _concrete(f"stepc{s}_{u:.0f}", seed=int(u), tone=0.3)))
            for k in range(rnd.randint(1, 3)):  # pipes down the wall, a rusty one now and then
                pu = u + rnd.uniform(0.2, w - 0.2)
                keep(_cyl(f"pipe{s}_{u:.0f}_{k}", rnd.uniform(0.03, 0.06), h, (pu, front - s * 0.08, h / 2),
                          rust if rnd.random() < 0.3 else pipe_mat))
            if rnd.random() < 0.6:  # an air-conditioner unit
                au, az = u + rnd.uniform(0.6, w - 0.6), rnd.choice([0.35, rnd.uniform(2.2, 4.0)])
                keep(_box(f"ac{s}_{u:.0f}", (0.8, 0.32, 0.6), (au, front - s * 0.17, az), ac_mat))
                keep(_cyl(f"fan{s}_{u:.0f}", 0.22, 0.02, (au - 0.12, front - s * 0.34, az), grille,
                          rotation=(math.pi / 2, 0.0, 0.0)))
            if rnd.random() < 0.5:  # meters and a conduit
                mu = u + rnd.uniform(0.5, w - 0.5)
                keep(_box(f"meter{s}_{u:.0f}", (0.3, 0.12, 0.4), (mu, front - s * 0.06, 1.6), _mat("meter", (0.6, 0.6, 0.58), 0.5)))
                keep(_cyl(f"conduit{s}_{u:.0f}", 0.018, w * 0.9, (u + w / 2, front - s * 0.05, 2.5), pipe_mat,
                          rotation=(0.0, math.pi / 2, 0.0)))
            # rubbish bags against the wall, in heaps
            if rnd.random() < 0.7:
                bu = u + rnd.uniform(0.6, w - 0.6)
                for k in range(rnd.randint(2, 6)):
                    sc = (rnd.uniform(0.22, 0.32), rnd.uniform(0.2, 0.28), rnd.uniform(0.2, 0.3))
                    for o in _bag(f"bag{s}_{u:.0f}_{k}", (bu + rnd.uniform(-0.6, 0.6), front - s * rnd.uniform(0.25, 0.6),
                                                           0.0 if k < 3 else 0.35), sc, rnd, bag_mat):
                        keep(o)
            u += w
    # paper notices pasted on the concrete, generic words only
    paper = _mat("paper", (0.55, 0.53, 0.48), 0.9)
    ink = _mat("ink", (0.02, 0.02, 0.02), 0.8)
    for k, (word, cu, s) in enumerate((("立入禁止", 3.2, -1), ("貸倉庫", 8.2, 1), ("駐輪禁止", 13.6, -1),
                                       ("関係者以外\n立入禁止", 17.5, 1))):
        front = s * half
        keep(_box(f"notice{k}", (0.42, 0.006, 0.6), (cu, front - s * 0.004, 1.5), paper,
                  rotation=(0.0, rnd.uniform(-0.05, 0.05), 0.0)))
        t = keep(_text(f"notice_text{k}", "\n".join(word) if "\n" not in word else word, 0.09,
                       (cu, front - s * 0.009, 1.5), (math.pi / 2, 0.0, math.pi if s < 0 else 0.0), ink, font))
        t.data.extrude = 0.0005
    # a skip, and crates
    skip = _mat("skip", (0.04, 0.12, 0.07), 0.55, 0.3)
    keep(_box("skip", (1.6, 0.9, 1.0), (9.5, -(half - 0.5), 0.5), skip))
    keep(_box("skip_lid", (1.65, 0.95, 0.05), (9.5, -(half - 0.5), 1.03), skip, rotation=(0.12, 0.0, 0.0)))
    for k in range(3):
        keep(_box(f"crate{k}", (0.45, 0.35, 0.3), (14.0 + 0.5 * k, half - 0.35, 0.15 + 0.3 * (k == 1)),
                  _mat("crate", (0.12, 0.2, 0.35), 0.6)))
    # the outside stair up the left wall, with a rail
    stair = _concrete("stair_concrete", seed=5, tone=0.3)
    for k in range(13):
        keep(_box(f"stair{k}", (0.3, 1.0, 0.18 * (k + 1)), (5.0 + 0.28 * k, half - 0.5, 0.09 * (k + 1)), stair))
    keep(_box("landing", (1.2, 1.0, 2.34), (5.0 + 0.28 * 13 + 0.5, half - 0.5, 1.17), stair))
    rail = _mat("rail", (0.08, 0.08, 0.085), 0.4, 0.8)
    for k in range(0, 14, 3):
        keep(_cyl(f"post{k}", 0.015, 0.9, (5.0 + 0.28 * k, half - 1.0, 0.18 * (k + 1) + 0.45), rail))
    keep(_cyl("rail_top", 0.02, 4.2, (5.0 + 1.8, half - 1.0, 0.18 * 7 + 0.95), rail,
              rotation=(0.0, math.pi / 2 - math.atan2(0.18, 0.28), 0.0)))
    # cables across, sagging
    cable = _mat("cable", (0.01, 0.01, 0.01), 0.5)
    for k in range(7):
        cu, z = rnd.uniform(0.0, u1 - 2.0), rnd.uniform(3.5, 6.5)
        pts = [Vector((cu, -half, z)), Vector((cu + rnd.uniform(-0.5, 0.5), 0.0, z - rnd.uniform(0.3, 0.7))),
               Vector((cu + rnd.uniform(-1, 1), half, z + rnd.uniform(-0.3, 0.3)))]
        curve = bpy.data.curves.new(f"cable{k}", "CURVE")
        curve.dimensions, curve.bevel_depth = "3D", 0.012
        sp = curve.splines.new("BEZIER")
        sp.bezier_points.add(len(pts) - 1)
        for bp, p in zip(sp.bezier_points, pts):
            bp.co = p
            bp.handle_left_type = bp.handle_right_type = "AUTO"
        obj = bpy.data.objects.new(f"cable{k}", curve)
        bpy.context.collection.objects.link(obj)
        curve.materials.append(cable)
        keep(obj)
    # light: bare bulbs on brackets, one cold tube over a door, a single sign far down the alley
    bulb = _glow("bulb", (1.0, 0.7, 0.4), 30.0)
    for k, (cu, s) in enumerate(((1.5, 1), (7.0, -1), (12.5, 1), (19.0, -1))):
        keep(_cyl(f"bracket{k}", 0.01, 0.35, (cu, s * (half - 0.17), 3.0), rail, rotation=(math.pi / 2, 0.0, 0.0)))
        bpy.ops.mesh.primitive_uv_sphere_add(radius=0.05, location=(cu, s * (half - 0.35), 2.92))
        b_ = bpy.context.active_object
        b_.name = f"bulb{k}"
        b_.data.materials.append(bulb)
        keep(b_)
        keep(_light(f"bulb_light{k}", "POINT", (cu, s * (half - 0.35), 2.85), 60.0, (1.0, 0.68, 0.38), size=0.05))
    keep(_box("tube", (1.2, 0.06, 0.05), (16.0, -(half - 0.06), 2.5), _glow("tube", (0.75, 0.9, 1.0), 12.0)))
    keep(_light("tube_light", "AREA", (16.0, -(half - 0.3), 2.45), 70.0, (0.75, 0.9, 1.0), size=1.2,
                rotation=(0.0, 0.0, 0.0)))
    word, rgb = "路地", (1.0, 0.1, 0.4)
    keep(_box("far_signbox", (0.15, 0.6, 1.3), (u1 - 2.0, half - 0.35, 3.2), _mat("sign_back", (0.02, 0.02, 0.025), 0.4)))
    t = keep(_text("far_sign", "\n".join(word), 0.45, (u1 - 2.0 - 0.08, half - 0.35, 3.2), (math.pi / 2, 0.0, -math.pi / 2),
                   _glow("neon_far", rgb, 8.0), font))
    t.data.space_line = 0.9
    keep(_light("far_sign_light", "AREA", (u1 - 2.6, half - 0.6, 3.0), 40.0, rgb, size=0.6))
    keep(_light("alley_end", "AREA", (u1 + 2.0, 0.0, 4.0), 900.0, (0.35, 0.45, 0.9), size=4.0,
                rotation=(0.0, math.radians(-90), 0.0)))
    try:
        bpy.ops.object.lightprobe_add(type="PLANE", location=((u0 + u1) / 2, 0.0, 0.001))
        probe = bpy.context.active_object
        probe.name = "alley_mirror"
        probe.scale = ((u1 - u0) / 2, half, 1.0)
        keep(probe)
    except Exception as err:  # noqa: BLE001 -- the probe is an embellishment
        print(f"[alley] no planar probe: {err}")
    return root


def setup_render(scene, samples: int = 32, haze: float = 0.012, res=(1920, 1080), engine: str = "eevee"):
    """EEVEE for the street (ray-traced reflections, volumetric haze in a near-black world, AgX Punchy),
    or Cycles on every GPU (OptiX, else CUDA) with adaptive sampling, denoising and motion blur."""
    scene.render.resolution_x, scene.render.resolution_y = res
    if engine == "cycles":
        scene.render.engine = "CYCLES"
        prefs = bpy.context.preferences.addons["cycles"].preferences
        for kind in ("OPTIX", "CUDA"):
            try:
                prefs.compute_device_type = kind
                break
            except TypeError:
                continue
        prefs.get_devices()
        for d in prefs.devices:  # the chosen backend only: OptiX and CUDA list the same GPU twice
            d.use = d.type == prefs.compute_device_type
        c = scene.cycles
        c.device = "GPU"
        c.samples = samples
        c.use_adaptive_sampling = True
        c.use_denoising = True
        c.max_bounces, c.glossy_bounces, c.transparent_max_bounces = 8, 4, 8
        scene.render.use_motion_blur = True
        scene.render.motion_blur_shutter = 0.5
        print(f"[alley] Cycles on {[d.name for d in prefs.devices if d.use]} ({prefs.compute_device_type})")
    else:
        scene.render.engine = "BLENDER_EEVEE"
    e = scene.eevee
    e.taa_render_samples = samples
    if hasattr(e, "use_raytracing"):
        e.use_raytracing = True
    if hasattr(e, "volumetric_tile_size"):
        e.volumetric_tile_size = "8"
        e.volumetric_samples = 48
        e.volumetric_end = 60.0
        e.use_volumetric_shadows = False
    world = bpy.data.worlds.new("night")
    world.use_nodes = True
    nt = world.node_tree
    bg = next((n for n in nt.nodes if n.type == "BACKGROUND"), None) or nt.nodes.new("ShaderNodeBackground")
    out = next((n for n in nt.nodes if n.type == "OUTPUT_WORLD"), None) or nt.nodes.new("ShaderNodeOutputWorld")
    nt.links.new(bg.outputs["Background"], out.inputs["Surface"])
    bg.inputs[0].default_value = (0.004, 0.005, 0.012, 1.0)
    bg.inputs[1].default_value = 1.0
    if haze > 0:
        vol = nt.nodes.new("ShaderNodeVolumePrincipled")
        vol.inputs["Density"].default_value = haze
        vol.inputs["Anisotropy"].default_value = 0.35
        vol.inputs["Color"].default_value = (0.8, 0.85, 1.0, 1.0)
        nt.links.new(vol.outputs[0], out.inputs["Volume"])
    scene.world = world
    views = [v.identifier for v in bpy.types.ColorManagedViewSettings.bl_rna.properties["view_transform"].enum_items]
    if "AgX" in views:
        scene.view_settings.view_transform = "AgX"
        looks = [v.identifier for v in bpy.types.ColorManagedViewSettings.bl_rna.properties["look"].enum_items]
        punchy = next((k for k in looks if "Punchy" in k), None)
        if punchy:
            scene.view_settings.look = punchy
