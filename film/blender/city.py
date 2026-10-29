"""A near-future night city from PLATEAU (10/28): the establishing shot over Shinjuku.

Data: 3D都市モデル（Project PLATEAU）東京都23区（2020年度）, OBJ, MLIT (the film credits it as
modified). The tiles are JIS 3rd-level meshes in JGD2011 plane rectangular zone IX, metres, x east,
y north, z up. LOD2 tiles (Nishi-Shinjuku's towers, Kabukicho) are used where they exist and LOD1
blocks round them for the skyline; DEM for the ground and the road polygons for the streets.

LOD1 blocks get one procedural near-future facade -- dark glass and metal with a window grid lit
at random in warm and cool whites, set by floor height and bay width that vary block by block --
and the roofs a dark concrete. With --keep-textures (the user's call, 10/28) the LOD2 tiles keep
PLATEAU's own facade photos, darkened for night, with the same window grid lit over them; the
photos are first scrubbed of lettering (film/post/sign_scrub.py), since the film shows no real
shop names or brands. On top: red aircraft-warning lights blinking on
the towers, street lights along the roads, tall holographic banners with generic words on a few
tower faces, drone lights crossing the sky, searchlight beams, a hazy night sky with light
pollution on the horizon. The coordinates are moved to the city centre before anything else, so
the render keeps its precision.

    blender -b --python city.py -- --plateau <extracted 13100_..._op> --out city.blend --render still.png
"""

from __future__ import annotations

import argparse
import glob
import math
import random
import sys
from pathlib import Path

import bpy
from mathutils import Matrix, Vector

LOD2 = ["53394515", "53394524", "53394525", "53394526", "53394534", "53394536"]


def _node(nt, kind, **inputs):
    n = nt.nodes.new(kind)
    for k, v in inputs.items():
        n.inputs[k].default_value = v
    return n


def _math(nt, op, a=None, b=None, va=0.0, vb=0.0):
    n = nt.nodes.new("ShaderNodeMath")
    n.operation = op
    if a is not None:
        nt.links.new(a, n.inputs[0])
    else:
        n.inputs[0].default_value = va
    if b is not None:
        nt.links.new(b, n.inputs[1])
    else:
        n.inputs[1].default_value = vb
    return n.outputs[0]


def _noise(nt, x, y, z):
    """White noise (value, colour channels) of a 3D cell."""
    c = nt.nodes.new("ShaderNodeCombineXYZ")
    for k, v in zip("XYZ", (x, y, z)):
        if isinstance(v, (int, float)):
            c.inputs[k].default_value = v
        else:
            nt.links.new(v, c.inputs[k])
    n = nt.nodes.new("ShaderNodeTexWhiteNoise")
    n.noise_dimensions = "3D"
    nt.links.new(c.outputs[0], n.inputs["Vector"])
    sep = nt.nodes.new("ShaderNodeSeparateColor")
    nt.links.new(n.outputs["Color"], sep.inputs["Color"])
    return n.outputs["Value"], sep.outputs


def _ramp(nt, fac, stops):
    r = nt.nodes.new("ShaderNodeValToRGB")
    els = r.color_ramp.elements
    els[0].position, els[0].color = stops[0][0], (*stops[0][1], 1)
    els[1].position, els[1].color = stops[-1][0], (*stops[-1][1], 1)
    for pos, col in stops[1:-1]:
        e = els.new(pos)
        e.color = (*col, 1)
    r.color_ramp.interpolation = "CONSTANT"
    nt.links.new(fac, r.inputs["Fac"])
    return r.outputs["Color"]


def _lights(nt):
    """The city's lights in node tree `nt`, from world position only. Windows: offices lit a floor
    (or part of one) at a time, homes here and there, each window its own brightness, colour
    temperatures from warm 3000 K to cool 6500 K with the odd green fluorescent. Signs: low on the
    walls, vertical strips of saturated neon with no lettering, thicker in Kabukicho. Returns the
    emission colour and strength, the wall mask, the window mask and a per-block random."""
    geo = nt.nodes.new("ShaderNodeNewGeometry")
    pos = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(geo.outputs["Position"], pos.inputs[0])
    nrm = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(geo.outputs["Normal"], nrm.inputs[0])
    X, Y, Z = pos.outputs["X"], pos.outputs["Y"], pos.outputs["Z"]
    # the facade's horizontal axis: y on faces looking east-west, x on faces looking north-south
    ew = _math(nt, "GREATER_THAN", _math(nt, "ABSOLUTE", nrm.outputs["X"]), _math(nt, "ABSOLUTE", nrm.outputs["Y"]))
    hmix = nt.nodes.new("ShaderNodeMix")
    hmix.data_type = "FLOAT"
    nt.links.new(ew, hmix.inputs["Factor"])
    nt.links.new(X, hmix.inputs["A"])
    nt.links.new(Y, hmix.inputs["B"])
    h = hmix.outputs["Result"]
    # a block id (25 m cells) sets the bay width, floor height, how much is lit and the facade
    bx = _math(nt, "FLOOR", _math(nt, "DIVIDE", X, None, vb=25.0))
    by = _math(nt, "FLOOR", _math(nt, "DIVIDE", Y, None, vb=25.0))
    rnd, rc = _noise(nt, bx, by, 0.0)
    bay = _math(nt, "ADD", _math(nt, "MULTIPLY", rnd, None, vb=1.2), None, vb=1.5)  # 1.5-2.7 m
    flo = _math(nt, "ADD", _math(nt, "MULTIPLY", rc["Green"], None, vb=0.9), None, vb=3.2)  # 3.2-4.1 m
    u = _math(nt, "DIVIDE", h, bay)
    v = _math(nt, "DIVIDE", Z, flo)
    fu, fv = _math(nt, "FRACT", u), _math(nt, "FRACT", v)
    # mullions 6-20 %; on about half the blocks ribbon windows (offices lit floor by floor, as in the photos)
    ribbon = _math(nt, "GREATER_THAN", rc["Blue"], None, vb=0.5)
    mull = _math(nt, "ADD", _math(nt, "MULTIPLY", _math(nt, "MULTIPLY", rc["Blue"], None, vb=0.14),
                                  _math(nt, "SUBTRACT", None, ribbon, va=1.0)),
                 _math(nt, "ADD", _math(nt, "MULTIPLY", ribbon, None, vb=-0.045), None, vb=0.06))
    inu = _math(nt, "MULTIPLY", _math(nt, "GREATER_THAN", fu, mull),
                _math(nt, "LESS_THAN", fu, _math(nt, "SUBTRACT", None, mull, va=1.0)))
    inv = _math(nt, "MULTIPLY", _math(nt, "GREATER_THAN", fv, None, vb=0.24), _math(nt, "LESS_THAN", fv, None, vb=0.88))
    window = _math(nt, "MULTIPLY", inu, inv)
    wall = _math(nt, "LESS_THAN", _math(nt, "ABSOLUTE", nrm.outputs["Z"]), None, vb=0.5)
    fl = _math(nt, "FLOOR", v)
    # zones: a floor split into runs of 4-9 bays, lit together (an office floor, a shop's rooms)
    zone_w = _math(nt, "ADD", _math(nt, "ADD", _math(nt, "MULTIPLY", rc["Red"], None, vb=5.0), None, vb=4.0),
                   _math(nt, "MULTIPLY", ribbon, None, vb=30.0))  # a ribbon block lights whole floors
    zone, zc = _noise(nt, fl, _math(nt, "FLOOR", _math(nt, "DIVIDE", u, zone_w)), _math(nt, "ADD", rnd, None, vb=7.0))
    one, oc = _noise(nt, _math(nt, "FLOOR", u), fl, _math(nt, "ADD", rnd, None, vb=3.0))
    ratio = _math(nt, "ADD", _math(nt, "MULTIPLY", rnd, None, vb=0.32), None, vb=0.08)  # 8-40 % of zones lit
    lit = _math(nt, "MULTIPLY", _math(nt, "LESS_THAN", zone, ratio), _math(nt, "LESS_THAN", one, None, vb=0.9))
    lit = _math(nt, "MAXIMUM", lit, _math(nt, "LESS_THAN", one, None, vb=0.05))  # and a few lone windows
    bright = _math(nt, "ADD", _math(nt, "MULTIPLY", oc["Red"], None, vb=0.9), None, vb=0.3)
    glow = _math(nt, "MULTIPLY", _math(nt, "MULTIPLY", window, lit), _math(nt, "MULTIPLY", wall, bright))
    wcol = _ramp(nt, zc["Green"], [(0.0, (1.0, 0.62, 0.32)), (0.22, (1.0, 0.8, 0.58)), (0.45, (0.92, 0.95, 1.0)),
                                    (0.8, (0.72, 0.85, 1.0)), (0.95, (0.8, 1.0, 0.82))])
    # neon: vertical strips in 3 m columns, from 3 m up to 6-20 m, low on the walls
    col = _math(nt, "FLOOR", _math(nt, "DIVIDE", h, None, vb=3.0))
    sn, sc = _noise(nt, col, rnd, 11.0)
    kab = _math(nt, "MULTIPLY",
                _math(nt, "MULTIPLY", _math(nt, "GREATER_THAN", X, None, vb=250.0), _math(nt, "LESS_THAN", X, None, vb=1050.0)),
                _math(nt, "MULTIPLY", _math(nt, "GREATER_THAN", Y, None, vb=120.0), _math(nt, "LESS_THAN", Y, None, vb=850.0)))
    p_sign = _math(nt, "ADD", _math(nt, "MULTIPLY", kab, None, vb=0.3), None, vb=0.12)
    top = _math(nt, "ADD", _math(nt, "MULTIPLY", sc["Red"], None, vb=14.0), None, vb=6.0)
    fx = _math(nt, "FRACT", _math(nt, "DIVIDE", h, None, vb=3.0))
    width = _math(nt, "ADD", _math(nt, "MULTIPLY", sc["Green"], None, vb=0.25), None, vb=0.12)
    in_x = _math(nt, "MULTIPLY", _math(nt, "GREATER_THAN", fx, width),
                 _math(nt, "LESS_THAN", fx, _math(nt, "SUBTRACT", None, width, va=1.0)))
    in_z = _math(nt, "MULTIPLY", _math(nt, "GREATER_THAN", Z, None, vb=3.0), _math(nt, "LESS_THAN", Z, top))
    sign = _math(nt, "MULTIPLY", _math(nt, "MULTIPLY", _math(nt, "LESS_THAN", sn, p_sign), in_x),
                 _math(nt, "MULTIPLY", in_z, wall))
    scol = _ramp(nt, sc["Blue"], [(0.0, (1.0, 0.08, 0.6)), (0.22, (0.05, 0.85, 1.0)), (0.4, (0.6, 0.2, 1.0)),
                                   (0.55, (1.0, 0.1, 0.4)), (0.7, (0.35, 0.3, 1.0)), (0.85, (1.0, 0.5, 0.1)),
                                   (0.94, (0.9, 0.95, 1.0))])
    cmix = nt.nodes.new("ShaderNodeMix")
    cmix.data_type = "RGBA"
    nt.links.new(sign, cmix.inputs["Factor"])
    nt.links.new(wcol, cmix.inputs["A"])
    nt.links.new(scol, cmix.inputs["B"])
    strength = _math(nt, "ADD", _math(nt, "MULTIPLY", glow, None, vb=2.2), _math(nt, "MULTIPLY", sign, None, vb=7.0))
    # LED bands round the ribbon towers every 9 floors, above 40 m: magenta, violet or cyan by block
    band = _math(nt, "MULTIPLY", _math(nt, "LESS_THAN", _math(nt, "FRACT", _math(nt, "DIVIDE", fl, None, vb=9.0)), None, vb=0.06),
                 _math(nt, "LESS_THAN", fv, None, vb=0.18))
    band = _math(nt, "MULTIPLY", _math(nt, "MULTIPLY", band, ribbon), _math(nt, "MULTIPLY", wall, _math(nt, "GREATER_THAN", Z, None, vb=40.0)))
    bcol = _ramp(nt, rc["Green"], [(0.0, (1.0, 0.1, 0.7)), (0.4, (0.5, 0.2, 1.0)), (0.75, (0.1, 0.9, 1.0))])
    # big LED screens on about one block in eight, 18-46 m up, animated abstract colour (no lettering)
    scr_u = _math(nt, "FRACT", _math(nt, "DIVIDE", h, None, vb=60.0))
    screen = _math(nt, "MULTIPLY", _math(nt, "MULTIPLY", _math(nt, "GREATER_THAN", scr_u, None, vb=0.2), _math(nt, "LESS_THAN", scr_u, None, vb=0.8)),
                   _math(nt, "MULTIPLY", _math(nt, "GREATER_THAN", Z, None, vb=18.0), _math(nt, "LESS_THAN", Z, None, vb=46.0)))
    east = _math(nt, "MULTIPLY",
                 _math(nt, "MULTIPLY", _math(nt, "GREATER_THAN", X, None, vb=430.0), _math(nt, "LESS_THAN", X, None, vb=1150.0)),
                 _math(nt, "MULTIPLY", _math(nt, "GREATER_THAN", Y, None, vb=120.0), _math(nt, "LESS_THAN", Y, None, vb=900.0)))
    p_screen = _math(nt, "ADD", _math(nt, "MULTIPLY", east, None, vb=0.42), None, vb=0.025)  # dense at the east exit and Kabukicho
    screen = _math(nt, "MULTIPLY", _math(nt, "MULTIPLY", screen, wall), _math(nt, "LESS_THAN", rc["Red"], p_screen))
    clock = nt.nodes.new("ShaderNodeValue")
    clock.name = "screen_clock"
    clock.outputs[0].default_value = 0.0
    clock.outputs[0].keyframe_insert("default_value", frame=1)
    clock.outputs[0].default_value = 6.0
    clock.outputs[0].keyframe_insert("default_value", frame=150)
    anim = nt.nodes.new("ShaderNodeTexNoise")
    anim.noise_dimensions = "4D"
    anim.inputs["Scale"].default_value = 0.05
    nt.links.new(geo.outputs["Position"], anim.inputs["Vector"])
    nt.links.new(_math(nt, "ADD", clock.outputs[0], _math(nt, "MULTIPLY", rnd, None, vb=40.0)), anim.inputs["W"])
    scrcol = _ramp(nt, anim.outputs["Fac"], [(0.0, (0.2, 0.1, 1.0)), (0.35, (1.0, 0.1, 0.65)), (0.55, (0.1, 0.85, 1.0)),
                                             (0.7, (1.0, 0.9, 0.95)), (0.85, (0.6, 0.2, 1.0))])
    c2 = nt.nodes.new("ShaderNodeMix")
    c2.data_type = "RGBA"
    nt.links.new(band, c2.inputs["Factor"])
    nt.links.new(cmix.outputs["Result"], c2.inputs["A"])
    nt.links.new(bcol, c2.inputs["B"])
    c3 = nt.nodes.new("ShaderNodeMix")
    c3.data_type = "RGBA"
    nt.links.new(screen, c3.inputs["Factor"])
    nt.links.new(c2.outputs["Result"], c3.inputs["A"])
    nt.links.new(scrcol, c3.inputs["B"])
    strength = _math(nt, "ADD", strength, _math(nt, "ADD", _math(nt, "MULTIPLY", band, None, vb=9.0),
                                                 _math(nt, "MULTIPLY", screen, None, vb=6.0)))
    region = {}
    for key, v in (("proj_x0", 1e9), ("proj_x1", -1e9), ("proj_y0", 1e9), ("proj_y1", -1e9)):
        n_ = nt.nodes.new("ShaderNodeValue")
        n_.name = n_.label = key
        n_.outputs[0].default_value = v
        region[key] = n_.outputs[0]
    inside = _math(nt, "MULTIPLY",
                   _math(nt, "MULTIPLY", _math(nt, "GREATER_THAN", X, region["proj_x0"]), _math(nt, "LESS_THAN", X, region["proj_x1"])),
                   _math(nt, "MULTIPLY", _math(nt, "GREATER_THAN", Y, region["proj_y0"]), _math(nt, "LESS_THAN", Y, region["proj_y1"])))
    east_face = _math(nt, "GREATER_THAN", nrm.outputs["X"], None, vb=0.6)
    high = _math(nt, "GREATER_THAN", Z, None, vb=45.0)
    proj = _math(nt, "MULTIPLY", _math(nt, "MULTIPLY", inside, east_face), high)
    film = nt.nodes.new("ShaderNodeTexNoise")
    film.noise_dimensions = "4D"
    film.inputs["Scale"].default_value = 0.02
    film.inputs["Detail"].default_value = 3.0
    nt.links.new(geo.outputs["Position"], film.inputs["Vector"])
    nt.links.new(_math(nt, "MULTIPLY", clock.outputs[0], None, vb=0.6), film.inputs["W"])
    waves = nt.nodes.new("ShaderNodeTexWave")
    waves.wave_type, waves.bands_direction = "BANDS", "Z"
    waves.inputs["Scale"].default_value = 0.04
    waves.inputs["Distortion"].default_value = 4.0
    nt.links.new(geo.outputs["Position"], waves.inputs["Vector"])
    nt.links.new(_math(nt, "MULTIPLY", clock.outputs[0], None, vb=3.0), waves.inputs["Phase Offset"])
    pcol = _ramp(nt, _math(nt, "MULTIPLY", film.outputs["Fac"], waves.outputs["Fac"]),
                 [(0.0, (0.15, 0.25, 1.0)), (0.2, (0.9, 0.3, 1.0)), (0.4, (1.0, 0.75, 0.3)), (0.6, (0.3, 0.95, 1.0)), (0.8, (1.0, 1.0, 1.0))])
    c4 = nt.nodes.new("ShaderNodeMix")
    c4.data_type = "RGBA"
    nt.links.new(proj, c4.inputs["Factor"])
    nt.links.new(c3.outputs["Result"], c4.inputs["A"])
    nt.links.new(pcol, c4.inputs["B"])
    strength = _math(nt, "ADD", strength, _math(nt, "MULTIPLY", proj, None, vb=3.5))
    c3 = nt.nodes.new("ShaderNodeMix")  # (the colour returned below is the mapped one)
    c3.data_type = "RGBA"
    c3.inputs["Factor"].default_value = 0.0
    nt.links.new(c4.outputs["Result"], c3.inputs["A"])
    _lights.extra = {"fu": fu, "fv": fv, "ribbon": ribbon, "geo": geo, "rc": rc}  # for the facade's materials
    return c3.outputs["Result"], strength, wall, window, rnd


def facade_material() -> bpy.types.Material:
    """(10/28, the user: the buildings read as plain grey) Real facade materials by block: glass curtain
    walls with aluminium spandrels on the ribbon-window blocks (mirror-smooth, so they carry the city
    lights and the sky), and on the rest white tile, beige stone, concrete, brick or dark granite with
    clear glass. Floor-slab lines, grime and rain streaks, a fine surface relief and panel seams; lit by
    `_lights`. Roofs dark concrete with a little relief."""
    mat = bpy.data.materials.new("city_facade")
    mat.use_nodes = True
    nt = mat.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    color, strength, wall, window, rnd = _lights(nt)
    ex = _lights.extra
    fu, fv, ribbon, geo, rc = ex["fu"], ex["fv"], ex["ribbon"], ex["geo"], ex["rc"]

    def mix(fac, a, b, kind="RGBA"):
        m = nt.nodes.new("ShaderNodeMix")
        m.data_type = kind
        nt.links.new(fac, m.inputs["Factor"])
        for sock, v in (("A", a), ("B", b)):
            if isinstance(v, tuple):
                m.inputs[sock].default_value = (*v, 1.0) if kind == "RGBA" else v
            elif isinstance(v, float):
                m.inputs[sock].default_value = v
            else:
                nt.links.new(v, m.inputs[sock])
        return m.outputs["Result"]

    masonry = _ramp(nt, rnd, [(0.0, (0.52, 0.51, 0.48)), (0.22, (0.42, 0.36, 0.29)), (0.45, (0.3, 0.3, 0.3)),
                               (0.62, (0.26, 0.15, 0.1)), (0.78, (0.09, 0.09, 0.1)), (0.9, (0.58, 0.57, 0.55))])
    panel = _ramp(nt, rc["Green"], [(0.0, (0.34, 0.36, 0.38)), (0.5, (0.2, 0.21, 0.23)), (0.8, (0.45, 0.43, 0.4))])
    wallcol = mix(ribbon, masonry, panel)
    # floor slabs: a darker line at the top of each floor
    slab = _math(nt, "MULTIPLY", _math(nt, "GREATER_THAN", fv, None, vb=0.9), _math(nt, "LESS_THAN", fv, None, vb=0.96))
    wallcol = mix(slab, wallcol, (0.08, 0.08, 0.085))
    # grime: broad blotches and rain streaks running down from the sills
    blot = _node(nt, "ShaderNodeTexNoise", Scale=0.08, Detail=8.0, Roughness=0.6)
    nt.links.new(geo.outputs["Position"], blot.inputs["Vector"])
    streak = nt.nodes.new("ShaderNodeTexWave")
    streak.wave_type, streak.bands_direction = "BANDS", "X"
    streak.inputs["Scale"].default_value = 0.6
    streak.inputs["Distortion"].default_value = 9.0
    streak.inputs["Detail"].default_value = 6.0
    nt.links.new(geo.outputs["Position"], streak.inputs["Vector"])
    grime = nt.nodes.new("ShaderNodeMapRange")
    nt.links.new(_math(nt, "MULTIPLY", blot.outputs["Fac"], streak.outputs["Fac"]), grime.inputs["Value"])
    grime.inputs["From Min"].default_value, grime.inputs["From Max"].default_value = 0.1, 0.6
    grime.inputs["To Min"].default_value, grime.inputs["To Max"].default_value = 0.62, 1.05
    dirty = nt.nodes.new("ShaderNodeMix")
    dirty.data_type, dirty.blend_type = "RGBA", "MULTIPLY"
    dirty.inputs["Factor"].default_value = 1.0
    nt.links.new(wallcol, dirty.inputs["A"])
    gcol = nt.nodes.new("ShaderNodeCombineColor")
    for ch in ("Red", "Green", "Blue"):
        nt.links.new(grime.outputs["Result"], gcol.inputs[ch])
    nt.links.new(gcol.outputs["Color"], dirty.inputs["B"])
    glasscol = mix(ribbon, (0.025, 0.03, 0.034), (0.018, 0.034, 0.042))
    # (10/28 afternoon, the user: the aerial's details read blotchy) at night a wall a kilometre or more
    # away is dark and only its windows show; its sky-lit albedo massed the distant blocks into a beige wash
    camd = nt.nodes.new("ShaderNodeCameraData")
    far_k = nt.nodes.new("ShaderNodeMapRange")
    nt.links.new(camd.outputs["View Distance"], far_k.inputs["Value"])
    far_k.inputs["From Min"].default_value, far_k.inputs["From Max"].default_value = 600.0, 2400.0
    far_k.inputs["To Min"].default_value, far_k.inputs["To Max"].default_value = 0.0, 0.85
    walldark = mix(far_k.outputs["Result"], dirty.outputs["Result"], (0.02, 0.02, 0.024))
    base = mix(window, walldark, glasscol)
    wall_rough = mix(ribbon, 0.72, 0.28, "FLOAT")
    rough = mix(window, wall_rough, 0.035, "FLOAT")
    metal = mix(window, mix(ribbon, 0.0, 0.9, "FLOAT"), mix(ribbon, 0.0, 0.75, "FLOAT"), "FLOAT")
    # relief: a fine stone/concrete grain and the seams round every window and panel
    grain = _node(nt, "ShaderNodeTexNoise", Scale=3.0, Detail=10.0)
    nt.links.new(geo.outputs["Position"], grain.inputs["Vector"])
    seam = _math(nt, "MAXIMUM", _math(nt, "LESS_THAN", _math(nt, "ABSOLUTE", _math(nt, "SUBTRACT", fu, None, vb=0.5)), None, vb=0.02),
                 _math(nt, "LESS_THAN", fv, None, vb=0.02))
    height = _math(nt, "ADD", _math(nt, "MULTIPLY", grain.outputs["Fac"], None, vb=0.6), _math(nt, "MULTIPLY", seam, None, vb=-0.8))
    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.25
    bump.inputs["Distance"].default_value = 0.02
    nt.links.new(_math(nt, "MULTIPLY", height, _math(nt, "SUBTRACT", None, window, va=1.0)), bump.inputs["Height"])
    glass = nt.nodes.new("ShaderNodeBsdfPrincipled")
    nt.links.new(base, glass.inputs["Base Color"])
    nt.links.new(rough, glass.inputs["Roughness"])
    nt.links.new(metal, glass.inputs["Metallic"])
    nt.links.new(bump.outputs["Normal"], glass.inputs["Normal"])
    nt.links.new(color, glass.inputs["Emission Color"])
    nt.links.new(strength, glass.inputs["Emission Strength"])
    roof = _node(nt, "ShaderNodeBsdfPrincipled", **{"Base Color": (0.07, 0.07, 0.075, 1), "Roughness": 0.85})
    rb = nt.nodes.new("ShaderNodeBump")
    rb.inputs["Strength"].default_value = 0.3
    nt.links.new(blot.outputs["Fac"], rb.inputs["Height"])
    nt.links.new(rb.outputs["Normal"], roof.inputs["Normal"])
    mix_s = nt.nodes.new("ShaderNodeMixShader")
    nt.links.new(wall, mix_s.inputs["Fac"])
    nt.links.new(roof.outputs[0], mix_s.inputs[1])
    nt.links.new(glass.outputs[0], mix_s.inputs[2])
    nt.links.new(mix_s.outputs[0], out.inputs["Surface"])
    return mat


def night_windows_group():
    """A shader group giving the window grid's emission colour and strength, for photo-textured buildings."""
    g = bpy.data.node_groups.new("night_windows", "ShaderNodeTree")
    g.interface.new_socket(name="Color", in_out="OUTPUT", socket_type="NodeSocketColor")
    g.interface.new_socket(name="Strength", in_out="OUTPUT", socket_type="NodeSocketFloat")
    out = g.nodes.new("NodeGroupOutput")
    color, strength, _wall, _window, _rnd = _lights(g)
    g.links.new(color, out.inputs["Color"])
    g.links.new(strength, out.inputs["Strength"])
    return g


def nightify(objects, group, darken: float = 0.62) -> int:
    """Keep each imported material's photo texture, darkened for night, and add the window lights."""
    done = set()
    for o in objects:
        for m in o.data.materials:
            if m is None or m.name in done or not m.use_nodes:
                continue
            nt = m.node_tree
            b = next((n for n in nt.nodes if n.type == "BSDF_PRINCIPLED"), None)
            if b is None:
                continue
            link = next((l for l in nt.links if l.to_socket == b.inputs["Base Color"]), None)
            mul = nt.nodes.new("ShaderNodeMix")
            mul.data_type, mul.blend_type = "RGBA", "MULTIPLY"
            mul.inputs["Factor"].default_value = 1.0
            mul.inputs["B"].default_value = (darken, darken, darken * 1.05, 1.0)
            if link is not None:
                nt.links.new(link.from_socket, mul.inputs["A"])
            else:
                mul.inputs["A"].default_value = b.inputs["Base Color"].default_value
            nt.links.new(mul.outputs["Result"], b.inputs["Base Color"])
            b.inputs["Roughness"].default_value = 0.55
            gn = nt.nodes.new("ShaderNodeGroup")
            gn.node_tree = group
            nt.links.new(gn.outputs["Color"], b.inputs["Emission Color"])
            nt.links.new(gn.outputs["Strength"], b.inputs["Emission Strength"])
            done.add(m.name)
    return len(done)


def _emit(name, rgb, strength):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    em = _node(nt, "ShaderNodeEmission", Color=(*rgb, 1.0), Strength=strength)
    o = nt.nodes.new("ShaderNodeOutputMaterial")
    nt.links.new(em.outputs[0], o.inputs["Surface"])
    return m


def _plain(name, rgb, rough):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    b = next(n for n in m.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
    b.inputs["Base Color"].default_value = (*rgb, 1.0)
    b.inputs["Roughness"].default_value = rough
    return m


def import_obj(path: str, origin: Vector, mat) -> list:
    before = set(bpy.data.objects)
    bpy.ops.wm.obj_import(filepath=path, forward_axis="Y", up_axis="Z")
    new = [o for o in bpy.data.objects if o not in before and o.type == "MESH"]
    for o in new:
        o.data.transform(Matrix.Translation(-origin))
        if mat is not None:
            o.data.materials.clear()
            o.data.materials.append(mat)
    return new


def build(plateau: Path, seed: int = 5, keep_textures: bool = False) -> dict:
    rnd = random.Random(seed)
    lod2 = {Path(p).name[:8]: p for p in glob.glob(str(plateau / "bldg/lod2/*/*.obj"))}
    lod1 = {Path(p).name[:8]: p for p in glob.glob(str(plateau / "bldg/lod1/*/*.obj"))}
    facade = facade_material()
    night = night_windows_group() if keep_textures else None
    # the centre: Nishi-Shinjuku's tile centre, at street level
    origin = Vector((-12640.0, -34640.0, 35.0))
    towers = []  # the building tiles (PLATEAU's OBJ has one object per tile, no per-building groups)
    for tile in sorted(set(lod1) | set(lod2)):
        # within about 3 km of Nishi-Shinjuku (tile 53394525: row 2, column 5 of mesh 5339-45)
        if tile[:6] != "533945" or abs(int(tile[6]) - 2) > 2 or abs(int(tile[7]) - 5) > 3:
            continue
        path = lod2.get(tile) if tile in LOD2 and tile in lod2 else lod1.get(tile)
        if path is None:
            continue
        textured = keep_textures and tile in LOD2 and tile in lod2
        objs = import_obj(path, origin, None if textured else facade)
        if textured:
            print(f"[city] {tile}: {nightify(objs, night)} photo materials kept for night", flush=True)
        towers += objs
        print(f"[city] {tile} {'LOD2' if tile in LOD2 and tile in lod2 else 'LOD1'} {len(objs)} objects", flush=True)
    bpy.context.preferences.edit.keyframe_new_interpolation_type = "LINEAR"  # traffic and drones at steady speed
    ground = far_city_material()  # (10/28) the ground carries the city's lights out to the horizon
    for p in glob.glob(str(plateau / "dem/*.obj")):
        import_obj(p, origin, ground)
    road = _plain("city_road", (0.012, 0.012, 0.014), 0.12)
    cache = Path("/content/ph_cache")
    if (cache / "credits.json").exists():  # (10/28) the roads in scanned asphalt, wet, as the alley's ground
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import photoreal

        assets = photoreal.fetch(cache)
        road = photoreal.pbr("city_asphalt", assets["floor"],
                             tile=3.5, space=None, tint=(0.55, 0.55, 0.58), bump=0.8, wet=0.7)
        _road_far(road)
    road_objs = []
    for p in glob.glob(str(plateau / "tran/*.obj")):
        road_objs += import_obj(p, origin, road)
    cars = traffic(road_objs, rnd)
    far_city(facade=facade)
    # aircraft-warning lights on the tower tops (blinking), from the tallest vertices per 40 m cell
    red = _emit("warning_red", (1.0, 0.04, 0.02), 60.0)
    tops = {}
    roofs = {}  # the highest point in every 40 m cell: warning lights above 110 m, billboards on 35-90 m roofs
    for o in towers:
        for v in o.data.vertices:
            if v.co.z > 30.0:
                key = (int(v.co.x // 40), int(v.co.y // 40))
                if v.co.z > roofs.get(key, Vector((0, 0, 0))).z:
                    roofs[key] = v.co.copy()
    tops = {k: p for k, p in roofs.items() if p.z > 110.0}
    sph = bpy.data.meshes.new("warning_light")
    import bmesh

    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=12, v_segments=8, radius=0.9)
    bm.to_mesh(sph)
    bm.free()
    sph.materials.append(red)
    for k, p in enumerate(tops.values()):
        o = bpy.data.objects.new(f"warning{k}", sph)
        o.location = p + Vector((0, 0, 1.5))
        bpy.context.scene.collection.objects.link(o)
    # blink: the red's strength on and off, 1 s period (all in step, as real warning lights are)
    em = next(n for n in red.node_tree.nodes if n.type == "EMISSION")
    for f, s in ((1, 60.0), (12, 60.0), (13, 0.0), (25, 0.0), (26, 60.0)):
        em.inputs["Strength"].default_value = s
        em.inputs["Strength"].keyframe_insert("default_value", frame=f)
    # street lights: warm points 7 m above the roads, every ~25 m along the road polygons' edges
    lamp = _emit("street_lamp", (1.0, 0.78, 0.5), 1500.0)
    lamp_light = bpy.data.lights.new("street_light", "POINT")
    lamp_light.energy, lamp_light.color, lamp_light.shadow_soft_size = 2500.0, (1.0, 0.8, 0.56), 0.3
    lm = bpy.data.meshes.new("lamp")
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=8, v_segments=6, radius=0.35)
    bm.to_mesh(lm)
    bm.free()
    lm.materials.append(lamp)
    placed = 0
    seen = set()
    for o in road_objs:
        for e in o.data.edges:
            a, b = o.data.vertices[e.vertices[0]].co, o.data.vertices[e.vertices[1]].co
            if (a - b).length < 20.0 or abs(a.x) > 1500 or abs(a.y) > 1500:
                continue
            key = (int(a.x // 25), int(a.y // 25))
            if key in seen:
                continue
            seen.add(key)
            lo = bpy.data.objects.new(f"lamp{placed}", lm)
            lo.location = a + Vector((0, 0, 7.0))
            bpy.context.scene.collection.objects.link(lo)
            pl = bpy.data.objects.new(f"lamp_light{placed}", lamp_light)  # the pool of light it casts on the road
            pl.location = a + Vector((0, 0, 6.6))
            bpy.context.scene.collection.objects.link(pl)
            placed += 1
    # holographic banners on a few towers: tall emissive planes with generic words
    holo_cols = [(0.1, 0.9, 1.0), (1.0, 0.2, 0.7), (1.0, 0.6, 0.15)]
    banners = 0
    mid = [p for p in roofs.values() if 35.0 < p.z < 90.0 and abs(p.x) < 1200 and abs(p.y) < 1200]
    for p in rnd.sample(mid, min(14, len(mid))):  # rooftop billboards facing the camera's side (south-west)
        me = bpy.data.meshes.new(f"banner{banners}")
        hgt, wid = rnd.uniform(6, 12), rnd.uniform(10, 18)
        me.from_pydata([(-wid / 2, 0, 0), (wid / 2, 0, 0), (wid / 2, 0, hgt), (-wid / 2, 0, hgt)], [], [(0, 1, 2, 3)])
        b = bpy.data.objects.new(f"banner{banners}", me)
        b.location = (p.x, p.y, p.z + 3.0)
        b.rotation_euler = (0.0, 0.0, math.radians(-45))
        me.materials.append(_emit(f"holo{banners}", holo_cols[banners % 3], rnd.uniform(4, 9)))
        bpy.context.scene.collection.objects.link(b)
        banners += 1
    # (10/28) air taxis on corridors over the city -- the near future Tokyo plans, not decoration: each a
    # dark craft with a red light to port, green to starboard and a white strobe blinking once a second
    strobe = _emit("taxi_strobe", (1.0, 1.0, 1.0), 400.0)
    n_ = next(n for n in strobe.node_tree.nodes if n.type == "EMISSION")
    for f0 in range(1, 222, 25):  # a flash a second, through the whole shot
        for f, v in ((f0 - 1, 0.0), (f0, 400.0), (f0 + 2, 0.0)):
            n_.inputs["Strength"].default_value = v
            n_.inputs["Strength"].keyframe_insert("default_value", frame=max(f, 0))
    nav = {"port": _emit("taxi_port", (1.0, 0.05, 0.03), 60.0), "star": _emit("taxi_star", (0.05, 1.0, 0.25), 60.0)}
    body_m = _plain("taxi_body", (0.03, 0.03, 0.035), 0.3)

    def dot(name, r, mat):
        me = bpy.data.meshes.new(name)
        bm = bmesh.new()
        bmesh.ops.create_uvsphere(bm, u_segments=8, v_segments=6, radius=r)
        bm.to_mesh(me)
        bm.free()
        me.materials.append(mat)
        return me

    m_body, m_port, m_star, m_strobe = dot("taxi_body", 1.4, body_m), dot("taxi_p", 0.35, nav["port"]), dot("taxi_s", 0.35, nav["star"]), dot("taxi_w", 0.3, strobe)
    corridors = [((-1400, -300, 290), (1400, 500, 300)), ((-900, 900, 310), (900, -900, 295)), ((-1400, 300, 320), (1400, 300, 315)),
                 ((300, -1400, 285), (500, 1400, 290))]  # above the towers' tops (~250 m)
    for k in range(14):
        a_, b_ = corridors[k % len(corridors)]
        a_, b_ = Vector(a_), Vector(b_)
        if k % 2:
            a_, b_ = b_, a_
        d = (b_ - a_).normalized()
        side = Vector((-d.y, d.x, 0.0)) * (rnd.uniform(-40, 40) + (25 if k % 2 else -25))
        root = bpy.data.objects.new(f"taxi{k}", None)
        bpy.context.scene.collection.objects.link(root)
        root.rotation_euler = (0.0, 0.0, math.atan2(d.y, d.x))
        for me, off in ((m_body, (0, 0, 0)), (m_port, (0, 1.6, 0)), (m_star, (0, -1.6, 0)), (m_strobe, (0, 0, -1.2))):
            o = bpy.data.objects.new(f"taxi{k}_{me.name}", me)
            bpy.context.scene.collection.objects.link(o)
            o.parent = root
            o.location = off
        start = a_.lerp(b_, rnd.uniform(0.05, 0.6)) + side
        v = rnd.uniform(22.0, 32.0)
        for f in (1, 220):
            root.location = start + d * v * (f - 1) / 25.0
            root.keyframe_insert("location", frame=f)
    japan(roofs, road_objs, rnd)
    tocho_mapping(roofs)
    globals()["_SEARCH_ROOFS"] = sorted((r for r in roofs.values() if r.z > 150), key=lambda r: -r.z)[:12:2]
    # (10/28, the user passing on a critique: vertical structure, information, clutter -- the future
    # layered on today's Shinjuku, its landmarks kept)
    highways(rnd, towers)
    megatowers(rnd, facade)
    tower_screens(roofs, towers, rnd)
    roof_clutter(roofs, towers, rnd, red)
    sky_lanes()
    globals()["_DRONE_INPUT"] = (towers, road_objs, roofs)
    print(f"[city] {len(towers)} towers, {len(tops)} warning lights, {placed} street lights, {banners} banners, 14 air taxis, {cars} cars", flush=True)
    return {"origin": tuple(origin), "towers": len(towers)}


FAR_SIZE = 60000.0
FAR_GLOW = (0.14, 0.08, 0.085)  # the lit haze at the far plane's edge, as the sky below the horizon


def far_city(size=FAR_SIZE, z=-30.0, facade=None):
    """The lit plane beyond the terrain tiles (see `far_city_material`), and, given the facade
    material, the far skyline on it (`far_skyline`)."""
    me = bpy.data.meshes.new("far_city")
    h = size / 2
    me.from_pydata([(-h, -h, z), (h, -h, z), (h, h, z), (-h, h, z)], [], [(0, 1, 2, 3)])
    me.materials.append(bpy.data.materials.get("far_city_lights") or far_city_material())
    o = bpy.data.objects.new("far_city", me)
    bpy.context.scene.collection.objects.link(o)
    if facade is not None:
        far_skyline(facade, z)
    return o


def _hazed(mat, name):
    """A copy of `mat` that sinks into the far haze glow (`FAR_GLOW`) with distance from the centre,
    as the far plane does."""
    m = mat.copy()
    m.name = name
    nt = m.node_tree
    out = next(n for n in nt.nodes if n.type == "OUTPUT_MATERIAL")
    link = next(l for l in nt.links if l.to_socket == out.inputs["Surface"])
    src = link.from_socket
    nt.links.remove(link)
    geo = nt.nodes.new("ShaderNodeNewGeometry")
    r = nt.nodes.new("ShaderNodeVectorMath")
    r.operation = "LENGTH"
    nt.links.new(geo.outputs["Position"], r.inputs[0])
    far = nt.nodes.new("ShaderNodeMapRange")
    nt.links.new(r.outputs["Value"], far.inputs["Value"])
    far.inputs["From Min"].default_value, far.inputs["From Max"].default_value = 8000.0, 0.4 * FAR_SIZE
    glow = _node(nt, "ShaderNodeEmission", Color=(*FAR_GLOW, 1.0), Strength=1.0)
    fmix = nt.nodes.new("ShaderNodeMixShader")
    nt.links.new(far.outputs["Result"], fmix.inputs["Fac"])
    nt.links.new(src, fmix.inputs[1])
    nt.links.new(glow.outputs[0], fmix.inputs[2])
    nt.links.new(fmix.outputs[0], out.inputs["Surface"])
    return m


def _far_block_material():
    """Far blocks, seen from kilometres: dark walls and sparse bright windows, the share lit and their
    brightness and tint varying block by block (the near facade material averaged to a grey wash)."""
    m = bpy.data.materials.new("far_blocks")
    m.use_nodes = True
    nt = m.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    geo = nt.nodes.new("ShaderNodeNewGeometry")
    cell = nt.nodes.new("ShaderNodeTexVoronoi")  # one random draw per ~70 m: per block
    cell.inputs["Scale"].default_value = 1.0 / 70.0
    nt.links.new(geo.outputs["Position"], cell.inputs["Vector"])
    rgb = nt.nodes.new("ShaderNodeSeparateColor")
    nt.links.new(cell.outputs["Color"], rgb.inputs["Color"])
    grid = nt.nodes.new("ShaderNodeVectorMath")  # one window per 3 m x 3 m x 3.6 m
    grid.operation = "DIVIDE"
    nt.links.new(geo.outputs["Position"], grid.inputs[0])
    grid.inputs[1].default_value = (3.0, 3.0, 3.6)
    fl = nt.nodes.new("ShaderNodeVectorMath")
    fl.operation = "FLOOR"
    nt.links.new(grid.outputs["Vector"], fl.inputs[0])
    wn = nt.nodes.new("ShaderNodeTexWhiteNoise")
    wn.noise_dimensions = "3D"
    nt.links.new(fl.outputs["Vector"], wn.inputs["Vector"])
    sz = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(grid.outputs["Vector"], sz.inputs[0])
    band = _math(nt, "LESS_THAN", _math(nt, "FRACT", sz.outputs["Z"]), None, vb=0.55)
    # (10/28 afternoon, the user: the details read blotchy -- the far blocks massed into a beige band)
    share = _math(nt, "ADD", _math(nt, "MULTIPLY", rgb.outputs["Red"], None, vb=0.12), None, vb=0.02)
    lit = _math(nt, "MULTIPLY", band, _math(nt, "LESS_THAN", wn.outputs["Value"], share))
    gain = _math(nt, "ADD", _math(nt, "MULTIPLY", _math(nt, "POWER", rgb.outputs["Green"], None, vb=3.0), None, vb=5.0),
                 None, vb=0.8)
    col = _ramp(nt, rgb.outputs["Blue"], [(0.0, (1.0, 0.5, 0.2)), (0.6, (1.0, 0.72, 0.45)), (0.85, (0.8, 0.88, 1.0)),
                                          (0.95, (1.0, 0.35, 0.8))])
    em = nt.nodes.new("ShaderNodeEmission")
    nt.links.new(col, em.inputs["Color"])
    nt.links.new(_math(nt, "MULTIPLY", lit, gain), em.inputs["Strength"])
    wall = _node(nt, "ShaderNodeBsdfPrincipled", **{"Base Color": (0.015, 0.015, 0.02, 1), "Roughness": 0.7})
    add = nt.nodes.new("ShaderNodeAddShader")
    nt.links.new(wall.outputs[0], add.inputs[0])
    nt.links.new(em.outputs[0], add.inputs[1])
    nt.links.new(add.outputs[0], out.inputs["Surface"])
    return m


def far_skyline(facade, z=-30.0, r0=4500.0, n_low=3800, n_hub=1200, seed=1028):
    """(10/28 dawn) Tokyo beyond the modelled tiles is buildings to the horizon, not a flat glow: seen
    from the drone's height the far plane alone read as a flat brown band with a straight edge. Blocks
    of lit windows (the facade material, hazed with distance) from 4.5 km to 27 km -- low-rise
    everywhere, towers clustered round a dozen far hubs -- so the skyline is broken as in a real night
    panorama. Own random stream, so nothing else in the city moves."""
    rnd = random.Random(seed)
    r1 = 0.45 * FAR_SIZE
    hubs = []
    for _ in range(14):
        a = rnd.uniform(0.0, 2.0 * math.pi)
        rr = rnd.uniform(r0 + 2500.0, 0.8 * r1)
        hubs.append((rr * math.cos(a), rr * math.sin(a), rnd.uniform(0.0, math.pi / 2), rnd.uniform(500.0, 1100.0)))
    verts, faces = [], []

    def box(cx, cy, w, d, hgt, ang):
        c_, s_ = math.cos(ang), math.sin(ang)
        b = len(verts)
        for zz in (z, z + hgt):
            for i, j in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
                x, y = i * w / 2, j * d / 2
                verts.append((cx + c_ * x - s_ * y, cy + s_ * x + c_ * y, zz))
        faces.extend([(b, b + 1, b + 2, b + 3)[::-1], (b + 4, b + 5, b + 6, b + 7), (b, b + 1, b + 5, b + 4),
                      (b + 1, b + 2, b + 6, b + 5), (b + 2, b + 3, b + 7, b + 6), (b + 3, b, b + 4, b + 7)])

    for _ in range(n_low):
        rr = math.sqrt(rnd.uniform(r0 * r0, r1 * r1))
        a = rnd.uniform(0.0, 2.0 * math.pi)
        box(rr * math.cos(a), rr * math.sin(a), rnd.uniform(18.0, 55.0), rnd.uniform(15.0, 45.0),
            50.0 + 55.0 * rnd.random() ** 2, rnd.uniform(0.0, math.pi / 2))
    for _ in range(n_hub):
        hx, hy, ang, sig = rnd.choice(hubs)
        x, y = hx + rnd.gauss(0.0, sig), hy + rnd.gauss(0.0, sig)
        if math.hypot(x, y) < r0:
            continue
        box(x, y, rnd.uniform(25.0, 70.0), rnd.uniform(25.0, 60.0), 100.0 + 210.0 * rnd.random() ** 1.6, ang)
    me = bpy.data.meshes.new("far_skyline")
    me.from_pydata(verts, [], faces)
    me.materials.append(_hazed(_far_block_material(), "far_skyline_blocks"))
    o = bpy.data.objects.new("far_skyline", me)
    bpy.context.scene.collection.objects.link(o)
    print(f"[city] far skyline: {len(faces) // 6} blocks, {len(hubs)} hubs", flush=True)
    return o


def _road_far(mat, near=35.0, far=110.0):
    """(10/28 evening, the user: from the air the streets looked like a blotchy river, not roads) The
    scanned asphalt's puddle mask and its 3.5 m repeat are made for the street; seen from 75-400 m they
    massed into mirror-dark blotches. Past `near` metres from the camera the road becomes even, damp,
    dark asphalt -- a soft sheen that carries the street lights and the cars' lights, as night roads
    look from a tower -- fully so by `far`."""
    nt = mat.node_tree
    out = next(n for n in nt.nodes if n.type == "OUTPUT_MATERIAL")
    link = next(l for l in nt.links if l.to_socket == out.inputs["Surface"])
    src = link.from_socket
    nt.links.remove(link)
    plain = _node(nt, "ShaderNodeBsdfPrincipled", **{"Base Color": (0.032, 0.032, 0.035, 1.0), "Roughness": 0.32})
    grain = _node(nt, "ShaderNodeTexNoise", Scale=0.6, Detail=4.0)  # a faint unevenness, no blotches
    rough = nt.nodes.new("ShaderNodeMapRange")
    nt.links.new(grain.outputs["Fac"], rough.inputs["Value"])
    rough.inputs["To Min"].default_value, rough.inputs["To Max"].default_value = 0.28, 0.4
    nt.links.new(rough.outputs["Result"], plain.inputs["Roughness"])
    camd = nt.nodes.new("ShaderNodeCameraData")
    k = nt.nodes.new("ShaderNodeMapRange")
    nt.links.new(camd.outputs["View Distance"], k.inputs["Value"])
    k.inputs["From Min"].default_value, k.inputs["From Max"].default_value = near, far
    mix = nt.nodes.new("ShaderNodeMixShader")
    nt.links.new(k.outputs["Result"], mix.inputs["Fac"])
    nt.links.new(src, mix.inputs[1])
    nt.links.new(plain.outputs[0], mix.inputs[2])
    nt.links.new(mix.outputs[0], out.inputs["Surface"])
    return mat


def far_city_material():
    """(10/28) Tokyo's lights to the horizon beyond the modelled tiles, as in the user's reference
    photos: a plane under the model lit by a shader -- points of light (one per ~14 m cell, kept by a
    district density), warm sodium-orange and white LED, and orange arterial roads every ~700 m,
    curving a little. The haze fades them into the horizon glow."""
    m = bpy.data.materials.new("far_city_lights")
    m.use_nodes = True
    nt = m.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    geo = nt.nodes.new("ShaderNodeNewGeometry")
    warp = _node(nt, "ShaderNodeTexNoise", Scale=0.0006, Detail=3.0)
    nt.links.new(geo.outputs["Position"], warp.inputs["Vector"])
    wv = nt.nodes.new("ShaderNodeVectorMath")
    wv.operation = "SCALE"
    nt.links.new(warp.outputs["Color"], wv.inputs[0])
    wv.inputs["Scale"].default_value = 260.0
    pos = nt.nodes.new("ShaderNodeVectorMath")
    nt.links.new(geo.outputs["Position"], pos.inputs[0])
    nt.links.new(wv.outputs["Vector"], pos.inputs[1])
    vor = nt.nodes.new("ShaderNodeTexVoronoi")
    vor.voronoi_dimensions = "2D"
    vor.inputs["Scale"].default_value = 1.0 / 14.0
    nt.links.new(geo.outputs["Position"], vor.inputs["Vector"])
    dot = nt.nodes.new("ShaderNodeMapRange")
    nt.links.new(vor.outputs["Distance"], dot.inputs["Value"])
    dot.inputs["From Min"].default_value, dot.inputs["From Max"].default_value = 0.14, 0.04
    sepc = nt.nodes.new("ShaderNodeSeparateColor")
    nt.links.new(vor.outputs["Color"], sepc.inputs["Color"])
    dist = _node(nt, "ShaderNodeTexNoise", Scale=0.0022, Detail=4.0)
    nt.links.new(geo.outputs["Position"], dist.inputs["Vector"])
    dens = nt.nodes.new("ShaderNodeMapRange")
    nt.links.new(dist.outputs["Fac"], dens.inputs["Value"])
    dens.inputs["From Min"].default_value, dens.inputs["From Max"].default_value = 0.35, 0.65
    dens.inputs["To Min"].default_value, dens.inputs["To Max"].default_value = 0.12, 0.7
    keep = _math(nt, "LESS_THAN", sepc.outputs["Red"], dens.outputs["Result"])
    pts = _math(nt, "MULTIPLY", dot.outputs["Result"], keep)
    col = _ramp(nt, sepc.outputs["Green"], [(0.0, (1.0, 0.5, 0.16)), (0.45, (1.0, 0.82, 0.6)), (0.75, (0.95, 0.96, 1.0)),
                                             (0.95, (0.75, 0.88, 1.0))])
    sp = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(pos.outputs["Vector"], sp.inputs[0])
    roads = None
    for axis in ("X", "Y"):  # arterial roads every ~700 m: thin bright lines, warped
        f = _math(nt, "FRACT", _math(nt, "DIVIDE", sp.outputs[axis], None, vb=700.0))
        line = _math(nt, "LESS_THAN", f, None, vb=0.008)
        roads = line if roads is None else _math(nt, "MAXIMUM", roads, line)
    em_pts = nt.nodes.new("ShaderNodeEmission")
    nt.links.new(col, em_pts.inputs["Color"])
    nt.links.new(_math(nt, "MULTIPLY", pts, None, vb=60.0), em_pts.inputs["Strength"])
    em_road = _node(nt, "ShaderNodeEmission", Color=(1.0, 0.48, 0.14, 1.0))
    nt.links.new(_math(nt, "MULTIPLY", roads, None, vb=5.0), em_road.inputs["Strength"])
    add = nt.nodes.new("ShaderNodeAddShader")
    nt.links.new(em_pts.outputs[0], add.inputs[0])
    nt.links.new(em_road.outputs[0], add.inputs[1])
    nt.links.new(add.outputs[0], out.inputs["Surface"])
    # near the model the ground is parks, yards and streets between buildings: fewer, dimmer points
    r = nt.nodes.new("ShaderNodeVectorMath")
    r.operation = "LENGTH"
    nt.links.new(geo.outputs["Position"], r.inputs[0])
    near = nt.nodes.new("ShaderNodeMapRange")
    nt.links.new(r.outputs["Value"], near.inputs["Value"])
    near.inputs["From Min"].default_value, near.inputs["From Max"].default_value = 1800.0, 4500.0
    near.inputs["To Min"].default_value, near.inputs["To Max"].default_value = 0.0, 1.0
    mix = nt.nodes.new("ShaderNodeMixShader")
    dark = _node(nt, "ShaderNodeBsdfPrincipled", **{"Base Color": (0.012, 0.012, 0.016, 1), "Roughness": 0.85})
    nt.links.new(near.outputs["Result"], mix.inputs["Fac"])
    nt.links.new(dark.outputs[0], mix.inputs[1])
    nt.links.new(add.outputs[0], mix.inputs[2])
    # (10/28 dawn) seen from the drone's height the carpet beyond ~7 km averaged to one flat colour: far
    # off, districts -- bright hubs, dark parks and low-rise -- give it the patchiness of a real panorama
    district = _node(nt, "ShaderNodeTexNoise", Scale=0.00035, Detail=2.0)
    nt.links.new(geo.outputs["Position"], district.inputs["Vector"])
    dmod = nt.nodes.new("ShaderNodeMapRange")
    nt.links.new(district.outputs["Fac"], dmod.inputs["Value"])
    dmod.inputs["From Min"].default_value, dmod.inputs["From Max"].default_value = 0.3, 0.7
    dmod.inputs["To Min"].default_value, dmod.inputs["To Max"].default_value = 0.1, 2.4
    beyond = nt.nodes.new("ShaderNodeMapRange")
    nt.links.new(r.outputs["Value"], beyond.inputs["Value"])
    beyond.inputs["From Min"].default_value, beyond.inputs["From Max"].default_value = 5000.0, 9000.0
    dm = nt.nodes.new("ShaderNodeMix")
    dm.data_type = "FLOAT"
    nt.links.new(beyond.outputs["Result"], dm.inputs["Factor"])
    dm.inputs["A"].default_value = 1.0
    nt.links.new(dmod.outputs["Result"], dm.inputs["B"])
    nt.links.new(_math(nt, "MULTIPLY", _math(nt, "MULTIPLY", pts, None, vb=60.0), dm.outputs["Result"]),
                 em_pts.inputs["Strength"])
    # and the far lights sink into the haze glow with distance, so the plane's edge meets the sky below
    # the horizon (`FAR_GLOW`) without a line
    far = nt.nodes.new("ShaderNodeMapRange")
    nt.links.new(r.outputs["Value"], far.inputs["Value"])
    far.inputs["From Min"].default_value, far.inputs["From Max"].default_value = 12000.0, 0.48 * FAR_SIZE
    glow = _node(nt, "ShaderNodeEmission", Color=(*FAR_GLOW, 1.0), Strength=1.0)
    fmix = nt.nodes.new("ShaderNodeMixShader")
    nt.links.new(far.outputs["Result"], fmix.inputs["Fac"])
    nt.links.new(mix.outputs[0], fmix.inputs[1])
    nt.links.new(glow.outputs[0], fmix.inputs[2])
    for l in list(nt.links):
        if l.to_node == out:
            nt.links.remove(l)
    nt.links.new(fmix.outputs[0], out.inputs["Surface"])
    return m


def holograms(tops, rnd, n=5):
    """Near-future dressing: rings and a lattice globe of light hanging over a few towers, cyan and
    magenta, turning slowly. Generic shapes, no lettering."""
    import bmesh

    cols = [(0.1, 0.9, 1.0), (1.0, 0.15, 0.75), (0.55, 0.3, 1.0)]
    picks = rnd.sample(list(tops.values()), min(n, len(tops)))
    for k, p in enumerate(picks):
        me = bpy.data.meshes.new(f"holo_shape{k}")
        bm = bmesh.new()
        if k % 3 == 2:
            bmesh.ops.create_icosphere(bm, subdivisions=2, radius=22.0)
        else:
            bmesh.ops.create_circle(bm, cap_ends=False, radius=28.0 + 6.0 * (k % 3), segments=64)
        bm.to_mesh(me)
        bm.free()
        o = bpy.data.objects.new(f"holo{k}", me)
        bpy.context.scene.collection.objects.link(o)
        o.location = p + Vector((0, 0, 45.0 + 10 * (k % 2)))
        o.rotation_euler = (math.radians(rnd.uniform(-20, 20)), math.radians(rnd.uniform(-20, 20)), 0.0)
        wf = o.modifiers.new("lines", "WIREFRAME")
        wf.thickness = 0.25
        if k % 3 != 2:  # a ring is an edge loop: give it a tube
            sk = o.modifiers.new("skin", "SKIN")
            o.modifiers.remove(wf)
            for v in me.skin_vertices[0].data:
                v.radius = (0.6, 0.6)
        me.materials.append(_emit(f"holo_light{k}", cols[k % 3], 3.5))
        for f, z in ((1, 0.0), (150, math.radians(40 * (1 if k % 2 else -1)))):
            o.rotation_euler.z = z
            o.keyframe_insert("rotation_euler", frame=f)


FONT = next((f for f in ("C:/Windows/Fonts/NotoSansJP-VF.ttf", "/content/up/NotoSansJP-VF.ttf") if Path(f).exists()), "")
#: generic words only (no shop or brand names): bar, open, ramen, karaoke, pharmacy, pawn, hotel, night
SIGN_WORDS = ["居酒屋", "営業中", "ラーメン", "カラオケ", "薬", "質", "ホテル", "夜", "焼鳥", "喫茶"]
HOLO_WORDS = ["新宿", "電脳", "夜"]


SIGNS = Path(__file__).resolve().parent / "signs"  # bold lettering drawn by PIL (Blender loads the variable font Thin)


def _sign_plane(name, png, height, rgb, strength):
    """A sign as a glowing board: the lettering image (white on black) masks an emission; its top edge
    at the origin, facing -Y."""
    img = bpy.data.images.load(str(png), check_existing=True)
    w, h = img.size
    width = height * w / h
    me = bpy.data.meshes.new(name)
    me.from_pydata([(-width / 2, 0, -height), (width / 2, 0, -height), (width / 2, 0, 0), (-width / 2, 0, 0)], [], [(0, 1, 2, 3)])
    uv = me.uv_layers.new()
    for loop, co in zip(uv.data, ((0, 0), (1, 0), (1, 1), (0, 1))):
        loop.uv = co
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    tex = nt.nodes.new("ShaderNodeTexImage")
    tex.image = img
    em = _node(nt, "ShaderNodeEmission", Color=(*rgb, 1.0), Strength=strength)
    clear = nt.nodes.new("ShaderNodeBsdfTransparent")
    mix = nt.nodes.new("ShaderNodeMixShader")
    nt.links.new(tex.outputs["Color"], mix.inputs["Fac"])
    nt.links.new(clear.outputs[0], mix.inputs[1])
    nt.links.new(em.outputs[0], mix.inputs[2])
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    nt.links.new(mix.outputs[0], out.inputs["Surface"])
    me.materials.append(m)
    o = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(o)
    return o


_TREES = {}


def building_tree(towers):
    """A BVH of the PLATEAU tiles (their mesh data is already in the city frame)."""
    if "t" not in _TREES:
        import bmesh
        from mathutils.bvhtree import BVHTree

        bm = bmesh.new()
        for o in towers:
            bm.from_mesh(o.data)
        _TREES["t"] = BVHTree.FromBMesh(bm)
        bm.free()
    return _TREES["t"]


def _boxv(name, verts_min, verts_max, mat, col=None):
    """An axis-free box from 8 corner points given as (origin, ax, ay, az) -- a mesh in world coordinates."""
    o_, ax, ay, az = verts_min
    vs = [o_ + ax * i + ay * j + az * k for k in (0, 1) for j in (0, 1) for i in (0, 1)]
    faces = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    me = bpy.data.meshes.new(name)
    me.from_pydata([tuple(v) for v in vs], [], faces)
    me.materials.append(mat)
    o = bpy.data.objects.new(name, me)
    (col or bpy.context.scene.collection).objects.link(o)
    return o


def highways(rnd, towers, routes=(((-1500.0, -150.0), (1500.0, -150.0), 55.0),
                                   ((620.0, -1500.0), (620.0, 1500.0), 40.0),
                                   ((-1300.0, 1500.0), (1300.0, 100.0), 85.0))):
    """Elevated expressways over today's streets at 40-85 m, crossing at three levels, one running through
    the towers' middle floors: decks with barriers, orange lamps, piers where the ground is open, and
    lanes of head- and tail-lights."""
    import bmesh

    tree = building_tree(towers)
    col = bpy.data.collections.new("highways")
    bpy.context.scene.collection.children.link(col)
    deck = _plain("hw_concrete", (0.32, 0.31, 0.3), 0.75)
    barrier = _plain("hw_barrier", (0.55, 0.55, 0.53), 0.5)
    lamp = _emit("hw_lamp", (1.0, 0.62, 0.25), 60.0)
    lm = bpy.data.meshes.new("hw_lamp")
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=6, v_segments=4, radius=0.5)
    bm.to_mesh(lm)
    bm.free()
    lm.materials.append(lamp)
    head = bpy.data.materials.get("headlights") or _emit("headlights", (1.0, 0.93, 0.8), 260.0)
    tail = bpy.data.materials.get("taillights") or _emit("taillights", (1.0, 0.04, 0.02), 200.0)
    hm, tm = bpy.data.meshes.get("car_head"), bpy.data.meshes.get("car_tail")
    n_pier = n_car = 0
    for k, (a2, b2, z) in enumerate(routes):
        a, b = Vector((*a2, z)), Vector((*b2, z))
        d = (b - a).normalized()
        side = Vector((-d.y, d.x, 0.0))
        length = (b - a).length
        w = 9.0
        _boxv(f"hw_deck{k}", (a - side * w - Vector((0, 0, 1.8)), d * length, side * 2 * w, Vector((0, 0, 1.8))), None, deck, col)
        for sgn in (-1, 1):
            _boxv(f"hw_barrier{k}{sgn}", (a + side * (w - 0.3) * sgn - side * 0.15, d * length, side * 0.3, Vector((0, 0, 1.1))), None,
                  barrier, col)
        for t in range(0, int(length), 40):
            for sgn in (-1, 1):
                o = bpy.data.objects.new(f"hw_lampo{k}_{t}{sgn}", lm)
                col.objects.link(o)
                o.location = a + d * t + side * (w - 0.6) * sgn + Vector((0, 0, 9.0))
        for t in range(35, int(length), 70):  # piers, only where the ground under the deck is open
            q = a + d * t
            hit = tree.find_nearest(Vector((q.x, q.y, z * 0.5)))
            up = tree.ray_cast(Vector((q.x, q.y, 1.0)), Vector((0.0, 0.0, 1.0)), z)
            if (hit[0] is not None and hit[3] < 6.0) or (up[0] is not None and up[3] < z - 3):
                continue
            _boxv(f"hw_pier{k}_{t}", (Vector((q.x - 1.6, q.y - 1.6, -10.0)), Vector((3.2, 0, 0)), Vector((0, 3.2, 0)),
                                      Vector((0, 0, z - 1.8 + 10.0))), None, deck, col)
            n_pier += 1
        if hm is not None and tm is not None:  # traffic on the deck, both ways
            for c_ in range(70):
                kind = c_ % 2
                lane = side * ((2.4 if kind else -2.4) + (rnd.choice((0.0, 3.4)) * (1 if kind else -1)))
                v = rnd.uniform(18.0, 28.0) * (1 if kind else -1)
                run_ = abs(v) * 219 / 25.0
                s0 = rnd.uniform(0, length - run_) if v > 0 else rnd.uniform(run_, length)  # stays on the deck
                car = bpy.data.objects.new(f"hw_car{k}_{c_}", hm if kind else tm)
                col.objects.link(car)
                car.rotation_euler = (0.0, 0.0, math.atan2(d.y, d.x) + math.pi / 2)
                for f in (1, 220):
                    sv = s0 + v * (f - 1) / 25.0
                    car.location = a + d * sv + lane + Vector((0, 0, 0.0))
                    car.keyframe_insert("location", frame=f)
                n_car += 1
    print(f"[city] highways: {len(routes)} elevated routes, {n_pier} piers, {n_car} cars on them", flush=True)


def megatowers(rnd, facade):
    """Two new megatowers, deliberately unlike each other (the user, 10/28): over the east exit a tower
    twisting a quarter turn as it tapers, LED edges every other slab, a spire with a red light; over
    Kabukicho a gate -- two slender legs joined high up by a lit sky block, one crown above. The city's
    window shader on both."""
    col = bpy.data.collections.new("megatowers")
    bpy.context.scene.collection.children.link(col)
    edge = _emit("mega_edge", (0.25, 0.85, 1.0), 12.0)
    crown = _emit("mega_crown", (1.0, 0.2, 0.7), 18.0)
    red = bpy.data.materials.get("warning_red")

    def slab(name, cx, cy, w, d, z0, z1, rot, mat):
        c, s_ = math.cos(rot), math.sin(rot)
        ax, ay = Vector((c * w, s_ * w, 0)), Vector((-s_ * d, c * d, 0))
        o_ = Vector((cx, cy, z0)) - ax / 2 - ay / 2
        return _boxv(name, (o_, ax, ay, Vector((0, 0, z1 - z0))), None, mat, col)

    # the twisting tower, 26 slabs, 520 m with the spire
    x, y, h, n = 760.0, 150.0, 470.0, 26
    for i in range(n):
        u = i / n
        w = 74.0 * (1 - 0.45 * u)
        z0, z1 = -10 + h * u, -10 + h * (i + 1) / n - 1.2
        rot = math.radians(90) * u
        slab(f"mega_twist_{i}", x, y, w, w * 0.82, z0, z1, rot, facade)
        if i % 2 == 0:
            slab(f"mega_twist_led_{i}", x, y, w + 1.0, w * 0.82 + 1.0, z1 - 0.6, z1 + 0.6, rot, edge)
    slab("mega_twist_spire", x, y, 3.0, 3.0, h - 10, h + 50, 0.0, edge)
    if red is not None:
        slab("mega_twist_tip", x, y, 3.0, 3.0, h + 50, h + 53, 0.0, red)
    # the gate tower over Kabukicho: two legs, a sky block, a crown
    x, y = 860.0, 790.0
    for k, dx in enumerate((-30.0, 30.0)):
        slab(f"mega_gate_leg{k}", x + dx, y, 30.0, 34.0, -10, 300, 0.0, facade)
        slab(f"mega_gate_legled{k}", x + dx + 15.6 * (1 if dx > 0 else -1), y, 1.2, 34.0, -10, 300, 0.0, edge)
    slab("mega_gate_block", x, y, 100.0, 40.0, 300, 360, 0.0, facade)
    slab("mega_gate_band", x, y, 101.0, 41.0, 326, 334, 0.0, crown)
    slab("mega_gate_crown", x, y, 46.0, 36.0, 360, 430, math.radians(15), facade)
    slab("mega_gate_crownled", x, y, 47.0, 37.0, 425, 431, math.radians(15), edge)
    if red is not None:
        for dx in (-23.0, 23.0):
            slab(f"mega_gate_red{dx}", x + dx * math.cos(math.radians(15)), y + dx * math.sin(math.radians(15)), 1.6, 1.6, 431, 433, 0.0, red)
    print("[city] megatowers: a twisting tower (520 m with its spire) and a gate tower (430 m)", flush=True)


def _ad_plane(name, img_a, img_b, width, height, strength=6.0, swap=(80, 96)):
    """A giant screen: two ads crossfading, emissive, with a faint scanline."""
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    uvn = nt.nodes.new("ShaderNodeTexCoord")
    ta, tb = nt.nodes.new("ShaderNodeTexImage"), nt.nodes.new("ShaderNodeTexImage")
    ta.image = bpy.data.images.load(str(img_a), check_existing=True)
    tb.image = bpy.data.images.load(str(img_b), check_existing=True)
    for t_ in (ta, tb):
        nt.links.new(uvn.outputs["UV"], t_.inputs["Vector"])
    mx = nt.nodes.new("ShaderNodeMix")
    mx.data_type = "RGBA"
    nt.links.new(ta.outputs["Color"], mx.inputs["A"])
    nt.links.new(tb.outputs["Color"], mx.inputs["B"])
    fac = mx.inputs["Factor"]
    for f, v in ((1, 0.0), (swap[0], 0.0), (swap[1], 1.0), (swap[0] + 100, 1.0), (swap[1] + 100, 0.0)):
        fac.default_value = v
        fac.keyframe_insert("default_value", frame=f)
    sep_uv = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(uvn.outputs["UV"], sep_uv.inputs[0])
    cols_, rows_ = 300.0, 300.0 * height / width  # the LED pitch: 300 across
    fx = _math(nt, "FRACT", _math(nt, "MULTIPLY", sep_uv.outputs["X"], None, vb=cols_ * 3))
    fy = _math(nt, "FRACT", _math(nt, "MULTIPLY", sep_uv.outputs["Y"], None, vb=rows_))
    sub_i = _math(nt, "FLOOR", _math(nt, "MULTIPLY", _math(nt, "FRACT", _math(nt, "MULTIPLY", sep_uv.outputs["X"], None, vb=cols_)), None, vb=3.0))
    mask = nt.nodes.new("ShaderNodeCombineColor")
    for ch, idx in (("Red", 0.0), ("Green", 1.0), ("Blue", 2.0)):
        nt.links.new(_math(nt, "COMPARE", sub_i, None, vb=idx), mask.inputs[ch])
    gap = _math(nt, "MULTIPLY", _math(nt, "LESS_THAN", _math(nt, "ABSOLUTE", _math(nt, "SUBTRACT", fx, None, vb=0.5)), None, vb=0.38),
                _math(nt, "LESS_THAN", _math(nt, "ABSOLUTE", _math(nt, "SUBTRACT", fy, None, vb=0.5)), None, vb=0.4))
    sub_col = nt.nodes.new("ShaderNodeMix")
    sub_col.data_type, sub_col.blend_type = "RGBA", "MULTIPLY"
    sub_col.inputs["Factor"].default_value = 1.0
    nt.links.new(mx.outputs["Result"], sub_col.inputs["A"])
    nt.links.new(mask.outputs["Color"], sub_col.inputs["B"])
    scan = nt.nodes.new("ShaderNodeTexWave")  # a refresh band rolling down
    scan.wave_type, scan.bands_direction = "BANDS", "Y"
    scan.inputs["Scale"].default_value = 0.6
    nt.links.new(uvn.outputs["UV"], scan.inputs["Vector"])
    ph = scan.inputs["Phase Offset"]
    for f, v in ((1, 0.0), (220, 40.0)):
        ph.default_value = v
        ph.keyframe_insert("default_value", frame=f)
    bright = _math(nt, "ADD", _math(nt, "MULTIPLY", scan.outputs["Fac"], None, vb=0.35), None, vb=0.85)
    em = nt.nodes.new("ShaderNodeEmission")
    nt.links.new(_math(nt, "MULTIPLY", _math(nt, "MULTIPLY", gap, bright), None, vb=strength * 3.2), em.inputs["Strength"])
    nt.links.new(sub_col.outputs["Result"], em.inputs["Color"])
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    nt.links.new(em.outputs[0], out.inputs["Surface"])
    me = bpy.data.meshes.new(name)
    me.from_pydata([(-width / 2, 0, -height), (width / 2, 0, -height), (width / 2, 0, 0), (-width / 2, 0, 0)], [], [(0, 1, 2, 3)])
    uv = me.uv_layers.new()
    for loop, co in zip(uv.data, ((0, 0), (1, 0), (1, 1), (0, 1))):
        loop.uv = co
    me.materials.append(m)
    o = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(o)
    return o


def _flat_box(name, sx, sy, sz, mat):
    """A box mesh sx x sy x sz, its base centred on the origin."""
    me = bpy.data.meshes.new(name)
    vs = [(x, y, z) for z in (0.0, sz) for y in (-sy / 2, sy / 2) for x in (-sx / 2, sx / 2)]
    me.from_pydata(vs, [], [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)])
    me.materials.append(mat)
    return me


def tower_screens(roofs, towers, rnd, cam=Vector((-800.0, -700.0, 0.0)), n=6):
    """Screens tens of metres to a hundred tall on the faces of the west side's towers that look toward
    the camera: placed by casting at each tower from outside, so they sit on the real facade."""
    tree = building_tree(towers)
    tocho = Vector((-160.0, 200.0, 0.0))
    tall = sorted([r for r in roofs.values() if r.z > 120.0 and (Vector((r.x, r.y, 0)) - tocho).length > 120],
                  key=lambda r: -r.z)
    picked = []
    for r in tall:
        if all((Vector((r.x, r.y, 0)) - Vector((q.x, q.y, 0))).length > 110 for q in picked):
            picked.append(r)
        if len(picked) >= n:
            break
    placed = 0
    for k, r in enumerate(picked):
        c = Vector((r.x, r.y, 0.0))
        to_cam = (cam - c).normalized()
        height = max(50.0, min(100.0, r.z - 60.0))
        mid = r.z - 20.0 - height / 2
        hit = tree.ray_cast(c + to_cam * 140.0 + Vector((0, 0, mid)), -to_cam, 160.0)
        if hit[0] is None or (Vector((hit[0].x, hit[0].y, 0)) - c).length > 45:
            continue
        nrm = Vector((hit[1].x, hit[1].y, 0.0)).normalized()
        o = _ad_plane(f"tower_screen{k}", SIGNS / f"ad_{k % 4:02d}.png", SIGNS / f"ad_{(k + 1) % 4:02d}.png", height / 2.2, height,
                      strength=5.0, swap=(60 + 9 * k, 76 + 9 * k))
        o.location = hit[0] + nrm * 1.6 + Vector((0, 0, height / 2))
        o.rotation_euler = (0.0, 0.0, math.atan2(nrm.y, nrm.x) + math.radians(90))
        # the housing: a dark frame round it, brackets back to the wall, the light it throws
        frame_m = bpy.data.materials.get("screen_frame") or _plain("screen_frame", (0.03, 0.03, 0.035), 0.35)
        w_ = height / 2.2
        for dz, sx, sz in ((0.0, w_ + 2.0, 1.0), (-height, w_ + 2.0, 1.0)):
            fr = bpy.data.objects.new(f"{o.name}_frame{dz}", _flat_box(f"{o.name}_fb{dz}", sx, 1.2, sz, frame_m))
            bpy.context.scene.collection.objects.link(fr)
            fr.parent = o
            fr.location = (0.0, 0.3, dz - 0.5)
        for dx in (-w_ / 2 - 0.5, w_ / 2 + 0.5):
            fr = bpy.data.objects.new(f"{o.name}_side{dx}", _flat_box(f"{o.name}_sb{dx}", 1.0, 1.2, height + 2.0, frame_m))
            bpy.context.scene.collection.objects.link(fr)
            fr.parent = o
            fr.location = (dx, 0.3, -height - 1.0)
        glow = bpy.data.lights.new(f"{o.name}_spill", "AREA")
        glow.shape, glow.size, glow.size_y = "RECTANGLE", w_, height
        glow.energy, glow.color = 9.0e4 * height / 100.0, [(1.0, 0.3, 0.6), (0.3, 0.8, 1.0), (1.0, 0.6, 0.2), (0.4, 1.0, 0.6)][k % 4]
        gl = bpy.data.objects.new(f"{o.name}_spill", glow)
        bpy.context.scene.collection.objects.link(gl)
        gl.location = hit[0] + nrm * 3.0
        gl.rotation_euler = (math.radians(90), 0.0, math.atan2(nrm.y, nrm.x) + math.radians(90))
        placed += 1
    print(f"[city] tower screens: {placed} of {len(picked)} towers", flush=True)


def roof_clutter(roofs, towers, rnd, red_mat, cap=3500):
    """What covers Tokyo's roofs: AC units, water tanks, antenna masts with red lights -- dropped onto the
    real roof surfaces by casting down."""
    import bmesh

    tree = building_tree(towers)

    def mesh(name, build, mat):
        me = bpy.data.meshes.new(name)
        bm = bmesh.new()
        build(bm)
        bm.to_mesh(me)
        bm.free()
        me.materials.append(mat)
        return me

    grey = _plain("roof_unit", (0.45, 0.46, 0.47), 0.5)
    white = _plain("roof_tank", (0.6, 0.6, 0.58), 0.45)
    metal = _plain("roof_mast", (0.25, 0.25, 0.27), 0.4)
    ac = mesh("roof_ac", lambda bm: (bmesh.ops.create_cube(bm, size=1.0), bmesh.ops.scale(bm, vec=(2.2, 1.2, 1.3), verts=bm.verts),
                                     bmesh.ops.translate(bm, vec=(0, 0, 0.65), verts=bm.verts)), grey)
    tank = mesh("roof_tank", lambda bm: bmesh.ops.create_cone(bm, cap_ends=True, segments=14, radius1=1.3, radius2=1.3, depth=2.6,
                                                              matrix=Matrix.Translation((0, 0, 1.3))), white)
    mast = mesh("roof_mast", lambda bm: bmesh.ops.create_cone(bm, cap_ends=True, segments=6, radius1=0.12, radius2=0.05, depth=12.0,
                                                              matrix=Matrix.Translation((0, 0, 6.0))), metal)
    tip = mesh("roof_tip", lambda bm: bmesh.ops.create_uvsphere(bm, u_segments=6, v_segments=4, radius=0.35,
                                                                matrix=Matrix.Translation((0, 0, 12.2))), red_mat)
    col = bpy.data.collections.new("roof_clutter")
    bpy.context.scene.collection.children.link(col)
    n = 0
    for r in roofs.values():
        if r.z < 18.0 or n > cap:
            continue
        for _ in range(rnd.randint(2, 6)):
            q = Vector((r.x + rnd.uniform(-17, 17), r.y + rnd.uniform(-17, 17), r.z + 25.0))
            hit = tree.ray_cast(q, Vector((0.0, 0.0, -1.0)), 60.0)
            if hit[0] is None or hit[1].z < 0.85 or hit[0].z < r.z - 30.0:
                continue
            roll = rnd.random()
            meshes = [ac] if roll < 0.6 else ([tank] if roll < 0.85 else [mast, tip])
            for me in meshes:
                o = bpy.data.objects.new("roofthing", me)
                col.objects.link(o)
                o.location = hit[0]
                o.rotation_euler = (0.0, 0.0, rnd.uniform(0, math.pi))
            n += 1
    print(f"[city] roof clutter: {n} units, tanks and masts", flush=True)


def sky_lanes():
    """Guidance lights marking the air-taxi corridors: green and amber beacons in pairs, every 60 m."""
    import bmesh

    green, amber = _emit("lane_green", (0.1, 1.0, 0.3), 14.0), _emit("lane_amber", (1.0, 0.6, 0.1), 14.0)
    meshes = {}
    for nm, mat in (("g", green), ("a", amber)):
        me = bpy.data.meshes.new(f"lane_{nm}")
        bm = bmesh.new()
        bmesh.ops.create_uvsphere(bm, u_segments=6, v_segments=4, radius=0.7)
        bm.to_mesh(me)
        bm.free()
        me.materials.append(mat)
        meshes[nm] = me
    corridors = [((-1400, -300, 290), (1400, 500, 300)), ((300, -1400, 285), (500, 1400, 290))]
    n = 0
    for a2, b2 in corridors:
        a, b = Vector(a2), Vector(b2)
        d = (b - a).normalized()
        side = Vector((-d.y, d.x, 0.0))
        for t in range(0, int((b - a).length), 140):
            for sgn, nm in ((-1, "g"), (1, "a")):
                o = bpy.data.objects.new("lane_light", meshes[nm])
                bpy.context.scene.collection.objects.link(o)
                o.location = a + d * t + side * 35.0 * sgn - Vector((0, 0, 12.0))
                n += 1
    print(f"[city] sky lanes: {n} guidance lights", flush=True)


def tocho_mapping(roofs, near=Vector((-160.0, 200.0, 0.0))):
    """Find the Tokyo Metropolitan Government Building's twin towers (the tallest roofs within 450 m of
    where it stands in this frame) and switch the projection-mapping region of every building shader onto
    their east faces."""
    tall = [r for r in roofs.values() if r.z > 180.0 and (Vector((r.x, r.y, 0)) - near).length < 450]
    if not tall:
        print("[city] tocho: not found; no projection mapping", flush=True)
        return
    # the tallest cell nearest the Tocho's site, and its neighbours in the same pair of towers only
    near_xy = Vector((near.x, near.y, 0))
    best = min(tall, key=lambda r: (Vector((r.x, r.y, 0)) - near_xy).length - 0.5 * r.z)
    top = best.z
    twin = [r for r in tall if (Vector((r.x, r.y, 0)) - Vector((best.x, best.y, 0))).length < 85 and r.z > top - 60.0]
    x0, x1 = min(r.x for r in twin) - 30, max(r.x for r in twin) + 30
    y0, y1 = min(r.y for r in twin) - 30, max(r.y for r in twin) + 30
    trees = [m.node_tree for m in bpy.data.materials if m.use_nodes and m.node_tree] + list(bpy.data.node_groups)
    n = 0
    for nt in trees:
        for key, v in (("proj_x0", x0), ("proj_x1", x1), ("proj_y0", y0), ("proj_y1", y1)):
            node = nt.nodes.get(key)
            if node is not None:
                node.outputs[0].default_value = v
                n += 1
    print(f"[city] tocho: {len(twin)} roof cells up to {top:.0f} m, mapping x {x0:.0f}..{x1:.0f}, y {y0:.0f}..{y1:.0f} ({n} nodes)", flush=True)


def japan(roofs, road_objs, rnd, cam=Vector((-800.0, -760.0, 0.0))):
    """(10/28, the user: a Night City feel but keep it Japanese) Tall vertical neon signs on the roofs of
    the low and mid blocks round Kabukicho and the station, in generic words; three big kanji floating
    as holograms over the towers; strings of red paper lanterns along a few streets there. All face the
    aerial camera's side (south-west)."""
    cols = [(1.0, 0.1, 0.55), (0.1, 0.9, 1.0), (1.0, 0.25, 0.08), (0.65, 0.25, 1.0), (1.0, 0.75, 0.2)]
    zone = [q for q in roofs.values() if 250 < q.x < 1100 and 100 < q.y < 900 and 12 < q.z < 70]
    zone += [q for q in roofs.values() if -150 < q.x < 450 and -250 < q.y < 350 and 12 < q.z < 60]
    picks = rnd.sample(zone, min(22, len(zone)))
    for k, q in enumerate(picks):
        i = k % len(SIGN_WORDS)
        size = rnd.uniform(10.0, 16.0)  # a character's height: legible from the aerial camera, a kilometre or two off
        height = size * (len(SIGN_WORDS[i]) + 0.2)
        o = _sign_plane(f"jp_sign{k}", SIGNS / f"sign_{i:02d}.png", height, cols[k % len(cols)], rnd.uniform(9.0, 16.0))
        o.location = q + Vector((0, 0, height + 2.0))
        d = cam - q
        o.rotation_euler = (0.0, 0.0, math.atan2(d.y, d.x) + math.radians(90))  # facing the camera
    # red lanterns: strings along road edges in Kabukicho, 4 m up, every 3 m
    import bmesh

    lm = bpy.data.meshes.new("lantern")
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=8, v_segments=6, radius=0.45)
    bm.to_mesh(lm)
    bm.free()
    lm.materials.append(_emit("lantern_red", (1.0, 0.12, 0.04), 30.0))
    n = 0
    for o in road_objs:
        vs = o.data.vertices
        for e in o.data.edges:
            a, b = vs[e.vertices[0]].co.copy(), vs[e.vertices[1]].co.copy()
            if not (250 < a.x < 1100 and 100 < a.y < 900) or (a - b).length < 25 or rnd.random() > 0.35:
                continue
            steps = int((a - b).length / 3.0)
            for i in range(steps):
                ob = bpy.data.objects.new(f"lantern{n}", lm)
                ob.location = a + (b - a) * (i / max(steps, 1)) + Vector((0, 0, 4.0))
                bpy.context.scene.collection.objects.link(ob)
                n += 1
            if n > 3000:
                break
        if n > 3000:
            break
    print(f"[city] japan: {len(picks)} vertical signs, {len(HOLO_WORDS)} floating kanji, {n} lanterns", flush=True)


def hero_robot(scene_npz, take, mjcf, start, along, f0, f1, every=8, step0=64, prefix="hero"):
    """(10/28, the user: keep the focus on the crab) The robot in the street at the end of the drone move:
    its recorded walk (`take`, real time: `every` control steps per frame from `step0`), moved and turned
    so it starts at `start` heading `along`, keyed on frames f0..f1, painted as in the alley. Returns the
    base's position on the last frame."""
    import json

    import numpy as np
    from mathutils import Quaternion

    here = Path(__file__).resolve().parent
    sys.path.insert(0, str(here))
    src = (here / "import_take.py").read_text(encoding="utf-8").rsplit("\nmain()", 1)[0]
    lib = {"__name__": "import_take_lib", "__file__": str(here / "import_take.py")}
    exec(compile(src, "import_take.py", "exec"), lib)  # its helpers, without running its main()
    z = np.load(scene_npz)
    meta = json.loads(str(z["meta"]))
    pos, quat = z["body_pos"][take], z["body_quat"][take]
    q0 = quat[step0, 0]
    yaw0 = math.atan2(2 * (q0[0] * q0[3] + q0[1] * q0[2]), 1 - 2 * (q0[2] ** 2 + q0[3] ** 2))
    dyaw = math.atan2(along.y, along.x) - yaw0
    c, s_ = math.cos(dyaw), math.sin(dyaw)
    qz = Quaternion((math.cos(dyaw / 2), 0.0, 0.0, math.sin(dyaw / 2)))
    p0 = pos[step0, 0]
    geoms = lib["visual_geoms"](Path(mjcf))
    last = None
    for b, name in enumerate(meta["body_names"]):
        holder = bpy.data.objects.new(f"{prefix}:{name}", None)
        bpy.context.scene.collection.objects.link(holder)
        holder.rotation_mode = "QUATERNION"
        for mesh_file, rgba, gpos, gquat in geoms.get(name, []):
            bpy.ops.wm.stl_import(filepath=str(mesh_file))
            part = bpy.context.selected_objects[0]
            part.name = f"{prefix}:{name}:mesh"
            part.data.materials.append(lib["material"](rgba))
            part.parent = holder
            part.location = gpos
            part.rotation_mode = "QUATERNION"
            part.rotation_quaternion = gquat
        for f in range(f0, f1 + 1):
            st = step0 + (f - f0) * every
            rx, ry = pos[st, b, 0] - p0[0], pos[st, b, 1] - p0[1]
            holder.location = (start.x + c * rx - s_ * ry, start.y + s_ * rx + c * ry, start.z + float(pos[st, b, 2]))
            holder.rotation_quaternion = qz @ Quaternion(tuple(float(v) for v in quat[st, b]))
            holder.keyframe_insert("location", frame=f)
            holder.keyframe_insert("rotation_quaternion", frame=f)
        if b == 0:
            last = Vector(holder.location)
    import photoreal

    photoreal.upgrade_robot(prefix)
    # a street lamp's pool of light on it (the alley's key, as the city's lamps are too far apart)
    key = bpy.data.lights.new(f"{prefix}_streetlight", "AREA")
    key.energy, key.size, key.color = 60.0, 1.2, (1.0, 0.82, 0.62)
    kl = bpy.data.objects.new(f"{prefix}_streetlight", key)
    bpy.context.scene.collection.objects.link(kl)
    kl.location = last + Vector((0.0, 0.0, 3.5)) + along * 0.6
    print(f"[city] hero: the take's walk, steps {step0}-{step0 + (f1 - f0) * every}, on frames {f0}-{f1}", flush=True)
    return last


def _pavers(name, w, h, cols, dots=False):
    """Paving blocks w x h metres in world XY: dark joints in relief, each block its own shade, a fine
    grain; `dots` adds the truncated domes of tactile paving."""
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    b = next(n for n in nt.nodes if n.type == "BSDF_PRINCIPLED")
    geo = nt.nodes.new("ShaderNodeNewGeometry")
    brick = nt.nodes.new("ShaderNodeTexBrick")
    brick.offset = 0.5
    brick.inputs["Scale"].default_value = 1.0
    brick.inputs["Brick Width"].default_value = w
    brick.inputs["Row Height"].default_value = h
    brick.inputs["Mortar Size"].default_value = 0.006
    brick.inputs["Color1"].default_value = (*cols[0], 1)
    brick.inputs["Color2"].default_value = (*cols[1], 1)
    brick.inputs["Mortar"].default_value = (0.05, 0.05, 0.05, 1)
    nt.links.new(geo.outputs["Position"], brick.inputs["Vector"])
    grain = _node(nt, "ShaderNodeTexNoise", Scale=30.0, Detail=10.0)
    nt.links.new(geo.outputs["Position"], grain.inputs["Vector"])
    shade = nt.nodes.new("ShaderNodeMix")
    shade.data_type, shade.blend_type = "RGBA", "MULTIPLY"
    shade.inputs["Factor"].default_value = 0.35
    nt.links.new(brick.outputs["Color"], shade.inputs["A"])
    nt.links.new(grain.outputs["Color"], shade.inputs["B"])
    nt.links.new(shade.outputs["Result"], b.inputs["Base Color"])
    b.inputs["Roughness"].default_value = 0.55
    height = _math(nt, "ADD", _math(nt, "SUBTRACT", None, brick.outputs["Fac"], va=1.0), _math(nt, "MULTIPLY", grain.outputs["Fac"], None, vb=0.3))
    if dots:
        vor = nt.nodes.new("ShaderNodeTexVoronoi")
        vor.inputs["Scale"].default_value = 1.0 / 0.05
        nt.links.new(geo.outputs["Position"], vor.inputs["Vector"])
        dome = nt.nodes.new("ShaderNodeMapRange")
        nt.links.new(vor.outputs["Distance"], dome.inputs["Value"])
        dome.inputs["From Min"].default_value, dome.inputs["From Max"].default_value = 0.35, 0.2
        height = _math(nt, "ADD", height, _math(nt, "MULTIPLY", dome.outputs["Result"], None, vb=1.5))
    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.7
    bump.inputs["Distance"].default_value = 0.004
    nt.links.new(height, bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], b.inputs["Normal"])
    return m


def hero_patch(street, along, half, yaw, length=(14.0, 44.0)):
    """The asphalt where the camera comes down to the robot, truly displaced (adaptive subdivision): the
    scanned surface's aggregate and ruts in relief, as a lens a few centimetres off the ground sees it."""
    cache = Path("/content/ph_cache")
    if not (cache / "credits.json").exists():
        return
    import photoreal

    maps = photoreal.fetch(cache)["floor"]
    mat = photoreal.pbr("hero_asphalt", maps, tile=2.5, space=None, tint=(0.55, 0.55, 0.58), bump=0.8, wet=0.7)
    nt = mat.node_tree
    out = next(n for n in nt.nodes if n.type == "OUTPUT_MATERIAL")
    tex = next(n for n in nt.nodes if n.type == "TEX_IMAGE" and n.image and "disp" in n.image.name)
    disp = nt.nodes.new("ShaderNodeDisplacement")
    disp.inputs["Scale"].default_value = 0.025
    disp.inputs["Midlevel"].default_value = 0.5
    nt.links.new(tex.outputs["Color"], disp.inputs["Height"])
    nt.links.new(disp.outputs["Displacement"], out.inputs["Displacement"])
    if hasattr(mat, "displacement_method"):
        mat.displacement_method = "BOTH"
    elif hasattr(mat.cycles, "displacement_method"):
        mat.cycles.displacement_method = "BOTH"
    if hasattr(mat, "max_vertex_displacement"):
        mat.max_vertex_displacement = 0.03
    side = Vector((-along.y, along.x, 0.0))
    L0, L1 = length
    me = bpy.data.meshes.new("hero_asphalt")
    corners = [street + along * L0 - side * half, street + along * L1 - side * half,
               street + along * L1 + side * half, street + along * L0 + side * half]
    me.from_pydata([tuple(c + Vector((0, 0, 0.004))) for c in corners], [], [(0, 1, 2, 3)])
    me.materials.append(mat)
    o = bpy.data.objects.new("hero_asphalt", me)
    bpy.context.scene.collection.objects.link(o)
    sub_ = o.modifiers.new("dice", "SUBSURF")
    sub_.subdivision_type = "SIMPLE"
    sub_.levels = sub_.render_levels = 6
    if hasattr(sub_, "use_adaptive_subdivision"):
        sub_.use_adaptive_subdivision = True
        if hasattr(sub_, "adaptive_pixel_size"):
            sub_.adaptive_pixel_size = 1.0


def street_dressing(street, along, rnd, half=4.6, length=(-50.0, 70.0)):
    """(10/28, the user: detail wherever it can go) The street the drone lands in, dressed as a Japanese
    street at night: concrete utility poles with crossarms, transformers and sagging wires; white road
    lines (a dashed centre line, solid edge lines) and manholes on the wet asphalt; glowing vending
    machines; projecting signs with generic words on both sides. +along is down the street."""
    import bmesh

    side = Vector((-along.y, along.x, 0.0))
    yaw = math.atan2(along.y, along.x)
    col = bpy.data.collections.new("street_dressing")
    bpy.context.scene.collection.children.link(col)

    def put(me, loc, rot_z=0.0, name="dress"):
        o = bpy.data.objects.new(name, me)
        col.objects.link(o)
        o.location = loc
        o.rotation_euler = (0.0, 0.0, rot_z)
        return o

    def box(name, sx, sy, sz, mat):
        me = bpy.data.meshes.new(name)
        bm = bmesh.new()
        bmesh.ops.create_cube(bm, size=1.0)
        bmesh.ops.scale(bm, vec=(sx, sy, sz), verts=bm.verts)
        bmesh.ops.translate(bm, vec=(0, 0, sz / 2), verts=bm.verts)
        bm.to_mesh(me)
        bm.free()
        me.materials.append(mat)
        return me

    concrete = _plain("pole_concrete", (0.35, 0.34, 0.32), 0.7)
    dark = _plain("pole_metal", (0.05, 0.05, 0.055), 0.4)
    white = _plain("road_paint", (0.75, 0.75, 0.72), 0.35)
    iron = _plain("manhole_iron", (0.06, 0.055, 0.05), 0.3)
    # the poles: 12 m, every ~28 m on alternate sides, crossarms and a transformer can on some
    pole = bpy.data.meshes.new("utility_pole")
    bm = bmesh.new()
    bmesh.ops.create_cone(bm, cap_ends=True, segments=16, radius1=0.17, radius2=0.12, depth=12.0,
                          matrix=Matrix.Translation((0, 0, 6.0)))
    bm.to_mesh(pole)
    bm.free()
    pole.materials.append(concrete)
    arm = box("crossarm", 0.12, 1.8, 0.12, dark)
    can = bpy.data.meshes.new("transformer")
    bm = bmesh.new()
    bmesh.ops.create_cone(bm, cap_ends=True, segments=12, radius1=0.32, radius2=0.32, depth=1.1)
    bm.to_mesh(can)
    bm.free()
    can.materials.append(_plain("transformer_grey", (0.42, 0.44, 0.45), 0.5))
    tops = []
    t = length[0]
    k = 0
    while t < length[1]:
        sgn = 1 if k % 2 == 0 else -1
        base = street + along * t + side * (half + 0.6) * sgn
        put(pole, base, yaw, f"pole{k}")
        for h in (10.6, 9.6):
            put(arm, base + Vector((0, 0, h)), yaw + math.pi / 2 * 0, f"arm{k}_{h}")
        if k % 3 == 1:
            put(can, base + side * 0.45 * sgn + Vector((0, 0, 8.2)), 0.0, f"can{k}")
        tops.append((base, sgn))
        t += 28.0 + rnd.uniform(-3, 3)
        k += 1
    # wires: three per span, sagging, between poles on the same side and across
    wire_mat = _plain("wire", (0.01, 0.01, 0.01), 0.5)
    n_w = 0
    for (a, sa), (b, sb) in zip(tops, tops[2:]):  # along each side (none across the drone's way down)
        for h, off in ((10.65, -0.7), (10.65, 0.7), (9.65, 0.0)):
            p0 = a + Vector((0, 0, h)) + side * off * 0.3
            p1 = b + Vector((0, 0, h)) + side * off * 0.3
            cu = bpy.data.curves.new(f"wire{n_w}", "CURVE")
            cu.dimensions, cu.bevel_depth, cu.bevel_resolution = "3D", 0.012, 2
            sp = cu.splines.new("POLY")
            pts = 12
            sp.points.add(pts - 1)
            for i in range(pts):
                u = i / (pts - 1)
                q = p0.lerp(p1, u) - Vector((0, 0, 0.9 * 4 * u * (1 - u)))
                sp.points[i].co = (q.x, q.y, q.z, 1.0)
            cu.materials.append(wire_mat)
            o = bpy.data.objects.new(f"wire{n_w}", cu)
            col.objects.link(o)
            n_w += 1
    # road lines on the wet asphalt (a hair above it), manholes
    for i in range(int((length[1] - length[0]) / 10)):
        t = length[0] + i * 10.0
        put(box(f"dash{i}", 5.0, 0.15, 0.006, white), street + along * (t + 2.5) + Vector((0, 0, 0.004)), yaw, f"dash{i}")
    for sgn in (-1, 1):
        line = box(f"edge{sgn}", length[1] - length[0], 0.15, 0.006, white)
        put(line, street + along * ((length[0] + length[1]) / 2) + side * (half - 0.4) * sgn + Vector((0, 0, 0.004)), yaw)
    disc = bpy.data.meshes.new("manhole")
    bm = bmesh.new()
    bmesh.ops.create_cone(bm, cap_ends=True, segments=32, radius1=0.32, radius2=0.32, depth=0.012)
    bm.to_mesh(disc)
    bm.free()
    disc.materials.append(iron)
    for t in (12.0, 33.0, 52.0):
        put(disc, street + along * t + side * rnd.uniform(-1.5, 1.5) + Vector((0, 0, 0.004)), 0.0, "manhole")
    # vending machines, glowing, against the walls' line
    vend_face = _emit("vending_face", (0.75, 0.88, 1.0), 6.0)
    vend_body = _plain("vending_body", (0.6, 0.62, 0.65), 0.4)
    for j, t in enumerate((8.0, 21.0, 46.0)):
        sgn = -1 if j % 2 else 1
        loc = street + along * t + side * (half + 1.6) * sgn
        put(box(f"vend{j}", 0.95, 0.8, 1.85, vend_body), loc, yaw + (math.pi / 2 if sgn > 0 else -math.pi / 2), f"vend{j}")
        put(box(f"vendface{j}", 0.8, 0.02, 1.2, vend_face), loc - side * 0.41 * sgn + Vector((0, 0, 0.5)), yaw, f"vendface{j}")
    # projecting signs (袖看板) at 3-6 m on both sides, generic words, lit
    cols = [(1.0, 0.92, 0.75), (1.0, 0.2, 0.5), (0.2, 0.9, 1.0), (1.0, 0.5, 0.15)]
    for j in range(10):
        sgn = 1 if j % 2 == 0 else -1
        t = rnd.uniform(length[0] + 10, length[1] - 5)
        i = j % len(SIGN_WORDS)
        o = _sign_plane(f"street_sign{j}", SIGNS / f"sign_{i:02d}.png", 0.55 * (len(SIGN_WORDS[i]) + 0.2), cols[j % 4],
                        rnd.uniform(5.0, 9.0))
        o.location = street + along * t + side * (half + 1.2) * sgn + Vector((0, 0, rnd.uniform(4.5, 6.5)))
        o.rotation_euler = (0.0, 0.0, yaw + math.pi / 2)  # edge-on to the street, facing up and down it
    # (10/28, the user: the asphalt reads as a board, not a road) what makes it a road: kerbs, raised
    # pavements, a zebra crossing, tyre tracks and patched asphalt, gutter grates, white guardrails
    L0, L1 = length
    mid = (L0 + L1) / 2
    run = L1 - L0
    cache = Path("/content/ph_cache")
    pave = _pavers("pavement", 0.3, 0.3, ((0.3, 0.29, 0.27), (0.4, 0.38, 0.35)))
    tactile = _pavers("tactile", 0.3, 0.3, ((0.72, 0.52, 0.06), (0.78, 0.58, 0.08)), dots=True)
    kerb = _plain("kerb_granite", (0.36, 0.36, 0.35), 0.55)
    for sgn in (-1, 1):
        o = put(box(f"pavement{sgn}", run, 3.2, 0.15, pave), street + along * mid + side * (half + 1.6) * sgn, yaw, f"pavement{sgn}")
        put(box(f"tactile{sgn}", run, 0.3, 0.155, tactile), street + along * mid + side * (half + 1.3) * sgn, yaw, f"tactile{sgn}")
        o = put(box(f"kerb{sgn}", run, 0.2, 0.17, kerb), street + along * mid + side * (half + 0.1) * sgn, yaw, f"kerb{sgn}")
        bev = o.modifiers.new("bevel", "BEVEL")
        bev.width, bev.segments = 0.02, 2
    hero_patch(street, along, half, yaw)
    # gutter grates along both kerbs
    grate_mat = _plain("grate_iron", (0.02, 0.02, 0.02), 0.45)
    for i in range(int(run / 12)):
        for sgn in (-1, 1):
            put(box(f"grate{i}{sgn}", 0.9, 0.35, 0.008, grate_mat), street + along * (L0 + 6 + 12 * i) + side * (half - 0.25) * sgn
                + Vector((0, 0, 0.003)), yaw, "grate")
    # the zebra crossing just beyond the robot, across the whole road
    zebra_at = 36.0
    n_str = int(2 * half / 0.9)
    for i in range(n_str):
        off = -half + 0.45 + i * 0.9
        put(box(f"zebra{i}", 3.0, 0.45, 0.006, white), street + along * zebra_at + side * off + Vector((0, 0, 0.005)), yaw, "zebra")
    put(box("stopline", 0.45, 2 * half, 0.006, white), street + along * (zebra_at - 2.4) + Vector((0, 0, 0.005)), yaw, "stopline")
    # tyre tracks (darker, polished by tyres) and patched asphalt
    track = _plain("tyre_track", (0.018, 0.018, 0.02), 0.12)
    for off in (-3.4, -1.6, 1.6, 3.4):
        put(box(f"track{off}", run, 0.55, 0.002, track), street + along * mid + side * off + Vector((0, 0, 0.002)), yaw, "track")
    patch = _plain("asphalt_patch", (0.03, 0.03, 0.032), 0.55)
    for j in range(5):
        t = rnd.uniform(L0 + 8, L1 - 8)
        put(box(f"patch{j}", rnd.uniform(1.5, 4.0), rnd.uniform(0.8, 2.2), 0.003, patch),
            street + along * t + side * rnd.uniform(-half + 1, half - 1) + Vector((0, 0, 0.0025)), yaw + rnd.uniform(-0.05, 0.05), "patch")
    # white guardrails on the pavement edge, in runs with gaps, and their posts
    rail_mat = _plain("guardrail_white", (0.8, 0.8, 0.78), 0.35)
    n_r = 0
    for sgn in (-1, 1):
        t = L0 + 4
        while t < L1 - 6:
            seg = rnd.uniform(8, 16)
            for h in (0.55, 0.85):
                put(box(f"rail{n_r}", seg, 0.05, 0.07, rail_mat), street + along * (t + seg / 2) + side * (half + 0.45) * sgn
                    + Vector((0, 0, 0.15 + h)), yaw, "rail")
            for q in range(int(seg / 2) + 1):
                put(box(f"post{n_r}{q}", 0.06, 0.06, 0.95, rail_mat), street + along * (t + 2 * q) + side * (half + 0.45) * sgn
                    + Vector((0, 0, 0.15)), yaw, "railpost")
            n_r += 1
            t += seg + rnd.uniform(4, 9)
    print(f"[city] street dressing: {k} poles, {n_w} wires, kerbs and pavements, a zebra crossing, tracks, patches, "
          f"{n_r} guardrail runs, road lines, manholes, 3 vending machines, 10 signs", flush=True)


def air_car(cam, frame=95, frames=200):
    """A flying car crossing close in front of the camera mid-dive (the user's references): a dark
    streamlined body, a glass canopy, four ducted fans ringed in blue, tail lights and headlights."""
    import bmesh

    scene = bpy.context.scene
    scene.frame_set(frame)
    c_pos = cam.matrix_world.translation.copy()
    fwd = (cam.matrix_world.to_3x3() @ Vector((0.0, 0.0, -1.0))).normalized()
    right = (cam.matrix_world.to_3x3() @ Vector((1.0, 0.0, 0.0))).normalized()
    centre = c_pos + fwd * 38.0 - Vector((0, 0, 4.0))

    def ell(name, sx, sy, sz, mat, loc, parent):
        me = bpy.data.meshes.new(name)
        bm = bmesh.new()
        bmesh.ops.create_uvsphere(bm, u_segments=24, v_segments=12, radius=1.0)
        bmesh.ops.scale(bm, vec=(sx, sy, sz), verts=bm.verts)
        bm.to_mesh(me)
        bm.free()
        for f_ in me.polygons:
            f_.use_smooth = True
        me.materials.append(mat)
        o = bpy.data.objects.new(name, me)
        scene.collection.objects.link(o)
        o.parent, o.location = parent, loc
        return o

    root = bpy.data.objects.new("air_car", None)
    scene.collection.objects.link(root)
    paint = bpy.data.materials.new("air_car_paint")
    paint.use_nodes = True
    bsdf = next(n for n in paint.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
    bsdf.inputs["Base Color"].default_value = (0.02, 0.025, 0.03, 1)
    bsdf.inputs["Metallic"].default_value, bsdf.inputs["Roughness"].default_value = 0.8, 0.15
    glass = _plain("air_car_glass", (0.01, 0.02, 0.03), 0.04)
    ring = _emit("air_car_ring", (0.2, 0.8, 1.0), 25.0)
    tail = _emit("air_car_tail", (1.0, 0.03, 0.02), 60.0)
    head = _emit("air_car_head", (1.0, 0.95, 0.85), 120.0)
    ell("air_car_body", 2.6, 1.05, 0.55, paint, (0, 0, 0), root)
    ell("air_car_canopy", 1.2, 0.75, 0.42, glass, (0.3, 0, 0.35), root)
    for dx in (-1.9, 1.9):
        for dy in (-1.45, 1.45):
            me = bpy.data.meshes.new("air_car_fan")
            bm = bmesh.new()
            bmesh.ops.create_cone(bm, cap_ends=False, segments=24, radius1=0.65, radius2=0.65, depth=0.22)
            bm.to_mesh(me)
            bm.free()
            me.materials.append(ring)
            o = bpy.data.objects.new("air_car_fan", me)
            scene.collection.objects.link(o)
            o.parent, o.location = root, (dx, dy, 0.05)
    ell("air_car_tail_l", 0.08, 0.5, 0.05, tail, (-2.55, 0, 0.1), root)
    ell("air_car_head_l", 0.06, 0.35, 0.05, head, (2.6, 0, 0.0), root)
    v = 42.0  # metres a second across the view
    for f in (frame - 30, frame + 30):
        root.location = centre + right * v * (f - frame) / 25.0 * (-1)
        root.rotation_euler = (0.08, 0.0, math.atan2(-right.y, -right.x))
        root.keyframe_insert("location", frame=f)
        root.keyframe_insert("rotation_euler", frame=f)
    print(f"[city] air car crosses 38 m ahead of the camera around frame {frame}", flush=True)


def street_lanterns_and_steam(street, along, half=4.6, length=(-30.0, 60.0)):
    """Red lanterns strung along both pavements, and steam rising from three grates near the robot."""
    import bmesh

    side = Vector((-along.y, along.x, 0.0))
    lm = bpy.data.meshes.new("street_lantern")
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=12, v_segments=8, radius=0.22)
    bmesh.ops.scale(bm, vec=(1.0, 1.0, 1.35), verts=bm.verts)
    bm.to_mesh(lm)
    bm.free()
    lm.materials.append(bpy.data.materials.get("lantern_red") or _emit("lantern_red", (1.0, 0.12, 0.04), 30.0))
    n = 0
    for sgn in (-1, 1):
        t = length[0]
        while t < length[1]:
            o = bpy.data.objects.new("street_lantern", lm)
            bpy.context.scene.collection.objects.link(o)
            o.location = street + along * t + side * (half + 2.6) * sgn + Vector((0, 0, 3.3 + 0.15 * math.sin(t)))
            t += 1.6
            n += 1
    steam = bpy.data.materials.new("steam")
    steam.use_nodes = True
    snt = steam.node_tree
    for nd in list(snt.nodes):
        snt.nodes.remove(nd)
    sv = snt.nodes.new("ShaderNodeVolumePrincipled")
    sv.inputs["Color"].default_value = (0.8, 0.8, 0.85, 1)
    tex = snt.nodes.new("ShaderNodeTexNoise")
    tex.noise_dimensions = "4D"
    tex.inputs["Scale"].default_value = 1.6
    tex.inputs["Detail"].default_value = 6.0
    tc = snt.nodes.new("ShaderNodeTexCoord")
    snt.links.new(tc.outputs["Object"], tex.inputs["Vector"])
    w_in = tex.inputs["W"]
    for f, v in ((1, 0.0), (220, 6.0)):
        w_in.default_value = v
        w_in.keyframe_insert("default_value", frame=f)
    grad = snt.nodes.new("ShaderNodeSeparateXYZ")
    snt.links.new(tc.outputs["Object"], grad.inputs[0])
    fade = snt.nodes.new("ShaderNodeMapRange")
    snt.links.new(grad.outputs["Z"], fade.inputs["Value"])
    fade.inputs["From Min"].default_value, fade.inputs["From Max"].default_value = 0.0, 1.0
    fade.inputs["To Min"].default_value, fade.inputs["To Max"].default_value = 1.0, 0.0
    dens = snt.nodes.new("ShaderNodeMapRange")
    snt.links.new(tex.outputs["Fac"], dens.inputs["Value"])
    dens.inputs["From Min"].default_value, dens.inputs["From Max"].default_value = 0.45, 0.75
    dens.inputs["To Max"].default_value = 1.6
    mul = snt.nodes.new("ShaderNodeMath")
    mul.operation = "MULTIPLY"
    snt.links.new(dens.outputs["Result"], mul.inputs[0])
    snt.links.new(fade.outputs["Result"], mul.inputs[1])
    snt.links.new(mul.outputs[0], sv.inputs["Density"])
    so = snt.nodes.new("ShaderNodeOutputMaterial")
    snt.links.new(sv.outputs[0], so.inputs["Volume"])
    for j, (t, off) in enumerate(((20.0, -3.5), (34.0, 2.8), (46.0, -1.2))):
        me = bpy.data.meshes.new(f"steam{j}")
        bm = bmesh.new()
        bmesh.ops.create_cone(bm, cap_ends=True, segments=16, radius1=0.6, radius2=2.2, depth=1.0,
                              matrix=Matrix.Translation((0, 0, 0.5)))
        bm.to_mesh(me)
        bm.free()
        me.materials.append(steam)
        o = bpy.data.objects.new(f"steam{j}", me)
        bpy.context.scene.collection.objects.link(o)
        o.location = street + along * t + side * off
        o.scale = (1.0, 1.0, 5.5)
    print(f"[city] street: {n} lanterns, 3 steam vents", flush=True)


def drone_shot(frames=200, lens=24.0, hero=None):
    """(10/28, the user: closer to the buildings, and from the ground looking up, seamless like drone
    footage) One continuous move: high over the west side, down among the towers, past their flanks,
    into a street between them and on along it at 3 m, the camera tilting up to the towers. The path
    is checked against the buildings themselves (a BVH of the tiles): every frame keeps 8 m clear, and
    the street is chosen where the air is open from 150 m down to the road. Fast while high, slow near
    the ground, as a drone pilot flies a reveal."""
    import bmesh
    from mathutils.bvhtree import BVHTree

    towers, road_objs, roofs = globals()["_DRONE_INPUT"]
    bm = bmesh.new()
    for o in towers + [o for o in bpy.data.objects if o.type == "MESH" and o.name.startswith(("hw_", "mega_"))]:
        bm.from_mesh(o.data)
    tree = BVHTree.FromBMesh(bm)

    def clear(q):  # distance to the nearest building surface; 0 inside one (an upward ray meets a roof from below)
        up = tree.ray_cast(q, Vector((0.0, 0.0, 1.0)))
        if up[0] is not None and up[1].z > 0.0:
            return 0.0
        hit = tree.find_nearest(q)
        return hit[3] if hit and hit[0] is not None else 1e9

    tall = [r for r in roofs.values() if r.z > 140.0]
    c = sum(tall, Vector((0, 0, 0))) / max(len(tall), 1)
    best = None
    for o in road_objs:
        vs = o.data.vertices
        for e in o.data.edges:
            a, b = vs[e.vertices[0]].co, vs[e.vertices[1]].co
            if (a - b).length < 40.0:
                continue
            m = (a + b) / 2
            if (Vector((m.x, m.y, 0)) - Vector((c.x, c.y, 0))).length > 420:
                continue
            d = (b - a).normalized()
            side = Vector((-d.y, d.x, 0.0))
            for off in (7.0, -7.0, 11.0, -11.0):
                q = m + side * off
                if min(clear(q + Vector((0, 0, h))) for h in (3.0, 12.0, 30.0, 60.0, 100.0, 150.0)) < 9.0:
                    continue
                if min(clear(q + d * t + Vector((0, 0, 3.0))) for t in (10.0, 20.0, 30.0, 40.0)) < 6.0:
                    continue
                score = sum(1 for r in tall if (Vector((r.x, r.y, 0)) - Vector((q.x, q.y, 0))).length < 260)
                if best is None or score > best[0]:
                    best = (score, q.copy(), d.copy())
    if best is None:
        print("[city] drone: no open street found; keeping the aerial move", flush=True)
        return False
    _score, street, along = best
    if (Vector((street.x, street.y, 0)) - Vector((-900.0, -760.0, 0))).dot(along) < 0:
        along = -along  # fly on along the street away from where we came from
    A = Vector((-950.0, -800.0, 280.0))
    toward = Vector((street.x - A.x, street.y - A.y, 0.0)).normalized()
    B = street - toward * 260.0 + Vector((0, 0, 0))
    B.z = street.z + 150.0
    D = street + Vector((0, 0, 45.0)) - along * 25.0
    E1 = street + Vector((0, 0, 3.0))
    E2 = street + along * 36.0 + Vector((0, 0, 3.0))
    hero_end = None
    if hero is not None:  # land behind the robot: it walks from 30 m down the street, the camera low behind it
        side = Vector((-along.y, along.x, 0.0))
        r0 = street + along * 30.0 + side * 1.4  # in the lane, clear of the centre line
        hero_end = hero_robot(*hero, start=r0, along=along, f0=frames - 28, f1=frames)
        street_dressing(street, along, random.Random(9))
        street_lanterns_and_steam(street, along)
        E2 = hero_end - along * 1.1 + side * 0.3 + Vector((0, 0, 0.3))
        E1 = street + along * 18.0 + Vector((0, 0, 2.5))
    pts = [A, B, D, E1, E2]

    def catmull(ps, n):
        ps = [ps[0]] + ps + [ps[-1]]
        out = []
        for i in range(1, len(ps) - 2):
            p0, p1, p2, p3 = ps[i - 1], ps[i], ps[i + 1], ps[i + 2]
            for k in range(n):
                t = k / n
                out.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t * t
                                  + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3))
        out.append(ps[-2])
        return out

    dense = catmull(pts, 400)
    for _ in range(40):  # lift any part of the flight that comes within 8 m of a building
        moved = False
        for i, q in enumerate(dense[:-90]):
            if clear(q) < 8.0:
                dense[i] = q + Vector((0, 0, 10.0))
                moved = True
        if not moved:
            break
    seg = [0.0]
    for i in range(1, len(dense)):
        seg.append(seg[-1] + (dense[i] - dense[i - 1]).length)
    total = seg[-1]

    def at(sv):
        import bisect

        j = min(max(bisect.bisect_left(seg, sv), 1), len(seg) - 1)
        f = (sv - seg[j - 1]) / max(seg[j] - seg[j - 1], 1e-6)
        return dense[j - 1].lerp(dense[j], f)

    look_far = Vector((1600.0, 1300.0, -60.0))
    up_at = max((r for r in tall if (Vector((r.x, r.y, 0)) - Vector((E2.x, E2.y, 0))).length < 300),
                key=lambda r: r.z - 0.3 * (Vector((r.x, r.y, 0)) - Vector((E2.x, E2.y, 0))).length, default=c)
    scene = bpy.context.scene
    cam = scene.camera
    cam.data.lens = lens
    cam.data.clip_start = 0.1
    cam.data.clip_end = 100000.0  # the far plane and skyline to 30 km
    cam.animation_data_clear()
    cam.constraints.clear()
    target = bpy.data.objects.get("aerial_target")
    target.animation_data_clear()
    c_ = cam.constraints.new("TRACK_TO")
    c_.target, c_.track_axis, c_.up_axis = target, "TRACK_NEGATIVE_Z", "UP_Y"
    for f in range(1, frames + 1):
        u = (f - 1) / (frames - 1)
        e = 1 - (1 - u) ** 2.4  # fast high up, slowing into the street
        pos = at(total * e)
        cam.location = pos
        cam.keyframe_insert("location", frame=f)
        # where it looks: the far city, then down into the street ahead, then up at the towers
        ahead = at(min(total, total * e + 60.0))
        if u < 0.35:
            w = u / 0.35
            look = look_far.lerp(street, w * w)
        elif u < 0.75:
            w = (u - 0.35) / 0.4
            look = street.lerp(ahead + Vector((0, 0, 4.0)), w)
        else:
            w = (u - 0.75) / 0.25
            w = w * w * (3 - 2 * w)
            end = up_at + Vector((0, 0, -20.0)) if hero_end is None else hero_end + along * 1.2 + Vector((0, 0, 0.33))
            look = (ahead + Vector((0, 0, 4.0))).lerp(end, w)
        target.location = look
        target.keyframe_insert("location", frame=f)
    if hero_end is not None:  # the landing: focus pulled onto the robot, the lens opened so the street melts to bokeh
        cam.data.dof.use_dof = True
        cam.data.dof.focus_object = bpy.data.objects["hero:base_link"]
        for f, stop in ((1, 22.0), (frames - 34, 22.0), (frames - 14, 1.4), (frames, 1.4)):
            cam.data.dof.aperture_fstop = stop
            cam.data.dof.keyframe_insert("aperture_fstop", frame=f)
    scene.frame_start, scene.frame_end = 1, frames
    print(f"[city] drone: {total:.0f} m in {frames} frames, street at ({street.x:.0f}, {street.y:.0f}), "
          f"looking up at a {up_at.z + 35:.0f} m roof; nearest building on the path {min(clear(q) for q in dense[:-60]):.1f} m",
          flush=True)
    return True


def _paving_material():
    """Pavement: grey stone slabs in a staggered bond with darker joints, a little worn and damp."""
    m = bpy.data.materials.new("city_paving")
    m.use_nodes = True
    nt = m.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    geo = nt.nodes.new("ShaderNodeNewGeometry")
    brick = _node(nt, "ShaderNodeTexBrick", Scale=1.0, **{"Mortar Size": 0.02, "Brick Width": 0.6, "Row Height": 0.3})
    brick.offset = 0.5
    nt.links.new(geo.outputs["Position"], brick.inputs["Vector"])
    brick.inputs["Color1"].default_value = (0.25, 0.25, 0.24, 1.0)
    brick.inputs["Color2"].default_value = (0.19, 0.19, 0.185, 1.0)
    brick.inputs["Mortar"].default_value = (0.07, 0.07, 0.07, 1.0)
    bsdf = _node(nt, "ShaderNodeBsdfPrincipled", Roughness=0.55)
    nt.links.new(brick.outputs["Color"], bsdf.inputs["Base Color"])
    nt.links.new(bsdf.outputs[0], out.inputs["Surface"])
    return m


def road_dressing(road_objs, center, radius=1600.0, rnd=None):
    """(10/28 evening, the user: from the air the even asphalt read as black tiles -- give the streets
    pavements, white lines, paving stones and objects) From PLATEAU's road polygons: their outline
    (the kerb), the road's width across from every kerb edge (a ray to the far kerb), and from that
    raised paved footways with a kerb on roads 9 m or wider, the carriageway's edge line, dashed lane
    lines on wide roads, a centre line (white dashed, or solid yellow on the widest), and street trees
    along the wide ones. Returns the lanes [(a, b, inward normal, width, footway)] for the traffic."""
    import bmesh
    from mathutils.bvhtree import BVHTree

    rnd = rnd or random.Random(17)
    bm = bmesh.new()
    for o in road_objs:
        me = o.data.copy()
        me.transform(o.matrix_world)
        bm.from_mesh(me)
        bpy.data.meshes.remove(me)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=0.05)
    bm.edges.ensure_lookup_table()
    kerbs = []
    for e in bm.edges:
        if not e.is_boundary or len(e.link_faces) != 1:
            continue
        a, b = e.verts[0].co.copy(), e.verts[1].co.copy()
        mid = (a + b) / 2
        if (Vector((mid.x, mid.y, 0)) - Vector((center.x, center.y, 0))).length > radius or (b - a).length < 1.0:
            continue
        d = (b - a).normalized()
        n = Vector((-d.y, d.x, 0.0))
        fc = e.link_faces[0].calc_center_median()
        if n.dot(fc - mid) < 0:
            n = -n
        kerbs.append((a, b, n))
    # the far kerb across: rays against the outline raised into 2 m walls
    verts, polys = [], []
    for a, b, _n in kerbs:
        k = len(verts)
        verts += [a - Vector((0, 0, 1)), b - Vector((0, 0, 1)), b + Vector((0, 0, 1)), a + Vector((0, 0, 1))]
        polys.append((k, k + 1, k + 2, k + 3))
    walls = BVHTree.FromPolygons(verts, polys) if polys else None
    lanes = []
    mats = {"pave": _paving_material(), "white": _plain("road_line_white", (0.75, 0.75, 0.72), 0.5),
            "yellow": _plain("road_line_yellow", (0.8, 0.55, 0.08), 0.5), "kerb": _plain("kerb_stone", (0.42, 0.42, 0.4), 0.6)}
    geo = {k: bmesh.new() for k in mats}

    def quad(kind, p0, p1, n, off0, off1, z):
        g = geo[kind]
        vs = [g.verts.new(p0 + n * off0 + Vector((0, 0, z))), g.verts.new(p1 + n * off0 + Vector((0, 0, z))),
              g.verts.new(p1 + n * off1 + Vector((0, 0, z))), g.verts.new(p0 + n * off1 + Vector((0, 0, z)))]
        g.faces.new(vs)

    def dashed(kind, a, b, n, off, w, z, on=4.0, off_len=6.0):
        d = b - a
        length = d.length
        d = d / max(length, 1e-6)
        s = rnd.uniform(0, on + off_len)
        while s < length:
            e = min(s + on, length)
            quad(kind, a + d * s, a + d * e, n, off - w / 2, off + w / 2, z)
            s = e + off_len

    trees = []
    for a, b, n in kerbs:
        mid = (a + b) / 2
        hit = walls.ray_cast(mid + n * 0.3 + Vector((0, 0, 0.0)), n, 60.0) if walls else (None,)
        width = (hit[3] + 0.3) if hit[0] is not None else 8.0
        sw = min(max(width * 0.16, 2.0), 4.0) if width >= 9.0 else 0.0
        if sw:
            quad("pave", a, b, n, 0.0, sw, 0.15)  # the footway, raised
            quad("kerb", a, b, n, sw - 0.15, sw, 0.16)  # its kerb stones
            quad("white", a, b, n, sw + 0.3, sw + 0.45, 0.02)  # the carriageway's edge line
        carriage = width - 2 * sw
        if carriage >= 5.5 and (n.x > 1e-6 or (abs(n.x) <= 1e-6 and n.y > 0)):  # the centre line, from one kerb only
            if carriage >= 16.0:
                quad("yellow", a, b, n, width / 2 - 0.08, width / 2 + 0.08, 0.02)
            else:
                dashed("white", a, b, n, width / 2, 0.15, 0.02)
        lanes_side = int(max(0.0, carriage / 2 - 0.5) // 3.3)
        for k in range(1, lanes_side):  # dashed lane lines on this side of the centre
            dashed("white", a, b, n, sw + 0.4 + 3.3 * k, 0.15, 0.02)
        if width >= 14.0 and sw >= 2.5:  # street trees along the footway
            length = (b - a).length
            s = rnd.uniform(0, 9.0)
            while s < length:
                trees.append(a + (b - a).normalized() * s + n * (sw * 0.55))
                s += rnd.uniform(8.0, 11.0)
        if carriage >= 5.5:
            lanes.append((a, b, n, width, sw))
    col = bpy.data.collections.new("road_dressing")
    bpy.context.scene.collection.children.link(col)
    for kind, g in geo.items():
        me = bpy.data.meshes.new(f"road_{kind}")
        g.to_mesh(me)
        g.free()
        me.materials.append(mats[kind])
        ob = bpy.data.objects.new(f"road_{kind}", me)
        col.objects.link(ob)
    # trees: one trunk and canopy mesh, instanced; the canopy a lumpy dark green ball
    tg = bmesh.new()
    bmesh.ops.create_cone(tg, cap_ends=True, segments=6, radius1=0.15, radius2=0.12, depth=3.0,
                          matrix=Matrix.Translation((0, 0, 1.5)))
    trunk = bpy.data.meshes.new("tree_trunk")
    tg.to_mesh(trunk)
    tg.free()
    trunk.materials.append(_plain("tree_bark", (0.06, 0.045, 0.035), 0.8))
    cg = bmesh.new()
    bmesh.ops.create_icosphere(cg, subdivisions=2, radius=2.3, matrix=Matrix.Translation((0, 0, 4.3)))
    for v in cg.verts:
        v.co += v.normal * rnd.uniform(-0.35, 0.35)
    canopy = bpy.data.meshes.new("tree_canopy")
    cg.to_mesh(canopy)
    cg.free()
    canopy.materials.append(_plain("tree_leaves", (0.02, 0.06, 0.025), 0.7))
    for i, p in enumerate(trees[:4000]):
        for me in (trunk, canopy):
            t = bpy.data.objects.new(f"{me.name}_{i}", me)
            t.location = p
            s = rnd.uniform(0.8, 1.2)
            t.scale = (s, s, s)
            col.objects.link(t)
    bm.free()
    print(f"[city] roads: {len(kerbs)} kerb edges, {len(lanes)} with lanes, {min(len(trees), 4000)} trees", flush=True)
    return lanes


def traffic_on_lanes(lanes, rnd, speed=(7.0, 13.0), shot_s=12.0, cap=2400) -> int:
    """Cars with bodies in the lanes, keeping left as in Japan: white headlights in front, red tail
    lights behind, five paints; at 25-47 km/h through the shot where the kerb runs long enough."""
    import bmesh

    head = _emit("car_headlight", (1.0, 0.93, 0.8), 120.0)
    tail = _emit("car_taillight", (1.0, 0.04, 0.02), 90.0)
    glass = _plain("car_glass", (0.01, 0.012, 0.015), 0.08)
    paints = [_plain(f"car_paint{k}", rgb, 0.25) for k, rgb in enumerate(
        ((0.6, 0.6, 0.62), (0.02, 0.02, 0.022), (0.75, 0.75, 0.74), (0.05, 0.08, 0.16), (0.18, 0.02, 0.02)))]
    meshes = []
    for k, paint in enumerate(paints):  # +X forward
        g = bmesh.new()
        bmesh.ops.create_cube(g, size=1.0, matrix=Matrix.Translation((0, 0, 0.62)) @ Matrix.Diagonal((4.4, 1.75, 0.75, 1.0)))
        bmesh.ops.create_cube(g, size=1.0, matrix=Matrix.Translation((-0.25, 0, 1.22)) @ Matrix.Diagonal((2.3, 1.55, 0.5, 1.0)))
        n_body = len(g.faces)
        for y in (-0.62, 0.62):
            bmesh.ops.create_uvsphere(g, u_segments=6, v_segments=4, radius=0.13, matrix=Matrix.Translation((2.2, y, 0.72)))
        n_head = len(g.faces)
        for y in (-0.65, 0.65):
            bmesh.ops.create_uvsphere(g, u_segments=6, v_segments=4, radius=0.12, matrix=Matrix.Translation((-2.2, y, 0.8)))
        me = bpy.data.meshes.new(f"car_body{k}")
        g.to_mesh(me)
        g.free()
        for m in (paint, head, tail, glass):
            me.materials.append(m)
        for i, poly in enumerate(me.polygons):
            poly.material_index = 0 if i < 6 else (3 if i < n_body else (1 if i < n_head else 2))
        meshes.append(me)
    n = 0
    order = list(lanes)
    rnd.shuffle(order)
    for a, b, nrm, width, sw in order:
        length = (b - a).length
        # PLATEAU's kerbs come in short pieces: one car on a piece by chance in proportion to its length
        # (on a short piece it all but stands, as in a queue at the lights)
        if length < 6.0 or rnd.random() > length / 26.0:
            continue
        f = Vector((-nrm.y, nrm.x, 0.0))  # keeping left: the kerb on the driver's left
        if f.dot(b - a) < 0:
            a, b = b, a
        d = (b - a).normalized()
        for _ in range(max(1, int(length / 35))):
            if n >= cap:
                return n
            v = rnd.uniform(*speed)
            travel = min(v * shot_s, length * 0.9)
            s0 = rnd.uniform(0, max(length - travel, 0.1))
            lane = nrm * (sw + 1.9 + 3.3 * rnd.randint(0, max(0, int((width / 2 - sw - 1) // 3.3) - 1)))
            car = bpy.data.objects.new(f"carb{n}", rnd.choice(meshes))
            bpy.context.scene.collection.objects.link(car)
            car.rotation_euler = (0.0, 0.0, math.atan2(d.y, d.x))
            for fr, s in ((1, s0), (int(shot_s * 25), s0 + travel)):
                car.location = a + d * s + lane + Vector((0, 0, 0.02))
                car.keyframe_insert("location", frame=fr)
            n += 1
    return n


def aerial_overview(frames=300, lens=26.0):
    """(10/28 afternoon, the user: slower, smooth -- the old dive was jerky -- and show the whole city
    first.) One slow, even move of about half a kilometre: it opens far off the towers and nearly level,
    all of Shinjuku and the lights to the horizon under the sky, drifts in and down, and tilts down into
    a street of the district ahead, 45 degrees below the horizon -- the cut goes to the alley on the
    ground. Ease in and out with a long cruise between (no speed jumps); the path keeps 30 m from every
    building."""
    import bisect

    import bmesh
    from mathutils.bvhtree import BVHTree

    towers, road_objs, roofs = globals()["_DRONE_INPUT"]
    bm = bmesh.new()
    for o in towers + [o for o in bpy.data.objects if o.type == "MESH" and o.name.startswith(("hw_", "mega_"))]:
        bm.from_mesh(o.data)
    tree = BVHTree.FromBMesh(bm)

    def clear(q):
        up = tree.ray_cast(q, Vector((0.0, 0.0, 1.0)))
        if up[0] is not None and up[1].z > 0.0:
            return 0.0
        hit = tree.find_nearest(q)
        return hit[3] if hit and hit[0] is not None else 1e9

    tall = [r for r in roofs.values() if r.z > 140.0]
    c = sum(tall, Vector((0, 0, 0))) / max(len(tall), 1)
    c.z = 0.0
    # (10/28 evening) the streets dressed as streets, and cars with bodies in their lanes in place of the
    # light pairs that ran along any road edge
    for o in [o for o in bpy.data.objects if o.name.startswith("car") and o.name[3:].isdigit()]:
        bpy.data.objects.remove(o, do_unlink=True)
    lanes = road_dressing(road_objs, c, 1600.0)
    print(f"[city] traffic: {traffic_on_lanes(lanes, random.Random(23))} cars in lanes", flush=True)
    start_dir = Vector((-950.0, -800.0, 0.0)) - c  # the old opening side, south-west
    start_dir.z = 0.0
    start_dir.normalize()

    def rot(v, deg):
        a = math.radians(deg)
        return Vector((v.x * math.cos(a) - v.y * math.sin(a), v.x * math.sin(a) + v.y * math.cos(a), 0.0))

    A = c + start_dir * 1250.0 + Vector((0, 0, 330.0))
    B = c + rot(start_dir, 6.0) * 1030.0 + Vector((0, 0, 285.0))
    D = c + rot(start_dir, 10.0) * 760.0 + Vector((0, 0, 210.0))
    # the end: a street about 220 m ahead of the last position, toward the towers, open to the sky
    ahead = Vector((c.x - D.x, c.y - D.y, 0.0)).normalized()
    best = None
    for o in road_objs:
        vs = o.data.vertices
        for e in o.data.edges:
            a, b = vs[e.vertices[0]].co, vs[e.vertices[1]].co
            if (a - b).length < 25.0:
                continue
            m = (a + b) / 2
            rel = Vector((m.x - D.x, m.y - D.y, 0.0))
            if rel.length < 120 or rel.length > 340 or rel.normalized().dot(ahead) < 0.9:
                continue
            if min(clear(m + Vector((0, 0, h))) for h in (15.0, 40.0)) < 8.0:
                continue
            score = -abs(rel.length - 220.0) - 300.0 * (1 - rel.normalized().dot(ahead))
            if best is None or score > best[0]:
                best = (score, m.copy(), (b - a).normalized())
    if best is None:
        print("[city] aerial: no street found ahead", flush=True)
        return False
    _score, street, along = best
    end_xy = Vector((street.x, street.y, 0.0))
    approach = ahead
    pts = [A, B, D]

    def catmull(ps, n):
        ps = [ps[0] + (ps[0] - ps[1])] + ps + [ps[-1] + (ps[-1] - ps[-2])]
        out = []
        for i in range(1, len(ps) - 2):
            p0, p1, p2, p3 = ps[i - 1], ps[i], ps[i + 1], ps[i + 2]
            for k in range(n):
                t = k / n
                out.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t * t
                                  + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3))
        out.append(ps[-2])
        return out

    dense = catmull(pts, 600)
    for _ in range(40):  # lift any part that comes within 30 m of a building
        moved = False
        for i, q in enumerate(dense[:-30]):
            if clear(q) < 30.0:
                dense[i] = q + Vector((0, 0, 8.0))
                moved = True
        if not moved:
            break
    seg = [0.0]
    for i in range(1, len(dense)):
        seg.append(seg[-1] + (dense[i] - dense[i - 1]).length)
    total = seg[-1]

    def at(sv):
        j = min(max(bisect.bisect_left(seg, sv), 1), len(seg) - 1)
        f = (sv - seg[j - 1]) / max(seg[j] - seg[j - 1], 1e-6)
        return dense[j - 1].lerp(dense[j], f)

    # distance share against time share: the speed rises as a smoothstep over the first 30 %, cruises,
    # and falls the same way over the last 30 % (no jumps in speed or acceleration)
    ramp, n = 0.3, 600

    def speed(x):
        w = x / ramp if x < ramp else (1 - x) / ramp if x > 1 - ramp else 1.0
        return w * w * (3 - 2 * w)

    acc = [0.0]
    for i in range(1, n + 1):
        acc.append(acc[-1] + 0.5 * (speed((i - 1) / n) + speed(i / n)) / n)

    def travel(u):
        k = min(int(u * n), n - 1)
        return (acc[k] + (acc[k + 1] - acc[k]) * (u * n - k)) / acc[-1]

    scene = bpy.context.scene
    cam = scene.camera
    cam.data.lens = lens
    cam.data.clip_start = 0.5
    cam.data.clip_end = 100000.0  # the far plane and skyline to 30 km
    cam.data.dof.use_dof = False
    cam.animation_data_clear()
    cam.constraints.clear()
    target = bpy.data.objects.get("aerial_target")
    target.animation_data_clear()
    c_ = cam.constraints.new("TRACK_TO")
    c_.target, c_.track_axis, c_.up_axis = target, "TRACK_NEGATIVE_Z", "UP_Y"
    look_city = c + Vector((0, 0, 140.0))
    look_down = end_xy + Vector((0, 0, street.z))
    for f in range(1, frames + 1):
        u = (f - 1) / (frames - 1)
        cam.location = at(total * travel(u))
        cam.keyframe_insert("location", frame=f)
        w = min(max((u - 0.35) / 0.65, 0.0), 1.0)  # the gaze stays on the towers, then sinks into the street
        w = w * w * w * (w * (w * 6 - 15) + 10)
        target.location = look_city.lerp(look_down, w)
        target.keyframe_insert("location", frame=f)
    for ob in (cam, target):  # a key on every frame already; Bezier handles between them would add wobble
        for fc in _fcurves(ob):
            for kp in fc.keyframe_points:
                kp.interpolation = "LINEAR"
    street_dressing(street, along, random.Random(9))
    street_lanterns_and_steam(street, along)
    scene.frame_start, scene.frame_end = 1, frames
    # (10/28 evening) one of the three G4s came back as an A100 (eight times slower in Cycles): the slow
    # move needs fewer samples than the dive did -- at most 96, stopping at a looser noise level, denoised
    scene.cycles.samples = min(scene.cycles.samples, 96)
    scene.cycles.use_adaptive_sampling = True
    scene.cycles.adaptive_threshold = 0.03
    print(f"[city] aerial: {total:.0f} m in {frames} frames ({total / (frames / 25):.0f} m/s on average), "
          f"street at ({street.x:.0f}, {street.y:.0f}); nearest building {min(clear(q) for q in dense[:-30]):.0f} m",
          flush=True)
    return True


def _fcurves(ob):
    """An object's F-curves, from the legacy action or the layered (Blender 4.4+) one."""
    ad = ob.animation_data
    act = ad.action if ad else None
    if act is None:
        return []
    if hasattr(act, "fcurves") and len(act.fcurves):
        return list(act.fcurves)
    out = []
    for layer in getattr(act, "layers", []):
        for strip in layer.strips:
            for bag in getattr(strip, "channelbags", []):
                out += list(bag.fcurves)
    return out


def traffic(road_objs, rnd, speed=(8.0, 15.0), shot_s=12.0, cap=2600) -> int:
    """Cars as their lights: along every long straight road edge two lanes, white headlights coming
    one way and red tail lights going the other, at 30-55 km/h through the shot."""
    head = _emit("headlights", (1.0, 0.93, 0.8), 260.0)
    tail = _emit("taillights", (1.0, 0.04, 0.02), 200.0)
    import bmesh

    meshes = {}
    for name, mat in (("head", head), ("tail", tail)):
        me = bpy.data.meshes.new(f"car_{name}")
        bm = bmesh.new()
        for side in (-0.75, 0.75):  # a pair of lamps 1.5 m apart
            bmesh.ops.create_uvsphere(bm, u_segments=6, v_segments=4, radius=0.4,
                                      matrix=Matrix.Translation((0.0, side, 0.7)))
        bm.to_mesh(me)
        bm.free()
        me.materials.append(mat)
        meshes[name] = me
    edges = []
    for o in road_objs:
        vs = o.data.vertices
        for e in o.data.edges:
            a, b = vs[e.vertices[0]].co.copy(), vs[e.vertices[1]].co.copy()
            if (a - b).length > 25.0 and max(abs(a.x), abs(a.y)) < 1800:
                edges.append((a, b))
    rnd.shuffle(edges)
    n = 0
    for a, b in edges:
        d = b - a
        length = d.length
        d.normalize()
        side = Vector((-d.y, d.x, 0.0))
        for _ in range(max(1, int(length / 30))):
            if n >= cap:
                return n
            kind = rnd.choice(("head", "tail"))
            v = rnd.uniform(*speed)
            travel = min(v * shot_s, length * 0.9)
            s0 = rnd.uniform(0, length - travel)
            lane = side * (2.2 if kind == "head" else 5.2)  # inward from the kerb edge
            car = bpy.data.objects.new(f"car{n}", meshes[kind])
            bpy.context.scene.collection.objects.link(car)
            car.rotation_euler = (0.0, 0.0, math.atan2(d.y, d.x) + math.pi / 2)
            for f, s in ((1, s0), (int(shot_s * 25), s0 + travel)):
                car.location = a + d * s + lane
                car.keyframe_insert("location", frame=f)
            n += 1
    return n


def sky_and_render(res=(2560, 1440), samples=128, fog=0.000012) -> None:
    scene = bpy.context.scene
    world = bpy.data.worlds.new("city_night")
    scene.world = world
    world.use_nodes = True
    nt = world.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    tc = nt.nodes.new("ShaderNodeTexCoord")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(tc.outputs["Generated"], sep.inputs[0])
    # (10/28, after the user's reference photos of Shinjuku from the towers) a blue night sky, not a
    # black one: warm grey at the horizon where the city lights the haze, deep blue overhead, and a
    # broken cloud deck lit from below. Generated Z in a world shader is the view direction's z, the sine
    # of the elevation (0 at the horizon): the ramp used to sit at 0.49-0.88, so the whole visible sky was
    # its first stop -- the flat mauve, then salmon, the user saw (fixed 10/28 morning).
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    els = ramp.color_ramp.elements
    els[0].position, els[0].color = 0.0, (0.2, 0.11, 0.1, 1)
    els[1].position, els[1].color = 0.5, (0.003, 0.008, 0.04, 1)  # a deep navy overhead
    for pos, col in ((0.012, (0.09, 0.055, 0.12)), (0.035, (0.03, 0.045, 0.14)), (0.12, (0.012, 0.025, 0.1))):
        e = els.new(pos)
        e.color = (*col, 1)
    nt.links.new(sep.outputs["Z"], ramp.inputs["Fac"])
    # (10/28 dawn) below the horizon, past the far plane's edge as seen from the drone's height, the
    # ramp's first stop showed as a flat brown band 3 degrees tall: there the glow fades into the lit
    # haze the far plane fades into
    below = nt.nodes.new("ShaderNodeMapRange")
    nt.links.new(sep.outputs["Z"], below.inputs["Value"])
    below.inputs["From Min"].default_value, below.inputs["From Max"].default_value = 0.0, -0.012
    under = nt.nodes.new("ShaderNodeMix")
    under.data_type = "RGBA"
    nt.links.new(below.outputs["Result"], under.inputs["Factor"])
    nt.links.new(ramp.outputs["Color"], under.inputs["A"])
    under.inputs["B"].default_value = (*FAR_GLOW, 1)
    stretch = nt.nodes.new("ShaderNodeMapping")  # clouds: wide, flat banks
    stretch.inputs["Scale"].default_value = (1.0, 1.0, 4.0)
    nt.links.new(tc.outputs["Generated"], stretch.inputs["Vector"])
    cloud = _node(nt, "ShaderNodeTexNoise", Scale=3.0, Detail=12.0, Roughness=0.6)
    nt.links.new(stretch.outputs["Vector"], cloud.inputs["Vector"])
    cmask = nt.nodes.new("ShaderNodeMapRange")
    nt.links.new(cloud.outputs["Fac"], cmask.inputs["Value"])
    cmask.inputs["From Min"].default_value, cmask.inputs["From Max"].default_value = 0.52, 0.6
    band_ = nt.nodes.new("ShaderNodeMapRange")  # clouds above the horizon haze only
    nt.links.new(sep.outputs["Z"], band_.inputs["Value"])
    band_.inputs["From Min"].default_value, band_.inputs["From Max"].default_value = 0.01, 0.06
    cm = _math(nt, "MULTIPLY", cmask.outputs[0], band_.outputs[0])
    ccol = nt.nodes.new("ShaderNodeMix")
    ccol.data_type = "RGBA"
    nt.links.new(_math(nt, "MULTIPLY", cm, None, vb=0.75), ccol.inputs["Factor"])
    nt.links.new(under.outputs["Result"], ccol.inputs["A"])
    ccol.inputs["B"].default_value = (0.1, 0.075, 0.11, 1)  # cloud underside, lit by the city
    bg = _node(nt, "ShaderNodeBackground", Strength=1.0)
    nt.links.new(ccol.outputs["Result"], bg.inputs["Color"])
    vol = _node(nt, "ShaderNodeVolumePrincipled", Density=0.0, Anisotropy=0.5)  # (the haze is the box below now)
    vol.inputs["Color"].default_value = (0.5, 0.55, 0.9, 1)
    out = nt.nodes.new("ShaderNodeOutputWorld")
    nt.links.new(bg.outputs[0], out.inputs["Surface"])
    # (10/28) the haze only in the air over the city, up to 180 m: a world-wide volume caught the light
    # of thirty kilometres of city and turned the whole sky a reddish grey; now the sky above is navy
    haze = bpy.data.meshes.new("haze")
    hz = 3500.0  # (not to the horizon: long grazing paths through it made a flat brown band)
    top_ = 520.0
    haze.from_pydata([(-hz, -hz, -20), (hz, -hz, -20), (hz, hz, -20), (-hz, hz, -20), (-hz, -hz, top_), (hz, -hz, top_), (hz, hz, top_),
                      (-hz, hz, top_)], [], [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)])
    hm = bpy.data.materials.new("haze")
    hm.use_nodes = True
    hnt = hm.node_tree
    for n in list(hnt.nodes):
        hnt.nodes.remove(n)
    hv = _node(hnt, "ShaderNodeVolumePrincipled", Anisotropy=0.6)
    hgeo = hnt.nodes.new("ShaderNodeNewGeometry")
    hsep = hnt.nodes.new("ShaderNodeSeparateXYZ")
    hnt.links.new(hgeo.outputs["Position"], hsep.inputs[0])
    hfall = hnt.nodes.new("ShaderNodeMapRange")  # thick low down, thin by 500 m
    hnt.links.new(hsep.outputs["Z"], hfall.inputs["Value"])
    hfall.inputs["From Min"].default_value, hfall.inputs["From Max"].default_value = 0.0, 500.0
    hfall.inputs["To Min"].default_value, hfall.inputs["To Max"].default_value = fog * 3.0, 0.0  # to nothing at the top
    hnt.links.new(hfall.outputs["Result"], hv.inputs["Density"])
    hv.inputs["Color"].default_value = (0.55, 0.6, 0.9, 1)
    ho = hnt.nodes.new("ShaderNodeOutputMaterial")
    hnt.links.new(hv.outputs[0], ho.inputs["Volume"])
    haze.materials.append(hm)
    bpy.context.scene.collection.objects.link(bpy.data.objects.new("haze", haze))
    scene.render.engine = "CYCLES"
    prefs = bpy.context.preferences.addons["cycles"].preferences
    for kind in ("OPTIX", "CUDA"):
        try:
            prefs.compute_device_type = kind
            break
        except TypeError:
            continue
    prefs.get_devices()
    for d in prefs.devices:
        d.use = d.type == prefs.compute_device_type
    scene.cycles.device = "GPU"
    scene.cycles.samples = samples
    scene.cycles.use_denoising = True
    scene.cycles.volume_step_rate = 4.0
    scene.render.resolution_x, scene.render.resolution_y = res
    scene.view_settings.view_transform = "AgX"
    scene.view_settings.exposure = 0.7  # (10/28, the user: too dark) a lifted night, as a night photo of Tokyo is
    for look in ("AgX - Punchy", "Punchy"):
        try:
            scene.view_settings.look = look
            break
        except TypeError:
            continue
    scene.render.use_motion_blur = True
    scene.render.motion_blur_shutter = 0.5
    scene.render.fps = 25
    for k, r in enumerate(globals().get("_SEARCH_ROOFS", [])):  # (10/28) beams up from the tallest roofs
        sl = bpy.data.lights.new(f"beam{k}", "SPOT")
        sl.energy, sl.spot_size, sl.spot_blend, sl.color = 4.0e7, math.radians(2.5), 0.3, (0.85, 0.92, 1.0)
        sl.shadow_soft_size = 0.5
        so = bpy.data.objects.new(f"beam{k}", sl)
        bpy.context.scene.collection.objects.link(so)
        so.location = r + Vector((0, 0, 3.0))
        for f, a_ in ((1, -0.35), (110, 0.35), (220, -0.35)):
            so.rotation_euler = (a_ * (1 if k % 2 else -1) + 0.1 * k, 0.25 * math.sin(k), 0.0)
            so.keyframe_insert("rotation_euler", frame=f)
    # searchlights from two rooftops sweeping
    for k, (x, y) in enumerate(((-120.0, 80.0), (260.0, -60.0))):
        sl = bpy.data.lights.new(f"searchlight{k}", "SPOT")
        sl.energy, sl.spot_size, sl.color = 3.0e6, math.radians(4), (0.85, 0.9, 1.0)
        so = bpy.data.objects.new(f"searchlight{k}", sl)
        bpy.context.scene.collection.objects.link(so)
        so.location = (x, y, 200.0)
        for f, ang in ((1, -0.25), (150, 0.25)):
            so.rotation_euler = (math.radians(18) * (1 if k else -1), ang, 0.0)
            so.keyframe_insert("rotation_euler", frame=f)
    # the aerial camera: from over the south-west of the towers, descending toward Kabukicho
    cam = bpy.data.objects.new("aerial", bpy.data.cameras.new("aerial"))
    bpy.context.scene.collection.objects.link(cam)
    cam.data.lens = 28
    cam.data.clip_end = 100000.0  # (10/28 dawn) was 6 km: past it the sky showed through as a flat brown band
    scene.camera = cam
    target = bpy.data.objects.new("aerial_target", None)
    bpy.context.scene.collection.objects.link(target)
    c = cam.constraints.new("TRACK_TO")
    c.target, c.track_axis, c.up_axis = target, "TRACK_NEGATIVE_Z", "UP_Y"
    # (10/28) at about the height of the towers' observation floors, looking over them to the horizon
    for f, p, t in ((1, (-1100, -900, 300), (1500, 1300, -60)), (150, (-520, -620, 240), (1700, 1500, -70))):
        cam.location = p
        cam.keyframe_insert("location", frame=f)
        target.location = t
        target.keyframe_insert("location", frame=f)
    scene.frame_start, scene.frame_end = 1, 150


def main() -> None:
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--plateau", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--render", default="", help="a still of --frame to this PNG")
    p.add_argument("--frame", type=int, default=75)
    p.add_argument("--movie", default="", help="render the shot's frames into this folder")
    p.add_argument("--shard", default="0/1")
    p.add_argument("--range", default="", help="first,last frame of the shot to render (default all 150)")
    p.add_argument("--res", default="2560x1440")
    p.add_argument("--samples", type=int, default=128)
    p.add_argument("--hero", default="", help="scene npz,take,mjcf: the robot in the street at the drone's end")
    p.add_argument("--drone", action="store_true", help="the drone move into a street between the towers (200 frames)")
    p.add_argument("--reverse", action="store_true", help="with --movie: from the last frame back")
    p.add_argument("--aerial", type=int, default=0, help="the slow overview move ending above a street, this many frames")
    p.add_argument("--keep-textures", action="store_true", help="PLATEAU photo textures (scrubbed of signs) at night")
    args = p.parse_args(argv)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    info = build(Path(args.plateau), keep_textures=args.keep_textures)
    sky_and_render(tuple(int(v) for v in args.res.split("x")), args.samples)
    if args.drone:
        h = args.hero.split(",") if args.hero else None
        drone_shot(hero=(h[0], int(h[1]), h[2]) if h else None)
        air_car(bpy.context.scene.camera)
    elif args.aerial:
        aerial_overview(frames=args.aerial)
    bpy.ops.wm.save_as_mainfile(filepath=str(Path(args.out).resolve()))
    print(f"CITY_BUILT {args.out} {info}", flush=True)
    scene = bpy.context.scene
    if args.render:
        scene.frame_set(args.frame)
        scene.render.filepath = args.render
        bpy.ops.render.render(write_still=True)
        print(f"CITY_RENDERED {args.render}", flush=True)
    if args.movie:
        import time

        i, n = (int(v) for v in args.shard.split("/"))
        folder = Path(args.movie)
        folder.mkdir(parents=True, exist_ok=True)
        first, last = (int(v) for v in args.range.split(",")) if args.range else (scene.frame_start, scene.frame_end)
        for f in (range(last, first - 1, -1) if args.reverse else range(first, last + 1)):
            path = folder / f"frame_{f:04d}.png"
            if (f - 1) % n != i or path.exists():
                continue
            scene.frame_set(f)
            scene.render.filepath = str(path)
            t = time.time()
            bpy.ops.render.render(write_still=True)
            print(f"[render] {path.name} {time.time() - t:.1f} s", flush=True)
        print(f"CITY_MOVIE {folder}", flush=True)


main()
