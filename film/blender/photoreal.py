"""Photo-real look for the back-alley takes (10/28): scanned CC0 materials, a night HDRI, bevelled
edges, wet ground, haze, and the robots' paint and eyes. Meant for Cycles on the GPU (Colab).

The user's bar (10/28): live-action / blockbuster-game quality, no shortcuts on detail. What this
does to a scene `import_take.py --set backalley` built:

- **Materials** from Poly Haven (CC0; `fetch` downloads through its public API into a cache and
  records name, URL and size in `credits.json`): concrete walls, worn wet asphalt with puddles,
  worn concrete steps, rusty painted metal, rusty pipes. They are box-projected in the set's own
  frame (the alley has no UVs), at a real-world tile size, with the height map as bump.
- **Edges** bevelled (a few millimetres): nothing in a photograph has a mathematically sharp edge.
- **Ground**: puddles from a large noise mask -- mirror-smooth and darker where wet -- and a damp
  sheen elsewhere.
- **World**: a night city HDRI, dim, for the sky slit and the reflections; a thin haze so the
  practical lights throw beams.
- **Robots**: glossy red and white plastics under a clear coat, satin grey metal, a gloss-black
  visor, and the eyes: the official art's two slanted eye shapes glowing on the visor (cyan for the
  hero, red for the rival). Only materials and decals change; the motion is the take's.
"""

from __future__ import annotations

import json
import math
import urllib.request
from pathlib import Path

import bpy
from mathutils import Vector

API = "https://api.polyhaven.com/files/"
#: Poly Haven refuses the bare Python user agent (403, 10/28); its API asks clients to name themselves.
UA = {"User-Agent": "jumper-film-asset-fetch/1.0 (CC0 textures for a robot short film)"}

#: material role -> (Poly Haven asset, tile size in metres)
ASSETS = {
    "wall_a": ("concrete_slab_wall", 2.4),
    "wall_b": ("concrete_wall_006", 2.0),
    "wall_c": ("dirty_concrete", 2.0),
    "floor": ("worn_asphalt", 2.5),
    "steps": ("concrete_floor_worn_001", 1.5),
    "painted_metal": ("rusty_painted_metal", 1.2),
    "pipe": ("rusty_metal_02", 0.8),
}
HDRI = ("shanghai_bund", "4k")
#: the maps used and the API's names for them
MAPS = {"diff": ("Diffuse", "jpg"), "rough": ("Rough", "jpg"), "disp": ("Displacement", "png")}


def fetch(cache: Path, res: str = "4k") -> dict:
    """Download every asset's colour, roughness and height maps and the HDRI into `cache`; returns
    role -> {map: path} and writes the credits (name, URL, bytes) beside them."""
    cache.mkdir(parents=True, exist_ok=True)
    credits, out = [], {}

    def get(url: str, path: Path) -> None:
        if not path.exists():
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA)) as r, open(path, "wb") as f:
                f.write(r.read())
        credits.append({"file": path.name, "url": url, "bytes": path.stat().st_size,
                        "source": "Poly Haven", "license": "CC0"})

    for role, (asset, _tile) in ASSETS.items():
        files = json.loads(urllib.request.urlopen(urllib.request.Request(API + asset, headers=UA)).read())
        out[role] = {}
        for key, (name, ext) in MAPS.items():
            entry = files.get(name, {}).get(res, {}).get(ext) or files.get(name, {}).get(res, {}).get("jpg")
            if not entry:
                continue
            path = cache / f"{asset}_{key}_{res}.{entry['url'].rsplit('.', 1)[-1]}"
            get(entry["url"], path)
            out[role][key] = path
    files = json.loads(urllib.request.urlopen(urllib.request.Request(API + HDRI[0], headers=UA)).read())
    hdr = files["hdri"][HDRI[1]]["hdr"]["url"]
    path = cache / f"{HDRI[0]}_{HDRI[1]}.hdr"
    get(hdr, path)
    out["hdri"] = path
    (cache / "credits.json").write_text(json.dumps(credits, indent=1))
    return out


def _bsdf(mat):
    nt = mat.node_tree
    b = next((n for n in nt.nodes if n.type == "BSDF_PRINCIPLED"), None)
    if b is None:
        b = nt.nodes.new("ShaderNodeBsdfPrincipled")
        o = next((n for n in nt.nodes if n.type == "OUTPUT_MATERIAL"), None) or nt.nodes.new("ShaderNodeOutputMaterial")
        nt.links.new(b.outputs[0], o.inputs["Surface"])
    return b


def _img(nt, path: Path, color: bool, vec):
    img = bpy.data.images.load(str(path), check_existing=True)
    img.colorspace_settings.name = "sRGB" if color else "Non-Color"
    node = nt.nodes.new("ShaderNodeTexImage")
    node.image = img
    node.projection = "BOX"
    node.projection_blend = 0.25
    nt.links.new(vec, node.inputs["Vector"])
    return node


def pbr(name: str, maps: dict, tile: float, space, tint=(1.0, 1.0, 1.0), bump: float = 0.35,
        rough_scale: float = 1.0, wet: float = 0.0):
    """A scanned material, box-projected in `space`'s frame at `tile` metres per repeat. `wet` > 0
    adds puddles (a noise mask: mirror-smooth and darker inside) and a damp sheen."""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    b = _bsdf(mat)
    coord = nt.nodes.new("ShaderNodeTexCoord")
    coord.object = space
    mapping = nt.nodes.new("ShaderNodeMapping")
    mapping.inputs["Scale"].default_value = (1.0 / tile,) * 3
    nt.links.new(coord.outputs["Object"], mapping.inputs["Vector"])
    vec = mapping.outputs["Vector"]
    col = _img(nt, maps["diff"], True, vec)
    tinted = nt.nodes.new("ShaderNodeMix")
    tinted.data_type, tinted.blend_type = "RGBA", "MULTIPLY"
    tinted.inputs["Factor"].default_value = 1.0
    tinted.inputs["B"].default_value = (*tint, 1.0)
    nt.links.new(col.outputs["Color"], tinted.inputs["A"])
    rough = nt.nodes.new("ShaderNodeMath")
    rough.operation = "MULTIPLY"
    rough.inputs[1].default_value = rough_scale
    if "rough" in maps:
        nt.links.new(_img(nt, maps["rough"], False, vec).outputs["Color"], rough.inputs[0])
    else:
        rough.inputs[0].default_value = 0.8
    base, rough_out = tinted.outputs["Result"], rough.outputs[0]
    height = _img(nt, maps["disp"], False, vec).outputs["Color"] if "disp" in maps else None
    if wet > 0:
        mask_map = nt.nodes.new("ShaderNodeTexNoise")  # puddles: large, soft-edged
        mask_map.inputs["Scale"].default_value = 0.35
        mask_map.inputs["Detail"].default_value = 3.0
        nt.links.new(coord.outputs["Object"], mask_map.inputs["Vector"])
        ramp = nt.nodes.new("ShaderNodeValToRGB")
        ramp.color_ramp.elements[0].position = 0.56 - 0.1 * wet
        ramp.color_ramp.elements[1].position = 0.6 - 0.1 * wet
        nt.links.new(mask_map.outputs["Fac"], ramp.inputs["Fac"])
        if height is not None:  # water collects in the low parts first
            low = nt.nodes.new("ShaderNodeMath")
            low.operation = "SUBTRACT"
            nt.links.new(ramp.outputs["Color"], low.inputs[0])
            nt.links.new(height, low.inputs[1])
            low.inputs[1].default_value = 0.0
            puddle = low.outputs[0]
        else:
            puddle = ramp.outputs["Color"]
        clamp = nt.nodes.new("ShaderNodeClamp")
        nt.links.new(puddle, clamp.inputs["Value"])
        damp = nt.nodes.new("ShaderNodeMath")  # a damp sheen everywhere, puddles mirror-smooth
        damp.operation = "MULTIPLY"
        damp.inputs[1].default_value = 0.55
        nt.links.new(rough_out, damp.inputs[0])
        rmix = nt.nodes.new("ShaderNodeMix")
        rmix.data_type = "FLOAT"
        nt.links.new(clamp.outputs[0], rmix.inputs["Factor"])
        nt.links.new(damp.outputs[0], rmix.inputs["A"])
        rmix.inputs["B"].default_value = 0.02
        rough_out = rmix.outputs["Result"]
        dark = nt.nodes.new("ShaderNodeMix")
        dark.data_type, dark.blend_type = "RGBA", "MULTIPLY"
        nt.links.new(clamp.outputs[0], dark.inputs["Factor"])
        nt.links.new(base, dark.inputs["A"])
        dark.inputs["B"].default_value = (0.45, 0.45, 0.47, 1.0)
        base = dark.outputs["Result"]
        if height is not None:  # flat water over the puddles
            flat = nt.nodes.new("ShaderNodeMix")
            flat.data_type = "FLOAT"
            nt.links.new(clamp.outputs[0], flat.inputs["Factor"])
            nt.links.new(height, flat.inputs["A"])
            flat.inputs["B"].default_value = 0.2
            height = flat.outputs["Result"]
    nt.links.new(base, b.inputs["Base Color"])
    nt.links.new(rough_out, b.inputs["Roughness"])
    if height is not None:
        bn = nt.nodes.new("ShaderNodeBump")
        bn.inputs["Strength"].default_value = bump
        bn.inputs["Distance"].default_value = 0.02 * tile
        nt.links.new(height, bn.inputs["Height"])
        nt.links.new(bn.outputs["Normal"], b.inputs["Normal"])
    return mat


def _bevel(obj, width: float) -> None:
    m = obj.modifiers.new("bevel", "BEVEL")
    m.width, m.segments, m.limit_method = width, 2, "ANGLE"
    m.harden_normals = False


def upgrade_set(assets: dict, set_root=None, haze: float = 0.006, hdri_strength: float = 0.25) -> None:
    """Swap the procedural alley's materials for scanned ones, bevel its boxes, and light the world."""
    root = set_root or bpy.data.objects.get("set")
    space = bpy.data.objects.new("texture_space", None)  # the set's frame: box projection follows the alley
    bpy.context.scene.collection.objects.link(space)
    if root is not None:
        space.parent = root
    mats = {
        "floor": pbr("pr_floor", assets["floor"], ASSETS["floor"][1], space, tint=(0.75, 0.75, 0.78), wet=1.0),
        "steps": pbr("pr_steps", assets["steps"], ASSETS["steps"][1], space, tint=(0.8, 0.8, 0.8), wet=0.3),
        "metal": pbr("pr_metal", assets["painted_metal"], ASSETS["painted_metal"][1], space, bump=0.25),
        "pipe": pbr("pr_pipe", assets["pipe"], ASSETS["pipe"][1], space, bump=0.2),
    }
    walls = [pbr(f"pr_wall_{k}", assets[k], ASSETS[k][1], space, tint=t, bump=0.5, rough_scale=1.0)
             for k, t in (("wall_a", (0.85, 0.85, 0.86)), ("wall_b", (0.8, 0.8, 0.82)), ("wall_c", (0.9, 0.88, 0.86)))]
    n_wall = 0
    for obj in list(bpy.data.objects):
        if obj.type != "MESH" or not obj.data.materials:
            continue
        name = obj.name
        old = obj.data.materials[0].name if obj.data.materials[0] else ""
        new = None
        if name.startswith("wall"):
            new = walls[n_wall % len(walls)]
            n_wall += 1
            _bevel(obj, 0.02)
        elif name == "alley_floor":
            new = mats["floor"]
        elif name.startswith(("stair", "landing", "step")):
            new = mats["steps"]
            _bevel(obj, 0.01)
        elif name.startswith(("door", "skip", "ac", "meter")):
            new = mats["metal"]
            _bevel(obj, 0.006)
        elif name.startswith("pipe") or name.startswith("conduit"):
            new = mats["pipe"]
        elif name.startswith(("crate", "win")):
            _bevel(obj, 0.004)
        if new is not None:
            obj.data.materials.clear()
            obj.data.materials.append(new)
    world = bpy.context.scene.world or bpy.data.worlds.new("night")
    bpy.context.scene.world = world
    world.use_nodes = True
    nt = world.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    env = nt.nodes.new("ShaderNodeTexEnvironment")
    env.image = bpy.data.images.load(str(assets["hdri"]), check_existing=True)
    bg = nt.nodes.new("ShaderNodeBackground")
    bg.inputs["Strength"].default_value = hdri_strength
    out = nt.nodes.new("ShaderNodeOutputWorld")
    nt.links.new(env.outputs["Color"], bg.inputs["Color"])
    nt.links.new(bg.outputs["Background"], out.inputs["Surface"])
    if haze > 0:
        vol = nt.nodes.new("ShaderNodeVolumePrincipled")
        vol.inputs["Density"].default_value = haze
        vol.inputs["Anisotropy"].default_value = 0.4
        nt.links.new(vol.outputs[0], out.inputs["Volume"])
    print(f"[photoreal] set: {n_wall} walls, scanned materials, HDRI {Path(assets['hdri']).name}, haze {haze}")


def _emit(name, rgb, strength):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    em = nt.nodes.new("ShaderNodeEmission")
    em.inputs["Color"].default_value = (*rgb, 1.0)
    em.inputs["Strength"].default_value = strength
    o = nt.nodes.new("ShaderNodeOutputMaterial")
    nt.links.new(em.outputs[0], o.inputs["Surface"])
    return mat


def dress_set(seed: int = 7, font_path: str = "") -> None:
    """The alley's far end and floor dressed for a close camera: wrinkled bin bags, vertical neon
    signs with generic words, a lit vending machine (no brand), cans and papers on the ground, a
    steam plume from a vent. Positions are in the set's frame (+x down the alley, +y to its left)."""
    import random

    from mathutils import Vector

    rnd = random.Random(seed)
    root = bpy.data.objects.get("set")
    floor = bpy.data.objects.get("alley_floor")
    if root is None or floor is None:
        print("[photoreal] no set to dress")
        return
    length = floor.dimensions.x - 4.0
    half = (floor.dimensions.y - 1.0) / 2
    u0 = floor.location.x - (length + 4.0) / 2 + 2.0
    u1 = u0 + length

    def put(obj):
        bpy.context.scene.collection.objects.link(obj) if obj.name not in bpy.context.scene.collection.objects else None
        obj.parent = root
        return obj

    # bin bags: wrinkles (a displacement of fine noise) and a soft plastic sheen
    tex = bpy.data.textures.new("bag_wrinkles", "CLOUDS")
    tex.noise_scale = 0.06
    plastic = bpy.data.materials.new("bag_plastic")
    plastic.use_nodes = True
    b = _bsdf(plastic)
    b.inputs["Base Color"].default_value = (0.012, 0.012, 0.014, 1.0)
    b.inputs["Roughness"].default_value = 0.18
    for key in ("Coat Weight", "Clearcoat"):
        if key in b.inputs:
            b.inputs[key].default_value = 0.4
            break
    for obj in bpy.data.objects:
        if obj.type == "MESH" and obj.name.startswith("bag"):
            sub = obj.modifiers.new("subdiv", "SUBSURF")
            sub.levels = sub.render_levels = 2
            d = obj.modifiers.new("wrinkles", "DISPLACE")
            d.texture, d.strength, d.mid_level = tex, 0.025, 0.5
            obj.data.materials.clear()
            obj.data.materials.append(plastic)
    # vertical signs down the far end, generic words only
    font = bpy.data.fonts.load(font_path) if font_path else None
    for k, (word, rgb, side) in enumerate((("居酒屋", (1.0, 0.25, 0.1), 1), ("営業中", (0.2, 0.9, 1.0), -1),
                                           ("スナック", (1.0, 0.15, 0.55), 1), ("ラーメン", (1.0, 0.75, 0.3), -1))):
        u = u1 - 9.0 + 2.2 * k
        y = side * (half - 0.25)
        box = bpy.data.objects.new(f"sign_box{k}", bpy.data.meshes.new(f"sign_box{k}"))
        bpy.context.scene.collection.objects.link(box)
        import bmesh

        bm = bmesh.new()
        bmesh.ops.create_cube(bm, size=1.0)
        bm.to_mesh(box.data)
        bm.free()
        box.scale = (0.14, 0.4, 0.38 * len(word))
        box.location = (u, y, 3.0 + 0.2 * len(word))
        box.data.materials.append(_emit(f"sign_face{k}", (0.9, 0.9, 0.85), 1.5))
        put(box)
        if font is not None:
            curve = bpy.data.curves.new(f"sign_text{k}", "FONT")
            curve.body = "\n".join(word)
            curve.font = font
            curve.size = 0.26
            curve.align_x, curve.align_y = "CENTER", "CENTER"
            curve.space_line = 0.95
            t = bpy.data.objects.new(f"sign_text{k}", curve)
            bpy.context.scene.collection.objects.link(t)
            t.location = (u - 0.075, y, 3.0 + 0.2 * len(word))
            t.rotation_euler = (math.pi / 2, 0.0, -math.pi / 2)
            curve.materials.append(_emit(f"sign_glow{k}", rgb, 18.0))
            put(t)
        light = bpy.data.lights.new(f"sign_light{k}", "AREA")
        light.energy, light.color, light.size = 60.0, rgb, 0.8
        lo = bpy.data.objects.new(f"sign_light{k}", light)
        bpy.context.scene.collection.objects.link(lo)
        lo.location = (u - 0.4, y - side * 0.4, 2.8)
        lo.rotation_euler = (0.0, math.radians(-90), 0.0)
        put(lo)
    # a lit vending machine against the right wall, near the far end (blank, glowing front)
    vm = bpy.data.objects.new("vending", bpy.data.meshes.new("vending"))
    bpy.context.scene.collection.objects.link(vm)
    import bmesh

    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    bm.to_mesh(vm.data)
    bm.free()
    vm.scale = (0.9, 0.75, 1.85)
    vm.location = (u1 - 6.0, -(half - 0.45), 0.925)
    vm.data.materials.append(_emit("vending_glow", (0.85, 0.92, 1.0), 4.0))
    put(vm)
    # cans and papers on the ground along the walls
    can_cols = [(0.55, 0.05, 0.05), (0.1, 0.25, 0.55), (0.7, 0.7, 0.72), (0.1, 0.4, 0.15)]
    for k in range(18):
        side = rnd.choice((-1, 1))
        u = rnd.uniform(u0 + 1.0, u1 - 2.0)
        y = side * (half - rnd.uniform(0.15, 0.6))
        bpy.ops.mesh.primitive_cylinder_add(radius=0.033, depth=0.122, vertices=24, location=(u, y, 0.033))
        can = bpy.context.active_object
        can.name = f"can{k}"
        can.rotation_euler = (math.pi / 2, 0.0, rnd.uniform(0, math.pi))
        m = bpy.data.materials.new(f"can_paint{k}")
        m.use_nodes = True
        _paint(m, rnd.choice(can_cols), 0.3, coat=0.3, metallic=0.6)
        can.data.materials.append(m)
        put(can)
    paper = bpy.data.materials.new("litter_paper")
    paper.use_nodes = True
    _paint(paper, (0.6, 0.58, 0.52), 0.85, coat=0.0)
    for k in range(14):
        u = rnd.uniform(u0 + 1.0, u1 - 2.0)
        y = rnd.uniform(-half + 0.2, half - 0.2)
        bpy.ops.mesh.primitive_plane_add(size=rnd.uniform(0.12, 0.25), location=(u, y, 0.002))
        pl = bpy.context.active_object
        pl.name = f"paper{k}"
        pl.rotation_euler = (0.0, 0.0, rnd.uniform(0, math.pi))
        pl.data.materials.append(paper)
        put(pl)
    # a steam plume from a vent low on the left wall, caught by the signs' light
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(u1 - 7.5, half - 0.6, 1.2))
    steam = bpy.context.active_object
    steam.name = "steam"
    steam.scale = (0.5, 0.5, 1.2)
    sm = bpy.data.materials.new("steam")
    sm.use_nodes = True
    nt = sm.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    noise = nt.nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 3.0
    noise.inputs["Detail"].default_value = 6.0
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].position, ramp.color_ramp.elements[1].position = 0.5, 0.75
    vol = nt.nodes.new("ShaderNodeVolumePrincipled")
    vol.inputs["Color"].default_value = (0.9, 0.9, 0.92, 1.0)
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    scale = nt.nodes.new("ShaderNodeMath")
    scale.operation = "MULTIPLY"
    scale.inputs[1].default_value = 0.6
    nt.links.new(noise.outputs["Fac"], ramp.inputs["Fac"])
    nt.links.new(ramp.outputs["Color"], scale.inputs[0])
    nt.links.new(scale.outputs[0], vol.inputs["Density"])
    nt.links.new(vol.outputs[0], out.inputs["Volume"])
    steam.data.materials.append(sm)
    put(steam)
    print(f"[photoreal] dressed: bags wrinkled, 4 signs, vending machine, 18 cans, 14 papers, steam (alley {length:.1f} m)")


def _paint(mat, rgb, rough, coat=1.0, metallic=0.0):
    mat.use_nodes = True
    b = _bsdf(mat)
    b.inputs["Base Color"].default_value = (*rgb, 1.0)
    b.inputs["Roughness"].default_value = rough
    b.inputs["Metallic"].default_value = metallic
    for key in ("Coat Weight", "Clearcoat"):
        if key in b.inputs:
            b.inputs[key].default_value = coat
            break
    for key in ("Coat Roughness", "Clearcoat Roughness"):
        if key in b.inputs:
            b.inputs[key].default_value = 0.04
            break


def upgrade_robot(prefix: str = "jumper", eyes=(0.0, 0.85, 1.0), eye_strength: float = 9.0,
                  shell=None) -> None:
    """The robot's plastics and the eyes on its visor. `shell` (r, g, b) repaints the red parts (the
    rival is black); `eyes` is the eyes' colour. Materials are shared by colour, so a rival needs
    its own prefix and is repainted after this has run for the hero."""
    for mat in bpy.data.materials:
        if not mat.name.startswith("jumper_"):
            continue
        r, g, bl = (float(v) for v in mat.name.split("_")[1:4])
        if r > 0.6 and g < 0.2:
            _paint(mat, shell or (0.62, 0.02, 0.015), 0.28)
        elif r > 0.75 and g > 0.75:
            _paint(mat, (0.82, 0.82, 0.8), 0.32, coat=0.6)
        elif r > 0.5:
            _paint(mat, (0.55, 0.56, 0.58), 0.38, coat=0.0, metallic=0.7)
        else:
            _paint(mat, (0.008, 0.008, 0.01), 0.12, coat=1.0)
    for obj in bpy.data.objects:  # smooth the scanned-looking facets, keep the moulded edges
        if obj.type == "MESH" and obj.name.startswith(f"{prefix}:") and obj.name.endswith(":mesh"):
            obj.data.shade_smooth()
            if hasattr(obj.data, "set_sharp_from_angle"):
                obj.data.set_sharp_from_angle(angle=math.radians(35))
    holder = bpy.data.objects.get(f"{prefix}:display_module_link")
    part = bpy.data.objects.get(f"{prefix}:display_module_link:mesh")
    if holder is None or part is None:
        print("[photoreal] no visor found; eyes skipped")
        return
    # the visor's front face in its holder's frame (the part may carry its own offset)
    xs = [(part.matrix_local @ v.co) for v in part.data.vertices]
    x_front = max(p.x for p in xs)
    ys = [p.y for p in xs]
    zs = [p.z for p in xs]
    yc, zc = (max(ys) + min(ys)) / 2, (max(zs) + min(zs)) / 2
    w, h = (max(ys) - min(ys)), (max(zs) - min(zs))
    # (10/28, the user: the eyes looked stuck on, floating) The eyes are the display module's own
    # surface lit, not decals in front of it: its material becomes a gloss-black screen whose
    # emission is masked to the two slanted eye shapes, worked out in the part's own coordinates, on
    # the faces looking forward -- so the glow lies on the curved visor and turns with it. A faint
    # backlight bleeds round them, as on a real panel.
    R = part.matrix_local.to_3x3().inverted()
    t = part.matrix_local.translation
    fwd, lat, up = (R @ Vector(a) for a in ((1, 0, 0), (0, 1, 0), (0, 0, 1)))
    centre = R @ (Vector((x_front, yc, zc)) - t)
    mat = bpy.data.materials.new(f"{prefix}_screen")
    mat.use_nodes = True
    nt = mat.node_tree
    b = next(n for n in nt.nodes if n.type == "BSDF_PRINCIPLED")
    b.inputs["Base Color"].default_value = (0.006, 0.006, 0.008, 1)
    b.inputs["Roughness"].default_value = 0.08
    if "Coat Weight" in b.inputs:
        b.inputs["Coat Weight"].default_value = 1.0
    tc = nt.nodes.new("ShaderNodeTexCoord")

    def vec(op, a, bb=None, value=None):
        n = nt.nodes.new("ShaderNodeVectorMath")
        n.operation = op
        nt.links.new(a, n.inputs[0])
        if bb is not None:
            if isinstance(bb, Vector):
                n.inputs[1].default_value = bb
            else:
                nt.links.new(bb, n.inputs[1])
        return n.outputs["Value" if op == "DOT_PRODUCT" else "Vector"]

    def math_(op, a, bb=None, vb=0.0):
        n = nt.nodes.new("ShaderNodeMath")
        n.operation = op
        nt.links.new(a, n.inputs[0])
        if bb is None:
            n.inputs[1].default_value = vb
        else:
            nt.links.new(bb, n.inputs[1])
        return n.outputs[0]

    def edge(x, half, soft):  # 1 inside |x| < half, 0 outside, a soft rim
        mr = nt.nodes.new("ShaderNodeMapRange")
        mr.interpolation_type = "SMOOTHSTEP"
        nt.links.new(math_("ABSOLUTE", x), mr.inputs["Value"])
        mr.inputs["From Min"].default_value, mr.inputs["From Max"].default_value = half, half - soft
        return mr.outputs["Result"]

    rel = vec("SUBTRACT", tc.outputs["Object"], centre)
    u0 = vec("DOT_PRODUCT", rel, lat)
    v0 = vec("DOT_PRODUCT", rel, up)
    facing = nt.nodes.new("ShaderNodeMapRange")
    nt.links.new(vec("DOT_PRODUCT", tc.outputs["Normal"], fwd), facing.inputs["Value"])
    facing.inputs["From Min"].default_value, facing.inputs["From Max"].default_value = 0.35, 0.6
    ew, eh = 0.16 * w, 0.42 * h
    eyes_mask = None
    for side in (-1, 1):  # two slanted eyes, the outer edge higher (the official art's look)
        u = math_("SUBTRACT", u0, vb=side * 0.2 * w)
        v = math_("SUBTRACT", math_("SUBTRACT", v0, math_("MULTIPLY", u, vb=side * 0.3 * eh / ew)), vb=0.1 * eh)
        m = math_("MULTIPLY", edge(u, ew / 2, 0.0012), edge(v, eh / 2, 0.0012))
        eyes_mask = m if eyes_mask is None else math_("MAXIMUM", eyes_mask, m)
    lit = math_("MULTIPLY", eyes_mask, facing.outputs["Result"])
    bleed = math_("MULTIPLY", facing.outputs["Result"], vb=0.015)
    strength = math_("ADD", math_("MULTIPLY", lit, vb=eye_strength), bleed)
    b.inputs["Emission Color"].default_value = (*eyes, 1.0)
    nt.links.new(strength, b.inputs["Emission Strength"])
    part.data.materials.clear()
    part.data.materials.append(mat)
    print(f"[photoreal] robot {prefix}: paint, eyes {eyes} on a {w * 100:.1f} x {h * 100:.1f} cm visor")
