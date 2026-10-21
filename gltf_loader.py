"""Chargeur GLB minimal (glTF 2.0 binaire), sans dépendance autre que numpy.

Ne garde que la géométrie (positions, normales, indices) et une couleur unie par
matériau : baseColorFactor, ou couleur moyenne de la texture baseColor si Pillow
est installé. Suffisant pour afficher le drone et calculer sa boîte englobante.
"""

import io
import json
import struct

import numpy as np

_COMPONENT_DTYPES = {5120: "i1", 5121: "u1", 5122: "i2", 5123: "u2", 5125: "u4", 5126: "f4"}
_TYPE_SIZES = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}


def _read_glb(path):
    with open(path, "rb") as f:
        data = f.read()
    magic, _, _ = struct.unpack_from("<4sII", data, 0)
    if magic != b"glTF":
        raise ValueError(f"{path} n'est pas un fichier GLB")
    json_len, _ = struct.unpack_from("<II", data, 12)
    gltf = json.loads(data[20:20 + json_len])
    bin_start = 20 + json_len
    bin_len, _ = struct.unpack_from("<II", data, bin_start)
    return gltf, data[bin_start + 8:bin_start + 8 + bin_len]


def _accessor(gltf, binary, index):
    acc = gltf["accessors"][index]
    view = gltf["bufferViews"][acc["bufferView"]]
    dtype = np.dtype(_COMPONENT_DTYPES[acc["componentType"]])
    ncomp = _TYPE_SIZES[acc["type"]]
    offset = view.get("byteOffset", 0) + acc.get("byteOffset", 0)
    stride = view.get("byteStride", 0)
    count = acc["count"]
    if stride and stride != dtype.itemsize * ncomp:
        raw = np.frombuffer(binary, dtype="u1", count=stride * (count - 1) + dtype.itemsize * ncomp, offset=offset)
        rows = np.lib.stride_tricks.as_strided(raw, shape=(count, dtype.itemsize * ncomp), strides=(stride, 1))
        arr = rows.copy().view(dtype).reshape(count, ncomp)
    else:
        arr = np.frombuffer(binary, dtype=dtype, count=count * ncomp, offset=offset).reshape(count, ncomp)
    return arr if ncomp > 1 else arr.ravel()


def _node_matrix(node):
    if "matrix" in node:
        return np.array(node["matrix"], dtype="f8").reshape(4, 4).T  # glTF est column-major
    t = np.identity(4)
    t[:3, 3] = node.get("translation", (0, 0, 0))
    x, y, z, w = node.get("rotation", (0, 0, 0, 1))
    r = np.identity(4)
    r[:3, :3] = [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]
    s = np.diag([*node.get("scale", (1, 1, 1)), 1.0])
    return t @ r @ s


def _material_color(gltf, binary, material_index):
    if material_index is None:
        return (0.7, 0.7, 0.7)
    pbr = gltf["materials"][material_index].get("pbrMetallicRoughness", {})
    factor = np.array(pbr.get("baseColorFactor", (1, 1, 1, 1))[:3])
    if "baseColorTexture" in pbr:
        factor = factor * _texture_mean(gltf, binary, pbr["baseColorTexture"]["index"])
    return tuple(float(c) for c in factor)


def _texture_mean(gltf, binary, texture_index):
    try:
        from PIL import Image
    except ImportError:
        return np.array([0.5, 0.5, 0.5])
    image = gltf["images"][gltf["textures"][texture_index]["source"]]
    view = gltf["bufferViews"][image["bufferView"]]
    start = view.get("byteOffset", 0)
    img = Image.open(io.BytesIO(binary[start:start + view["byteLength"]]))
    img.draft("RGB", (64, 64))  # décodage JPEG réduit, bien plus rapide
    rgb = np.asarray(img.convert("RGB").resize((32, 32)), dtype="f4") / 255.0
    return rgb.reshape(-1, 3).mean(axis=0) ** 2.2  # sRGB -> linéaire (approx.)


def _get_texture_image(gltf, binary, material_index):
    if material_index is None:
        return None
    pbr = gltf["materials"][material_index].get("pbrMetallicRoughness", {})
    if "baseColorTexture" not in pbr:
        return None
    try:
        from PIL import Image
    except ImportError:
        return None
    tex_idx = pbr["baseColorTexture"]["index"]
    src = gltf["textures"][tex_idx]["source"]
    img_info = gltf["images"][src]
    if "bufferView" in img_info:
        bv = gltf["bufferViews"][img_info["bufferView"]]
        start = bv.get("byteOffset", 0)
        data = binary[start:start + bv["byteLength"]]
        try:
            return Image.open(io.BytesIO(data)).convert("RGB")
        except Exception:
            return None
    return None


def load_glb(path):
    """Retourne une liste de parties : (verts[pos(3), norm(3), uv(2)], idx, color, texture_img)."""
    gltf, binary = _read_glb(path)
    parts = []

    def visit(node_index, parent):
        node = gltf["nodes"][node_index]
        world = parent @ _node_matrix(node)
        if "mesh" in node:
            normal_mat = np.linalg.inv(world[:3, :3]).T
            for prim in gltf["meshes"][node["mesh"]]["primitives"]:
                if prim.get("mode", 4) != 4:  # triangles uniquement
                    continue
                pos = _accessor(gltf, binary, prim["attributes"]["POSITION"]).astype("f8")
                pos = pos @ world[:3, :3].T + world[:3, 3]
                if "NORMAL" in prim["attributes"]:
                    nrm = _accessor(gltf, binary, prim["attributes"]["NORMAL"]).astype("f8") @ normal_mat.T
                    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
                else:
                    nrm = np.tile((0.0, 1.0, 0.0), (len(pos), 1))
                if "TEXCOORD_0" in prim["attributes"]:
                    uv = _accessor(gltf, binary, prim["attributes"]["TEXCOORD_0"]).astype("f4")
                else:
                    uv = np.zeros((len(pos), 2), dtype="f4")
                if "indices" in prim:
                    idx = _accessor(gltf, binary, prim["indices"]).astype("i4")
                else:
                    idx = np.arange(len(pos), dtype="i4")
                verts = np.hstack([pos, nrm, uv]).astype("f4")
                mat_idx = prim.get("material")
                col = _material_color(gltf, binary, mat_idx)
                tex_img = _get_texture_image(gltf, binary, mat_idx)
                parts.append((verts, idx, col, tex_img))
        for child in node.get("children", []):
            visit(child, world)

    scene = gltf["scenes"][gltf.get("scene", 0)]
    for root in scene["nodes"]:
        visit(root, np.identity(4))
    return parts


_IMAGE_CACHE = {}


def load_gltf(path):
    """Charge un .gltf (JSON + .bin et images externes, ex. pack Quaternius).

    Retourne une liste de primitives sous forme de dicts :
    verts [pos(3), norm(3), uv(2)], idx, color, image (PIL RGBA ou None),
    alpha_mask (feuilles découpées par la transparence), material (nom).
    """
    import os
    base = os.path.dirname(os.path.abspath(path))
    with open(path, encoding="utf-8") as f:
        gltf = json.load(f)
    buffers = []
    for b in gltf["buffers"]:
        with open(os.path.join(base, b["uri"]), "rb") as f:
            buffers.append(f.read())
    if len(buffers) != 1:
        raise ValueError(f"{path} : un seul buffer attendu")
    binary = buffers[0]

    def image_for(material_index):
        if material_index is None:
            return None
        tex = gltf["materials"][material_index].get("pbrMetallicRoughness", {}).get("baseColorTexture")
        if tex is None:
            return None
        from PIL import Image
        uri = gltf["images"][gltf["textures"][tex["index"]]["source"]]["uri"]
        full = os.path.join(base, uri)
        if full not in _IMAGE_CACHE:  # les textures sont partagées entre modèles
            _IMAGE_CACHE[full] = Image.open(full).convert("RGBA")
        return _IMAGE_CACHE[full]

    prims = []

    def visit(node_index, parent):
        node = gltf["nodes"][node_index]
        world = parent @ _node_matrix(node)
        if "mesh" in node:
            normal_mat = np.linalg.inv(world[:3, :3]).T
            for prim in gltf["meshes"][node["mesh"]]["primitives"]:
                if prim.get("mode", 4) != 4:
                    continue
                attrs = prim["attributes"]
                pos = _accessor(gltf, binary, attrs["POSITION"]).astype("f8") @ world[:3, :3].T + world[:3, 3]
                nrm = _accessor(gltf, binary, attrs["NORMAL"]).astype("f8") @ normal_mat.T
                nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
                uv = (_accessor(gltf, binary, attrs["TEXCOORD_0"]).astype("f4")
                      if "TEXCOORD_0" in attrs else np.zeros((len(pos), 2), dtype="f4"))
                idx = (_accessor(gltf, binary, prim["indices"]).astype("i4")
                       if "indices" in prim else np.arange(len(pos), dtype="i4"))
                mat_index = prim.get("material")
                material = gltf["materials"][mat_index] if mat_index is not None else {}
                factor = material.get("pbrMetallicRoughness", {}).get("baseColorFactor", (1, 1, 1, 1))
                prims.append({
                    "verts": np.hstack([pos, nrm, uv]).astype("f4"),
                    "idx": idx,
                    "color": tuple(float(c) for c in factor[:3]),
                    "image": image_for(mat_index),
                    "alpha_mask": material.get("alphaMode") == "MASK",
                    "material": material.get("name", ""),
                })
        for child in node.get("children", []):
            visit(child, world)

    for root in gltf["scenes"][gltf.get("scene", 0)]["nodes"]:
        visit(root, np.identity(4))
    return prims


def bounds(parts):
    """Boîte englobante (min, max) de toutes les parties."""
    pts = np.vstack([p[0][:, :3] for p in parts])
    return pts.min(axis=0), pts.max(axis=0)
