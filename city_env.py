"""Environnement « ville » : quartier low-poly téléchargé (Assets/ville/source/Untitled.glb, export Sketchfab
« neighbourhood-city-modular-lowpoly », environ 42 x 39 m, immeubles de 10 à 14 m, rues de 2 à 6 m de large,
lampadaires, arbres, plots, barrières, une voiture).

- Rendu : géométrie du GLB regroupée par matériau (une vingtaine d'appels de rendu), textures de couleur
  de base, transparence des feuillages traitée en découpe (alpha < 0,5 jeté), éclairage et brouillard
  comme la forêt. La ville est relevée de 2 cm pour ne pas scintiller avec le sol quadrillé de la scène.
- Collisions : les triangles sont échantillonnés (pas de 5 cm) et voxelisés à 10 cm entre 0,15 m et
  COLLISION_Y_MAX (texels transparents des feuillages ignorés), puis les voxels sont fusionnés en boîtes
  (fusion gloutonne). Les boîtes enveloppent la surface vue à 10 cm près. Résultat mis en cache dans
  Assets/ville/ville_collision.npz (recalculé si le GLB change).
- Enceinte : murs de ENCLOSURE_H m aux textures procédurales des murs troués (une par côté) autour de l'emprise du quartier (comme l'enceinte des murs
  troués) et plafond invisible à cette hauteur : pas de contournement par l'extérieur ni par le haut.
- Départ et cible tirés à partir de la graine n'importe où dans l'espace libre de la ville (colonne libre entre
  0,6 et 4,4 m, à plus de CLEARANCE m de tout obstacle), dans la même composante connexe, à au moins MIN_GEODESIC m
  par les rues. La cible n'est jamais devant le drone au départ : son lacet fait au moins MIN_ANGLE_CIBLE° avec la
  direction de la cible (de côté ou dans le dos), face à au moins DEGAGE_AVANT m d'espace libre.

Ville placée en origin (par défaut x = 0, z = 120), loin des autres environnements.
"""

import math
import os
import random
from collections import deque

import moderngl
import numpy as np

from collision import Box
from realisme import inject_shadow

HERE = os.path.dirname(os.path.abspath(__file__))
CITY_DIR = os.path.join(HERE, "Assets", "ville")
CITY_GLB = os.path.join(CITY_DIR, "source", "Untitled.glb")
CACHE_PATH = os.path.join(CITY_DIR, "ville_collision.npz")

VOXEL = 0.10               # résolution de la voxelisation des collisions (m)
SAMPLE_STEP = 0.05         # pas d'échantillonnage des triangles (m)
COLLISION_Y_MIN = 0.15     # en dessous : chaussée et trottoirs, le sol (Ground) suffit
COLLISION_Y_MAX = 6.0
ENCLOSURE_H = 6.0          # hauteur de l'enceinte et du plafond invisible (m), = COLLISION_Y_MAX
ENCLOSURE_T = 0.3
ENCLOSURE_MARGIN = 1.0     # marge entre l'emprise du GLB et l'enceinte (m)
LIFT = 0.02                # relève le rendu (évite le z-fighting avec le sol de la scène)
FLY_Y = (0.6, 4.4)         # tranche de hauteur où l'espace libre est cherché (= ymin, ymax de test_dagger)
CLEARANCE = 1.0            # distance min départ / cible - obstacle (m)
MIN_GEODESIC = 25.0        # distance min départ -> cible par l'espace libre (m)
MIN_ANGLE_CIBLE = 90.0     # angle min (°) entre l'axe du drone au départ et la direction de la cible : jamais devant
DEGAGE_AVANT = 3.0         # le drone part face à au moins 3 m d'espace libre
FREE_CELL = 0.20           # cellule de la carte 2D d'espace libre (m), multiple de VOXEL

CITY_VERTEX_SHADER = """
#version 330
uniform mat4 u_proj;
uniform mat4 u_view;
uniform vec3 u_offset;
in vec3 in_pos;
in vec3 in_normal;
in vec2 in_uv;
out vec3 v_world;
out vec3 v_normal;
out vec2 v_uv;
void main() {
    vec3 w = in_pos + u_offset;
    v_world = w;
    v_normal = in_normal;
    v_uv = in_uv;
    gl_Position = u_proj * u_view * vec4(w, 1.0);
}
"""

CITY_FRAGMENT_SHADER = """
#version 330
uniform sampler2D u_tex;
uniform bool u_use_tex;
uniform bool u_alpha_mask;
uniform vec3 u_color;
uniform vec3 u_light_dir;
uniform vec3 u_eye;
in vec3 v_world;
in vec3 v_normal;
in vec2 v_uv;
out vec4 f_color;
void main() {
    vec3 n = normalize(v_normal);
    if (!gl_FrontFacing) n = -n;   // matériaux à double face
    vec4 c = u_use_tex ? texture(u_tex, v_uv) : vec4(1.0);
    if (u_alpha_mask && c.a < 0.5) discard;
    vec3 base = c.rgb * u_color;
    vec3 l = normalize(-u_light_dir);
    float diff = max(dot(n, l), 0.0) * shadow_factor(v_world, n, l);
    vec3 ambient = vec3(0.36, 0.38, 0.40) * (0.6 + 0.4 * n.y) + vec3(0.12);
    vec3 col = base * (ambient + vec3(0.70 * diff));
    float d = length(u_eye - v_world);
    col = mix(col, vec3(0.55, 0.7, 0.9), clamp((d - 40.0) / 90.0, 0.0, 1.0));
    f_color = vec4(col, 1.0);
}
"""
CITY_FRAGMENT_SHADER = inject_shadow(CITY_FRAGMENT_SHADER)


# --- lecture du GLB -------------------------------------------------------------

def _load_city_glb(path):
    """Triangles du GLB en repère monde, regroupés par matériau.

    Retourne (groupes, images, image de béton ou None) ; groupes : clé -> dict(verts (n, 8) f4 [pos, normale, uv], tex (indice
    d'image ou None), color (3,), alpha (bool)), images : indice -> PIL.Image RGBA.
    """
    import io
    from PIL import Image
    import gltf_loader as gl

    gltf, binary = gl._read_glb(path)
    groups = {}

    def material_key(mat_index):
        if mat_index is None:
            return (None, (0.7, 0.7, 0.7), False)
        m = gltf["materials"][mat_index]
        pbr = m.get("pbrMetallicRoughness", {})
        col = tuple(round(float(c), 3) for c in pbr.get("baseColorFactor", (1, 1, 1, 1))[:3])
        tex = gltf["textures"][pbr["baseColorTexture"]["index"]]["source"] if "baseColorTexture" in pbr else None
        return (tex, col, m.get("alphaMode", "OPAQUE") != "OPAQUE")

    def visit(node_index, parent):
        node = gltf["nodes"][node_index]
        world = parent @ gl._node_matrix(node)
        if "mesh" in node:
            normal_mat = np.linalg.inv(world[:3, :3]).T
            for prim in gltf["meshes"][node["mesh"]]["primitives"]:
                if prim.get("mode", 4) != 4:
                    continue
                pos = gl._accessor(gltf, binary, prim["attributes"]["POSITION"]).astype("f8")
                pos = pos @ world[:3, :3].T + world[:3, 3]
                if "NORMAL" in prim["attributes"]:
                    nrm = gl._accessor(gltf, binary, prim["attributes"]["NORMAL"]).astype("f8") @ normal_mat.T
                    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
                else:
                    nrm = np.tile((0.0, 1.0, 0.0), (len(pos), 1))
                if "TEXCOORD_0" in prim["attributes"]:
                    uv = gl._accessor(gltf, binary, prim["attributes"]["TEXCOORD_0"]).astype("f8")
                else:
                    uv = np.zeros((len(pos), 2))
                idx = (gl._accessor(gltf, binary, prim["indices"]).astype("i8") if "indices" in prim
                       else np.arange(len(pos)))
                if np.linalg.det(world[:3, :3]) < 0:   # transformation miroir : on garde l'orientation des faces
                    idx = idx.reshape(-1, 3)[:, ::-1].ravel()
                v = np.hstack([pos, nrm, uv])[idx].astype("f4")   # triangles « à plat » (3 sommets par face)
                groups.setdefault(material_key(prim.get("material")), []).append(v)
        for child in node.get("children", []):
            visit(child, world)

    for root in gltf["scenes"][gltf.get("scene", 0)]["nodes"]:
        visit(root, np.identity(4))

    images = {}

    def image(i):
        if i not in images:
            view = gltf["bufferViews"][gltf["images"][i]["bufferView"]]
            start = view.get("byteOffset", 0)
            images[i] = Image.open(io.BytesIO(binary[start:start + view["byteLength"]])).convert("RGBA")
        return images[i]

    out = {}
    for key, vs in groups.items():
        tex, col, alpha = key
        if tex is not None:
            image(tex)
        out[key] = dict(verts=np.vstack(vs), tex=tex, color=np.array(col, "f4"), alpha=alpha)
    # texture de béton du pack, pour l'enceinte
    concrete = next((i for i, im in enumerate(gltf["images"]) if "granular_concrete_diff" in im.get("name", "")), None)
    return out, images, (image(concrete) if concrete is not None else None)


# --- voxelisation et fusion en boîtes (numba) -------------------------------------

def _numba_kernels():
    from numba import njit

    @njit(cache=True)
    def voxelize(tris, alpha_id, alphas, alpha_shapes, occ, lo, vox, step, y_min, y_max):
        nx, ny, nz = occ.shape
        for t in range(tris.shape[0]):
            a = tris[t, 0]
            b = tris[t, 1]
            c = tris[t, 2]
            if max(a[1], b[1], c[1]) < y_min or min(a[1], b[1], c[1]) > y_max:
                continue
            e = max(np.sqrt(((b[:3] - a[:3]) ** 2).sum()), np.sqrt(((c[:3] - a[:3]) ** 2).sum()),
                    np.sqrt(((c[:3] - b[:3]) ** 2).sum()))
            k = int(math.ceil(e / step)) + 1
            ai = alpha_id[t]
            for i in range(k + 1):
                for j in range(k + 1 - i):
                    u = i / k
                    w = j / k
                    p = a * (1.0 - u - w) + b * u + c * w
                    if p[1] < y_min or p[1] > y_max:
                        continue
                    if ai >= 0:   # feuillage : texel transparent ignoré
                        h = alpha_shapes[ai, 0]
                        wd = alpha_shapes[ai, 1]
                        uu = p[6] - math.floor(p[6])
                        vv = p[7] - math.floor(p[7])
                        if alphas[ai, min(int(vv * h), h - 1), min(int(uu * wd), wd - 1)] < 128:
                            continue
                    ix = int((p[0] - lo[0]) / vox)
                    iy = int((p[1] - lo[1]) / vox)
                    iz = int((p[2] - lo[2]) / vox)
                    if 0 <= ix < nx and 0 <= iy < ny and 0 <= iz < nz:
                        occ[ix, iy, iz] = True
        return occ

    @njit(cache=True)
    def greedy_boxes(occ):
        """Fusion gloutonne des voxels occupés en boîtes (x, puis z, puis y). Retourne (n, 6) : i0 j0 k0 i1 j1 k1."""
        nx, ny, nz = occ.shape
        used = np.zeros_like(occ)
        out = []
        for j in range(ny):
            for k in range(nz):
                for i in range(nx):
                    if not occ[i, j, k] or used[i, j, k]:
                        continue
                    i1 = i + 1
                    while i1 < nx and occ[i1, j, k] and not used[i1, j, k]:
                        i1 += 1
                    k1 = k + 1
                    while k1 < nz:
                        ok = True
                        for ii in range(i, i1):
                            if not occ[ii, j, k1] or used[ii, j, k1]:
                                ok = False
                                break
                        if not ok:
                            break
                        k1 += 1
                    j1 = j + 1
                    while j1 < ny:
                        ok = True
                        for kk in range(k, k1):
                            for ii in range(i, i1):
                                if not occ[ii, j1, kk] or used[ii, j1, kk]:
                                    ok = False
                                    break
                            if not ok:
                                break
                        if not ok:
                            break
                        j1 += 1
                    used[i:i1, j:j1, k:k1] = True
                    out.append((i, j, k, i1, j1, k1))
        res = np.empty((len(out), 6), np.int32)
        for n in range(len(out)):
            for m in range(6):
                res[n, m] = out[n][m]
        return res

    return voxelize, greedy_boxes


def _compute_collision(groups, images):
    """Voxels occupés (grille locale) et boîtes fusionnées, en repère du GLB."""
    voxelize, greedy_boxes = _numba_kernels()
    tris, aid = [], []
    alpha_keys = [k for k, g in groups.items() if g["alpha"] and g["tex"] is not None]
    shapes = np.array([images[groups[k]["tex"]].size[::-1] for k in alpha_keys] or [(1, 1)], np.int64)
    alphas = np.zeros((max(1, len(alpha_keys)), shapes[:, 0].max(), shapes[:, 1].max()), np.uint8)
    for n, k in enumerate(alpha_keys):
        a = np.asarray(images[groups[k]["tex"]])[:, :, 3]
        alphas[n, :a.shape[0], :a.shape[1]] = a
    for key, g in groups.items():
        v = g["verts"].astype("f8").reshape(-1, 3, 8)
        tris.append(v)
        aid.append(np.full(len(v), alpha_keys.index(key) if key in alpha_keys else -1, np.int64))
    tris, aid = np.concatenate(tris), np.concatenate(aid)
    pts = tris[:, :, :3].reshape(-1, 3)
    lo = np.floor(pts.min(axis=0) / VOXEL) * VOXEL - VOXEL
    lo[1] = 0.0
    hi = pts.max(axis=0) + VOXEL
    shape = tuple(int(math.ceil(x)) for x in (hi - lo) / VOXEL)
    shape = (shape[0], int(math.ceil(COLLISION_Y_MAX / VOXEL)), shape[2])
    occ = np.zeros(shape, np.bool_)
    voxelize(tris, aid, alphas, shapes, occ, lo, VOXEL, SAMPLE_STEP, COLLISION_Y_MIN, COLLISION_Y_MAX)
    boxes = greedy_boxes(occ)
    return occ, lo, boxes, pts.min(axis=0), pts.max(axis=0)


# --- carte 2D d'espace libre (tirage du départ et de la cible) ---------------------

def _free_map(occ, lo, footprint_lo, footprint_hi):
    """Grille 2D (cellules FREE_CELL) : distance (m) au plus proche obstacle de la tranche FLY_Y, ou à l'enceinte."""
    from scipy import ndimage
    j0, j1 = int(FLY_Y[0] / VOXEL), int(math.ceil(FLY_Y[1] / VOXEL))
    col = occ[:, j0:j1, :].any(axis=1)                                 # (nx, nz) à 10 cm
    f = int(round(FREE_CELL / VOXEL))
    assert abs(f * VOXEL - FREE_CELL) < 1e-9
    nx, nz = col.shape[0] // f, col.shape[1] // f
    col = col[:nx * f, :nz * f].reshape(nx, f, nz, f).any(axis=(1, 3))  # (nx, nz) à 25 cm
    xs = lo[0] + (np.arange(nx) + 0.5) * FREE_CELL
    zs = lo[2] + (np.arange(nz) + 0.5) * FREE_CELL
    inside = ((xs[:, None] > footprint_lo[0]) & (xs[:, None] < footprint_hi[0]) &
              (zs[None, :] > footprint_lo[2]) & (zs[None, :] < footprint_hi[2]))
    blocked = col | ~inside
    dist = ndimage.distance_transform_edt(~blocked) * FREE_CELL
    return dist, xs, zs


def _bfs(free, start):
    d = np.full(free.shape, -1, np.int32)
    d[start] = 0
    q = deque([start])
    nx, nz = free.shape
    while q:
        i, k = q.popleft()
        for di, dk in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            a, b = i + di, k + dk
            if 0 <= a < nx and 0 <= b < nz and free[a, b] and d[a, b] < 0:
                d[a, b] = d[i, k] + 1
                q.append((a, b))
    return d


class CityAssets:
    """GLB chargé une fois : buffers GPU par matériau, textures, boîtes de collision et carte d'espace libre."""

    def __init__(self, ctx):
        self.ctx = ctx
        self.prog = ctx.program(vertex_shader=CITY_VERTEX_SHADER, fragment_shader=CITY_FRAGMENT_SHADER)
        groups, images, concrete = _load_city_glb(CITY_GLB)
        self.textures = {}
        self.draws = []
        for key, g in groups.items():
            v = g["verts"].copy()
            v[:, 1] += LIFT
            tex = None
            if g["tex"] is not None:
                if g["tex"] not in self.textures:
                    img = images[g["tex"]]
                    t = ctx.texture(img.size, 4, img.tobytes())
                    t.build_mipmaps()
                    t.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
                    t.anisotropy = 4.0
                    self.textures[g["tex"]] = t
                tex = self.textures[g["tex"]]
            vbo = ctx.buffer(v.tobytes())
            vao = ctx.vertex_array(self.prog, [(vbo, "3f 3f 2f", "in_pos", "in_normal", "in_uv")])
            self.draws.append(dict(vao=vao, vbo=vbo, tex=tex, alpha=g["alpha"], color=tuple(float(c) for c in g["color"]),
                                   n=len(v)))
        self.n_triangles = sum(d["n"] for d in self.draws) // 3
        self.concrete = None
        if concrete is not None:
            self.concrete = ctx.texture(concrete.size, 4, concrete.tobytes())
            self.concrete.build_mipmaps()
            self.concrete.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
            self.concrete.anisotropy = 4.0
        self._load_collision(groups, images)

    def _load_collision(self, groups, images):
        stamp = os.path.getsize(CITY_GLB)   # taille plutôt que date : survit à une copie (scp, cp)
        if os.path.exists(CACHE_PATH):
            c = np.load(CACHE_PATH)
            if float(c["stamp"]) == stamp and float(c["voxel"]) == VOXEL and float(c["y_max"]) == COLLISION_Y_MAX:
                self.occ, self.lo, self.boxes = np.unpackbits(c["occ"], count=int(np.prod(c["shape"]))).reshape(
                    tuple(c["shape"])).astype(bool), c["lo"], c["boxes"]
                self.mesh_lo, self.mesh_hi = c["mesh_lo"], c["mesh_hi"]
                return
        self.occ, self.lo, self.boxes, self.mesh_lo, self.mesh_hi = _compute_collision(groups, images)
        tmp = CACHE_PATH + f".{os.getpid()}.tmp.npz"   # écriture atomique : plusieurs vols peuvent démarrer ensemble
        np.savez_compressed(tmp, stamp=stamp, voxel=VOXEL, y_max=COLLISION_Y_MAX, lo=self.lo, boxes=self.boxes,
                            occ=np.packbits(self.occ.ravel()), shape=np.array(self.occ.shape),
                            mesh_lo=self.mesh_lo, mesh_hi=self.mesh_hi)
        os.replace(tmp, CACHE_PATH)


class CityEnv:
    def __init__(self, ctx, assets, seed=0, origin=(0.0, 0.0, 120.0)):
        self.ctx = ctx
        self.assets = assets
        self.seed = int(seed)
        self.origin = np.asarray(origin, dtype="f8")
        a = assets
        # emprise : boîte englobante horizontale du GLB, enceinte juste autour
        self.fp_lo = a.mesh_lo - ENCLOSURE_MARGIN
        self.fp_hi = a.mesh_hi + ENCLOSURE_MARGIN
        self.colliders = []
        for i0, j0, k0, i1, j1, k1 in a.boxes:
            lo = a.lo + VOXEL * np.array([i0, j0, k0])
            hi = a.lo + VOXEL * np.array([i1, j1, k1])
            self.colliders.append(Box(self.origin + (lo + hi) / 2.0, (hi - lo) / 2.0))
        self._build_enclosure()
        self._pick_spawn_target()

    # --- points utiles ----------------------------------------------------------

    @property
    def spawn(self):
        return tuple(self._spawn), self._spawn_yaw

    @property
    def overview_camera(self):
        c = self.origin + (self.fp_lo + self.fp_hi) / 2.0
        return tuple(c + (30.0, 35.0, 30.0)), tuple(c * (1, 0, 1))

    def summary(self):
        return {"boites": len(self.assets.boxes), "triangles": self.assets.n_triangles}

    def clearance(self, p):
        """Distance (m) du point p (monde) au plus proche voxel occupé ou à l'enceinte (précision 10 cm)."""
        a = self.assets
        if getattr(a, "edt", None) is None:
            from scipy import ndimage
            a.edt = ndimage.distance_transform_edt(~a.occ).astype("f4") * VOXEL   # calculée une fois (quelques s)
        q = np.asarray(p, float) - self.origin
        i = np.clip(((q - a.lo) / VOXEL).astype(int), 0, np.array(a.occ.shape) - 1)
        d = float(a.edt[tuple(i)])
        walls = min(q[0] - self.fp_lo[0], self.fp_hi[0] - q[0], q[2] - self.fp_lo[2], self.fp_hi[2] - q[2])
        return min(d, walls)

    # --- génération -------------------------------------------------------------

    def _build_enclosure(self):
        lo, hi = self.fp_lo, self.fp_hi
        H, T = ENCLOSURE_H, ENCLOSURE_T
        cx, cz = (lo[0] + hi[0]) / 2.0, (lo[2] + hi[2]) / 2.0
        hx, hz = (hi[0] - lo[0]) / 2.0, (hi[2] - lo[2]) / 2.0
        walls = [((lo[0] - T / 2.0, H / 2.0, cz), (T / 2.0, H / 2.0, hz + T)),
                 ((hi[0] + T / 2.0, H / 2.0, cz), (T / 2.0, H / 2.0, hz + T)),
                 ((cx, H / 2.0, lo[2] - T / 2.0), (hx, H / 2.0, T / 2.0)),
                 ((cx, H / 2.0, hi[2] + T / 2.0), (hx, H / 2.0, T / 2.0))]
        # textures procédurales des murs troués (building.py), une par côté, deux voisins jamais identiques
        from building import _building_program, add_box_geometry
        from walls_env import WALL_MATERIALS
        mats = list(WALL_MATERIALS)
        random.Random(self.seed + 1000).shuffle(mats)
        verts, idx = [], []
        for k, (c, h) in enumerate(walls):
            self.colliders.append(Box(self.origin + c, h))
            add_box_geometry(verts, idx, self.origin + np.array(c), np.array(h), mat_id=mats[k % len(mats)])
        # plafond invisible (collision seule)
        self.colliders.append(Box(self.origin + (cx, H + 0.5, cz), (hx + 1.0, 0.5, hz + 1.0)))
        self.encl_prog = _building_program(self.ctx)
        self.encl_vbo = self.ctx.buffer(np.array(verts, dtype="f4").tobytes())
        self.encl_ibo = self.ctx.buffer(np.array(idx, dtype="i4").tobytes())
        self.encl_vao = self.ctx.vertex_array(
            self.encl_prog, [(self.encl_vbo, "3f 3f 2f 1f", "in_pos", "in_normal", "in_uv", "in_mat")], self.encl_ibo)

    def _pick_spawn_target(self):
        """Départ et cible n'importe où dans l'espace libre de la ville (≥ MIN_GEODESIC m par les rues), et drone
        orienté au départ à au moins MIN_ANGLE_CIBLE° de la direction de la cible (de côté ou dos à elle), face à une
        rue dégagée sur au moins DEGAGE_AVANT m."""
        a = self.assets
        dist, xs, zs = _free_map(a.occ, a.lo, self.fp_lo + 0.3, self.fp_hi - 0.3)
        free = dist >= 0.5                       # passage pour le drone (rayon 0,14 m + marge)
        ok = dist >= CLEARANCE
        rng = random.Random(self.seed)
        cand = np.argwhere(ok)

        def degage(s, yaw):
            """Longueur (m) libre devant le départ dans la direction yaw (couloir de 0,5 m)."""
            c, sn = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
            L = 0.0
            while L < DEGAGE_AVANT:
                L += FREE_CELL
                i, k = int(round(s[0] + c * L / FREE_CELL)), int(round(s[1] + sn * L / FREE_CELL))
                if not (0 <= i < free.shape[0] and 0 <= k < free.shape[1]) or not free[i, k]:
                    return L
            return L

        for _ in range(400):
            s = tuple(cand[rng.randrange(len(cand))])
            d = _bfs(free, s)
            far = np.argwhere(ok & (d * FREE_CELL >= MIN_GEODESIC))
            if not len(far):
                continue
            t = tuple(far[rng.randrange(len(far))])
            cap = math.degrees(math.atan2(zs[t[1]] - zs[s[1]], xs[t[0]] - xs[s[0]]))
            yaws = [y for y in range(-180, 180, 15)
                    if abs((cap - y + 180.0) % 360.0 - 180.0) >= MIN_ANGLE_CIBLE and degage(s, y) >= DEGAGE_AVANT]
            if yaws:
                yaw = float(rng.choice(yaws))
                break
        else:
            raise RuntimeError("ville : pas de couple départ / cible valable")
        self._spawn_yaw = yaw
        self._spawn = self.origin + (xs[s[0]], 1.5, zs[s[1]])
        self.target = self.origin + (xs[t[0]], rng.uniform(1.0, 3.0), zs[t[1]])
        self.geodesic = float(d[t] * FREE_CELL)
        self.angle_cible = abs((cap - yaw + 180.0) % 360.0 - 180.0)

    # --- rendu ------------------------------------------------------------------

    def render(self, camera, aspect, light_dir=(-0.4, -1.0, -0.3)):
        prog = self.assets.prog
        prog["u_proj"].write(camera.projection(aspect).T.copy().tobytes())
        prog["u_view"].write(camera.view().T.copy().tobytes())
        prog["u_light_dir"].value = tuple(light_dir)
        prog["u_eye"].value = tuple(float(c) for c in camera.position)
        prog["u_offset"].value = tuple(float(c) for c in self.origin)
        prog["u_tex"].value = 0
        for d in self.assets.draws:
            if d["tex"] is not None:
                d["tex"].use(location=0)
            prog["u_use_tex"].value = d["tex"] is not None
            prog["u_alpha_mask"].value = d["alpha"]
            prog["u_color"].value = d["color"]
            d["vao"].render()
        # enceinte : shader et matériaux des murs troués
        ep = self.encl_prog
        ep["u_proj"].write(camera.projection(aspect).T.copy().tobytes())
        ep["u_view"].write(camera.view().T.copy().tobytes())
        ep["u_model"].write(np.identity(4, dtype="f4").tobytes())
        ep["u_light_dir"].value = tuple(light_dir)
        ep["u_eye"].value = tuple(float(c) for c in camera.position)
        self.encl_vao.render()

    def release(self):
        self.encl_vao.release()
        self.encl_vbo.release()
        self.encl_ibo.release()


def _box_verts(center, half, uv_scale=1.0):
    """Triangles « à plat » d'une boîte [pos, normale, uv], uv en mètres x uv_scale (texture répétée)."""
    out = []
    for axis in range(3):
        u_ax, v_ax = [(2, 1), (0, 2), (0, 1)][axis]
        for sign in (-1.0, 1.0):
            n = np.zeros(3)
            n[axis] = sign
            quad = []
            for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
                p = center.copy()
                p[axis] += sign * half[axis]
                p[u_ax] += a * half[u_ax]
                p[v_ax] += b * half[v_ax]
                quad.append([*p, *n, p[u_ax] * uv_scale, p[v_ax] * uv_scale])
            out += [quad[0], quad[1], quad[2], quad[0], quad[2], quad[3]]
    return np.array(out)
