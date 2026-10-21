"""Environnement « forêt » : forêt dense de pins à tronc droit, dessinée en instanciation
(un seul appel de rendu par modèle).

- Arbres : pin « fir_spruce » téléchargé, préparé par tools/prepare_pine.py (objet des
  branches supprimé, il reste le fût et les cartes d'aiguilles, qui commencent vers 22 %
  de la hauteur), répartis aléatoirement avec une distance minimale entre troncs.
- Sol forestier procédural (terre et mousse), sans autre décor.
- Collisions : cylindre vertical par tronc, deux cylindres pour le feuillage.
- Départ en bordure sud, cible (coordonnée connue) en bordure nord.

Tout est tiré à partir d'une graine. Les modèles et textures sont chargés une seule
fois (ForestAssets) ; régénérer la forêt ne refait que le placement.
"""

import math
import os
import random

import moderngl
import numpy as np

from collision import VerticalCylinder

HERE = os.path.dirname(os.path.abspath(__file__))
PINE_DIR = os.path.join(HERE, "Assets", "models", "pine")

FOREST_VERTEX_SHADER = """
#version 330
uniform mat4 u_proj;
uniform mat4 u_view;
in vec3 in_pos;
in vec3 in_normal;
in vec2 in_uv;
in vec4 in_m0;
in vec4 in_m1;
in vec4 in_m2;
in vec4 in_m3;
out vec3 v_world;
out vec3 v_normal;
out vec2 v_uv;
void main() {
    mat4 m = mat4(in_m0, in_m1, in_m2, in_m3);
    vec4 w = m * vec4(in_pos, 1.0);
    v_world = w.xyz;
    v_normal = mat3(m) * in_normal;
    v_uv = in_uv;
    gl_Position = u_proj * u_view * w;
}
"""

FOREST_FRAGMENT_SHADER = """
#version 330
uniform sampler2D u_tex;
uniform bool u_use_tex;
uniform bool u_alpha_mask;
uniform bool u_ground;
uniform vec3 u_color;
uniform vec3 u_light_dir;
uniform vec3 u_eye;
in vec3 v_world;
in vec3 v_normal;
in vec2 v_uv;
out vec4 f_color;

float hash(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453123); }
float noise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    vec2 u = f * f * (3.0 - 2.0 * f);
    return mix(mix(hash(i), hash(i + vec2(1.0, 0.0)), u.x),
               mix(hash(i + vec2(0.0, 1.0)), hash(i + vec2(1.0, 1.0)), u.x), u.y);
}

void main() {
    vec3 n = normalize(v_normal);
    if (!gl_FrontFacing) n = -n;  // feuilles et herbes à double face
    vec4 c = u_use_tex ? texture(u_tex, v_uv) : vec4(1.0);
    if (u_alpha_mask && c.a < 0.5) discard;
    vec3 base = c.rgb * u_color;
    if (u_ground) {
        // sol forestier : terre, mousse et litière de feuilles
        vec2 p = v_world.xz;
        float big = noise(p * 0.15) * 0.6 + noise(p * 0.6) * 0.4;
        float fine = noise(p * 6.0);
        vec3 dirt = vec3(0.30, 0.22, 0.14);
        vec3 moss = vec3(0.24, 0.34, 0.14);
        vec3 litter = vec3(0.42, 0.30, 0.16);
        base = mix(dirt, moss, smoothstep(0.35, 0.65, big));
        base = mix(base, litter, smoothstep(0.7, 0.9, fine) * 0.5);
        base *= 0.9 + 0.2 * noise(p * 20.0);
    }
    vec3 l = normalize(-u_light_dir);
    float diff = max(dot(n, l), 0.0);
    vec3 ambient = vec3(0.36, 0.38, 0.40) * (0.6 + 0.4 * n.y) + vec3(0.12);
    vec3 col = base * (ambient + vec3(0.70 * diff));
    float d = length(u_eye - v_world);
    col = mix(col, vec3(0.55, 0.7, 0.9), clamp((d - 40.0) / 90.0, 0.0, 1.0));
    f_color = vec4(col, 1.0);
}
"""

# Pin normalisé à une hauteur de 1 (voir tools/prepare_pine.py) : échelle = hauteur en mètres
PINE_SCALE = (11.0, 14.0)        # hauteur des pins (m)
PINE_TRUNK_RADIUS = 0.016        # rayon du fût vers 15 % de la hauteur (au-dessus de l'évasement)
PINE_HEIGHT = 1.0
PINE_FOLIAGE = [(0.22, 0.6, 0.115), (0.6, 1.0, 0.075)]  # (y min, y max, rayon) des cylindres de feuillage
TREE_SPACING = 1.6               # demi-distance minimale entre deux troncs (m)



def _instance_matrix(x, z, yaw, scale, y=0.0):
    c, s = math.cos(yaw), math.sin(yaw)
    m = np.array([[c * scale, 0, s * scale, x],
                  [0, scale, 0, y],
                  [-s * scale, 0, c * scale, z],
                  [0, 0, 0, 1]], dtype="f4")
    return m


class ForestAssets:
    """Modèles du pack chargés une fois : buffers GPU, textures et mesures utiles."""

    def __init__(self, ctx):
        self.ctx = ctx
        self.prog = ctx.program(vertex_shader=FOREST_VERTEX_SHADER, fragment_shader=FOREST_FRAGMENT_SHADER)
        self.textures = {}
        self.models = {}  # nom -> {"prims": [...], "bbox": (lo, hi)}
        self.models["Pine"] = self._load_pine()
        # sol forestier : un grand quad
        quad = np.array([[-1, 0, -1, 0, 1, 0, 0, 0], [1, 0, -1, 0, 1, 0, 1, 0],
                         [1, 0, 1, 0, 1, 0, 1, 1], [-1, 0, 1, 0, 1, 0, 0, 1]], dtype="f4")
        self.ground_vbo = ctx.buffer(quad.tobytes())
        self.ground_ibo = ctx.buffer(np.array([0, 2, 1, 0, 3, 2], dtype="i4").tobytes())

    def _texture(self, img):
        key = id(img)
        if key not in self.textures:
            tex = self.ctx.texture(img.size, 4, img.tobytes())
            tex.build_mipmaps()
            self._preserve_alpha_coverage(tex, img)
            tex.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
            tex.anisotropy = 4.0
            self.textures[key] = tex
        return self.textures[key]

    @staticmethod
    def _preserve_alpha_coverage(tex, img, threshold=0.5):
        """Les mipmaps moyennent l'alpha : au loin, le feuillage passerait sous le seuil de
        découpe et disparaîtrait. On réécrit chaque niveau en amplifiant l'alpha pour garder
        la même proportion de pixels opaques qu'au niveau 0."""
        from PIL import Image
        alpha0 = np.asarray(img, dtype="f4")[:, :, 3] / 255.0
        coverage = (alpha0 >= threshold).mean()
        if coverage in (0.0, 1.0):
            return
        w, h = img.size
        level = 1
        while w > 1 or h > 1:
            w, h = max(1, w // 2), max(1, h // 2)
            small = np.asarray(img.resize((w, h), Image.BOX), dtype="f4")
            a = small[:, :, 3] / 255.0
            lo, hi = 1.0, 16.0
            for _ in range(12):  # recherche du facteur qui redonne la couverture d'origine
                k = (lo + hi) / 2.0
                if (np.clip(a * k, 0, 1) >= threshold).mean() < coverage:
                    lo = k
                else:
                    hi = k
            small[:, :, 3] = np.clip(a * hi, 0, 1) * 255.0
            tex.write(small.astype("u1").tobytes(), level=level)
            level += 1

    def _load_pine(self):
        from PIL import Image
        data = np.load(os.path.join(PINE_DIR, "pine.npz"))
        out = []
        for part, tex_name, alpha in (("trunk", "bark.jpg", False), ("foliage", "needles.png", True)):
            verts = data[part].astype("f4")
            img = Image.open(os.path.join(PINE_DIR, tex_name)).convert("RGBA")
            out.append({
                "vbo": self.ctx.buffer(verts.tobytes()),
                "ibo": self.ctx.buffer(np.arange(len(verts), dtype="i4").tobytes()),
                "texture": self._texture(img),
                "alpha_mask": alpha,
                "color": (1.0, 1.0, 1.0),
            })
        pts = np.vstack([data["trunk"][:, :3], data["foliage"][:, :3]])
        return {"prims": out, "bbox": (pts.min(axis=0), pts.max(axis=0))}


class ForestEnv:
    def __init__(self, ctx, assets, seed=0, origin=(-80.0, 0.0, 0.0), size=50.0, tree_density=0.07):
        self.ctx = ctx
        self.assets = assets
        self.seed = int(seed)
        self.origin = np.asarray(origin, dtype="f8")
        self.size = size
        self.tree_density = tree_density
        self.colliders = []
        self.obstacles = []  # vérité terrain : arbres
        self.instances = {}  # nom -> liste de matrices 4x4
        self._generate()
        self._build_vaos()

    # --- points utiles ----------------------------------------------------------

    @property
    def spawn(self):
        return tuple(self.origin + (0.0, 1.5, self.size / 2.0 - 2.0)), -90.0

    @property
    def overview_camera(self):
        h = self.size / 2.0
        return tuple(self.origin + (h + 12.0, 22.0, h + 12.0)), tuple(self.origin + (0.0, 0.0, 0.0))

    def summary(self):
        kinds = [o["kind"] for o in self.obstacles]
        return {k: kinds.count(k) for k in ("arbre",)}

    # --- génération -------------------------------------------------------------

    def _generate(self):
        rng = random.Random(self.seed)
        h = self.size / 2.0
        ox, oz = self.origin[0], self.origin[2]
        spawn = np.array([0.0, h - 2.0])
        self.target = self.origin + (rng.uniform(-h + 4.0, h - 4.0), 1.5, -h + 3.0)
        target = np.array([self.target[0] - ox, self.target[2] - oz])
        occupied = []  # (x, z, rayon) en coordonnées locales

        def is_free(p, r, clear_spawn=4.0, clear_target=2.5):
            if np.hypot(*(p - spawn)) < clear_spawn + r or np.hypot(*(p - target)) < clear_target + r:
                return False
            return all(np.hypot(p[0] - x, p[1] - z) >= r + rr for x, z, rr in occupied)

        def add(name, x, z, yaw, scale):
            self.instances.setdefault(name, []).append(_instance_matrix(ox + x, oz + z, yaw, scale))

        def rotate(lx, lz, yaw):
            c, s = math.cos(yaw), math.sin(yaw)
            return c * lx + s * lz, -s * lx + c * lz

        # Pins
        n_trees = int(self.tree_density * self.size * self.size)
        for _ in range(n_trees * 40):
            if len(self.obstacles) >= n_trees:
                break
            p = np.array([rng.uniform(-h, h), rng.uniform(-h, h)])
            if not is_free(p, TREE_SPACING):
                continue
            scale, yaw = rng.uniform(*PINE_SCALE), rng.uniform(0, 2 * math.pi)
            add("Pine", p[0], p[1], yaw, scale)
            occupied.append((p[0], p[1], TREE_SPACING))
            wx, wz = ox + p[0], oz + p[1]
            self.colliders.append(VerticalCylinder((wx, wz), PINE_TRUNK_RADIUS * scale, -0.5, PINE_HEIGHT * scale))
            for y0, y1, r in PINE_FOLIAGE:
                self.colliders.append(VerticalCylinder((wx, wz), r * scale, y0 * scale, y1 * scale))
            self.obstacles.append({"kind": "arbre", "model": "Pine", "center": (wx, 0.0, wz),
                                   "radius": PINE_TRUNK_RADIUS * scale, "height": PINE_HEIGHT * scale,
                                   "foliage_from": PINE_FOLIAGE[0][0] * scale})

    def _build_vaos(self):
        prog = self.assets.prog
        self.draws = []  # (vao, prim, buffer d'instances, nombre)
        self.instance_buffers = []
        for name, mats in self.instances.items():
            data = np.stack([m.T for m in mats]).astype("f4")  # colonnes de la matrice (column-major)
            inst = self.ctx.buffer(data.tobytes())
            self.instance_buffers.append(inst)
            for prim in self.assets.models[name]["prims"]:
                vao = self.ctx.vertex_array(prog, [
                    (prim["vbo"], "3f 3f 2f", "in_pos", "in_normal", "in_uv"),
                    (inst, "4f 4f 4f 4f/i", "in_m0", "in_m1", "in_m2", "in_m3"),
                ], prim["ibo"])
                self.draws.append((vao, prim, inst, len(mats)))
        g = self.size / 2.0 + 6.0
        ground = np.stack([_instance_matrix(self.origin[0], self.origin[2], 0.0, 1.0, y=0.01) @
                           np.diag([g, 1.0, g, 1.0]).astype("f4")]).transpose(0, 2, 1).astype("f4")
        self.ground_inst = self.ctx.buffer(ground.tobytes())
        self.ground_vao = self.ctx.vertex_array(prog, [
            (self.assets.ground_vbo, "3f 3f 2f", "in_pos", "in_normal", "in_uv"),
            (self.ground_inst, "4f 4f 4f 4f/i", "in_m0", "in_m1", "in_m2", "in_m3"),
        ], self.assets.ground_ibo)

    # --- rendu ------------------------------------------------------------------

    def render(self, camera, aspect, light_dir=(-0.4, -1.0, -0.3)):
        prog = self.assets.prog
        prog["u_proj"].write(camera.projection(aspect).T.copy().tobytes())
        prog["u_view"].write(camera.view().T.copy().tobytes())
        prog["u_light_dir"].value = tuple(light_dir)
        prog["u_eye"].value = tuple(float(c) for c in camera.position)
        prog["u_tex"].value = 0

        prog["u_ground"].value = True
        prog["u_use_tex"].value = False
        prog["u_alpha_mask"].value = False
        prog["u_color"].value = (1.0, 1.0, 1.0)
        self.ground_vao.render()
        prog["u_ground"].value = False

        for vao, prim, _, count in self.draws:
            tex = prim["texture"]
            if tex is not None:
                tex.use(location=0)
            prog["u_use_tex"].value = tex is not None
            prog["u_alpha_mask"].value = prim["alpha_mask"]
            prog["u_color"].value = prim["color"]
            vao.render(instances=count)

    def release(self):
        for vao, _, _, _ in self.draws:
            vao.release()
        for inst in self.instance_buffers:
            inst.release()
        self.ground_vao.release()
        self.ground_inst.release()
