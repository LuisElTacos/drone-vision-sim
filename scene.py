"""Scène de test moderngl : un drone (DJI Avata 2) muni de 12 caméras, soit 6 paires
stéréo réparties tous les 60° pour voir à 360° (voir STEREO_PAIRS), dans une scène
avec un sol, un bâtiment procédural, une sphère et un mur percé.

Un panneau latéral permet de cocher les vues affichées : les 12 caméras du drone,
la caméra fixe placée dans le monde, la caméra de suivi. Les vues cochées se
partagent le reste de la fenêtre en grille. Les collisions drone / obstacles
sont gérées (voir collision.py).

Installation : pip install -r requirements.txt
Lancement    : python scene.py

Contrôles :
  Z/Q/S/D ou W/A/S/D : avancer / gauche / reculer / droite (repère du drone)
  Espace / Ctrl      : monter / descendre
  Shift              : aller plus vite
  Souris             : X = lacet du drone, Y = inclinaison des caméras
                       (clic gauche dans une vue pour capturer, Échap pour libérer)
  Tab                : afficher / masquer la caméra de suivi
  Échap (non capturé): quitter
"""

import argparse
import math
import os
import random
import sys

import glfw
import moderngl
import numpy as np

from building import ProceduralBuilding
from collision import Box, Ground, Sphere
from gltf_loader import bounds, load_glb
from forest_env import ForestAssets, ForestEnv
from ui_panel import PANEL_WIDTH, SidePanel
from mujoco_flight import MujocoFlight
from walls_env import WallMaze

HERE = os.path.dirname(os.path.abspath(__file__))
DRONE_MODEL = os.path.join(HERE, "Assets", "models", "DJI_Avata2_Sketchfab.glb")

STEREO_BASELINE = 0.18     # distance entre les deux caméras d'une paire (m)
CAMERA_FOVY = 60.0         # champ vertical des caméras du drone (degrés)
# Paires stéréo réparties autour du drone : (nom, lacet relatif en degrés, + vers la droite).
# Avec fovy 60° en 4:3, chaque caméra voit ~75° en horizontal : 6 paires espacées de 60°
# couvrent les 360° avec du recouvrement entre paires voisines.
STEREO_PAIRS = [
    ("Avant", 0.0),
    ("Avant-droite", 60.0),
    ("Arrière-droite", 120.0),
    ("Arrière", 180.0),
    ("Arrière-gauche", 240.0),
    ("Avant-gauche", 300.0),
]
CAMERA_SETBACK = 0.03      # retrait des caméras derrière le bord du drone (m)
SPLIT = "grid"             # vues en grille ("grid"), en colonnes ("side_by_side") ou en lignes ("stacked")
WINDOW_SIZE = (1860, 800)  # panneau de 270 px + vues
WORLD_CAMERA_POSITION = (12.0, 9.0, 6.0)   # caméra fixe du monde
WORLD_CAMERA_TARGET = (0.0, 3.0, -10.0)

# Environnements : valeur de settings["building_mode"]
ENV_BUILDING, ENV_WALLS, ENV_EMPTY, ENV_FOREST = 0, 1, 2, 3
ENV_NAMES = {"batiment": ENV_BUILDING, "murs": ENV_WALLS, "vide": ENV_EMPTY, "foret": ENV_FOREST}
BUILDING_SPAWN = ((-2.7, 1.4, 2.0), -90.0)


# ---------------------------------------------------------------------------
# Maths caméra
# ---------------------------------------------------------------------------

def perspective(fovy_deg, aspect, near, far):
    f = 1.0 / math.tan(math.radians(fovy_deg) / 2.0)
    m = np.zeros((4, 4), dtype="f4")
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = 2.0 * far * near / (near - far)
    m[3, 2] = -1.0
    return m


def look_at(eye, target, up):
    f = target - eye
    f /= np.linalg.norm(f)
    s = np.cross(f, up)
    s /= np.linalg.norm(s)
    u = np.cross(s, f)
    m = np.identity(4, dtype="f4")
    m[0, :3] = s
    m[1, :3] = u
    m[2, :3] = -f
    m[:3, 3] = -m[:3, :3] @ eye
    return m


def translation(x, y, z):
    m = np.identity(4, dtype="f4")
    m[:3, 3] = (x, y, z)
    return m


class Camera:
    """Caméra yaw/pitch. Repère monde : Y vers le haut.

    Si `rotation` est donnée (colonnes droite, haut, avant en repère monde), elle remplace
    yaw/pitch : caméra fixée à un châssis qui s'incline (tangage et roulis).
    """

    def __init__(self, position, yaw=-90.0, pitch=0.0, fovy=60.0, near=0.1):
        self.position = np.array(position, dtype="f4")
        self.yaw = yaw
        self.pitch = pitch
        self.fovy = fovy
        self.near = near
        self.rotation = None

    @property
    def forward(self):
        if self.rotation is not None:
            return self.rotation[:, 2].astype("f4")
        y, p = math.radians(self.yaw), math.radians(self.pitch)
        return np.array([math.cos(p) * math.cos(y), math.sin(p), math.cos(p) * math.sin(y)], dtype="f4")

    @property
    def right(self):
        if self.rotation is not None:
            return self.rotation[:, 0].astype("f4")
        r = np.cross(self.forward, np.array([0, 1, 0], dtype="f4"))
        return r / np.linalg.norm(r)

    @property
    def up(self):
        if self.rotation is not None:
            return self.rotation[:, 1].astype("f4")
        return np.cross(self.right, self.forward)

    def view(self):
        up = self.rotation[:, 1] if self.rotation is not None else (0.0, 1.0, 0.0)
        return look_at(self.position, self.position + self.forward, np.array(up, dtype="f4"))

    def projection(self, aspect):
        return perspective(self.fovy, aspect, self.near, 500.0)

    def intrinsics(self, width, height):
        """Matrice K (convention vision) pour une image width x height."""
        f = (height / 2.0) / math.tan(math.radians(self.fovy) / 2.0)
        return np.array([[f, 0, width / 2.0], [0, f, height / 2.0], [0, 0, 1]])


class ChaseCamera:
    """Caméra extérieure qui suit le drone (derrière et au-dessus), orientable en inclinaison."""

    def __init__(self, drone, distance=1.1, height=0.35, fovy=60.0):
        self.drone, self.distance, self.height, self.fovy = drone, distance, height, fovy

    @property
    def position(self):
        p = math.radians(self.drone.camera_pitch)
        # Orbite verticale autour du drone selon l'inclinaison de caméra
        fwd_dist = self.distance * math.cos(p * 0.7)
        h = self.height - math.sin(p) * self.distance * 0.6
        return (self.drone.position - self.drone.forward * fwd_dist
                + np.array([0, max(0.1, h), 0])).astype("f4")

    def view(self):
        p = math.radians(self.drone.camera_pitch)
        # La cible s'élève ou s'abaisse pour orienter le regard dans la direction visée
        target = (self.drone.position + np.array([0, 0.15 + math.sin(p) * 2.0, 0])
                  + self.drone.forward * max(0.0, math.cos(p)) * 2.5).astype("f4")
        return look_at(self.position, target, np.array([0, 1, 0], dtype="f4"))

    def projection(self, aspect):
        return perspective(self.fovy, aspect, 0.05, 500.0)


class LookAtCamera:
    """Caméra fixe placée dans le monde, qui regarde un point donné."""

    def __init__(self, position, target, fovy=60.0):
        self.position = np.array(position, dtype="f4")
        self.target = np.array(target, dtype="f4")
        self.fovy = fovy

    def view(self):
        return look_at(self.position, self.target, np.array([0, 1, 0], dtype="f4"))

    def projection(self, aspect):
        return perspective(self.fovy, aspect, 0.05, 500.0)


# ---------------------------------------------------------------------------
# Géométrie
# ---------------------------------------------------------------------------

def make_sphere(radius=1.0, stacks=32, slices=64):
    """Retourne (vertices[pos(3), normal(3), uv(2)], indices)."""
    verts = []
    for i in range(stacks + 1):
        phi = math.pi * i / stacks
        v = i / stacks
        for j in range(slices + 1):
            theta = 2.0 * math.pi * j / slices
            u = j / slices
            n = (math.sin(phi) * math.cos(theta), math.cos(phi), math.sin(phi) * math.sin(theta))
            verts.extend((radius * n[0], radius * n[1], radius * n[2], *n, u, v))
    idx = []
    for i in range(stacks):
        for j in range(slices):
            a = i * (slices + 1) + j
            b = a + slices + 1
            idx.extend((a, b, a + 1, a + 1, b, b + 1))
    return np.array(verts, dtype="f4"), np.array(idx, dtype="i4")


def make_ground(size=300.0):
    s = size / 2.0
    verts = [
        -s, 0, -s,  0, 1, 0,  0, 0,
         s, 0, -s,  0, 1, 0,  s, 0,
         s, 0,  s,  0, 1, 0,  s, s,
        -s, 0,  s,  0, 1, 0,  0, s,
    ]
    idx = [0, 2, 1, 0, 3, 2]
    return np.array(verts, dtype="f4"), np.array(idx, dtype="i4")


def make_box(half):
    """Boîte centrée à l'origine, demi-dimensions half = (hx, hy, hz)."""
    verts, idx = [], []
    for axis in range(3):
        u_ax, v_ax = [(1, 2), (2, 0), (0, 1)][axis]
        for sign in (-1.0, 1.0):
            n = [0.0, 0.0, 0.0]
            n[axis] = sign
            base = len(verts) // 8
            for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
                p = [0.0, 0.0, 0.0]
                p[axis], p[u_ax], p[v_ax] = sign, a, b
                uv = ((a + 1.0) * 0.5, (b + 1.0) * 0.5)
                verts.extend((p[0] * half[0], p[1] * half[1], p[2] * half[2], *n, *uv))
            idx.extend((base, base + 1, base + 2, base, base + 2, base + 3))
    return np.array(verts, dtype="f4"), np.array(idx, dtype="i4")


VERTEX_SHADER = """
#version 330
uniform mat4 u_proj;
uniform mat4 u_view;
uniform mat4 u_model;
in vec3 in_pos;
in vec3 in_normal;
in vec2 in_uv;
out vec3 v_world;
out vec3 v_normal;
out vec2 v_uv;
void main() {
    vec4 w = u_model * vec4(in_pos, 1.0);
    v_world = w.xyz;
    v_normal = mat3(u_model) * in_normal;
    v_uv = in_uv;
    gl_Position = u_proj * u_view * w;
}
"""

FRAGMENT_SHADER = """
#version 330
uniform vec3 u_color;
uniform vec3 u_light_dir;
uniform vec3 u_eye;
uniform bool u_grid;
uniform bool u_use_texture;
uniform sampler2D u_texture;
in vec3 v_world;
in vec3 v_normal;
in vec2 v_uv;
out vec4 f_color;
void main() {
    vec3 n = normalize(v_normal);
    vec3 l = normalize(-u_light_dir);
    float diff = max(dot(n, l), 0.0);
    vec3 v = normalize(u_eye - v_world);
    float spec = pow(max(dot(reflect(-l, n), v), 0.0), 32.0) * 0.25;
    vec3 base = u_color;
    if (u_use_texture) {
        base = texture(u_texture, v_uv).rgb;
    }
    if (u_grid) {
        vec2 g = abs(fract(v_world.xz) - 0.5);
        float line = step(0.48, max(g.x, g.y));
        base = mix(base, vec3(0.15), line);
    }
    vec3 c = base * (0.25 + 0.75 * diff) + vec3(spec);
    // brouillard léger pour la profondeur
    float d = length(u_eye - v_world);
    c = mix(c, vec3(0.55, 0.7, 0.9), clamp((d - 45.0) / 100.0, 0.0, 1.0));
    f_color = vec4(c, 1.0);
}
"""


class Scene:
    """Contient les objets (et leurs collisionneurs) et sait se dessiner depuis une caméra."""

    def __init__(self, ctx, maze_seed=0):
        self.ctx = ctx
        self.prog = ctx.program(vertex_shader=VERTEX_SHADER, fragment_shader=FRAGMENT_SHADER)
        self.objects = []  # objets de base (sol, sphère, etc.)
        self.base_colliders = [Ground(0.0)]
        self.texture_cache = {}

        # Sol quadrillé par défaut
        v, i = make_ground(300.0)
        self.add(v, i, np.identity(4, dtype="f4"), (0.75, 0.75, 0.75), grid=True)

        # Bâtiment industriel procédural (vol intérieur + shader architectural)
        self.procedural_building = ProceduralBuilding(self.ctx, origin=(0.0, 0.0, -12.0))

        # Environnement « murs troués » (placé à l'écart, en x = 60)
        self.wall_maze = WallMaze(self.ctx, seed=maze_seed)
        # Forêt (placée en x = -80) : chargée à la première sélection (modèles et textures)
        self.forest_assets = None
        self.forest = None
        self.forest_seed = maze_seed
        v, i = make_sphere(0.25)
        self.target_vao = self.make_vao(v, i)

        # Sphère et mur de test conservés dans la scène
        v, i = make_sphere(1.0)
        self.add(v, i, translation(14.0, 1.0, -5.0), (0.85, 0.25, 0.2), collider=Sphere((14.0, 1.0, -5.0), 1.0))
        self.add_wall_with_hole(z=-28.0)

        # Environnement actif : ENV_BUILDING, ENV_WALLS ou ENV_EMPTY
        self.building_mode = ENV_BUILDING

    @property
    def colliders(self):
        """Collisionneurs dynamiques selon le modèle de bâtiment actif."""
        if self.building_mode == ENV_BUILDING and self.procedural_building is not None:
            return [*self.base_colliders, *self.procedural_building.colliders]
        if self.building_mode == ENV_WALLS:
            return [*self.base_colliders, *self.wall_maze.colliders]
        if self.building_mode == ENV_FOREST and self.forest is not None:
            return [*self.base_colliders, *self.forest.colliders]
        return self.base_colliders

    def ensure_forest(self):
        if self.forest is None:
            self.forest_assets = ForestAssets(self.ctx)
            self.forest = ForestEnv(self.ctx, self.forest_assets, seed=self.forest_seed)
        return self.forest

    def regenerate_forest(self, seed):
        self.ensure_forest().release()
        self.forest_seed = seed
        self.forest = ForestEnv(self.ctx, self.forest_assets, seed=seed)

    def regenerate_maze(self, seed):
        self.wall_maze.release()
        self.wall_maze = WallMaze(self.ctx, seed=seed)

    def spawn(self, mode):
        if mode == ENV_WALLS:
            return self.wall_maze.spawn
        if mode == ENV_FOREST:
            return self.ensure_forest().spawn
        return BUILDING_SPAWN

    def overview_camera(self, mode):
        if mode == ENV_WALLS:
            return self.wall_maze.overview_camera
        if mode == ENV_FOREST:
            return self.ensure_forest().overview_camera
        return WORLD_CAMERA_POSITION, WORLD_CAMERA_TARGET

    def target(self):
        """Cible de l'environnement actif, ou None."""
        if self.building_mode == ENV_WALLS:
            return self.wall_maze.target
        if self.building_mode == ENV_FOREST and self.forest is not None:
            return self.forest.target
        return None

    def target_items(self):
        """Repère vert de la cible (murs, forêt), pour les vues extérieures."""
        target = self.target()
        if target is None:
            return []
        return [(self.target_vao, translation(*target), (0.1, 0.9, 0.2), False)]

    def make_texture(self, img):
        """Convertit une image PIL en texture ModernGL avec mipmapping."""
        if img is None:
            return None
        key = id(img)
        if key in self.texture_cache:
            return self.texture_cache[key]
        tex = self.ctx.texture(img.size, 3, img.convert("RGB").tobytes())
        tex.build_mipmaps()
        tex.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
        self.texture_cache[key] = tex
        return tex

    def make_vao(self, vertices, indices):
        vbo = self.ctx.buffer(vertices.tobytes())
        ibo = self.ctx.buffer(indices.tobytes())
        return self.ctx.vertex_array(self.prog, [(vbo, "3f 3f 2f", "in_pos", "in_normal", "in_uv")], ibo)

    def add(self, vertices, indices, model, color, grid=False, texture=None, collider=None):
        self.objects.append((self.make_vao(vertices, indices), model, color, grid, texture))
        if collider is not None:
            self.base_colliders.append(collider)

    def add_box(self, center, half, color):
        v, i = make_box(half)
        self.add(v, i, translation(*center), color, collider=Box(center, half))

    def add_wall_with_hole(self, z, width=6.0, height=3.0, thickness=0.2,
                           hole=(0.6, 0.6), hole_center=(0.0, 1.5), color=(0.6, 0.55, 0.5)):
        """Mur de test perpendiculaire à Z, percé d'un trou rectangulaire (4 boîtes)."""
        hw, hh = hole[0] / 2.0, hole[1] / 2.0
        cx, cy = hole_center
        t = thickness / 2.0
        left, right = -width / 2.0, width / 2.0
        self.add_box(((left + cx - hw) / 2.0, height / 2.0, z), ((cx - hw - left) / 2.0, height / 2.0, t), color)
        self.add_box(((right + cx + hw) / 2.0, height / 2.0, z), ((right - cx - hw) / 2.0, height / 2.0, t), color)
        self.add_box((cx, (cy - hh) / 2.0, z), (hw, (cy - hh) / 2.0, t), color)
        self.add_box((cx, (cy + hh + height) / 2.0, z), (hw, (height - cy - hh) / 2.0, t), color)

    def render(self, camera, aspect, extra=(), building_mode=0):
        """Dessine la scène dans le viewport courant."""
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.building_mode = building_mode

        # 1. Rendu du bâtiment procédural avec son shader architectural dédié
        if self.building_mode == ENV_BUILDING and self.procedural_building is not None:
            self.procedural_building.render(camera, aspect, light_dir=(-0.4, -1.0, -0.3))
        elif self.building_mode == ENV_WALLS:
            self.wall_maze.render(camera, aspect, light_dir=(-0.4, -1.0, -0.3))
        elif self.building_mode == ENV_FOREST and self.forest is not None:
            self.forest.render(camera, aspect, light_dir=(-0.4, -1.0, -0.3))

        # 2. Rendu des objets standard (sol, sphère, drone)
        self.prog["u_proj"].write(camera.projection(aspect).T.copy().tobytes())
        self.prog["u_view"].write(camera.view().T.copy().tobytes())
        self.prog["u_light_dir"].value = (-0.4, -1.0, -0.3)
        self.prog["u_eye"].value = tuple(camera.position)

        items_to_render = [*self.objects, *extra]

        for item in items_to_render:
            vao, model, color, grid = item[:4]
            tex = item[4] if len(item) > 4 else None
            self.prog["u_model"].write(model.T.copy().tobytes())
            self.prog["u_color"].value = color
            self.prog["u_grid"].value = grid
            if tex is not None:
                tex.use(location=0)
                self.prog["u_use_texture"].value = True
            else:
                self.prog["u_use_texture"].value = False
            vao.render()


# ---------------------------------------------------------------------------
# Drone
# ---------------------------------------------------------------------------

class Drone:
    """Drone commandé en vitesse (m/s, repère drone) et en vitesse de lacet.

    La dynamique et les chocs sont calculés par MuJoCo (mujoco_flight.py : quadrotor à 4 moteurs,
    contrôleur en cascade). Les caméras sont fixées au châssis (elles suivent tangage et roulis)
    ou stabilisées (lacet seul).
    """

    def __init__(self, scene, position, yaw=-90.0):
        self.scene = scene
        self.position = np.array(position, dtype="f8")
        self.velocity = np.zeros(3, dtype="f8")   # vitesse en repère monde (m/s)
        self.yaw = float(yaw)                     # lacet (degrés)
        self.pitch = 0.0                          # tangage châssis (degrés)
        self.roll = 0.0                           # roulis châssis (degrés)
        self.camera_pitch = 0.0                   # inclinaison caméra (degrés)
        self.contact = False

        parts = load_glb(DRONE_MODEL)
        lo, hi = bounds(parts)
        self.center_offset = (lo + hi) / 2.0           # centre de la boîte dans le repère modèle
        size = hi - lo                                 # (gauche-droite, hauteur, longueur)
        self.half = size / 2.0                         # demi-dimensions (droite, haut, avant)
        self.parts = [
            (scene.make_vao(v, i), tuple(max(c, 0.03) for c in color), scene.make_texture(tex_img) if tex_img else None)
            for v, i, color, tex_img in parts
        ]

        # Deux caméras (gauche, droite) par paire stéréo, dans l'ordre de STEREO_PAIRS.
        self.cameras = [Camera(position, fovy=CAMERA_FOVY, near=0.02) for _ in range(2 * len(STEREO_PAIRS))]
        self.cams_on_chassis = True   # False : caméras stabilisées (lacet seul)
        self.flight = MujocoFlight(self.half, scene.colliders, self.position, self.yaw)
        self.flight_pos = self.position.copy()   # dernière position donnée par MuJoCo (détecte les téléportations)
        self.update_cameras()

    def body_axes(self):
        """Axes du châssis en repère monde : (avant, gauche, haut)."""
        R = self.flight.rotation
        return R[:, 0], R[:, 1], R[:, 2]

    @property
    def forward(self):
        """Vecteur avant horizontal unitaire."""
        y = math.radians(self.yaw)
        return np.array([math.cos(y), 0.0, math.sin(y)])

    @property
    def right(self):
        """Vecteur droite horizontal unitaire."""
        return np.cross(self.forward, (0.0, 1.0, 0.0))

    def model_matrix(self):
        """Repère modèle (+X gauche, +Y haut, +Z avant) -> monde, orientation complète du châssis."""
        fwd, left, up = self.body_axes()
        rot4 = np.identity(4, dtype="f4")
        rot4[:3, :3] = np.column_stack([left, up, fwd])
        m = translation(*self.position) @ rot4 @ translation(*(-self.center_offset))
        return m.astype("f4")

    def update_cameras(self):
        """Chaque paire est posée sur le bord de la boîte du drone dans sa direction (en retrait de
        CAMERA_SETBACK pour rester dans la coque au contact d'un obstacle), ses deux caméras à
        ±baseline/2 de part et d'autre, parallèles (paire rectifiée).

        Fixées au châssis, les caméras suivent tangage et roulis ; stabilisées, le lacet seul."""
        if self.cams_on_chassis:
            fwd, left, up = self.body_axes()
            right = -left
        else:
            fwd, right, up = self.forward, self.right, np.array([0.0, 1.0, 0.0])
        hx, hz = self.half[0], self.half[2]
        p = math.radians(self.camera_pitch)
        for k, (_, offset) in enumerate(STEREO_PAIRS):
            a = math.radians(offset)
            # direction de la paire dans le repère drone (avant, droite)
            df, dr = math.cos(a), math.sin(a)
            # distance du centre au bord de la boîte dans cette direction
            t = min(hz / abs(df) if abs(df) > 1e-9 else np.inf, hx / abs(dr) if abs(dr) > 1e-9 else np.inf)
            direction = fwd * df + right * dr
            side_axis = np.cross(direction, up)  # droite de la paire
            cam_fwd = math.cos(p) * direction + math.sin(p) * up
            cam_up = -math.sin(p) * direction + math.cos(p) * up
            mount = self.position + direction * (t - CAMERA_SETBACK)
            for cam, side in zip(self.cameras[2 * k:2 * k + 2], (-0.5, 0.5)):  # gauche, droite
                cam.position = (mount + side_axis * side * STEREO_BASELINE).astype("f4")
                cam.rotation = np.column_stack([side_axis, cam_up, cam_fwd])
                cam.yaw = self.yaw + offset
                cam.pitch = self.camera_pitch

    def update(self, dt, cmd_velocity=None, cmd_yaw_rate=0.0):
        """Avance la simulation de dt secondes.

        cmd_velocity : vecteur [v_fwd, v_right, v_up] en m/s dans le repère drone (lacet seul).
        cmd_yaw_rate : vitesse angulaire en lacet (deg/s).
        Retourne True si le drone touche un obstacle.
        """
        v_fwd_cmd, v_right_cmd, v_up_cmd = np.zeros(3) if cmd_velocity is None else cmd_velocity
        # Vitesse désirée en repère monde
        v_target = (self.forward * v_fwd_cmd +
                    self.right * v_right_cmd +
                    np.array([0.0, v_up_cmd, 0.0]))

        flight = self.flight
        flight.sync_scene(self.scene.colliders)
        if not np.allclose(self.position, self.flight_pos):
            # Position changée de l'extérieur (téléportation, changement d'environnement)
            flight.reset(self.position, self.yaw, self.velocity)
        hit = flight.step(dt, v_target, cmd_yaw_rate)
        self.position = flight.position
        self.flight_pos = self.position.copy()
        self.velocity = flight.velocity
        self.yaw = flight.yaw_deg
        fwd, left, _ = self.body_axes()
        self.pitch = math.degrees(-math.asin(np.clip(fwd[1], -1.0, 1.0)))   # > 0 : nez vers le bas
        self.roll = math.degrees(math.asin(np.clip(left[1], -1.0, 1.0)))    # > 0 : penché à droite
        self.contact = hit
        self.update_cameras()
        return hit

    def render_items(self):
        m = self.model_matrix()
        return [(vao, m, color, False, tex) for vao, color, tex in self.parts]


# ---------------------------------------------------------------------------
# Contrôleur manette
# ---------------------------------------------------------------------------

def deadzone(val, threshold=0.15):
    if abs(val) < threshold:
        return 0.0
    return math.copysign((abs(val) - threshold) / (1.0 - threshold), val)


def read_gamepad():
    """Lit les entrées analogiques de la manette (si présente et reconnue)."""
    if not glfw.joystick_present(glfw.JOYSTICK_1):
        return None

    if glfw.joystick_is_gamepad(glfw.JOYSTICK_1):
        state = glfw.get_gamepad_state(glfw.JOYSTICK_1)
        axes = state.axes
        buttons = state.buttons

        # Stick gauche : translation horizontale (X = droite/gauche, Y = avant/arrière)
        fwd = -deadzone(axes[glfw.GAMEPAD_AXIS_LEFT_Y])
        right = deadzone(axes[glfw.GAMEPAD_AXIS_LEFT_X])

        # Stick droit : rotation lacet (X) et regard vertical haut/bas (Y)
        yaw_cmd = deadzone(axes[glfw.GAMEPAD_AXIS_RIGHT_X])
        cam_pitch_cmd = -deadzone(axes[glfw.GAMEPAD_AXIS_RIGHT_Y])

        # Altitude : Gâchettes RT (monter) / LT (descendre) + Bumpers RB/LB + Boutons A/B
        up = 0.0
        rt = axes[glfw.GAMEPAD_AXIS_RIGHT_TRIGGER]
        lt = axes[glfw.GAMEPAD_AXIS_LEFT_TRIGGER]
        if rt > 0.1:
            up += rt
        if lt > 0.1:
            up -= lt
        if buttons[glfw.GAMEPAD_BUTTON_RIGHT_BUMPER] or buttons[glfw.GAMEPAD_BUTTON_A]:
            up += 1.0
        if buttons[glfw.GAMEPAD_BUTTON_LEFT_BUMPER] or buttons[glfw.GAMEPAD_BUTTON_B]:
            up -= 1.0
        up = float(np.clip(up, -1.0, 1.0))

        boost = bool(buttons[glfw.GAMEPAD_BUTTON_LEFT_THUMB] or buttons[glfw.GAMEPAD_BUTTON_RIGHT_BUMPER])
        reset_cam = bool(buttons[glfw.GAMEPAD_BUTTON_Y])
    else:
        # Fallback pour manettes USB génériques
        raw_axes = glfw.get_joystick_axes(glfw.JOYSTICK_1)[0]
        raw_buttons = glfw.get_joystick_buttons(glfw.JOYSTICK_1)[0]
        fwd = -deadzone(raw_axes[1]) if len(raw_axes) > 1 else 0.0
        right = deadzone(raw_axes[0]) if len(raw_axes) > 0 else 0.0
        yaw_cmd = deadzone(raw_axes[2]) if len(raw_axes) > 2 else 0.0
        cam_pitch_cmd = -deadzone(raw_axes[3]) if len(raw_axes) > 3 else 0.0
        up = 0.0
        if len(raw_buttons) > 0 and raw_buttons[0]:
            up += 1.0
        if len(raw_buttons) > 1 and raw_buttons[1]:
            up -= 1.0
        boost = bool(len(raw_buttons) > 5 and raw_buttons[5])
        reset_cam = bool(len(raw_buttons) > 3 and raw_buttons[3])

    return {
        "fwd": fwd,
        "right": right,
        "up": up,
        "yaw": yaw_cmd,
        "cam_pitch": cam_pitch_cmd,
        "boost": boost,
        "reset_cam": reset_cam,
    }


# ---------------------------------------------------------------------------
# Fenêtre et boucle principale
# ---------------------------------------------------------------------------

def tile_viewports(x0, w, h, n):
    """Découpe la zone [x0, x0 + w] x [0, h] en n vues (grille, colonnes ou lignes selon SPLIT)."""
    if n == 0:
        return []
    if SPLIT == "grid":
        # nombre de colonnes choisi pour des vues proches du 4:3
        cols = max(1, min(n, round(math.sqrt(n * (w / h) / (4.0 / 3.0)))))
        rows = math.ceil(n / cols)
        xs = [round(w * c / cols) for c in range(cols + 1)]
        ys = [round(h * r / rows) for r in range(rows + 1)]
        return [(x0 + xs[i % cols], h - ys[i // cols + 1], xs[i % cols + 1] - xs[i % cols],
                 ys[i // cols + 1] - ys[i // cols]) for i in range(n)]
    if SPLIT == "stacked":
        edges = [round(h * k / n) for k in range(n + 1)]
        return [(x0, h - edges[k + 1], w, edges[k + 1] - edges[k]) for k in range(n)]
    edges = [round(w * k / n) for k in range(n + 1)]
    return [(x0 + edges[k], 0, edges[k + 1] - edges[k], h) for k in range(n)]


def main():
    parser = argparse.ArgumentParser(description="Simulation de drone stéréo")
    parser.add_argument("--env", choices=sorted(ENV_NAMES), default="murs", help="environnement de départ")
    parser.add_argument("--seed", type=int, default=0, help="graine des environnements « murs » et « foret »")
    args = parser.parse_args()
    start_mode = ENV_NAMES[args.env]

    if not glfw.init():
        sys.exit("Impossible d'initialiser GLFW")
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, True)
    window = glfw.create_window(*WINDOW_SIZE, "Simulation drone", None, None)
    if not window:
        glfw.terminate()
        sys.exit("Impossible de créer la fenêtre")
    glfw.make_context_current(window)
    glfw.swap_interval(1)

    ctx = moderngl.create_context()
    scene = Scene(ctx, maze_seed=args.seed)
    scene.building_mode = start_mode
    spawn_pos, spawn_yaw = scene.spawn(start_mode)
    drone = Drone(scene, position=spawn_pos, yaw=spawn_yaw)
    panel = SidePanel()

    # Paramètres de l'interface et du contrôle
    settings = {
        "max_speed": 8.0,          # vitesse max par défaut (m/s) -> ~29 km/h
        "yaw_sens": 1.0,           # sensibilité rotation lacet
        "cam_sens": 1.0,           # sensibilité caméra
        "gamepad_enabled": True,
        "gamepad_connected": False,
        "gamepad_name": "",
        "show_grid": True,         # sol requadrillé par défaut
        "building_mode": start_mode,  # ENV_BUILDING, ENV_WALLS ou ENV_EMPTY
        "teleport": None,          # tuple (pos, yaw) si téléportation demandée
        "maze_seed": args.seed,
        "regen_maze": False,       # bouton « Nouveaux murs »
        "maze_spawn": scene.wall_maze.spawn,
        "maze_info": [],
        "regen_forest": False,     # bouton « Nouvelle forêt »
        "forest_spawn": None,
        "forest_info": [],
        "cams_on_chassis": drone.cams_on_chassis,  # False : caméras stabilisées
    }
    last_mode = start_mode

    # [libellé, actif, caméra] : l'ordre est celui d'affichage, de gauche à droite.
    views = []
    for k, (name, _) in enumerate(STEREO_PAIRS):
        views.append([f"{name} G", True, drone.cameras[2 * k]])
        views.append([f"{name} D", True, drone.cameras[2 * k + 1]])
    chase_view = ["Caméra de suivi", False, ChaseCamera(drone)]
    world_cam = LookAtCamera(*scene.overview_camera(start_mode))
    views += [["Caméra fixe (monde)", True, world_cam], chase_view]

    def update_env_info():
        maze = scene.wall_maze
        counts = maze.summary()
        settings["maze_spawn"] = maze.spawn
        settings["maze_info"] = [
            f"Graine : {maze.seed}",
            f"{maze.n_walls} murs, {len(maze.openings)} ouvertures",
            f"{counts['passage']} passages, {counts['cul_de_sac']} culs-de-sac",
            f"{counts['trop_petit']} trop petites",
        ]

    def update_forest_info():
        forest = scene.forest
        if forest is None:
            return
        counts = forest.summary()
        settings["forest_spawn"] = forest.spawn
        settings["forest_info"] = [
            f"Graine : {forest.seed}",
            f"{counts['arbre']} pins",
        ]

    update_env_info()
    update_forest_info()
    onboard = set(id(c) for c in drone.cameras)

    state = {"captured": False, "last": None, "mouse_yaw_delta": 0.0}

    def set_capture(on):
        state["captured"] = on
        state["last"] = None
        glfw.set_input_mode(window, glfw.CURSOR, glfw.CURSOR_DISABLED if on else glfw.CURSOR_NORMAL)
        panel.set_mouse_enabled(not on)

    def on_mouse_button(win, button, action, mods):
        panel.mouse_button(button, action == glfw.PRESS)
        if button == glfw.MOUSE_BUTTON_LEFT and action == glfw.PRESS and not state["captured"]:
            x, _ = glfw.get_cursor_pos(win)
            if x > PANEL_WIDTH and not panel.wants_mouse():
                set_capture(True)

    def on_key(win, key, scancode, action, mods):
        if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
            if state["captured"]:
                set_capture(False)
            else:
                glfw.set_window_should_close(win, True)
        if key == glfw.KEY_TAB and action == glfw.PRESS:
            chase_view[1] = not chase_view[1]

    def on_cursor(win, x, y):
        panel.mouse_pos(x, y)
        if not state["captured"]:
            return
        if state["last"] is not None:
            dx, dy = x - state["last"][0], y - state["last"][1]
            yaw_sens = float(settings.get("yaw_sens", 1.0))
            cam_sens = float(settings.get("cam_sens", 1.0))
            state["mouse_yaw_delta"] += dx * 0.1 * yaw_sens
            drone.camera_pitch = float(np.clip(drone.camera_pitch - dy * 0.1 * cam_sens, -60.0, 60.0))
        state["last"] = (x, y)

    glfw.set_mouse_button_callback(window, on_mouse_button)
    glfw.set_key_callback(window, on_key)
    glfw.set_cursor_pos_callback(window, on_cursor)

    def pressed(*keys):
        return any(glfw.get_key(window, k) == glfw.PRESS for k in keys)

    last_t = glfw.get_time()
    while not glfw.window_should_close(window):
        t = glfw.get_time()
        dt, last_t = min(t - last_t, 0.1), t
        glfw.poll_events()

        # Détection de l'état de la manette
        if glfw.joystick_present(glfw.JOYSTICK_1):
            name = glfw.get_joystick_name(glfw.JOYSTICK_1)
            if isinstance(name, bytes):
                name = name.decode("utf-8", errors="replace")
            settings["gamepad_connected"] = True
            settings["gamepad_name"] = name or "Manette"
        else:
            settings["gamepad_connected"] = False
            settings["gamepad_name"] = ""

        # Paramètres de vol issus des sliders
        base_h = float(settings.get("max_speed", 8.0))
        yaw_sens = float(settings.get("yaw_sens", 1.0))
        cam_sens = float(settings.get("cam_sens", 1.0))

        # Détermination des vitesses maximales
        boost = pressed(glfw.KEY_LEFT_SHIFT)
        boost_mult = 1.75 if boost else 1.0
        max_h = base_h * boost_mult
        max_v = (base_h * 0.6) * boost_mult
        max_yaw = (80.0 * yaw_sens) * (1.3 if boost else 1.0)

        # Consignes de déplacement
        cmd_fwd, cmd_right, cmd_up = 0.0, 0.0, 0.0
        cmd_yaw_rate = 0.0

        # Clavier (AZERTY compatible via codes physiques GLFW)
        if pressed(glfw.KEY_W, glfw.KEY_UP):
            cmd_fwd += 1.0
        if pressed(glfw.KEY_S, glfw.KEY_DOWN):
            cmd_fwd -= 1.0
        if pressed(glfw.KEY_A, glfw.KEY_LEFT):
            cmd_right -= 1.0
        if pressed(glfw.KEY_D, glfw.KEY_RIGHT):
            cmd_right += 1.0
        if pressed(glfw.KEY_SPACE):
            cmd_up += 1.0
        if pressed(glfw.KEY_LEFT_CONTROL):
            cmd_up -= 1.0

        # Normalisation diagonale clavier
        kh_norm = math.hypot(cmd_fwd, cmd_right)
        if kh_norm > 1.0:
            cmd_fwd /= kh_norm
            cmd_right /= kh_norm

        # Entrées manette si activée
        if settings["gamepad_enabled"] and settings["gamepad_connected"]:
            gp = read_gamepad()
            if gp is not None:
                if gp["boost"]:
                    max_h *= 1.75
                    max_v *= 1.75
                    max_yaw *= 1.3
                if abs(gp["fwd"]) > 0.01 or abs(gp["right"]) > 0.01:
                    cmd_fwd = gp["fwd"]
                    cmd_right = gp["right"]
                if abs(gp["up"]) > 0.01:
                    cmd_up = gp["up"]
                if abs(gp["yaw"]) > 0.01:
                    cmd_yaw_rate += gp["yaw"] * max_yaw
                if abs(gp["cam_pitch"]) > 0.01:
                    drone.camera_pitch = float(np.clip(
                        drone.camera_pitch + gp["cam_pitch"] * 90.0 * dt * cam_sens, -60.0, 60.0))
                if gp["reset_cam"]:
                    drone.camera_pitch = 0.0

        # Application de la souris (quand capturée) sur le lacet
        if state["mouse_yaw_delta"] != 0.0:
            cmd_yaw_rate += state["mouse_yaw_delta"] / dt
            state["mouse_yaw_delta"] = 0.0

        # Téléportation immédiate si demandée via un bouton de l'interface
        if settings.get("teleport") is not None:
            t_pos, t_yaw = settings["teleport"]
            drone.position = np.array(t_pos, dtype="f8")
            drone.velocity = np.zeros(3, dtype="f8")
            drone.yaw = float(t_yaw)
            drone.pitch = 0.0
            drone.roll = 0.0
            drone.camera_pitch = 0.0
            drone.update_cameras()
            settings["teleport"] = None

        # Synchronisation de l'environnement sélectionné
        b_mode = int(settings.get("building_mode", 0))
        scene.building_mode = b_mode
        if settings.get("regen_maze"):
            settings["regen_maze"] = False
            settings["maze_seed"] = random.randrange(100000)
            scene.regenerate_maze(settings["maze_seed"])
            update_env_info()
            last_mode = None  # force le retour au départ ci-dessous
        if settings.get("regen_forest"):
            settings["regen_forest"] = False
            scene.regenerate_forest(random.randrange(100000))
            last_mode = None
        if b_mode != last_mode:
            last_mode = b_mode
            if b_mode == ENV_FOREST:
                scene.ensure_forest()
                update_forest_info()
            world_cam.position, world_cam.target = (np.array(p, dtype="f4") for p in scene.overview_camera(b_mode))
            t_pos, t_yaw = scene.spawn(b_mode)
            drone.position = np.array(t_pos, dtype="f8")
            drone.velocity = np.zeros(3, dtype="f8")
            drone.yaw, drone.pitch, drone.roll = float(t_yaw), 0.0, 0.0
            drone.update_cameras()

        drone.cams_on_chassis = bool(settings["cams_on_chassis"])

        # Vecteur de consigne de vitesse [m/s]
        v_cmd = np.array([cmd_fwd * max_h, cmd_right * max_h, cmd_up * max_v])

        # Mise à jour dynamique du drone
        drone.update(dt, cmd_velocity=v_cmd, cmd_yaw_rate=cmd_yaw_rate)

        w, h = glfw.get_framebuffer_size(window)
        ww, wh = glfw.get_window_size(window)
        if w > 0 and h > 0:
            # Mise à jour dynamique de l'état du quadrillage du sol
            if len(scene.objects) > 0:
                show_g = bool(settings.get("show_grid", True))
                o0 = scene.objects[0]
                scene.objects[0] = (o0[0], o0[1], o0[2], show_g, o0[4])

            ctx.viewport = (0, 0, w, h)
            ctx.clear(0.12, 0.12, 0.12)
            panel_px = round(PANEL_WIDTH * w / max(ww, 1))
            active = [v for v in views if v[1]]
            target_items = scene.target_items()
            drone_items = drone.render_items() + target_items
            tiles = tile_viewports(panel_px, w - panel_px, h, len(active))
            for (_, _, cam), vp in zip(active, tiles):
                ctx.viewport = vp
                ctx.clear(0.55, 0.7, 0.9, viewport=vp)
                # Le drone n'est pas dessiné dans ses propres caméras (images propres pour la vision).
                extra = () if id(cam) in onboard else drone_items
                scene.render(cam, vp[2] / vp[3], extra=extra, building_mode=b_mode)
            # traits de séparation : bord gauche et bas de chaque vue
            ctx.viewport = (0, 0, w, h)
            for vp in tiles:
                if vp[0] > panel_px:
                    ctx.clear(0.05, 0.05, 0.05, viewport=(vp[0] - 1, vp[1], 2, vp[3]))
                if vp[1] > 0:
                    ctx.clear(0.05, 0.05, 0.05, viewport=(vp[0], vp[1] - 1, vp[2], 2))

            x, y, z = drone.position
            speed_kmh = float(np.linalg.norm(drone.velocity)) * 3.6
            info = [
                f"Position : {x:6.2f} {y:6.2f} {z:6.2f}",
                f"Vitesse  : {speed_kmh:5.1f} km/h",
                f"Lacet    : {drone.yaw % 360:6.1f} deg",
                f"Tangage  : {drone.pitch:5.1f} deg",
                f"Roulis   : {drone.roll:5.1f} deg",
                f"Cam. inc.: {drone.camera_pitch:5.1f} deg",
                f"Base st. : {STEREO_BASELINE * 100:.0f} cm",
                f"Caméras  : {len(drone.cameras)} ({len(STEREO_PAIRS)} paires)",
                "Contact  : OUI" if drone.contact else "Contact  : non",
            ]
            if scene.target() is not None:
                d_target = float(np.linalg.norm(scene.target() - drone.position))
                info.append(f"Cible    : {d_target:5.1f} m")
            info += [
                "",
                "Clic vue : capturer souris",
                "Échap    : libérer souris",
                "Tab      : caméra de suivi",
            ]
            panel.draw((ww, wh), (w, h), dt, views, info, settings=settings)
            glfw.swap_buffers(window)

    panel.shutdown()
    glfw.terminate()


if __name__ == "__main__":
    main()
