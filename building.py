"""Bâtiment industriel procédural avec collision exacte et shader architectural.

Ce module génère un bâtiment industriel modulaire inspiré de l'usine (Nickel Factory),
structuré avec exactitude géométrique pour permettre au drone de :
1. Pénétrer à l'intérieur via de multiples ouvertures réparties sur les façades.
2. Naviguer entre les différentes pièces (grand hall hangar, atelier, étage mezzanine).
3. Traverser des ouvertures de dimensions variées (grand portail, portes standard,
   fenêtres larges, fenêtre étroite de précision, fente d'aération infranchissable).
4. Distinguer par vision stéréo les passages réels des niches aveugles (recesses muraux
   fermés par un fond à faible profondeur).
5. Fournir des vérités terrain (ground truth) géométriques précises pour l'évaluation
   des métriques du cours CSC_51073 (détection de coins Harris, RANSAC plan, stéréo).

Toutes les parties solides sont décomposées en boîtes alignées sur les axes (Box de collision.py),
assurant une détection de collision 100 % exacte et sans aucun polygone invisible bloquant le vol.
"""

import math
import numpy as np
import moderngl
from collision import Box

# ---------------------------------------------------------------------------
# Identifiants de matériaux pour le shader procédural
# ---------------------------------------------------------------------------
MAT_CONCRETE = 0.0       # Parpaings de béton industriels avec joints de mortier
MAT_STEEL_DARK = 1.0     # Acier industriel foncé (poutres en I, linteaux, cadres)
MAT_STEEL_YELLOW = 2.0   # Zébrures de sécurité jaunes et noires (portail principal)
MAT_FLOOR = 3.0          # Dalle béton lissée avec joints de dilatation
MAT_ROOF = 4.0           # Toiture industrielle étanche / bac acier
MAT_INTERIOR_WALL = 5.0  # Parpaings peints clairs pour murs intérieurs
MAT_PLAIN = 6.0          # Enduit uni sans texture (environnement « murs troués »)

# ---------------------------------------------------------------------------
# Shaders GLSL 330
# ---------------------------------------------------------------------------
BUILDING_VERTEX_SHADER = """
#version 330
uniform mat4 u_proj;
uniform mat4 u_view;
uniform mat4 u_model;

in vec3 in_pos;
in vec3 in_normal;
in vec2 in_uv;
in float in_mat;

out vec3 v_world;
out vec3 v_normal;
out vec2 v_uv;
out float v_mat;

void main() {
    vec4 w = u_model * vec4(in_pos, 1.0);
    v_world = w.xyz;
    v_normal = normalize(mat3(u_model) * in_normal);
    v_uv = in_uv;
    v_mat = in_mat;
    gl_Position = u_proj * u_view * w;
}
"""

BUILDING_FRAGMENT_SHADER = """
#version 330
uniform vec3 u_light_dir;
uniform vec3 u_eye;

in vec3 v_world;
in vec3 v_normal;
in vec2 v_uv;
in float v_mat;

out vec4 f_color;

// Générateur pseudo-aléatoire rapide
float hash(vec2 p) {
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453123);
}

// Bruit procédural pour la texture du béton
float noise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    vec2 u = f * f * (3.0 - 2.0 * f);
    return mix(mix(hash(i + vec2(0.0, 0.0)), hash(i + vec2(1.0, 0.0)), u.x),
               mix(hash(i + vec2(0.0, 1.0)), hash(i + vec2(1.0, 1.0)), u.x), u.y);
}

void main() {
    vec3 n = normalize(v_normal);
    vec3 l = normalize(-u_light_dir);
    vec3 v = normalize(u_eye - v_world);
    vec3 h = normalize(l + v);

    float diff = max(dot(n, l), 0.0);
    float spec = 0.0;
    vec3 base = vec3(0.55);

    // Coordonnées de projection planaire selon la normale dominante
    vec2 coord;
    if (abs(n.x) > 0.5) {
        coord = v_world.zy;
    } else if (abs(n.z) > 0.5) {
        coord = v_world.xy;
    } else {
        coord = v_world.xz;
    }

    if (v_mat < 0.5) {
        // MAT 0: Béton / Parpaings industriels extérieurs
        float bw = 0.50; // largeur d'un parpaing (m)
        float bh = 0.25; // hauteur d'un parpaing (m)
        float row = floor(coord.y / bh);
        float x_off = mod(row, 2.0) * (bw * 0.5);
        vec2 bcoord = vec2(coord.x + x_off, coord.y);
        vec2 bid = floor(bcoord / vec2(bw, bh));
        vec2 f = fract(bcoord / vec2(bw, bh));

        float mortar_w = 0.035;
        float is_mortar = step(f.x, mortar_w) + step(1.0 - mortar_w, f.x)
                        + step(f.y, mortar_w * 1.5) + step(1.0 - mortar_w * 1.5, f.y);
        is_mortar = clamp(is_mortar, 0.0, 1.0);

        float block_rnd = hash(bid) * 0.14 - 0.07;
        float grain = (noise(coord * 50.0) - 0.5) * 0.08;

        vec3 block_col = vec3(0.58 + block_rnd + grain, 0.57 + block_rnd + grain, 0.55 + block_rnd + grain);
        vec3 mortar_col = vec3(0.32, 0.32, 0.34);

        base = mix(block_col, mortar_col, is_mortar);
        spec = pow(max(dot(n, h), 0.0), 16.0) * 0.10;
    } else if (v_mat < 1.5) {
        // MAT 1: Acier de structure / Cadres / Poutres
        base = vec3(0.18, 0.20, 0.22);
        // Simulation de rivets réguliers
        vec2 r_grid = fract(coord * 2.5) - 0.5;
        float r_dist = length(r_grid);
        if (r_dist < 0.14 && r_dist > 0.07) {
            base += vec3(0.18); // anneau métallique du rivet
        }
        spec = pow(max(dot(n, h), 0.0), 64.0) * 0.45;
    } else if (v_mat < 2.5) {
        // MAT 2: Zébrures d'avertissement de sécurité (jaune / noir)
        float stripe = fract((coord.x + coord.y) * 4.0);
        base = stripe > 0.5 ? vec3(0.94, 0.74, 0.08) : vec3(0.10, 0.10, 0.11);
        spec = pow(max(dot(n, h), 0.0), 32.0) * 0.30;
    } else if (v_mat < 3.5) {
        // MAT 3: Sol en dalle béton industrielle
        vec2 fg = abs(fract(v_world.xz * 0.5) - 0.5);
        float seam = step(0.488, max(fg.x, fg.y));
        float grain = (noise(v_world.xz * 30.0) - 0.5) * 0.05;
        vec3 slab = vec3(0.50 + grain, 0.51 + grain, 0.52 + grain);
        base = mix(slab, vec3(0.25), seam);
        spec = pow(max(dot(n, h), 0.0), 32.0) * 0.28;
    } else if (v_mat < 4.5) {
        // MAT 4: Toiture industrielle
        float rib = abs(fract(v_world.x * 2.0) - 0.5);
        base = mix(vec3(0.24, 0.25, 0.27), vec3(0.16, 0.17, 0.19), step(0.4, rib));
        spec = pow(max(dot(n, h), 0.0), 24.0) * 0.20;
    } else if (v_mat < 5.5) {
        // MAT 5: Murs intérieurs (parpaings peints clairs pour visibilité intérieure)
        float bw = 0.50;
        float bh = 0.25;
        float row = floor(coord.y / bh);
        float x_off = mod(row, 2.0) * (bw * 0.5);
        vec2 bcoord = vec2(coord.x + x_off, coord.y);
        vec2 bid = floor(bcoord / vec2(bw, bh));
        vec2 f = fract(bcoord / vec2(bw, bh));
        float mortar_w = 0.03;
        float is_mortar = step(f.x, mortar_w) + step(1.0 - mortar_w, f.x)
                        + step(f.y, mortar_w * 1.5) + step(1.0 - mortar_w * 1.5, f.y);
        is_mortar = clamp(is_mortar, 0.0, 1.0);
        float rnd = hash(bid) * 0.08 - 0.04;
        vec3 wall_col = vec3(0.72 + rnd, 0.73 + rnd, 0.70 + rnd);
        base = mix(wall_col, vec3(0.48, 0.48, 0.46), is_mortar);
        spec = pow(max(dot(n, h), 0.0), 16.0) * 0.10;
    } else {
        // MAT 6: Enduit uni, sans texture
        base = vec3(0.78, 0.74, 0.68);
        spec = pow(max(dot(n, h), 0.0), 12.0) * 0.05;
    }

    // Éclairage hémisphérique ciel/sol + illumination intérieure de rebond
    vec3 sky_amb = vec3(0.38, 0.40, 0.44) * max(n.y * 0.5 + 0.5, 0.0);
    vec3 gnd_amb = vec3(0.24, 0.22, 0.20) * max(-n.y * 0.5 + 0.5, 0.0);
    vec3 ambient = sky_amb + gnd_amb + vec3(0.32); // base claire pour l'intérieur

    vec3 color = base * (ambient + vec3(diff * 0.75)) + vec3(spec);

    // Atténuation atmosphérique avec la distance (fondue vers le ciel bleu-gris)
    float d = length(u_eye - v_world);
    color = mix(color, vec3(0.55, 0.7, 0.9), clamp((d - 45.0) / 100.0, 0.0, 1.0));

    // Préservation active de v_uv pour le layout des sommets
    color += vec3(v_uv.x * 1e-7, v_uv.y * 1e-7, 0.0);

    f_color = vec4(color, 1.0);
}
"""


class OpeningInfo:
    """Informations de vérité terrain (ground truth) pour une ouverture du bâtiment."""

    def __init__(self, name, kind, center, size, passable=True, is_niche=False, depth_behind=10.0, description=""):
        self.name = name
        self.kind = kind                 # "hangar_gate", "door", "window", "vent", "niche", "roof_hatch", "floor_hatch"
        self.center = np.asarray(center, dtype="f8")  # [x, y, z] en coordonnées monde
        self.size = tuple(size)          # (largeur, hauteur)
        self.passable = passable         # True si le drone peut physiquement traverser
        self.is_niche = is_niche         # True si l'ouverture est fermée par un fond opaque
        self.depth_behind = depth_behind # Profondeur visible derrière l'ouverture (m)
        self.description = description

    def __repr__(self):
        status = "Passable" if self.passable else ("Niche (borgne)" if self.is_niche else "Infranchissable")
        return (f"<Opening '{self.name}' ({self.kind}) {self.size[0]:.2f}x{self.size[1]:.2f}m "
                f"at {self.center.round(2).tolist()} - {status}>")


class WallOpeningSpec:
    """Spécification d'une ouverture pour le découpage géométrique du mur."""

    def __init__(self, name, u_min, u_max, v_min, v_max, kind="window",
                 passable=True, is_niche=False, niche_depth=0.20, frame=True, warning_stripes=False):
        self.name = name
        self.u_min = float(u_min)
        self.u_max = float(u_max)
        self.v_min = float(v_min)
        self.v_max = float(v_max)
        self.kind = kind
        self.passable = passable
        self.is_niche = is_niche
        self.niche_depth = niche_depth
        self.frame = frame
        self.warning_stripes = warning_stripes

    @property
    def width(self):
        return self.u_max - self.u_min

    @property
    def height(self):
        return self.v_max - self.v_min

    @property
    def u_center(self):
        return (self.u_min + self.u_max) / 2.0

    @property
    def v_center(self):
        return (self.v_min + self.v_max) / 2.0


def add_box_geometry(verts, idx, center, half, mat_id=MAT_CONCRETE):
    """Ajoute une boîte alignée sur les axes à la liste de sommets et d'indices."""
    cx, cy, cz = center
    hx, hy, hz = half
    axes = [(1, 2), (2, 0), (0, 1)]  # paires d'axes u, v pour chaque normale d'axe

    for axis in range(3):
        u_ax, v_ax = axes[axis]
        for sign in (-1.0, 1.0):
            n = [0.0, 0.0, 0.0]
            n[axis] = sign
            base_idx = len(verts) // 9  # 9 floats par sommet : pos(3), normal(3), uv(2), mat(1)

            for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
                p = [0.0, 0.0, 0.0]
                p[axis] = sign
                p[u_ax] = a
                p[v_ax] = b

                wx = cx + p[0] * hx
                wy = cy + p[1] * hy
                wz = cz + p[2] * hz

                # UV dérivés des dimensions du panneau
                u = (a + 1.0) * 0.5
                v = (b + 1.0) * 0.5

                verts.extend((wx, wy, wz, *n, u, v, mat_id))

            idx.extend((base_idx, base_idx + 1, base_idx + 2,
                        base_idx, base_idx + 2, base_idx + 3))


class ProceduralBuilding:
    """Générateur et moteur de rendu du bâtiment industriel procédural.

    Structure :
    - Bâtiment principal (Hangar double hauteur) : 9.0 m (X) x 10.0 m (Z) x 5.8 m (Y)
    - Aile atelier / contrôle : 5.0 m (X) x 10.0 m (Z) x 5.8 m (Y)
    - Cloison intérieure séparatrice avec porte, baie d'observation et niche
    - Mezzanine à l'étage (Y = 3.1 m) avec trémie ouverte pour le vol vertical
    - Toiture avec puits de lumière / trappe zénithale
    - Poteaux métalliques de renfort intérieur
    - Collision exacte par boîtes découpées
    """

    def __init__(self, ctx, origin=(0.0, 0.0, -12.0)):
        self.ctx = ctx
        self.origin = np.asarray(origin, dtype="f8")
        self.prog = ctx.program(vertex_shader=BUILDING_VERTEX_SHADER,
                                fragment_shader=BUILDING_FRAGMENT_SHADER)

        self.colliders = []
        self.openings = []
        self.verts = []
        self.idx = []

        self._build_geometry()

        # Création du buffer GPU ModernGL
        v_arr = np.array(self.verts, dtype="f4")
        i_arr = np.array(self.idx, dtype="i4")
        self.vbo = ctx.buffer(v_arr.tobytes())
        self.ibo = ctx.buffer(i_arr.tobytes())
        self.vao = ctx.vertex_array(
            self.prog,
            [(self.vbo, "3f 3f 2f 1f", "in_pos", "in_normal", "in_uv", "in_mat")],
            self.ibo
        )

    def _add_solid_box(self, center, half, mat_id=MAT_CONCRETE, add_collider=True):
        """Ajoute un bloc solide à la géométrie et (optionnellement) aux collisionneurs."""
        center_w = np.asarray(center, dtype="f8") + self.origin
        half_w = np.asarray(half, dtype="f8")
        add_box_geometry(self.verts, self.idx, center_w, half_w, mat_id=mat_id)
        if add_collider:
            self.colliders.append(Box(center_w, half_w))

    def _build_wall_x(self, x_min, x_max, z_pos, y_min, y_max, thickness=0.25,
                      openings=(), is_interior=False, default_mat=MAT_CONCRETE):
        """Construit un mur parallèle à l'axe X avec découpe exacte des ouvertures."""
        hz = thickness / 2.0
        mat = MAT_INTERIOR_WALL if is_interior else default_mat

        # Trier les ouvertures le long de X
        sorted_ops = sorted(openings, key=lambda op: op.u_min)
        u_curr = x_min

        for op in sorted_ops:
            # 1. Poteau plein à gauche de l'ouverture
            if op.u_min > u_curr:
                hx = (op.u_min - u_curr) / 2.0
                cx = u_curr + hx
                hy = (y_max - y_min) / 2.0
                cy = y_min + hy
                self._add_solid_box((cx, cy, z_pos), (hx, hy, hz), mat_id=mat)

            # 2. Allège sous l'ouverture
            if op.v_min > y_min:
                hx = op.width / 2.0
                cx = op.u_center
                hy = (op.v_min - y_min) / 2.0
                cy = y_min + hy
                self._add_solid_box((cx, cy, z_pos), (hx, hy, hz), mat_id=mat)

            # 3. Linteau au-dessus de l'ouverture
            if op.v_max < y_max:
                hx = op.width / 2.0
                cx = op.u_center
                hy = (y_max - op.v_max) / 2.0
                cy = op.v_max + hy
                self._add_solid_box((cx, cy, z_pos), (hx, hy, hz), mat_id=mat)

            # 4. Cadre métallique de finition autour de l'ouverture (haut contraste Harris)
            if op.frame:
                frame_t = 0.06
                # Linteau acier foncé
                self._add_solid_box(
                    (op.u_center, op.v_max - frame_t / 2.0, z_pos),
                    (op.width / 2.0, frame_t / 2.0, hz * 1.05),
                    mat_id=MAT_STEEL_DARK, add_collider=False
                )
                # Appui acier foncé
                if op.v_min > y_min:
                    self._add_solid_box(
                        (op.u_center, op.v_min + frame_t / 2.0, z_pos),
                        (op.width / 2.0, frame_t / 2.0, hz * 1.05),
                        mat_id=MAT_STEEL_DARK, add_collider=False
                    )

            # 5. Bandeau de zébrures de sécurité (ex. au-dessus du portail)
            if op.warning_stripes:
                stripe_h = 0.40
                self._add_solid_box(
                    (op.u_center, op.v_max + stripe_h / 2.0, z_pos + (0.02 if z_pos > 0 else -0.02)),
                    (op.width / 2.0 + 0.20, stripe_h / 2.0, hz * 1.08),
                    mat_id=MAT_STEEL_YELLOW, add_collider=False
                )

            # 6. Fond de niche si c'est une ouverture aveugle
            if op.is_niche:
                back_hz = 0.05
                # Décalé vers l'arrière du mur pour laisser un creux visible
                back_z = z_pos - (hz - back_hz) if z_pos > 0 else z_pos + (hz - back_hz)
                self._add_solid_box(
                    (op.u_center, op.v_center, back_z),
                    (op.width / 2.0, op.height / 2.0, back_hz),
                    mat_id=mat, add_collider=True
                )

            # Enregistrement de la vérité terrain
            world_center = self.origin + np.array([op.u_center, op.v_center, z_pos])
            depth = op.niche_depth if op.is_niche else (12.0 if not is_interior else 5.0)
            self.openings.append(OpeningInfo(
                name=op.name,
                kind=op.kind,
                center=world_center,
                size=(op.width, op.height),
                passable=op.passable,
                is_niche=op.is_niche,
                depth_behind=depth,
                description=f"Ouverture {op.name} sur façade X={x_min:.1f}..{x_max:.1f}, Z={z_pos:.1f}"
            ))

            u_curr = op.u_max

        # Dernier tronçon plein à droite
        if u_curr < x_max:
            hx = (x_max - u_curr) / 2.0
            cx = u_curr + hx
            hy = (y_max - y_min) / 2.0
            cy = y_min + hy
            self._add_solid_box((cx, cy, z_pos), (hx, hy, hz), mat_id=mat)

    def _build_wall_z(self, z_min, z_max, x_pos, y_min, y_max, thickness=0.25,
                      openings=(), is_interior=False, default_mat=MAT_CONCRETE):
        """Construit un mur parallèle à l'axe Z avec découpe exacte des ouvertures."""
        hx = thickness / 2.0
        mat = MAT_INTERIOR_WALL if is_interior else default_mat

        sorted_ops = sorted(openings, key=lambda op: op.u_min)
        u_curr = z_min

        for op in sorted_ops:
            # 1. Poteau plein avant l'ouverture
            if op.u_min > u_curr:
                hz = (op.u_min - u_curr) / 2.0
                cz = u_curr + hz
                hy = (y_max - y_min) / 2.0
                cy = y_min + hy
                self._add_solid_box((x_pos, cy, cz), (hx, hy, hz), mat_id=mat)

            # 2. Allège sous l'ouverture
            if op.v_min > y_min:
                hz = op.width / 2.0
                cz = op.u_center
                hy = (op.v_min - y_min) / 2.0
                cy = y_min + hy
                self._add_solid_box((x_pos, cy, cz), (hx, hy, hz), mat_id=mat)

            # 3. Linteau au-dessus de l'ouverture
            if op.v_max < y_max:
                hz = op.width / 2.0
                cz = op.u_center
                hy = (y_max - op.v_max) / 2.0
                cy = op.v_max + hy
                self._add_solid_box((x_pos, cy, cz), (hx, hy, hz), mat_id=mat)

            # 4. Cadre métallique
            if op.frame:
                frame_t = 0.06
                self._add_solid_box(
                    (x_pos, op.v_max - frame_t / 2.0, op.u_center),
                    (hx * 1.05, frame_t / 2.0, op.width / 2.0),
                    mat_id=MAT_STEEL_DARK, add_collider=False
                )
                if op.v_min > y_min:
                    self._add_solid_box(
                        (x_pos, op.v_min + frame_t / 2.0, op.u_center),
                        (hx * 1.05, frame_t / 2.0, op.width / 2.0),
                        mat_id=MAT_STEEL_DARK, add_collider=False
                    )

            # 5. Fond de niche si besoin
            if op.is_niche:
                back_hx = 0.05
                back_x = x_pos - (hx - back_hx) if x_pos > 0 else x_pos + (hx - back_hx)
                self._add_solid_box(
                    (back_x, op.v_center, op.u_center),
                    (back_hx, op.height / 2.0, op.width / 2.0),
                    mat_id=mat, add_collider=True
                )

            # Vérité terrain
            world_center = self.origin + np.array([x_pos, op.v_center, op.u_center])
            depth = op.niche_depth if op.is_niche else (12.0 if not is_interior else 5.0)
            self.openings.append(OpeningInfo(
                name=op.name,
                kind=op.kind,
                center=world_center,
                size=(op.width, op.height),
                passable=op.passable,
                is_niche=op.is_niche,
                depth_behind=depth,
                description=f"Ouverture {op.name} sur paroi Z={z_min:.1f}..{z_max:.1f}, X={x_pos:.1f}"
            ))

            u_curr = op.u_max

        if u_curr < z_max:
            hz = (z_max - u_curr) / 2.0
            cz = u_curr + hz
            hy = (y_max - y_min) / 2.0
            cy = y_min + hy
            self._add_solid_box((x_pos, cy, cz), (hx, hy, hz), mat_id=mat)

    def _build_horizontal_slab(self, x_min, x_max, z_min, z_max, y_pos, thickness=0.20,
                               opening_xz=None, mat_id=MAT_FLOOR):
        """Construit une dalle horizontale (sol, mezzanine ou toit) avec découpe optionnelle."""
        hy = thickness / 2.0
        cy = y_pos + hy

        if opening_xz is None:
            # Dalle pleine sans découpe
            hx = (x_max - x_min) / 2.0
            cx = (x_min + x_max) / 2.0
            hz = (z_max - z_min) / 2.0
            cz = (z_min + z_max) / 2.0
            self._add_solid_box((cx, cy, cz), (hx, hy, hz), mat_id=mat_id)
            return

        # Découpe d'une trémie (ox_min, ox_max, oz_min, oz_max)
        ox_min, ox_max, oz_min, oz_max = opening_xz

        # 1. Dalle gauche
        if ox_min > x_min:
            hx = (ox_min - x_min) / 2.0
            cx = x_min + hx
            hz = (z_max - z_min) / 2.0
            cz = (z_min + z_max) / 2.0
            self._add_solid_box((cx, cy, cz), (hx, hy, hz), mat_id=mat_id)

        # 2. Dalle droite
        if ox_max < x_max:
            hx = (x_max - ox_max) / 2.0
            cx = ox_max + hx
            hz = (z_max - z_min) / 2.0
            cz = (z_min + z_max) / 2.0
            self._add_solid_box((cx, cy, cz), (hx, hy, hz), mat_id=mat_id)

        # 3. Dalle avant du trou (dans la bande centrale en X)
        if oz_max < z_max:
            hx = (ox_max - ox_min) / 2.0
            cx = (ox_min + ox_max) / 2.0
            hz = (z_max - oz_max) / 2.0
            cz = oz_max + hz
            self._add_solid_box((cx, cy, cz), (hx, hy, hz), mat_id=mat_id)

        # 4. Dalle arrière du trou (dans la bande centrale en X)
        if oz_min > z_min:
            hx = (ox_max - ox_min) / 2.0
            cx = (ox_min + ox_max) / 2.0
            hz = (oz_min - z_min) / 2.0
            cz = z_min + hz
            self._add_solid_box((cx, cy, cz), (hx, hy, hz), mat_id=mat_id)

    def _build_geometry(self):
        """Assemble l'ensemble du complexe industriel."""
        # Dimensions d'ensemble :
        # X : [-7.0, +7.0] (14.0 m)
        # Z : [-5.0, +5.0] (10.0 m)
        # Y : [0.0, 5.8 m]
        # Cloison intérieure à X = 2.0 m :
        #   - Hangar : X in [-7.0, 2.0] (9.0 m x 10.0 m)
        #   - Atelier : X in [2.0, 7.0] (5.0 m x 10.0 m)

        wall_thick = 0.25

        # -------------------------------------------------------------------
        # 1. FAÇADE SUD (Face avant Z = +5.0 m, visible depuis le spawn du drone)
        # -------------------------------------------------------------------
        # Découpe en 2 étages (RDC Y in [0, 3.1] et Étage Y in [3.3, 5.8])
        # avec poutre allège/linteau intermédiaire solide à Y in [3.1, 3.3]
        south_rdc = [
            WallOpeningSpec(
                name="Fenetre Ouest RDC",
                u_min=-6.2, u_max=-4.8, v_min=1.1, v_max=2.2,
                kind="window", passable=True
            ),
            WallOpeningSpec(
                name="Portail Principal Hangar",
                u_min=-4.2, u_max=-1.2, v_min=0.0, v_max=2.7,
                kind="hangar_gate", passable=True, warning_stripes=True
            ),
            WallOpeningSpec(
                name="Niche Borgne Sud",
                u_min=0.3, u_max=1.3, v_min=1.1, v_max=2.1,
                kind="niche", passable=False, is_niche=True, niche_depth=0.18
            ),
            WallOpeningSpec(
                name="Porte Service Atelier",
                u_min=3.5, u_max=4.7, v_min=0.0, v_max=2.2,
                kind="door", passable=True
            ),
            WallOpeningSpec(
                name="Fenetre Atelier Sud",
                u_min=5.3, u_max=6.5, v_min=1.1, v_max=2.2,
                kind="window", passable=True
            ),
        ]
        south_etage = [
            WallOpeningSpec(
                name="Fenetre Etroite Etage",
                u_min=-6.0, u_max=-5.5, v_min=3.8, v_max=4.5,
                kind="tight_window", passable=True
            ),
            WallOpeningSpec(
                name="Fenetre Haute Mezzanine",
                u_min=-4.0, u_max=-2.0, v_min=3.8, v_max=5.0,
                kind="window", passable=True
            ),
            WallOpeningSpec(
                name="Fente Aeration Haute",
                u_min=4.0, u_max=4.22, v_min=4.2, v_max=4.42,
                kind="vent", passable=False
            ),
        ]
        # RDC Sud
        self._build_wall_x(-7.0, 7.0, 5.0, 0.0, 3.1, thickness=wall_thick, openings=south_rdc)
        # Poutre intermédiaire d'étage
        self._add_solid_box((0.0, 3.2, 5.0), (7.0, 0.10, wall_thick / 2.0), mat_id=MAT_CONCRETE)
        # Étage Sud
        self._build_wall_x(-7.0, 7.0, 5.0, 3.3, 5.8, thickness=wall_thick, openings=south_etage)

        # -------------------------------------------------------------------
        # 2. FAÇADE NORD (Face arrière Z = -5.0 m)
        # -------------------------------------------------------------------
        north_rdc = [
            WallOpeningSpec(
                name="Fenetre Nord Hangar",
                u_min=-6.2, u_max=-4.6, v_min=1.2, v_max=2.2,
                kind="window", passable=True
            ),
            WallOpeningSpec(
                name="Porte Arriere Secours",
                u_min=-3.5, u_max=-2.1, v_min=0.0, v_max=2.3,
                kind="door", passable=True
            ),
            WallOpeningSpec(
                name="Baie Nord Atelier",
                u_min=3.5, u_max=5.5, v_min=1.2, v_max=2.2,
                kind="window", passable=True
            ),
        ]
        north_etage = [
            WallOpeningSpec(
                name="Fenetre Haute Nord 1",
                u_min=-4.5, u_max=-2.5, v_min=3.8, v_max=4.9,
                kind="window", passable=True
            ),
            WallOpeningSpec(
                name="Fenetre Haute Nord 2",
                u_min=3.5, u_max=5.5, v_min=3.8, v_max=4.9,
                kind="window", passable=True
            ),
        ]
        # RDC Nord
        self._build_wall_x(-7.0, 7.0, -5.0, 0.0, 3.1, thickness=wall_thick, openings=north_rdc)
        # Poutre intermédiaire d'étage
        self._add_solid_box((0.0, 3.2, -5.0), (7.0, 0.10, wall_thick / 2.0), mat_id=MAT_CONCRETE)
        # Étage Nord
        self._build_wall_x(-7.0, 7.0, -5.0, 3.3, 5.8, thickness=wall_thick, openings=north_etage)

        # -------------------------------------------------------------------
        # 3. FAÇADE OUEST (Mur gauche X = -7.0 m)
        # -------------------------------------------------------------------
        west_openings = [
            WallOpeningSpec(
                name="Fenetre Ouest 1",
                u_min=-3.8, u_max=-2.2, v_min=1.2, v_max=2.2,
                kind="window", passable=True
            ),
            # Porte quai de chargement latéral
            WallOpeningSpec(
                name="Porte Quai Chargement",
                u_min=-1.0, u_max=1.2, v_min=0.0, v_max=2.5,
                kind="door", passable=True
            ),
            WallOpeningSpec(
                name="Fenetre Ouest 2",
                u_min=2.2, u_max=3.8, v_min=1.2, v_max=2.2,
                kind="window", passable=True
            ),
        ]
        self._build_wall_z(-5.0, 5.0, -7.0, 0.0, 5.8, thickness=wall_thick,
                           openings=west_openings)

        # -------------------------------------------------------------------
        # 4. FAÇADE EST (Mur droit X = +7.0 m)
        # -------------------------------------------------------------------
        east_openings = [
            WallOpeningSpec(
                name="Fenetre Est Atelier",
                u_min=-3.8, u_max=-2.0, v_min=1.2, v_max=2.2,
                kind="window", passable=True
            ),
            # Porte de livraison atelier
            WallOpeningSpec(
                name="Porte Livraison Est",
                u_min=-0.8, u_max=0.8, v_min=0.0, v_max=2.3,
                kind="door", passable=True
            ),
            WallOpeningSpec(
                name="Fenetre Est Haute",
                u_min=2.0, u_max=3.8, v_min=3.8, v_max=4.9,
                kind="window", passable=True
            ),
        ]
        self._build_wall_z(-5.0, 5.0, 7.0, 0.0, 5.8, thickness=wall_thick,
                           openings=east_openings)

        # -------------------------------------------------------------------
        # 5. CLOISON INTÉRIEURE (X = +2.0 m, sépare Hangar et Atelier)
        # -------------------------------------------------------------------
        interior_openings = [
            # Porte de liaison haute à l'étage
            WallOpeningSpec(
                name="Porte Liaison Etage",
                u_min=-3.5, u_max=-2.3, v_min=3.3, v_max=5.3,
                kind="door", passable=True
            ),
            # Baie d'observation intérieure RDC (regard entre les pièces)
            WallOpeningSpec(
                name="Baie Observation Interieure",
                u_min=-1.5, u_max=0.0, v_min=1.2, v_max=2.2,
                kind="window", passable=True
            ),
            # Passage principal intérieur (arche / porte sans battant)
            WallOpeningSpec(
                name="Passage Principal Interieur",
                u_min=1.0, u_max=2.6, v_min=0.0, v_max=2.4,
                kind="door", passable=True
            ),
            # Niche murale intérieure aveugle
            WallOpeningSpec(
                name="Niche Murale Interieure",
                u_min=3.2, u_max=4.2, v_min=1.2, v_max=2.0,
                kind="niche", passable=False, is_niche=True, niche_depth=0.15
            ),
        ]
        self._build_wall_z(-5.0, 5.0, 2.0, 0.0, 5.8, thickness=wall_thick,
                           openings=interior_openings, is_interior=True)

        # -------------------------------------------------------------------
        # 6. DALLE DE SOL INTÉRIEURE (Béton poli avec joints)
        # -------------------------------------------------------------------
        self._build_horizontal_slab(-7.0, 7.0, -5.0, 5.0, y_pos=0.0, thickness=0.10,
                                    mat_id=MAT_FLOOR)

        # -------------------------------------------------------------------
        # 7. MEZZANINE / PLANCHER DE L'ÉTAGE (Y = 3.1 m)
        # -------------------------------------------------------------------
        # Couvre le fond du hangar (X in [-7.0, 2.0], Z in [-5.0, 0.5])
        # avec une trémie ouverte (2.0 m x 2.0 m) pour voler entre les niveaux !
        tremie_xz = (-2.5, -0.5, -2.0, 0.0)
        self._build_horizontal_slab(-7.0, 2.0, -5.0, 0.5, y_pos=3.1, thickness=0.20,
                                    opening_xz=tremie_xz, mat_id=MAT_FLOOR)

        # Plancher étage atelier (X in [2.0, 7.0], Z in [-5.0, 5.0])
        self._build_horizontal_slab(2.0, 7.0, -5.0, 5.0, y_pos=3.1, thickness=0.20,
                                    mat_id=MAT_FLOOR)

        # Enregistrement de la trémie comme passage vertical de vérité terrain
        tremie_center = self.origin + np.array([-1.5, 3.2, -1.0])
        self.openings.append(OpeningInfo(
            name="Tremie Mezzanine Hangar",
            kind="floor_hatch",
            center=tremie_center,
            size=(2.0, 2.0),
            passable=True,
            depth_behind=5.0,
            description="Ouverture horizontale dans le plancher pour navigation verticale entre niveaux"
        ))

        # -------------------------------------------------------------------
        # 8. TOITURE INDUSTRIELLE AVEC PUITS DE LUMIÈRE (Y = 5.8 m)
        # -------------------------------------------------------------------
        # Puits de lumière / Trappe zénithale de 2.0 m x 2.0 m au-dessus du grand hall
        skylight_xz = (-3.5, -1.5, 1.0, 3.0)
        self._build_horizontal_slab(-7.0, 7.0, -5.0, 5.0, y_pos=5.8, thickness=0.25,
                                    opening_xz=skylight_xz, mat_id=MAT_ROOF)

        # Acrotère / rebord de toiture en périphérie
        rim_h = 0.35
        rim_t = 0.20
        # Bordures de toit
        self._add_solid_box((0.0, 5.8 + rim_h / 2.0, 5.0), (7.0, rim_h / 2.0, rim_t), mat_id=MAT_STEEL_DARK, add_collider=True)
        self._add_solid_box((0.0, 5.8 + rim_h / 2.0, -5.0), (7.0, rim_h / 2.0, rim_t), mat_id=MAT_STEEL_DARK, add_collider=True)
        self._add_solid_box((-7.0, 5.8 + rim_h / 2.0, 0.0), (rim_t, rim_h / 2.0, 5.0), mat_id=MAT_STEEL_DARK, add_collider=True)
        self._add_solid_box((7.0, 5.8 + rim_h / 2.0, 0.0), (rim_t, rim_h / 2.0, 5.0), mat_id=MAT_STEEL_DARK, add_collider=True)

        # Enregistrement du puits de lumière
        skylight_center = self.origin + np.array([-2.5, 5.8, 2.0])
        self.openings.append(OpeningInfo(
            name="Puits de Lumiere Zenithal",
            kind="roof_hatch",
            center=skylight_center,
            size=(2.0, 2.0),
            passable=True,
            depth_behind=6.0,
            description="Ouverture zénithale permettant l'insertion ou l'extraction verticale du drone"
        ))

        # -------------------------------------------------------------------
        # 9. POTEAUX STRUCTURAUX MÉTALLIQUES (I-BEAMS INTÉRIEURS)
        # -------------------------------------------------------------------
        # Colonnes dans le grand hall du hangar (supportant la mezzanine et le toit)
        col_half = (0.18, 5.8 / 2.0, 0.18)
        self._add_solid_box((-2.5, 5.8 / 2.0, 0.5), col_half, mat_id=MAT_STEEL_DARK, add_collider=True)
        self._add_solid_box((-2.5, 3.1 / 2.0, -3.0), (0.18, 3.1 / 2.0, 0.18), mat_id=MAT_STEEL_DARK, add_collider=True)

    def render(self, camera, aspect, light_dir=(-0.4, -1.0, -0.3)):
        """Dessine le bâtiment dans le contexte ModernGL courant."""
        self.prog["u_proj"].write(camera.projection(aspect).T.copy().tobytes())
        self.prog["u_view"].write(camera.view().T.copy().tobytes())
        model = np.identity(4, dtype="f4")
        self.prog["u_model"].write(model.tobytes())
        self.prog["u_light_dir"].value = tuple(light_dir)
        self.prog["u_eye"].value = tuple(camera.position)

        self.vao.render()

    def get_openings(self):
        """Retourne la liste des ouvertures avec leurs vérités terrain."""
        return list(self.openings)
