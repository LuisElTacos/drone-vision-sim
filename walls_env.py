"""Environnement « murs troués » : une suite de murs unis percés de trous.

Le drone part devant le premier mur, la cible est derrière le dernier. Chaque mur
contient plusieurs ouvertures de types différents :
- passage     : franchissable, mène à l'espace entre ce mur et le suivant ;
- cul_de_sac  : franchissable, mais débouche dans une poche fermée (5 faces) ;
- trop_petit  : trop étroite pour le drone.

Chaque mur a au moins un vrai passage, décalé par rapport à celui du mur précédent :
un chemin vers la cible est donc garanti. La zone est fermée sur les côtés et aux
extrémités ; il n'y a pas de toit (la lumière reste), mais un plafond invisible à la
hauteur des murs empêche de passer par-dessus.

Tout est généré à partir d'une graine : même graine, même environnement. La vérité
terrain des ouvertures est disponible via get_openings().
"""

import random

import numpy as np

from building import MAT_PLAIN, ProceduralBuilding, WallOpeningSpec
from collision import Box


class WallMaze(ProceduralBuilding):
    """Murs troués successifs, perpendiculaires à Z ; le drone avance vers -Z."""

    def __init__(self, ctx, seed=0, origin=(60.0, 0.0, 0.0), n_walls=5, spacing=5.0,
                 width=30.0, height=5.0, thickness=0.3, n_slots=6, holes_per_wall=4,
                 p_dead_end=0.35, p_too_small=0.25):
        self.seed = int(seed)
        self.n_walls = n_walls
        self.spacing = spacing
        self.width = width
        self.height = height
        self.thickness = thickness
        self.n_slots = n_slots
        self.holes_per_wall = min(holes_per_wall, n_slots)
        self.p_dead_end = p_dead_end
        self.p_too_small = p_too_small
        super().__init__(ctx, origin=origin)

    # --- points utiles (coordonnées monde) -------------------------------------

    @property
    def spawn(self):
        """(position, lacet) de départ : devant le premier mur, face à -Z."""
        return tuple(self.origin + (0.0, 1.5, 2.0)), -90.0

    @property
    def overview_camera(self):
        """(position, cible) d'une caméra fixe qui voit tout l'environnement."""
        length = (self.n_walls + 1) * self.spacing
        pos = self.origin + (self.width / 2.0 + 10.0, self.height + 14.0, 6.0)
        target = self.origin + (0.0, 0.0, -length / 2.0)
        return tuple(pos), tuple(target)

    def summary(self):
        kinds = [o.kind for o in self.openings]
        return {k: kinds.count(k) for k in ("passage", "cul_de_sac", "trop_petit")}

    # --- génération -------------------------------------------------------------

    def _build_geometry(self):
        rng = random.Random(self.seed)
        half_w, H, T = self.width / 2.0, self.height, self.thickness
        slot_w = self.width / self.n_slots
        z_start, z_end = 4.0, -(self.n_walls + 1) * self.spacing

        prev_passage_slot = None
        for k in range(self.n_walls):
            z = -(k + 1) * self.spacing
            slots = list(range(self.n_slots))
            rng.shuffle(slots)
            passage_slot = next(s for s in slots if s != prev_passage_slot)
            prev_passage_slot = passage_slot
            chosen = [passage_slot] + [s for s in slots if s != passage_slot][:self.holes_per_wall - 1]

            specs, dead_ends = [], {}
            for i, slot in enumerate(chosen):
                if slot == passage_slot:
                    kind = "passage"
                else:
                    r = rng.random()
                    kind = ("cul_de_sac" if r < self.p_dead_end
                            else "trop_petit" if r < self.p_dead_end + self.p_too_small
                            else "passage")
                if kind == "trop_petit":
                    w, h = rng.uniform(0.10, 0.18), rng.uniform(0.25, 0.8)
                else:
                    w, h = rng.uniform(0.7, 1.6), rng.uniform(0.6, 1.4)
                # centre dans la case, avec de la marge pour la poche éventuelle
                margin = (slot_w - w) / 2.0 - 0.8
                u = -half_w + (slot + 0.5) * slot_w + rng.uniform(-margin, margin)
                v_min = rng.uniform(0.3, H - h - 0.6)
                name = f"M{k + 1}-{i + 1}"
                specs.append(WallOpeningSpec(name, u - w / 2.0, u + w / 2.0, v_min, v_min + h,
                                             kind=kind, passable=(kind == "passage"), frame=True))
                if kind == "cul_de_sac":
                    dead_ends[name] = rng.uniform(1.5, min(3.0, self.spacing - T - 1.5))

            first = len(self.openings)
            self._build_wall_x(-half_w, half_w, z, 0.0, H, thickness=T, openings=specs, default_mat=MAT_PLAIN)

            # Poches des culs-de-sac et vérité terrain
            for spec in specs:
                if spec.name in dead_ends:
                    self._build_pocket(spec, z, dead_ends[spec.name])
            for info in self.openings[first:]:
                if info.name in dead_ends:
                    info.is_niche, info.passable = True, False
                    info.depth_behind = dead_ends[info.name]
                    info.description = f"Cul-de-sac de {info.depth_behind:.1f} m derrière le mur {k + 1}"
                elif info.kind == "passage":
                    info.depth_behind = self.spacing - T
                    info.description = f"Passage vers l'espace derrière le mur {k + 1}"
                else:
                    info.depth_behind = self.spacing - T
                    info.description = f"Ouverture trop étroite ({info.size[0] * 100:.0f} cm) dans le mur {k + 1}"

        # Enceinte : murs latéraux, mur derrière le départ, mur du fond
        length = z_start - z_end
        zc = (z_start + z_end) / 2.0
        for side in (-1.0, 1.0):
            self._add_solid_box((side * (half_w + T / 2.0), H / 2.0, zc), (T / 2.0, H / 2.0, length / 2.0 + T),
                                mat_id=MAT_PLAIN)
        for zw in (z_start, z_end):
            self._add_solid_box((0.0, H / 2.0, zw), (half_w, H / 2.0, T / 2.0), mat_id=MAT_PLAIN)

        # Plafond invisible (collision seule) : pas de passage par-dessus les murs
        self.colliders.append(Box(self.origin + (0.0, H + 0.5, zc), (half_w + 1.0, 0.5, length / 2.0 + 1.0)))

        # Cible derrière le dernier mur
        self.target = self.origin + (rng.uniform(-half_w + 2.0, half_w - 2.0), rng.uniform(1.0, 3.0),
                                     -(self.n_walls + 0.5) * self.spacing)

    def _build_pocket(self, spec, z_wall, depth, margin=0.3, t=0.2):
        """Boîte fermée derrière l'ouverture : gauche, droite, haut, fond, et bas si surélevée."""
        x0, x1 = spec.u_min - margin, spec.u_max + margin
        y0, y1 = max(spec.v_min - margin, 0.0), spec.v_max + margin
        z_front = z_wall - self.thickness / 2.0
        z_back = z_front - depth
        zc, hz = (z_front + z_back) / 2.0, depth / 2.0
        xc, hx = (x0 + x1) / 2.0, (x1 - x0) / 2.0 + t
        yc, hy = (y0 + y1) / 2.0, (y1 - y0) / 2.0
        self._add_solid_box((x0 - t / 2.0, yc, zc), (t / 2.0, hy + t, hz), mat_id=MAT_PLAIN)
        self._add_solid_box((x1 + t / 2.0, yc, zc), (t / 2.0, hy + t, hz), mat_id=MAT_PLAIN)
        self._add_solid_box((xc, y1 + t / 2.0, zc), (hx, t / 2.0, hz), mat_id=MAT_PLAIN)
        if y0 > 0.0:
            self._add_solid_box((xc, y0 - t / 2.0, zc), (hx, t / 2.0, hz), mat_id=MAT_PLAIN)
        self._add_solid_box((xc, yc, z_back - t / 2.0), (hx, hy + t, t / 2.0), mat_id=MAT_PLAIN)

    def release(self):
        for obj in (self.vao, self.vbo, self.ibo, self.prog):
            obj.release()
