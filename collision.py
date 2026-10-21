"""Collisions cinématiques : le drone est une boîte orientée (lacet seulement),
les obstacles sont des boîtes alignées sur les axes, des sphères ou le sol.

Pas de moteur physique : à chaque pas, on déplace le drone puis on le repousse
hors des obstacles (le long de l'axe de pénétration minimale), ce qui le fait
glisser contre les surfaces.
"""

import numpy as np

UP = np.array([0.0, 1.0, 0.0])


class DroneBox:
    """Boîte orientée du drone : centre, demi-dimensions (droite, haut, avant), lacet."""

    def __init__(self, half_extents):
        self.half = np.asarray(half_extents, dtype="f8")

    @staticmethod
    def axes(forward):
        f = np.array([forward[0], 0.0, forward[2]])
        f /= np.linalg.norm(f)
        r = np.cross(f, UP)
        return r, UP, f


class Box:
    """Obstacle boîte alignée sur les axes."""

    def __init__(self, center, half_extents):
        self.center = np.asarray(center, dtype="f8")
        self.half = np.asarray(half_extents, dtype="f8")

    def push_out(self, pos, axes, half):
        """Vecteur à ajouter à pos pour sortir de la boîte, ou None."""
        d = pos - self.center
        # Axe vertical (commun aux deux boîtes).
        best = (half[1] + self.half[1] - abs(d[1]), UP * np.sign(d[1] or 1.0))
        if best[0] <= 0:
            return None
        # Axes horizontaux candidats (théorème des axes séparateurs, en 2D dans XZ).
        r, _, f = axes
        for a in (np.array([1.0, 0, 0]), np.array([0, 0, 1.0]), r, f):
            proj_drone = half[0] * abs(r @ a) + half[2] * abs(f @ a)
            proj_box = self.half[0] * abs(a[0]) + self.half[2] * abs(a[2])
            depth = proj_drone + proj_box - abs(d @ a)
            if depth <= 0:
                return None
            if depth < best[0]:
                best = (depth, a * np.sign(d @ a or 1.0))
        return best[1] * best[0]


class Sphere:
    def __init__(self, center, radius):
        self.center = np.asarray(center, dtype="f8")
        self.radius = float(radius)

    def push_out(self, pos, axes, half):
        rot = np.column_stack(axes)  # colonnes : droite, haut, avant
        local = rot.T @ (self.center - pos)
        closest = rot @ np.clip(local, -half, half) + pos
        delta = closest - self.center
        dist = np.linalg.norm(delta)
        if dist >= self.radius:
            return None
        if dist < 1e-9:  # centre de la sphère dans la boîte : on sort vers le haut
            return UP * (self.radius + half[1] - (pos[1] - self.center[1]))
        return delta / dist * (self.radius - dist)


class Ground:
    def __init__(self, height=0.0):
        self.height = height

    def push_out(self, pos, axes, half):
        depth = self.height + half[1] - pos[1]
        return UP * depth if depth > 0 else None


def move_with_collisions(pos, displacement, forward, drone_box, colliders, max_step=0.05, iterations=4):
    """Déplace pos de displacement par petits pas en résolvant les collisions.

    Retourne (nouvelle position, True si un contact a eu lieu).
    """
    pos = np.asarray(pos, dtype="f8").copy()
    displacement = np.asarray(displacement, dtype="f8")
    axes = DroneBox.axes(forward)
    steps = max(1, int(np.ceil(np.linalg.norm(displacement) / max_step)))
    hit = False
    for _ in range(steps):
        before = pos.copy()
        pos += displacement / steps
        for _ in range(iterations):
            moved = False
            for c in colliders:
                push = c.push_out(pos, axes, drone_box.half)
                if push is not None:
                    pos += push * 1.0001
                    moved = hit = True
            if not moved:
                break
        else:
            # Pas de position libre trouvée (ex. coincé entre les deux bords d'une
            # ouverture trop étroite : les poussées se compensent) : on annule le pas.
            if any(c.push_out(pos, axes, drone_box.half) is not None for c in colliders):
                pos = before
    return pos, hit
