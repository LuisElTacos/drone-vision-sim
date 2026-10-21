"""Formes de collision des environnements (repère du simulateur, Y vers le haut).

Ce sont de simples descriptions géométriques : mujoco_flight.py les convertit en
géométries MuJoCo, qui calcule les contacts avec le drone.
"""

import numpy as np


class Box:
    """Obstacle boîte alignée sur les axes."""

    def __init__(self, center, half_extents):
        self.center = np.asarray(center, dtype="f8")
        self.half = np.asarray(half_extents, dtype="f8")


class Sphere:
    def __init__(self, center, radius):
        self.center = np.asarray(center, dtype="f8")
        self.radius = float(radius)


class VerticalCylinder:
    """Cylindre vertical (tronc d'arbre) : centre (x, z), rayon, hauteurs min et max."""

    def __init__(self, center_xz, radius, y_min, y_max):
        self.center_xz = np.asarray(center_xz, dtype="f8")
        self.radius = float(radius)
        self.y_min, self.y_max = float(y_min), float(y_max)


class Ground:
    def __init__(self, height=0.0):
        self.height = height
