"""Prépare le pin de la forêt à partir du modèle du projet C++ de 2A (trunk.obj, foliage.obj).

- Tronc : on retire les branches en ne gardant que les triangles proches de l'axe
  (rayon inférieur à 1,35 x le rayon du fût mesuré à cette hauteur).
- Feuillage : on ne garde que les triangles au-dessus de FOLIAGE_CUT (pas de feuillage bas).
- Le modèle de branches (branches.obj) n'est pas utilisé.

Sortie : Assets/models/pine/pine.npz (sommets [pos, normale, uv] du tronc et du feuillage)
et copie des textures trunk.png et pine.png.

Usage : python tools/prepare_pine.py <dossier contenant trunk.obj, foliage.obj, trunk.png, pine.png>
"""

import os
import shutil
import sys

import numpy as np

FOLIAGE_CUT = 1.3  # hauteur (unités du modèle, pin de 3,9) sous laquelle on retire le feuillage

# rayon du fût en fonction de la hauteur, mesuré sur trunk.obj
CORE_Y = np.array([0.0, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.1, 2.4, 2.7, 3.0, 3.9])
CORE_R = np.array([0.14, 0.13, 0.115, 0.105, 0.095, 0.085, 0.078, 0.06, 0.05, 0.042, 0.035, 0.01])


def load_obj(path):
    """Triangles non indexés : tableau (n, 8) = position, normale, uv (v retourné pour OpenGL)."""
    V, T, N, out = [], [], [], []
    with open(path) as f:
        for line in f:
            p = line.split()
            if not p:
                continue
            if p[0] == "v":
                V.append([float(x) for x in p[1:4]])
            elif p[0] == "vt":
                T.append([float(x) for x in p[1:3]])
            elif p[0] == "vn":
                N.append([float(x) for x in p[1:4]])
            elif p[0] == "f":
                corners = []
                for w in p[1:]:
                    a = w.split("/")
                    corners.append((int(a[0]) - 1,
                                    int(a[1]) - 1 if len(a) > 1 and a[1] else -1,
                                    int(a[2]) - 1 if len(a) > 2 and a[2] else -1))
                for k in range(1, len(corners) - 1):  # triangulation en éventail
                    for vi, ti, ni in (corners[0], corners[k], corners[k + 1]):
                        out.append(V[vi] + (N[ni] if ni >= 0 else [0.0, 1.0, 0.0]) + (T[ti] if ti >= 0 else [0.0, 0.0]))
    a = np.array(out, dtype="f4")
    a[:, 7] = 1.0 - a[:, 7]
    return a


def main(src):
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dst = os.path.join(root, "Assets", "models", "pine")
    os.makedirs(dst, exist_ok=True)

    trunk = load_obj(os.path.join(src, "trunk.obj")).reshape(-1, 3, 8)
    r = np.hypot(trunk[:, :, 0], trunk[:, :, 2])
    keep = (r < 1.35 * np.interp(trunk[:, :, 1], CORE_Y, CORE_R) + 0.015).all(axis=1)
    trunk = trunk[keep].reshape(-1, 8)

    foliage = load_obj(os.path.join(src, "foliage.obj")).reshape(-1, 3, 8)
    foliage = foliage[foliage[:, :, 1].min(axis=1) > FOLIAGE_CUT].reshape(-1, 8)

    np.savez_compressed(os.path.join(dst, "pine.npz"), trunk=trunk, foliage=foliage,
                        core_y=CORE_Y, core_r=CORE_R, foliage_cut=FOLIAGE_CUT)
    for name in ("trunk.png", "pine.png"):
        shutil.copy(os.path.join(src, name), dst)
    print(f"tronc : {len(trunk) // 3} triangles, feuillage : {len(foliage) // 3} triangles -> {dst}")


if __name__ == "__main__":
    main(sys.argv[1])
