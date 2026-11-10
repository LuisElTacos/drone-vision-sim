"""test16.py : carte de voxels construite à partir de l'image 0 (paire stéréo gauche/droite).

Chaîne : capture -> coins -> appariement -> triangulation -> VoxelMap.integrate -> figures.

    python test16.py                       # murs troués, graine 0
    python test16.py --env foret --seed 1
    python test16.py --voxel 0.15 --save sorties_test16
"""

import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from test14 import capture
from vision.corners import harris_function, detect_corners
from vision.matching import to_gray
from test14 import match_along_rows
from vision.triangulation import triangulate
from vision.voxels import VoxelMap

import scene as sim


def build_points(frame, args):
    imgL, imgR = frame.images[0], frame.images[1]
    K = frame.K
    B = float(np.linalg.norm(frame.Cs[1] - frame.Cs[0]))
    R, C = frame.Rs[0], frame.Cs[0]
    RL, RR = harris_function(to_gray(imgL)), harris_function(to_gray(imgR))
    kw = dict(rel_thresh=args.seuil, max_n=args.max_coins, grid_cells=tuple(args.grille))
    cL, cR = detect_corners(RL, **kw), detect_corners(RR, **kw)
    if args.matcher == "lignes":
        ptsL, ptsR, _ = match_along_rows(imgL, imgR, cL, min_zncc=args.zncc, ratio=args.ratio)
    else:
        ptsL, ptsR, _ = match_along_rows(imgL, imgR, cL, min_zncc=args.zncc, ratio=args.ratio)
    P_w, ok = triangulate(ptsL[:, 0], ptsL[:, 1], ptsR[:, 0], ptsR[:, 1], K, B, R, C)
    return P_w[ok], C, len(cL), len(ptsL)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", choices=sorted(sim.ENV_NAMES), default="murs")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--avance", type=float, default=0.0)
    ap.add_argument("--lacet", type=float, default=0.0)
    ap.add_argument("--W", type=int, default=1730)
    ap.add_argument("--H", type=int, default=1300)
    ap.add_argument("--max-coins", type=int, default=10000)
    ap.add_argument("--seuil", type=float, default=1e-6)
    ap.add_argument("--grille", type=int, nargs=2, default=(12, 16))
    ap.add_argument("--matcher", choices=("lignes", "coins"), default="lignes")
    ap.add_argument("--zncc", type=float, default=0.8)
    ap.add_argument("--ratio", type=float, default=0.6)
    ap.add_argument("--voxel", type=float, default=0.25, help="taille d'un voxel (m)")
    ap.add_argument("--taille", type=float, nargs=3, default=(40.0, 10.0, 40.0),
                    help="taille de la carte en m (x, y vertical, z), centrée sur la caméra")
    ap.add_argument("--portee", type=float, default=15.0)
    ap.add_argument("--save", default=None)
    ap.add_argument("--no-show", action="store_true")
    args = ap.parse_args()

    frame, _ = capture(args)
    P_w, C, n_corners, n_match = build_points(frame, args)

    size = np.array(args.taille)
    origin = np.array([C[0] - size[0] / 2, C[1] - 2.0, C[2] - size[2] / 2])
    vm = VoxelMap(origin, size, args.voxel)
    vm.integrate(P_w, C, max_range=args.portee)
    L = vm.map

    occ, free, unk = (L > 0.4), (L < -0.2), (np.abs(L) <= 0.0)
    print(f"\n=== test16 : env={args.env} seed={args.seed}")
    print(f"  coins {n_corners}, appariés {n_match}, points 3D {len(P_w)}, caméra C = {np.round(C, 2)}")
    print(f"  carte {tuple(vm.shape)} voxels de {args.voxel} m, origine {np.round(origin, 1)}")
    print(f"  occupés {int(occ.sum())}, libres {int((L < 0).sum())}, inconnus {int((L == 0).sum())}"
          f"  ({100 * (L != 0).mean():.2f} % de la carte observée)")

    # Image d'état par voxel : 0 inconnu, 1 libre, 2 occupé
    state = np.zeros(L.shape, np.int8); state[L < 0] = 1; state[L > 0] = 2
    cmap = ListedColormap(["#d9d9d9", "#8fd18f", "#d62728"])
    ci = vm.world_to_idx(C).astype(int)

    fig, ax = plt.subplots(1, 3, figsize=(17, 5.6))
    top = state.max(axis=1)      # axe 1 = y vertical : vue de dessus (plus fort état de la colonne)
    ax[0].imshow(top.T, origin="lower", cmap=cmap, vmin=0, vmax=2,
                 extent=[origin[0], origin[0] + size[0], origin[2], origin[2] + size[2]])
    ax[0].plot(C[0], C[2], "b^", ms=9); ax[0].set_xlabel("x (m)"); ax[0].set_ylabel("z (m)")
    ax[0].set_title("Vue de dessus (état max sur la hauteur)")
    sl = state[:, min(max(ci[1], 0), vm.shape[1] - 1), :]
    ax[1].imshow(sl.T, origin="lower", cmap=cmap, vmin=0, vmax=2,
                 extent=[origin[0], origin[0] + size[0], origin[2], origin[2] + size[2]])
    ax[1].plot(C[0], C[2], "b^", ms=9); ax[1].set_xlabel("x (m)"); ax[1].set_ylabel("z (m)")
    ax[1].set_title(f"Coupe horizontale à la hauteur de la caméra (y = {C[1]:.1f} m)")
    ax[2].remove()
    ax3 = fig.add_subplot(1, 3, 3, projection="3d")
    io = np.argwhere(occ)
    if len(io):
        pw = vm.idx_to_world(io)
        ax3.scatter(pw[:, 0], pw[:, 2], pw[:, 1], s=4, c=pw[:, 1], cmap="viridis")
    ax3.scatter([C[0]], [C[2]], [C[1]], c="b", marker="^", s=50)
    ax3.set_xlabel("x"); ax3.set_ylabel("z"); ax3.set_zlabel("y (haut)")
    ax3.set_title(f"Voxels occupés ({int(occ.sum())})")
    fig.suptitle("gris : inconnu, vert : libre, rouge : occupé")
    fig.tight_layout()

    if args.save:
        import os
        os.makedirs(args.save, exist_ok=True)
        fig.savefig(os.path.join(args.save, "16_voxels.png"), dpi=110)
        print(f"  Figure enregistrée dans {args.save}/")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
