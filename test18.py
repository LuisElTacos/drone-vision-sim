"""test18.py : test16 (stéréo -> voxels) + planificateur (classify + A*) dans la forêt.

Boucle : capture stéréo -> points 3D -> VoxelMap.integrate -> classify -> A* vers la cible
-> le drone est téléporté de `--pas` mètres le long du chemin (comme test17, sans MuJoCo,
caméras stabilisées) -> nouvelle capture, replanification.

    python test18.py                         # forêt, graine 0, 12 itérations
    python test18.py --iterations 1          # une seule image (comme test16) + un plan
    python test18.py --seed 3 --w 1.5 --save sorties_test18
    python test18.py --env murs              # fonctionne aussi pour les murs troués

Vérité terrain : les arbres de scene.forest.obstacles (centre, rayon du tronc). On mesure la
distance minimale drone-tronc le long de la trajectoire parcourue.
"""

import argparse
import math
import time

import glfw
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Circle

import scene as sim
from test14 import make_hidden_context
from test16 import build_points
from vision.capture import StereoCapture
from vision.planner import classify, astar
from vision.voxels import VoxelMap


def to_tuple(idx):
    return tuple(int(x) for x in idx)


def plan(vm, C, goal_w, args):
    """Retourne (path_world ou None, infos). Tous les indices passent par la tranche d'altitude."""
    t0 = time.perf_counter()
    blocked, cost_extra, j_min, libre, dist = classify(vm, args.ymin, args.ymax, args.rayon + args.marge,
                                                       k_unk=args.k_unk, k_prox=args.k_prox, details=True,
                                                       **{k: getattr(args, k) for k in ('d_secu', 'seuil_free', 'seuil_occ')
                                                          if getattr(args, k, None) is not None})
    t1 = time.perf_counter()
    nx, nj, nz = blocked.shape

    def to_grid(p):
        i, j, k = vm.world_to_idx(np.asarray(p, float))
        j -= j_min
        return (int(np.clip(i, 0, nx - 1)), int(np.clip(j, 0, nj - 1)), int(np.clip(k, 0, nz - 1)))

    s, g = to_grid(C), to_grid(goal_w)
    grid_follow = (blocked.copy(), j_min, vm)   # pour le rayon de follow (avant déblocage du départ)
    # départ et voisinage immédiat libres, cible libre
    # voisinage du départ débloqué, sauf les voxels réellement occupés (sinon A* traverse l'obstacle collé au drone)
    nb = (slice(max(s[0] - 1, 0), s[0] + 2), slice(max(s[1] - 1, 0), s[1] + 2), slice(max(s[2] - 1, 0), s[2] + 2))
    blocked[nb] &= (dist[nb] == 0)
    blocked[g] = False
    path = astar(blocked, cost_extra, s, g, vm.voxel_size, w=args.w)
    t2 = time.perf_counter()
    info = dict(t_classify=t1 - t0, t_astar=t2 - t1, n_blocked=int(blocked.sum()), start=s, goal=g,
                shape=blocked.shape, grid=grid_follow)
    if path is None:
        return None, info
    idx = np.array(path)
    ii = tuple(idx.T)                       # indices dans la tranche (avant + j_min)
    info["libre"] = libre[ii]               # (N,) pour follow
    info["dist"] = dist[ii]                 # (N,) distance aux obstacles (m)
    idx[:, 1] += j_min
    return vm.idx_to_world(idx), info


def advance_along(path_w, pos, dist):
    """Point situé à `dist` mètres le long du chemin à partir de la position la plus proche."""
    d = np.linalg.norm(path_w - pos, axis=1)
    i = int(np.argmin(d))
    remaining = dist
    cur = pos.copy()
    for k in range(i, len(path_w) - 1):
        seg = path_w[k + 1] - cur
        L = np.linalg.norm(seg)
        if L >= remaining:
            return cur + seg / L * remaining
        remaining -= L
        cur = path_w[k + 1].copy()
    return path_w[-1].copy()


def clearance_trees(scene, pos):
    """Distance horizontale du drone au bord du tronc le plus proche (m), vérité terrain."""
    forest = scene.forest
    if forest is None or not forest.obstacles:
        return np.inf
    c = np.array([[o["center"][0], o["center"][2]] for o in forest.obstacles])
    r = np.array([o["radius"] for o in forest.obstacles])
    return float((np.hypot(c[:, 0] - pos[0], c[:, 1] - pos[2]) - r).min())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", choices=sorted(sim.ENV_NAMES), default="foret")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--iterations", type=int, default=12)
    ap.add_argument("--pas", type=float, default=2.0, help="avance entre deux images (m)")
    ap.add_argument("--W", type=int, default=865)
    ap.add_argument("--H", type=int, default=650)
    ap.add_argument("--max-coins", type=int, default=10000)
    ap.add_argument("--seuil", type=float, default=1e-6)
    ap.add_argument("--grille", type=int, nargs=2, default=(12, 16))
    ap.add_argument("--matcher", choices=("lignes", "coins"), default="lignes")
    ap.add_argument("--zncc", type=float, default=0.8)
    ap.add_argument("--ratio", type=float, default=0.6)
    ap.add_argument("--voxel", type=float, default=0.25)
    ap.add_argument("--portee", type=float, default=15.0)
    ap.add_argument("--marge-carte", type=float, default=8.0, help="marge autour du segment départ-cible (m)")
    ap.add_argument("--ymin", type=float, default=0.6, help="altitude min de vol (m)")
    ap.add_argument("--ymax", type=float, default=2.4, help="altitude max de vol (m)")
    ap.add_argument("--rayon", type=float, default=0.15, help="rayon du drone (m)")
    ap.add_argument("--marge", type=float, default=0.12, help="marge de sécurité (m)")
    ap.add_argument("--k-unk", type=float, default=0.3)
    ap.add_argument("--k-prox", type=float, default=3.0)
    ap.add_argument("--w", type=float, default=2.0, help="poids de l'heuristique A*")
    ap.add_argument("--save", default=None)
    ap.add_argument("--no-show", action="store_true")
    args = ap.parse_args()

    ctx, window = make_hidden_context()
    scene = sim.Scene(ctx, maze_seed=args.seed)
    mode = sim.ENV_NAMES[args.env]
    scene.building_mode = mode
    if mode == sim.ENV_FOREST:
        scene.forest_seed = args.seed
    pos, yaw = scene.spawn(mode)
    if mode == sim.ENV_FOREST:
        scene.ensure_forest()
    goal_w = np.array(scene.target(), float)
    drone = sim.Drone(scene, position=np.array(pos, float), yaw=float(yaw))
    drone.cams_on_chassis = False
    cams = drone.cameras[:2]
    cap = StereoCapture(ctx, scene, cams, W=args.W, H=args.H)

    # Carte centrée sur le segment départ-cible, taille = segment + marge
    p0 = np.array(pos, float)
    lo = np.minimum(p0, goal_w) - args.marge_carte
    hi = np.maximum(p0, goal_w) + args.marge_carte
    size = np.array([hi[0] - lo[0], 10.0, hi[2] - lo[2]])
    origin = np.array([lo[0], p0[1] - 2.0, lo[2]])
    vm = VoxelMap(origin, size, args.voxel)
    print(f"\n=== test18 : env={args.env} seed={args.seed}")
    print(f"  départ {np.round(p0, 1)}, cible {np.round(goal_w, 1)}, distance {np.linalg.norm(goal_w - p0):.1f} m")
    print(f"  carte {tuple(int(x) for x in vm.shape)} voxels de {args.voxel} m ({int(np.prod(vm.shape)) / 1e6:.2f} M)")

    traj = [drone.position.copy()]
    plans = []
    min_clear = np.inf
    last_path = first_path = None
    for it in range(args.iterations):
        drone.update_cameras()
        frame = cap.grab(cams, mode, 0.0)
        args_cap = argparse.Namespace(**vars(args))
        P_w, C, n_corners, n_match = build_points(frame, args_cap)
        vm.integrate(P_w, C, max_range=args.portee)

        path_w, info = plan(vm, drone.position, goal_w, args)
        info["n_points"] = len(P_w)
        plans.append(info)
        d_goal = float(np.linalg.norm(goal_w - drone.position))
        status = "pas de chemin" if path_w is None else f"chemin {len(path_w)} voxels"
        print(f"  it {it:2d} : pts 3D {len(P_w):5d}  classify {1000 * info['t_classify']:4.0f} ms  "
              f"A* {1000 * info['t_astar']:6.0f} ms  {status}  dist cible {d_goal:5.1f} m")
        if path_w is None:
            break
        last_path = path_w
        if first_path is None:
            first_path = path_w
        if d_goal < 1.0:
            print("  cible atteinte")
            break
        nxt = advance_along(path_w, drone.position, args.pas)
        nxt[1] = drone.position[1]                      # altitude conservée dans ce test
        d = nxt - drone.position
        if math.hypot(d[0], d[2]) > 1e-6:
            drone.yaw = math.degrees(math.atan2(d[2], d[0]))
        drone.position = nxt
        traj.append(nxt.copy())
        if scene.forest is not None:
            min_clear = min(min_clear, clearance_trees(scene, nxt))
    glfw.destroy_window(window)
    glfw.terminate()

    traj = np.array(traj)
    L = vm.map
    print("\n  Bilan")
    print(f"  distance parcourue {np.sum(np.linalg.norm(np.diff(traj, axis=0), axis=1)):.1f} m, "
          f"distance finale à la cible {np.linalg.norm(goal_w - traj[-1]):.1f} m")
    if np.isfinite(min_clear):
        print(f"  distance minimale drone - tronc (vérité terrain) : {100 * min_clear:.0f} cm "
              f"({'COLLISION' if min_clear < 0.0 else 'ok'})")
    print(f"  A* : médiane {1000 * np.median([p['t_astar'] for p in plans]):.0f} ms, "
          f"max {1000 * max(p['t_astar'] for p in plans):.0f} ms")

    # --- figure : vue de dessus -------------------------------------------------
    state = np.zeros(L.shape, np.int8); state[L < 0] = 1; state[L > 0] = 2
    j0 = int(np.ceil((args.ymin - vm.origin[1]) / args.voxel))
    j1 = int(np.floor((args.ymax - vm.origin[1]) / args.voxel))
    top = state[:, j0:j1 + 1, :].max(axis=1)
    cmap = ListedColormap(["#d9d9d9", "#8fd18f", "#d62728"])
    ext = [vm.origin[0], vm.origin[0] + size[0], vm.origin[2], vm.origin[2] + size[2]]
    fig, ax = plt.subplots(figsize=(8, 9))
    ax.imshow(top.T, origin="lower", cmap=cmap, vmin=0, vmax=2, extent=ext, alpha=0.8)
    if scene.forest is not None:
        for o in scene.forest.obstacles:
            ax.add_patch(Circle((o["center"][0], o["center"][2]), o["radius"], fill=False, ec="k", lw=0.8))
    if first_path is not None:
        ax.plot(first_path[:, 0], first_path[:, 2], "--", c="tab:blue", lw=1.5, label="premier chemin A*")
    ax.plot(traj[:, 0], traj[:, 2], "o-", c="tab:orange", ms=3, label="trajectoire parcourue")
    ax.plot(*goal_w[[0, 2]], "g*", ms=14, label="cible")
    ax.plot(*p0[[0, 2]], "b^", ms=9, label="départ")
    ax.set_xlabel("x (m)"); ax.set_ylabel("z (m)"); ax.set_aspect("equal")
    ax.set_title("Vue de dessus (gris inconnu, vert libre, rouge occupé ; cercles = troncs réels)")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    if args.save:
        import os
        os.makedirs(args.save, exist_ok=True)
        fig.savefig(os.path.join(args.save, "18_planner.png"), dpi=110)
        print(f"  Figure enregistrée dans {args.save}/")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
