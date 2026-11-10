"""test_full.py : pipeline complète en boucle, 8 images par seconde simulée, vidéo + temps.

Chaîne par image : capture stéréo -> coins (Harris) -> appariement (match_along_rows) ->
triangulation -> VoxelMap.integrate -> classify -> A* vers la cible.
Déplacement : téléportation le long du chemin A* (comme test18, sans MuJoCo, caméras
stabilisées), altitude comprise : le drone suit le chemin en 3D (--altitude-fixe pour garder
l'altitude comme test18). Entre deux images le drone avance de vitesse / img-par-s mètres et le temps
simulé avance de 1 / img-par-s secondes.

Sorties (dossier --save, par défaut sorties_test_full/) :
  test_full.mp4      une image vidéo par image traitée, jouée à img-par-s images/s (temps réel simulé) :
                     vue caméra gauche + points appariés (couleur = profondeur), vue de suivi,
                     vue de dessus (carte, chemin A*, trajectoire, troncs réels), temps de l'image
  temps.csv          temps de chaque étape pour chaque image (ms)
  temps.png          histogramme empilé des temps par image

    python test_full.py                          # forêt, graine 0, 1,5 m/s, 8 img/s
    python test_full.py --env murs --marge-carte 16 --duree 90
                                                 # murs : la carte doit couvrir toute l'enceinte
    python test_full.py --vitesse 1.0 --duree 40 --W 1280 --H 960
    python test_full.py --no-video               # seulement les temps

Les temps mesurés sont ceux du calcul (pas ceux du rendu de la vidéo). L'image 0 contient la
compilation numba si le cache n'existe pas encore ; elle est exclue des médianes.
"""

import argparse
import csv
import math
import os
import time

import cv2
import glfw
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Circle

import scene as sim
from test14 import make_hidden_context
from test18 import plan, advance_along, clearance_trees
from vision.capture import StereoCapture
from vision.corners import harris_function, detect_corners
from vision.matching import to_gray
from test14 import match_along_rows
from vision.triangulation import triangulate
from vision.voxels import VoxelMap

STEPS = ("capture", "coins", "appariement", "triangulation", "voxels", "classify", "A*")
PANEL_W, PANEL_H = 640, 480          # vue caméra et vue de suivi
TOP_W, TOP_H = 720, 2 * PANEL_H      # vue de dessus


# ---------------------------------------------------------------------------
# Pipeline vision (étapes chronométrées séparément)
# ---------------------------------------------------------------------------

def perceive(frame, args, t):
    """Coins -> appariement -> triangulation. Remplit t[1:4]. Retourne P_w, C, ptsL, Z, n_coins."""
    imgL, imgR = frame.images[0], frame.images[1]
    K = frame.K
    B = float(np.linalg.norm(frame.Cs[1] - frame.Cs[0]))
    R, C = frame.Rs[0], frame.Cs[0]

    t0 = time.perf_counter()
    gL, gR = to_gray(imgL), to_gray(imgR)
    kw = dict(rel_thresh=args.seuil, max_n=args.max_coins, grid_cells=tuple(args.grille))
    cL = detect_corners(harris_function(gL), **kw)
    cR = detect_corners(harris_function(gR), **kw) if args.matcher == "coins" else None
    t1 = time.perf_counter()
    if args.matcher == "lignes":
        ptsL, ptsR, _ = match_along_rows(imgL, imgR, cL, min_zncc=args.zncc, ratio=args.ratio,
                                         grayL=gL, grayR=gR)
    else:
        ptsL, ptsR, _ = match_along_rows(imgL, imgR, cL, min_zncc=args.zncc, ratio=args.ratio)
    t2 = time.perf_counter()
    P_w, ok = triangulate(ptsL[:, 0], ptsL[:, 1], ptsR[:, 0], ptsR[:, 1], K, B, R, C)
    t3 = time.perf_counter()
    t[1], t[2], t[3] = t1 - t0, t2 - t1, t3 - t2
    ptsL = ptsL[ok]
    disp = ptsL[:, 0] - ptsR[ok][:, 0]
    Z = K[0, 0] * B / np.maximum(disp, 1e-6)
    return P_w[ok], C, ptsL, Z, len(cL)


# ---------------------------------------------------------------------------
# Vidéo
# ---------------------------------------------------------------------------

def yaw_only_axes(drone):
    """Châssis à plat orienté selon le lacet (le drone est téléporté, MuJoCo n'est pas utilisé)."""
    def axes():
        fwd = drone.forward
        up = np.array([0.0, 1.0, 0.0])
        return fwd, -np.cross(fwd, up), up        # avant, gauche, haut
    return axes


class TopView:
    """Vue de dessus matplotlib, figure créée une fois et mise à jour à chaque image."""

    def __init__(self, vm, args, scene, goal_w, p0):
        self.vm, self.args = vm, args
        self.j0 = int(np.ceil((args.ymin - vm.origin[1]) / args.voxel))
        self.j1 = int(np.floor((args.ymax - vm.origin[1]) / args.voxel))
        sx, sz = vm.shape[0] * args.voxel, vm.shape[2] * args.voxel
        ext = [vm.origin[0], vm.origin[0] + sx, vm.origin[2], vm.origin[2] + sz]
        dpi = 100
        self.fig, self.ax = plt.subplots(figsize=(TOP_W / dpi, TOP_H / dpi), dpi=dpi)
        ax = self.ax
        cmap = ListedColormap(["#d9d9d9", "#8fd18f", "#d62728"])
        self.im = ax.imshow(np.zeros((vm.shape[2], vm.shape[0]), np.int8), origin="lower",
                            cmap=cmap, vmin=0, vmax=2, extent=ext, alpha=0.85, interpolation="nearest")
        if scene.forest is not None:
            for o in scene.forest.obstacles:
                ax.add_patch(Circle((o["center"][0], o["center"][2]), o["radius"], fill=False, ec="k", lw=0.6))
        self.path_line, = ax.plot([], [], "-", c="tab:blue", lw=1.5, label="chemin A* courant")
        self.traj_line, = ax.plot([], [], "-", c="tab:orange", lw=2, label="trajectoire")
        self.drone_pt, = ax.plot([], [], "o", c="tab:orange", mec="k", ms=7)
        self.fov_line, = ax.plot([], [], "-", c="k", lw=0.8)
        ax.plot(*goal_w[[0, 2]], "g*", ms=14, label="cible")
        ax.plot(*p0[[0, 2]], "b^", ms=8, label="départ")
        ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
        ax.set_aspect("equal"); ax.set_xlabel("x (m)"); ax.set_ylabel("z (m)")
        ax.legend(loc="upper right", fontsize=7)
        self.title = ax.set_title("Vue de dessus, t = 00.00 s\n(gris inconnu, vert libre, rouge occupé ; cercles = troncs réels)",
                                  fontsize=10)
        self.fig.tight_layout()

    def draw(self, traj, path_w, pos, yaw_deg, t_sim, half_fov_deg, rng):
        L = self.vm.map[:, self.j0:self.j1 + 1, :]
        state = np.zeros(L.shape, np.int8); state[L < 0] = 1; state[L > 0] = 2
        self.im.set_data(state.max(axis=1).T)
        tr = np.asarray(traj)
        self.traj_line.set_data(tr[:, 0], tr[:, 2])
        if path_w is not None:
            self.path_line.set_data(path_w[:, 0], path_w[:, 2])
        else:
            self.path_line.set_data([], [])
        self.drone_pt.set_data([pos[0]], [pos[2]])
        y = math.radians(yaw_deg); h = math.radians(half_fov_deg)
        a = (pos[0] + rng * math.cos(y - h), pos[2] + rng * math.sin(y - h))
        b = (pos[0] + rng * math.cos(y + h), pos[2] + rng * math.sin(y + h))
        self.fov_line.set_data([a[0], pos[0], b[0]], [a[1], pos[2], b[1]])
        self.title.set_text(f"Vue de dessus, t = {t_sim:5.2f} s\n(gris inconnu, vert libre, rouge occupé ; cercles = troncs réels)")
        self.fig.canvas.draw()
        img = np.asarray(self.fig.canvas.buffer_rgba())[:, :, :3]
        return cv2.resize(img, (TOP_W, TOP_H))


def camera_panel(img, ptsL, Z, z_max):
    """Image gauche réduite + points appariés colorés par profondeur (proche rouge, loin bleu)."""
    H, W = img.shape[:2]
    out = cv2.resize(img, (PANEL_W, PANEL_H), interpolation=cv2.INTER_AREA)
    if len(ptsL):
        s = PANEL_W / W
        q = np.clip(Z / z_max, 0, 1)
        cols = (plt.cm.jet_r(q)[:, :3] * 255).astype(np.uint8)
        for (u, v), c in zip(ptsL * s, cols):
            cv2.circle(out, (int(u), int(v)), 2, tuple(int(x) for x in c), -1)
    return out


def put_lines(img, lines, org=(8, 20), scale=0.5, col2=None):
    """Texte blanc cerclé de noir. Une ligne peut être (gauche, droite) : la droite est alignée à col2."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    for k, line in enumerate(lines):
        y = org[1] + int(22 * scale / 0.5) * k
        parts = [(line, org[0])] if isinstance(line, str) else []
        if not isinstance(line, str):
            (w, _), _ = cv2.getTextSize(line[1], font, scale, 1)
            parts = [(line[0], org[0]), (line[1], org[0] + col2 - w)]
        for txt, x in parts:
            cv2.putText(img, txt, (x, y), font, scale, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(img, txt, (x, y), font, scale, (255, 255, 255), 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", choices=sorted(sim.ENV_NAMES), default="foret")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--img-par-s", type=float, default=12.0, help="images traitées par seconde simulée")
    ap.add_argument("--vitesse", type=float, default=1.5, help="vitesse le long du chemin (m/s)")
    ap.add_argument("--duree", type=float, default=60.0, help="durée simulée max (s)")
    ap.add_argument("--W", type=int, default=865)
    ap.add_argument("--H", type=int, default=650)
    ap.add_argument("--max-coins", type=int, default=3000)
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
    ap.add_argument("--d-secu", type=float, default=None, help="distance de sécurité du coût de proximité (défaut : classify)")
    ap.add_argument("--seuil-free", type=float, default=None, help="seuil libre de classify (défaut : classify)")
    ap.add_argument("--seuil-occ", type=float, default=None, help="seuil occupé de classify (défaut : classify)")
    ap.add_argument("--w", type=float, default=2.0, help="poids de l'heuristique A*")
    ap.add_argument("--altitude-fixe", action="store_true",
                    help="garder l'altitude de départ (test18) au lieu de suivre le chemin A* en 3D")
    ap.add_argument("--save", default="sorties_test_full")
    ap.add_argument("--no-video", action="store_true")
    args = ap.parse_args()

    dt = 1.0 / args.img_par_s
    pas = args.vitesse * dt
    n_max = int(round(args.duree * args.img_par_s))
    os.makedirs(args.save, exist_ok=True)

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
    drone.body_axes = yaw_only_axes(drone)        # modèle 3D orienté selon le lacet dans la vue de suivi
    cams = drone.cameras[:2]
    cap = StereoCapture(ctx, scene, cams, W=args.W, H=args.H)

    p0 = np.array(pos, float)
    lo = np.minimum(p0, goal_w) - args.marge_carte
    hi = np.maximum(p0, goal_w) + args.marge_carte
    size = np.array([hi[0] - lo[0], 10.0, hi[2] - lo[2]])
    origin = np.array([lo[0], p0[1] - 2.0, lo[2]])
    vm = VoxelMap(origin, size, args.voxel)

    print(f"\n=== test_full : env={args.env} seed={args.seed}, {args.W}x{args.H}, "
          f"{args.img_par_s:g} img/s, {args.vitesse:g} m/s -> {pas:.2f} m entre deux images")
    print(f"  départ {np.round(p0, 1)}, cible {np.round(goal_w, 1)}, distance {np.linalg.norm(goal_w - p0):.1f} m")
    print(f"  carte {tuple(int(x) for x in vm.shape)} voxels de {args.voxel} m")

    writer = top = chase_fbo = None
    if not args.no_video:
        top = TopView(vm, args, scene, goal_w, p0)
        chase = sim.ChaseCamera(drone, distance=1.3, height=0.5)
        chase_fbo = ctx.framebuffer(ctx.renderbuffer((PANEL_W, PANEL_H)), ctx.depth_renderbuffer((PANEL_W, PANEL_H)))
        video_path = os.path.join(args.save, "test_full.mp4")
        writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), args.img_par_s,
                                 (PANEL_W + TOP_W, TOP_H))
        if not writer.isOpened():
            raise SystemExit(f"Impossible d'ouvrir {video_path} en écriture")
    half_fov_h = math.degrees(math.atan(math.tan(math.radians(sim.CAMERA_FOVY) / 2) * args.W / args.H))

    traj = [drone.position.copy()]
    times, rows = [], []
    min_clear = np.inf
    status = "durée max atteinte"
    t_video = 0.0
    for it in range(n_max):
        t_sim = it * dt
        t = np.zeros(len(STEPS))
        drone.update_cameras()

        t0 = time.perf_counter()
        frame = cap.grab(cams, mode, t_sim)
        t[0] = time.perf_counter() - t0
        P_w, C, ptsL, Z, n_coins = perceive(frame, args, t)
        t0 = time.perf_counter()
        vm.integrate(P_w, C, max_range=args.portee)
        t[4] = time.perf_counter() - t0
        path_w, info = plan(vm, drone.position, goal_w, args)
        t[5], t[6] = info["t_classify"], info["t_astar"]
        times.append(t)
        d_goal = float(np.linalg.norm(goal_w - drone.position))
        rows.append([it, f"{t_sim:.3f}", n_coins, len(ptsL), *[f"{1000 * x:.2f}" for x in t],
                     f"{1000 * t.sum():.2f}", f"{d_goal:.2f}"])
        print(f"  it {it:3d}  t {t_sim:5.1f} s  coins {n_coins:5d}  pts {len(ptsL):5d}  "
              f"total {1000 * t.sum():5.0f} ms  (A* {1000 * t[6]:4.0f})  dist cible {d_goal:5.1f} m"
              + ("" if path_w is not None else "  PAS DE CHEMIN"))

        # --- image vidéo (non chronométrée) ---
        if writer is not None:
            tv = time.perf_counter()
            camp = camera_panel(frame.images[0], ptsL, Z, args.portee)
            budget = 1000 * dt
            put_lines(camp, [f"camera gauche  t = {t_sim:5.2f} s  image {it}",
                             f"coins {n_coins}  apparies {len(ptsL)}  (couleur = profondeur 0-{args.portee:g} m)"])
            lines = [(n, f"{1000 * x:.0f} ms") for n, x in zip(STEPS, t)]
            lines.append(("TOTAL", f"{1000 * t.sum():.0f} ms"))
            lines.append(f"budget {budget:.0f} ms : " + ("OK" if 1000 * t.sum() <= budget else "TROP LENT"))
            chase_fbo.use()
            chase_fbo.clear(0.55, 0.7, 0.9)
            scene.render(chase, PANEL_W / PANEL_H, extra=drone.render_items() + scene.target_items(),
                         building_mode=mode)
            chp = np.frombuffer(chase_fbo.read(components=3), np.uint8).reshape(PANEL_H, PANEL_W, 3)[::-1].copy()
            ctx.screen.use()
            put_lines(chp, ["vue de suivi"])
            put_lines(chp, lines, org=(8, PANEL_H - 22 * len(lines)), scale=0.5, col2=190)
            topi = top.draw(traj, path_w, drone.position, drone.yaw, t_sim, half_fov_h, args.portee)
            img = np.zeros((TOP_H, PANEL_W + TOP_W, 3), np.uint8)
            img[:PANEL_H, :PANEL_W] = camp
            img[PANEL_H:2 * PANEL_H, :PANEL_W] = chp
            img[:, PANEL_W:] = topi
            writer.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            t_video += time.perf_counter() - tv

        if path_w is None:
            status = "pas de chemin"
            break
        if d_goal < 1.0:
            status = "cible atteinte"
            break
        nxt = advance_along(path_w, drone.position, pas)
        if args.altitude_fixe:
            nxt[1] = drone.position[1]                  # comme test18 (bloque si le chemin monte à la verticale)
        d = nxt - drone.position
        if math.hypot(d[0], d[2]) > 1e-6:
            drone.yaw = math.degrees(math.atan2(d[2], d[0]))
        if scene.forest is not None:                    # vérité terrain le long du segment parcouru
            for s in np.linspace(0, 1, 6)[1:]:
                min_clear = min(min_clear, clearance_trees(scene, drone.position + s * d))
        drone.position = nxt
        traj.append(nxt.copy())

    if writer is not None:
        writer.release()
    glfw.destroy_window(window)
    glfw.terminate()

    # --- bilan ---------------------------------------------------------------
    times = np.array(times)
    traj = np.array(traj)
    n = len(times)
    T = times[1:] if n > 1 else times
    budget = 1000 * dt
    print(f"\n  Bilan : {status} après {n} images ({(n - 1) * dt:.1f} s simulées)")
    print(f"  distance parcourue {np.sum(np.linalg.norm(np.diff(traj, axis=0), axis=1)):.1f} m, "
          f"distance finale à la cible {np.linalg.norm(goal_w - traj[-1]):.1f} m")
    if np.isfinite(min_clear):
        print(f"  distance minimale drone - tronc (vérité terrain) : {100 * min_clear:.0f} cm "
              f"({'COLLISION' if min_clear < 0.0 else 'ok'})")
    print(f"\n  Temps par image (ms), image 0 exclue :   médiane   moyenne       max")
    for i, name in enumerate(STEPS):
        print(f"    {name:<14s} {1000 * np.median(T[:, i]):22.1f} {1000 * T[:, i].mean():9.1f} {1000 * T[:, i].max():9.1f}")
    tot = T.sum(axis=1)
    print(f"    {'TOTAL':<14s} {1000 * np.median(tot):22.1f} {1000 * tot.mean():9.1f} {1000 * tot.max():9.1f}")
    print(f"  budget à {args.img_par_s:g} img/s : {budget:.0f} ms -> {100 * (tot <= budget / 1000).mean():.0f} % "
          f"des images dans le budget, {1 / np.median(tot):.2f} img/s possibles (médiane)")
    print(f"  image 0 (compilation numba éventuelle) : {1000 * times[0].sum():.0f} ms")

    with open(os.path.join(args.save, "temps.csv"), "w", newline="") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["image", "t_sim_s", "coins", "apparies", *[f"{s}_ms" for s in STEPS], "total_ms", "dist_cible_m"])
        w.writerows(rows)

    fig, ax = plt.subplots(figsize=(12, 5))
    bottom = np.zeros(n)
    cols = plt.cm.tab10.colors
    for i, name in enumerate(STEPS):
        ax.bar(np.arange(n), 1000 * times[:, i], bottom=bottom, label=name, color=cols[i])
        bottom += 1000 * times[:, i]
    ax.axhline(budget, c="k", ls="--", lw=1, label=f"budget {budget:.0f} ms ({args.img_par_s:g} img/s)")
    ax.set_ylim(0, max(1.3 * budget, 1.1 * 1000 * np.percentile(times.sum(axis=1), 98)))
    ax.set_xlabel("image"); ax.set_ylabel("ms"); ax.set_title(f"Temps de calcul par image ({args.env}, {args.W}x{args.H})")
    ax.legend(fontsize=8, ncol=4)
    fig.tight_layout()
    fig.savefig(os.path.join(args.save, "temps.png"), dpi=110)
    print(f"\n  Sorties dans {args.save}/ : "
          + ("test_full.mp4, " if not args.no_video else "") + "temps.csv, temps.png"
          + (f"  (rendu vidéo {t_video:.1f} s, non compté)" if not args.no_video else ""))


if __name__ == "__main__":
    main()
