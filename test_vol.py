"""test_vol.py : pipeline complète en vol MuJoCo avec les 12 caméras (6 paires stéréo à 360°).

Chaîne par image traitée (img-par-s fois par seconde simulée, 8 par défaut) :
  capture des 12 caméras -> pour chaque paire : coins (Harris) -> appariement le long des lignes
  (match_along_rows_ref) -> triangulation -> VoxelMap.integrate_multi (toutes les paires, une
  seule mise à jour par voxel) -> classify -> A* vers la cible.
Entre deux images, la physique MuJoCo avance par pas de --dt-phys et vision/follow.py transforme
le dernier chemin A* en commande (vitesse + vitesse de lacet). Avec la vue à 360°, follow est
appelé avec omni=True : le drone n'a pas besoin de regarder là où il va (--face pour revenir au
comportement d'avant).

Simplification : le chemin calculé sur une image est appliqué tout de suite (on ne simule pas
la latence du calcul).

Sorties (dossier --save, par défaut sorties_test_vol/<env>/ à côté du script) :
  test_vol.mp4   vidéo à --fps images/s en temps simulé réel :
                 en haut à gauche, les 6 images gauches + points appariés (couleur = profondeur),
                 disposées comme autour du drone (avant au centre en haut, arrière au centre en bas) ;
                 en bas à gauche, vue de suivi en continu ;
                 à droite, vue de dessus (couche de voxels à l'altitude du drone, chemin A*, trajectoire)
  temps.csv      temps de chaque étape pour chaque image traitée (ms, cumulés sur les 6 paires)
  vol.csv        état du drone à chaque image vidéo (position, vitesse, commande, contact)

    python test_vol.py                                   # forêt, graine 0, 8 img/s
    python test_vol.py --env murs --duree 90
    python test_vol.py --v-max 2.5 --cams-stab           # plus rapide, caméras stabilisées
"""

import argparse
import csv
import inspect
import math
import os
import time
import unicodedata

import cv2
import glfw
import moderngl
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Circle

import scene as sim
from vision.capture import StereoCapture
from vision.follow import follow
from vision.pipeline import perceive_all
from vision.planner import classify, astar
from vision.voxels import VoxelMap

# Valeurs par défaut lues directement dans vision/follow.py et vision/planner.py
_defaults = lambda f: {k: p.default for k, p in inspect.signature(f).parameters.items()
                       if p.default is not inspect.Parameter.empty}
FOLLOW_DEF = _defaults(follow)
CLASSIFY_DEF = _defaults(classify)

STEPS = ("capture", "coins", "appariement", "triangulation", "voxels", "classify", "A*")
TILE_W, TILE_H = 320, 240                    # une image gauche dans la mosaïque
MOS_W, MOS_H = 3 * TILE_W, 2 * TILE_H        # mosaïque des 6 paires
CHASE_W, CHASE_H = MOS_W, MOS_H              # vue de suivi
TOP_W, TOP_H = 720, MOS_H + CHASE_H          # vue de dessus
# Position des paires dans la mosaïque (indices dans scene.STEREO_PAIRS) : rangée du haut vers l'avant
# (avant-gauche, avant, avant-droite), rangée du bas vers l'arrière (arrière-gauche, arrière, arrière-droite).
MOSAIC = [[5, 0, 1], [4, 3, 2]]


# ---------------------------------------------------------------------------
# Contexte, planification, vérité terrain
# ---------------------------------------------------------------------------

def make_hidden_context():
    """Contexte OpenGL 3.3 dans une fenêtre glfw cachée (comme scene.py, sans affichage)."""
    if not glfw.init():
        raise SystemExit("Impossible d'initialiser GLFW")
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, True)
    glfw.window_hint(glfw.VISIBLE, False)
    window = glfw.create_window(64, 64, "test_vol", None, None)
    if not window:
        glfw.terminate()
        raise SystemExit("Impossible de créer la fenêtre cachée")
    glfw.make_context_current(window)
    return moderngl.create_context(), window


def plan(vm, C, goal_w, args):
    """classify + A* de C vers goal_w. Retourne (chemin monde (N, 3) ou None, infos)."""
    t0 = time.perf_counter()
    blocked, cost_extra, j_min, libre, dist = classify(vm, args.ymin, args.ymax, args.rayon + args.marge,
                                                       k_unk=args.k_unk, k_prox=args.k_prox, details=True,
                                                       r_v=args.rayon_v + args.marge)
    t1 = time.perf_counter()
    nx, nj, nz = blocked.shape

    def to_grid(p):
        i, j, k = vm.world_to_idx(np.asarray(p, float))
        return (int(np.clip(i, 0, nx - 1)), int(np.clip(j - j_min, 0, nj - 1)), int(np.clip(k, 0, nz - 1)))

    s, g = to_grid(C), to_grid(goal_w)
    # départ et voisinage immédiat libres, cible libre
    blocked[max(s[0] - 1, 0):s[0] + 2, max(s[1] - 1, 0):s[1] + 2, max(s[2] - 1, 0):s[2] + 2] = False
    blocked[g] = False
    path = astar(blocked, cost_extra, s, g, vm.voxel_size, w=args.w)
    t2 = time.perf_counter()
    info = dict(t_classify=t1 - t0, t_astar=t2 - t1)
    if path is None:
        return None, info
    idx = np.array(path)
    ii = tuple(idx.T)
    info["libre"] = libre[ii]               # (N,) pour follow
    info["dist"] = dist[ii]                 # (N,) distance aux obstacles (m)
    idx[:, 1] += j_min
    return vm.idx_to_world(idx), info


def clearance_trees(scene, pos):
    """Distance horizontale du drone au bord du tronc le plus proche (m), vérité terrain."""
    forest = scene.forest
    if forest is None or not forest.obstacles:
        return np.inf
    c = np.array([[o["center"][0], o["center"][2]] for o in forest.obstacles])
    r = np.array([o["radius"] for o in forest.obstacles])
    return float((np.hypot(c[:, 0] - pos[0], c[:, 1] - pos[2]) - r).min())


# ---------------------------------------------------------------------------
# Vidéo
# ---------------------------------------------------------------------------

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


def mosaic_panel(frame, res, z_max):
    """Les 6 images gauches réduites + points appariés colorés par profondeur (proche rouge, loin bleu)."""
    out = np.zeros((MOS_H, MOS_W, 3), np.uint8)
    for r, row in enumerate(MOSAIC):
        for c, k in enumerate(row):
            img = frame.images[2 * k]
            tile = cv2.resize(img, (TILE_W, TILE_H), interpolation=cv2.INTER_AREA)
            ptsL, Z = res[k]["ptsL"], res[k]["Z"]
            if len(ptsL):
                s = TILE_W / img.shape[1]
                cols = (plt.cm.jet_r(np.clip(Z / z_max, 0, 1))[:, :3] * 255).astype(np.uint8)
                for (u, v), col in zip(ptsL * s, cols):
                    cv2.circle(tile, (int(u), int(v)), 1, tuple(int(x) for x in col), -1)
            name = unicodedata.normalize("NFKD", sim.STEREO_PAIRS[k][0]).encode("ascii", "ignore").decode()   # cv2 : ASCII seul
            put_lines(tile, [f"{name}  {len(ptsL)} pts"], scale=0.45)
            cv2.rectangle(tile, (0, 0), (TILE_W - 1, TILE_H - 1), (0, 0, 0), 1)
            out[r * TILE_H:(r + 1) * TILE_H, c * TILE_W:(c + 1) * TILE_W] = tile
    return out


class TopView:
    """Vue de dessus = couche de voxels à l'altitude du drone, avec les seuils de classify.
    Cercle noir : portée de la stéréo (vue à 360°) ; trait : avant du drone."""

    def __init__(self, vm, scene, goal_w, p0):
        self.vm = vm
        sx, sz = vm.shape[0] * vm.voxel_size, vm.shape[2] * vm.voxel_size
        ext = [vm.origin[0], vm.origin[0] + sx, vm.origin[2], vm.origin[2] + sz]
        dpi = 100
        self.fig, ax = plt.subplots(figsize=(TOP_W / dpi, TOP_H / dpi), dpi=dpi)
        cmap = ListedColormap(["#d9d9d9", "#8fd18f", "#d62728"])
        self.im = ax.imshow(np.zeros((vm.shape[2], vm.shape[0]), np.int8), origin="lower",
                            cmap=cmap, vmin=0, vmax=2, extent=ext, alpha=0.85, interpolation="nearest")
        if scene.forest is not None:
            for o in scene.forest.obstacles:
                ax.add_patch(Circle((o["center"][0], o["center"][2]), o["radius"], fill=False, ec="k", lw=0.6))
        self.path_line, = ax.plot([], [], "-", c="tab:blue", lw=1.5, label="chemin A* courant")
        self.traj_line, = ax.plot([], [], "-", c="tab:orange", lw=2, label="trajectoire")
        self.drone_pt, = ax.plot([], [], "o", c="tab:orange", mec="k", ms=7)
        self.head_line, = ax.plot([], [], "-", c="k", lw=1.5)
        self.range_circle = Circle((0, 0), 1.0, fill=False, ec="k", lw=0.6, ls="--")
        ax.add_patch(self.range_circle)
        ax.plot(*goal_w[[0, 2]], "g*", ms=14, label="cible")
        ax.plot(*p0[[0, 2]], "b^", ms=8, label="départ")
        ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
        ax.set_aspect("equal"); ax.set_xlabel("x (m)"); ax.set_ylabel("z (m)")
        ax.legend(loc="upper right", fontsize=7)
        self.title = ax.set_title("", fontsize=10)
        self.fig.tight_layout()

    def draw(self, traj, path_w, pos, yaw_deg, t_sim, rng):
        vm = self.vm
        j = int(np.clip(vm.world_to_idx(np.asarray(pos, float))[1], 0, vm.shape[1] - 1))
        L = vm.map[:, j, :]
        state = np.zeros(L.shape, np.int8)
        state[L < -CLASSIFY_DEF["seuil_free"]] = 1
        state[L > CLASSIFY_DEF["seuil_occ"]] = 2
        self.im.set_data(state.T)
        tr = np.asarray(traj)
        self.traj_line.set_data(tr[:, 0], tr[:, 2])
        self.path_line.set_data(*((path_w[:, 0], path_w[:, 2]) if path_w is not None else ([], [])))
        self.drone_pt.set_data([pos[0]], [pos[2]])
        y = math.radians(yaw_deg)
        self.head_line.set_data([pos[0], pos[0] + 1.5 * math.cos(y)], [pos[2], pos[2] + 1.5 * math.sin(y)])
        self.range_circle.center, self.range_circle.radius = (pos[0], pos[2]), rng
        self.title.set_text(f"Vue de dessus, t = {t_sim:5.2f} s, couche y = {pos[1]:.2f} m (altitude du drone)\n"
                            "(gris inconnu, vert libre, rouge occupé, seuils de classify)")
        self.fig.canvas.draw()
        img = np.asarray(self.fig.canvas.buffer_rgba())[:, :, :3]
        return cv2.resize(img, (TOP_W, TOP_H))


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", choices=sorted(sim.ENV_NAMES), default="foret")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--img-par-s", type=float, default=8.0, help="images traitées par seconde simulée")
    ap.add_argument("--dt-phys", type=float, default=0.01, help="pas de commande (s) ; MuJoCo sous-échantillonne à 2 ms")
    ap.add_argument("--fps", type=float, default=30.0, help="images par seconde de la vidéo")
    ap.add_argument("--duree", type=float, default=60.0, help="durée simulée max (s)")
    ap.add_argument("--max-contacts", type=int, default=5, help="arrêt après ce nombre de contacts")
    ap.add_argument("--cams-stab", action="store_true", help="caméras stabilisées (lacet seul) au lieu de fixées au châssis")
    ap.add_argument("--face", action="store_true",
                    help="follow sans omni : vitesse réduite quand le drone ne regarde pas où il va")
    # follow : None = valeur par défaut de vision/follow.py
    ap.add_argument("--v-max", type=float, default=None)
    ap.add_argument("--lookahead", type=float, default=None)
    ap.add_argument("--a-frein", type=float, default=None)
    ap.add_argument("--v-min", type=float, default=None)
    # vision / planification (par paire stéréo)
    ap.add_argument("--W", type=int, default=865)
    ap.add_argument("--H", type=int, default=650)
    ap.add_argument("--max-coins", type=int, default=5000, help="coins max par image gauche")
    ap.add_argument("--seuil", type=float, default=1e-6)
    ap.add_argument("--grille", type=int, nargs=2, default=(12, 16))
    ap.add_argument("--zncc", type=float, default=0.8)
    ap.add_argument("--ratio", type=float, default=0.6)
    ap.add_argument("--voxel", type=float, default=0.25)
    ap.add_argument("--portee", type=float, default=15.0)
    ap.add_argument("--marge-carte", type=float, default=16.0, help="marge autour du segment départ-cible (m)")
    ap.add_argument("--ymin", type=float, default=0.6)
    ap.add_argument("--ymax", type=float, default=4.4, help="altitude max de vol (m), plafond de la mission")
    ap.add_argument("--rayon", type=float, default=0.14, help="rayon horizontal du drone (m), englobe le châssis à tout lacet")
    ap.add_argument("--rayon-v", type=float, default=0.035, help="demi-hauteur du drone (m) ; gonflement cylindrique")
    ap.add_argument("--marge", type=float, default=0.08, help="marge dure autour du drone (m) ; l éloignement vient du coût d_secu de classify")
    ap.add_argument("--k-unk", type=float, default=CLASSIFY_DEF["k_unk"], help="défaut : celui de classify")
    ap.add_argument("--k-prox", type=float, default=CLASSIFY_DEF["k_prox"], help="défaut : celui de classify")
    ap.add_argument("--w", type=float, default=2.0)
    ap.add_argument("--save", default=None,
                    help="dossier de sortie (défaut : sorties_test_vol/<env> à côté de test_vol.py)")
    ap.add_argument("--no-video", action="store_true")
    args = ap.parse_args()
    # seuls les paramètres donnés en ligne de commande remplacent les défauts de follow
    follow_kw = {k: v for k, v in dict(v_max=args.v_max, lookahead=args.lookahead,
                                       a_frein=args.a_frein, v_min=args.v_min).items() if v is not None}
    follow_kw["omni"] = not args.face
    follow_eff = {**FOLLOW_DEF, **follow_kw}
    corner_kw = dict(rel_thresh=args.seuil, max_n=args.max_coins, grid_cells=tuple(args.grille))
    match_kw = dict(min_zncc=args.zncc, ratio=args.ratio)

    dt_img, dt_vid, dt = 1.0 / args.img_par_s, 1.0 / args.fps, args.dt_phys
    if args.save is None:
        args.save = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sorties_test_vol", args.env)
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
    drone.cams_on_chassis = not args.cams_stab
    cams = drone.cameras                         # 12 caméras : paire k = (2k gauche, 2k+1 droite)
    cap = StereoCapture(ctx, scene, cams, W=args.W, H=args.H)

    p0 = np.array(pos, float)
    lo = np.minimum(p0, goal_w) - args.marge_carte
    hi = np.maximum(p0, goal_w) + args.marge_carte
    vm = VoxelMap(np.array([lo[0], p0[1] - 2.0, lo[2]]), np.array([hi[0] - lo[0], 10.0, hi[2] - lo[2]]), args.voxel)

    print(f"\n=== test_vol : env={args.env} seed={args.seed}, {len(cams)} caméras ({len(cams) // 2} paires), "
          f"{args.W}x{args.H}, {args.img_par_s:g} img/s, v_max {follow_eff['v_max']:g} m/s, "
          f"caméras {'stabilisées' if args.cams_stab else 'fixées au châssis'}")
    print("  follow : " + ", ".join(f"{k}={v:g}" for k, v in follow_eff.items()))
    print(f"  départ {np.round(p0, 1)}, cible {np.round(goal_w, 1)}, distance {np.linalg.norm(goal_w - p0):.1f} m")

    writer = top = chase_fbo = None
    if not args.no_video:
        top = TopView(vm, scene, goal_w, p0)
        chase = sim.ChaseCamera(drone, distance=1.3, height=0.5)
        chase_fbo = ctx.framebuffer(ctx.renderbuffer((CHASE_W, CHASE_H)), ctx.depth_renderbuffer((CHASE_W, CHASE_H)))
        video_path = os.path.join(args.save, "test_vol.mp4")
        writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (MOS_W + TOP_W, TOP_H))
        if not writer.isOpened():
            raise SystemExit(f"Impossible d'ouvrir {video_path} en écriture")

    path_w, libre, dist_obs = None, None, None
    mos, t_img = None, np.zeros(len(STEPS))
    traj = [drone.position.copy()]
    times, rows, vol_rows = [], [], []
    n_contacts, in_contact, min_clear = 0, False, np.inf
    cmd_v, cmd_yaw = np.zeros(3), 0.0
    status = "durée max atteinte"
    next_img, next_vid, t_sim, it = 0.0, 0.0, 0.0, 0
    t_video = 0.0
    t_wall = time.perf_counter()

    while t_sim <= args.duree:
        # --- perception + planification, img-par-s fois par seconde ---
        if t_sim >= next_img - 1e-9:
            next_img += dt_img
            t = np.zeros(len(STEPS))
            t0 = time.perf_counter()
            scene.target_items = lambda: []          # sphère verte invisible pour les caméras du drone
            frame = cap.grab(cams, mode, t_sim)
            del scene.target_items                   # elle reste visible dans la vue de suivi
            t[0] = time.perf_counter() - t0
            res, t[1:4] = perceive_all(frame, corner_kw, match_kw)
            t0 = time.perf_counter()
            vm.integrate_multi([r["P_w"] for r in res], [r["C"] for r in res], max_range=args.portee)
            t[4] = time.perf_counter() - t0
            path_w, info = plan(vm, drone.position, goal_w, args)
            libre, dist_obs = info.get("libre"), info.get("dist")
            t[5], t[6] = info["t_classify"], info["t_astar"]
            times.append(t)
            t_img = t
            n_coins = sum(r["n_coins"] for r in res)
            n_pts = sum(len(r["ptsL"]) for r in res)
            d_goal = float(np.linalg.norm(goal_w - drone.position))
            rows.append([it, f"{t_sim:.3f}", n_coins, n_pts, *[f"{1000 * x:.2f}" for x in t],
                         f"{1000 * t.sum():.2f}", f"{d_goal:.2f}"])
            print(f"  it {it:3d}  t {t_sim:5.1f} s  pts {n_pts:6d}  total {1000 * t.sum():5.0f} ms  "
                  f"v {np.linalg.norm(drone.velocity):4.2f} m/s  dist cible {d_goal:5.1f} m  contacts {n_contacts}"
                  + ("" if path_w is not None else "  PAS DE CHEMIN"))
            if writer is not None:
                mos = mosaic_panel(frame, res, args.portee)
                put_lines(mos, [f"image traitee {it}  t = {t_sim:5.2f} s  apparies {n_pts}"],
                          org=(TILE_W + 8, MOS_H - 10), scale=0.45)
            it += 1
            if d_goal < 1.0:
                status = "cible atteinte"
                break
            if path_w is None:                       # A* ne trouve pas de chemin : arrêt, sorties enregistrées
                status = "pas de chemin"
                break

        # --- image vidéo ---
        if writer is not None and t_sim >= next_vid - 1e-9:
            next_vid += dt_vid
            tv = time.perf_counter()
            chase_fbo.use()
            chase_fbo.clear(0.55, 0.7, 0.9)
            scene.render(chase, CHASE_W / CHASE_H, extra=drone.render_items() + scene.target_items(), building_mode=mode)
            chp = np.frombuffer(chase_fbo.read(components=3), np.uint8).reshape(CHASE_H, CHASE_W, 3)[::-1].copy()
            ctx.screen.use()
            v = drone.velocity
            put_lines(chp, [f"vue de suivi  t = {t_sim:5.2f} s",
                            f"vitesse {np.linalg.norm(v):4.2f} m/s  (commande {np.linalg.norm(cmd_v):4.2f})",
                            f"lacet {drone.yaw:5.1f} deg  tangage {drone.pitch:5.1f}  roulis {drone.roll:5.1f}",
                            f"contacts {n_contacts}" + ("  CONTACT" if drone.contact else "")])
            lines = [(n, f"{1000 * x:.0f} ms") for n, x in zip(STEPS, t_img)] + [("TOTAL", f"{1000 * t_img.sum():.0f} ms")]
            put_lines(chp, lines, org=(8, CHASE_H - 22 * len(lines)), scale=0.5, col2=190)
            topi = top.draw(traj, path_w, drone.position, drone.yaw, t_sim, args.portee)
            img = np.zeros((TOP_H, MOS_W + TOP_W, 3), np.uint8)
            img[:MOS_H, :MOS_W] = mos
            img[MOS_H:, :MOS_W] = chp
            img[:, MOS_W:] = topi
            writer.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            t_video += time.perf_counter() - tv
            vol_rows.append([f"{t_sim:.3f}", *np.round(drone.position, 3), *np.round(drone.velocity, 3),
                             *np.round(cmd_v, 3), round(cmd_yaw, 2), int(drone.contact)])

        # --- commande + physique ---
        cmd_v, cmd_yaw = follow(path_w, libre, dist_obs, drone.position, drone.yaw, goal_w, **follow_kw)
        hit = drone.update(dt, cmd_velocity=cmd_v, cmd_yaw_rate=cmd_yaw)
        if hit and not in_contact:
            n_contacts += 1
            if n_contacts >= args.max_contacts:      # arrêt, vidéo et sorties enregistrées
                status = f"{n_contacts} contacts"
                break
        in_contact = hit
        if scene.forest is not None:
            min_clear = min(min_clear, clearance_trees(scene, drone.position) - args.rayon)
        traj.append(drone.position.copy())
        t_sim += dt

    if writer is not None:
        writer.release()
    glfw.destroy_window(window)
    glfw.terminate()

    # --- bilan ---------------------------------------------------------------
    times, traj = np.array(times), np.array(traj)
    T = times[1:] if len(times) > 1 else times
    tot = T.sum(axis=1)
    print(f"\n  Bilan : {status} à t = {t_sim:.1f} s simulées ({len(times)} images traitées)")
    print(f"  distance parcourue {np.sum(np.linalg.norm(np.diff(traj, axis=0), axis=1)):.1f} m, "
          f"distance finale à la cible {np.linalg.norm(goal_w - traj[-1]):.1f} m")
    print(f"  contacts avec un obstacle : {n_contacts}")
    if np.isfinite(min_clear):
        print(f"  distance minimale bord du drone (rayon {args.rayon} m) - tronc : {100 * min_clear:.0f} cm")
    print(f"  temps de calcul par image : médiane {1000 * np.median(tot):.0f} ms, max {1000 * tot.max():.0f} ms "
          f"(budget {1000 * dt_img:.0f} ms)")
    tot_v = T[:, 1:].sum(axis=1)             # sans la capture (rendu simulé, remplacé par les caméras en vrai)
    print(f"  hors capture : médiane {1000 * np.median(tot_v):.0f} ms, max {1000 * tot_v.max():.0f} ms, "
          f"{100 * (tot_v <= dt_img).mean():.0f} % des images dans le budget")
    print(f"  temps réel écoulé {time.perf_counter() - t_wall:.0f} s"
          + (f" dont vidéo {t_video:.0f} s" if writer is not None else ""))

    with open(os.path.join(args.save, "temps.csv"), "w", newline="") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["image", "t_sim_s", "coins", "apparies", *[f"{s}_ms" for s in STEPS], "total_ms", "dist_cible_m"])
        w.writerows(rows)
    with open(os.path.join(args.save, "vol.csv"), "w", newline="") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["t_s", "x", "y", "z", "vx", "vy", "vz", "cmd_avant", "cmd_droite", "cmd_haut", "cmd_lacet_deg_s", "contact"])
        w.writerows(vol_rows)
    print(f"  Sorties dans {args.save}/")


if __name__ == "__main__":
    main()
