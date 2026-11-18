"""test_vslam.py : test de vol global (comme test_vol.py) où la pose vient d'une VIO style VINS-Fusion.

Différence avec test_vol.py : le drone ne connaît plus sa pose vraie. À chaque image traitée,
vision/vslam.py (suivi KLT, stéréo, préintégration IMU, fenêtre glissante) estime la pose ; entre
deux images, l'IMU simulé (vision/imu.py, parfait par défaut) propage l'état à chaque pas de
commande. Cette pose estimée sert partout : triangulation -> carte de voxels, départ de l'A*,
suivi du chemin (follow). La pose vraie (MuJoCo) ne sert qu'à initialiser la VIO et à mesurer son
erreur (--pose vraie : la VIO tourne mais le drone vole avec la pose vraie, pour comparer).

Chaîne par image : capture -> coins (Harris) -> VIO (KLT + stéréo + optimisation) -> appariement
-> triangulation (pose VIO) -> voxels -> classify -> A*.

Sorties (dossier --save, par défaut sorties_test_vslam/<env>/ à côté du script) :
  test_vslam.mp4  comme test_vol.mp4, avec la trajectoire estimée (violet) et l'erreur de la VIO
  temps.csv       temps de chaque étape par image traitée (ms)
  vol.csv         état vrai et estimé du drone à chaque image vidéo
  vslam.csv       par image traitée : pose vraie, pose estimée, erreurs, points suivis, image clé
  erreur_vio.png  erreur de position et de lacet au cours du vol
  recap_temps.txt récap des temps

    python test_vslam.py                                 # forêt, graine 0
    python test_vslam.py --env murs --duree 90
    python test_vslam.py --klt perso                     # ton KLT (vision/klt.py) au lieu du corrigé (klt_ref.py)
    python test_vslam.py --pose vraie                    # vol avec la pose vraie, VIO en observateur
"""

import argparse
import csv
import inspect
import math
import os
import time

import cv2
import glfw
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import scene as sim
from test14 import make_hidden_context, match_along_rows
from test18 import clearance_trees
from vision.planner import classify, classify_fenetre, astar
from test_full import PANEL_W, PANEL_H, TOP_W, TOP_H, camera_panel, put_lines
from test_vol import TopViewAltitude, FOLLOW_DEF, CLASSIFY_DEF
from vision.capture import StereoCapture
from vision.corners import harris_function, detect_corners
from vision.follow import follow
from vision.imu import SimIMU, so3_log
from vision import klt as klt_perso
from vision.matching import to_gray
from vision.triangulation import triangulate
from vision.voxels import VoxelMap
from vision.vslam import FrontEnd, VIO

STEPS = ("capture", "coins", "VIO", "appariement", "triangulation", "voxels", "classify", "A*")


class TopViewVIO(TopViewAltitude):
    """Vue de dessus de test_vol + trajectoire estimée par la VIO."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.est_line, = self.ax.plot([], [], "--", c="tab:purple", lw=1.5, label="trajectoire VIO")
        self.est_pt, = self.ax.plot([], [], "o", c="tab:purple", mec="k", ms=6)
        self.ax.legend(loc="upper right", fontsize=7)

    def draw_vio(self, traj, traj_est, path_w, pos, pos_est, yaw_est, t_sim, half_fov_deg, rng, err):
        te = np.asarray(traj_est)
        self.est_line.set_data(te[:, 0], te[:, 2])
        self.est_pt.set_data([pos_est[0]], [pos_est[2]])
        # le cône de vue et la couche affichée suivent la pose estimée (celle que le drone croit avoir)
        img = super().draw(traj, path_w, pos_est, yaw_est, t_sim, half_fov_deg, rng)
        self.drone_pt.set_data([pos[0]], [pos[2]])
        return img, err


def perceive_vio(frame, args, t, vio, use_true):
    """Coins -> VIO -> appariement -> triangulation avec la pose VIO. Remplit t[1:5]."""
    imgL, imgR = frame.images[0], frame.images[1]
    K = frame.K
    B = float(np.linalg.norm(frame.Cs[1] - frame.Cs[0]))
    t0 = time.perf_counter()
    gL, gR = to_gray(imgL), to_gray(imgR)
    RhL = harris_function(gL, version=args.harris)          # réponse de Harris calculée une seule fois, réutilisée par le front-end VIO
    cL = detect_corners(RhL, rel_thresh=args.seuil, max_n=args.max_coins, grid_cells=tuple(args.grille))
    t1 = time.perf_counter()
    vio.add_frame(frame.t, gL, gR, corners=cL, harris=RhL)
    R, C = (frame.Rs[0], frame.Cs[0]) if use_true else vio.cam_pose()
    t2 = time.perf_counter()
    ptsL, ptsR, _ = match_along_rows(imgL, imgR, cL, min_zncc=args.zncc, ratio=args.ratio, grayL=gL, grayR=gR,
                                     norms=vio.front.norms if args.normes == "partage" else None)
    t3 = time.perf_counter()
    P_w, ok = triangulate(ptsL[:, 0], ptsL[:, 1], ptsR[:, 0], ptsR[:, 1], K, B, R, C)
    t4 = time.perf_counter()
    t[1], t[2], t[3], t[4] = t1 - t0, t2 - t1, t3 - t2, t4 - t3
    ptsL = ptsL[ok]
    disp = ptsL[:, 0] - ptsR[ok][:, 0]
    Z = K[0, 0] * B / np.maximum(disp, 1e-6)
    return P_w[ok], C, ptsL, Z, len(cL)


def make_klt(args):
    if args.klt == "opencv":
        return klt_perso.KLTCV(max_res=args.klt_res)
    return klt_perso.KLT(max_res=args.klt_res)      # ref / perso : notre KLT (vision/klt.py)

def plan(vm, C, goal_w, args):
    """Copie de test18.plan, avec classify_fenetre (distances calculées dans une fenêtre de rayon
    --rayon-fenetre autour de la position estimée C). --rayon-fenetre 0 : fenêtre = toute la carte."""
    t0 = time.perf_counter()
    seuils = {k: getattr(args, k) for k in ("d_secu", "seuil_free", "seuil_occ") if getattr(args, k, None) is not None}
    if args.classify_classique:
        blocked, cost_extra, j_min, libre, dist = classify(
            vm, args.ymin, args.ymax, args.rayon + args.marge,
            k_unk=args.k_unk, k_prox=args.k_prox, details=True, **seuils)
    else:
        rayon_f = args.rayon_fenetre if args.rayon_fenetre > 0 else 1e6
        blocked, cost_extra, j_min, libre, dist = classify_fenetre(
            vm, np.asarray(C, float), args.ymin, args.ymax, args.rayon + args.marge, rayon=rayon_f,
            k_unk=args.k_unk, k_prox=args.k_prox, details=True, **seuils)
    t1 = time.perf_counter()
    nx, nj, nz = blocked.shape

    def to_grid(p):
        i, j, k = vm.world_to_idx(np.asarray(p, float))
        j -= j_min
        return (int(np.clip(i, 0, nx - 1)), int(np.clip(j, 0, nj - 1)), int(np.clip(k, 0, nz - 1)))

    s, g = to_grid(C), to_grid(goal_w)
    grid_follow = (blocked.copy(), j_min, vm)
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
    ii = tuple(idx.T)
    info["libre"] = libre[ii]
    info["dist"] = dist[ii]
    idx[:, 1] += j_min
    return vm.idx_to_world(idx), info


def vio_yaw_deg(vio):
    """Lacet (deg, [0, 360)) de la VIO : la VIO de Louis n'a pas de propriété yaw_deg."""
    fwd = vio.R[:, 0]
    return float(np.degrees(np.arctan2(fwd[2], fwd[0])) % 360.0)


def yaw_err_deg(a, b):
    return (a - b + 180.0) % 360.0 - 180.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", choices=sorted(sim.ENV_NAMES), default="foret")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--img-par-s", type=float, default=15.0, help="images traitées par seconde simulée")
    ap.add_argument("--dt-phys", type=float, default=0.01, help="pas de commande et de l'IMU (s) ; MuJoCo sous-échantillonne à 2 ms")
    ap.add_argument("--fps", type=float, default=30.0, help="images par seconde de la vidéo")
    ap.add_argument("--duree", type=float, default=60.0, help="durée simulée max (s)")
    ap.add_argument("--max-contacts", type=int, default=5, help="arrêt après ce nombre de contacts")
    ap.add_argument("--arret-immobile", type=float, default=2.0,
                    help="arrêt si la vitesse reste sous --v-immobile pendant cette durée (s) ; 0 désactive")
    ap.add_argument("--v-immobile", type=float, default=0.05, help="seuil de vitesse considérée nulle (m/s)")
    # VIO
    ap.add_argument("--pose", choices=("vio", "vraie"), default="vio",
                    help="pose utilisée pour la carte et la commande ; vraie = VIO en simple observateur")
    ap.add_argument("--klt", choices=("opencv", "ours", "perso", "ref"), default="opencv",
                    help="opencv (défaut) : cv2.calcOpticalFlowPyrLK ; ours (ou perso, ref) : notre KLT (vision/klt.py)")
    ap.add_argument("--harris", choices=("opencv", "ours"), default="opencv",
                    help="opencv (défaut) : cv2.cornerHarris ; ours : notre fonction de Harris (gaussiennes)")
    ap.add_argument("--klt-res", type=float, default=0.15, help="résidu KLT moyen max (gris dans [0, 1]) ; au-delà le point est perdu")
    ap.add_argument("--fenetre", type=int, default=15, help="nombre d'images clés de la fenêtre glissante")
    ap.add_argument("--n-points", type=int, default=250, help="nombre de points suivis par la VIO")
    ap.add_argument("--dist-min", type=int, default=15, help="distance min entre points suivis (px)")
    ap.add_argument("--kf-parallaxe", type=float, default=50.0, help="parallaxe moyenne (px) déclenchant une image clé")
    ap.add_argument("--vio-iters", type=int, default=5, help="itérations de Levenberg-Marquardt par image")
    ap.add_argument("--schur", choices=("dense", "creux"), default="creux",
                    help="complément de Schur du back-end VIO : dense (version d'origine) ou creux (numba, par couple image-point)")
    ap.add_argument("--normes", choices=("recalcul", "partage"), default="partage",
                    help="normes de fenêtres ZNCC : recalculées par l'appariement, ou reprises de la stéréo du front-end VIO")
    ap.add_argument("--no-ransac", action="store_true", help="pas de RANSAC (matrice fondamentale) dans le suivi")
    ap.add_argument("--bruit-imu", type=float, default=0.05,
                    help="bruit blanc gaussien relatif de l'IMU : écart-type = ce facteur x amplitude RMS du signal (gyro et accéléro) ; 0 = parfait")
    ap.add_argument("--gyro-bruit", type=float, default=0.0, help="densité de bruit gyro (rad/s/sqrt(Hz)), 0 = parfait")
    ap.add_argument("--acc-bruit", type=float, default=0.0, help="densité de bruit accéléro (m/s²/sqrt(Hz)), 0 = parfait")
    # follow : None = valeur par défaut de vision/follow.py
    ap.add_argument("--v-max", type=float, default=6.0)
    ap.add_argument("--lookahead", type=float, default=1.3)
    ap.add_argument("--a-frein", type=float, default=None)
    ap.add_argument("--v-min", type=float, default=None)
    ap.add_argument("--rayon-follow", choices=("avant", "vitesse", "aucun"), default=None, help="défaut : celui de follow")
    ap.add_argument("--arret-sans-chemin", type=float, default=2.0,
                    help="arrêt 'pas de chemin' après cette durée (s simulées) sans chemin A* consécutif ; 0 = dès la première image sans chemin")
    ap.add_argument("--a-lat-max", type=float, default=4.0, help="accélération latérale max (m/s2) ; 0 = désactivé")
    ap.add_argument("--courbure-lisse", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--yaw-cos2", action="store_true")
    ap.add_argument("--k-yaw", type=float, default=None)
    ap.add_argument("--a-lat", type=float, default=None)
    ap.add_argument("--a-front", type=float, default=None)
    ap.add_argument("--marge-front", type=float, default=None)
    # vision / planification (comme test_vol)
    ap.add_argument("--W", type=int, default=865)
    ap.add_argument("--H", type=int, default=650)
    ap.add_argument("--max-coins", type=int, default=3000)
    ap.add_argument("--seuil", type=float, default=1e-6)
    ap.add_argument("--grille", type=int, nargs=2, default=(12, 16))
    ap.add_argument("--zncc", type=float, default=0.8)
    ap.add_argument("--ratio", type=float, default=0.6)
    ap.add_argument("--voxel", type=float, default=0.25)
    ap.add_argument("--max-par-frame", type=int, default=1)
    ap.add_argument("--portee", type=float, default=15.0)
    ap.add_argument("--marge-carte", type=float, default=16.0)
    ap.add_argument("--ymin", type=float, default=0.6)
    ap.add_argument("--ymax", type=float, default=4.4)
    ap.add_argument("--rayon", type=float, default=0.14)
    ap.add_argument("--rayon-v", type=float, default=0.035)
    ap.add_argument("--marge", type=float, default=0.3)
    ap.add_argument("--k-unk", type=float, default=CLASSIFY_DEF["k_unk"])
    ap.add_argument("--k-prox", type=float, default=8.0)
    ap.add_argument("--d-secu", type=float, default=1.0)
    ap.add_argument("--seuil-free", type=float, default=0.6,
                    help="L < -seuil_free = libre ; 0.6 avec val_free -0.4 : 2 rayons libres (2 images) pour dire libre")
    ap.add_argument("--seuil-occ", type=float, default=1.2,
                    help="L > seuil_occ = occupé ; 1.2 avec val_occ 0.85 : 2 impacts (2 images) pour dire occupé")
    ap.add_argument("--w", type=float, default=2.0)
    ap.add_argument("--rayon-fenetre", type=float, default=8.0,
                    help="rayon (m) de la fenêtre autour du drone où classify calcule les distances ; 0 = toute la carte")
    ap.add_argument("--classify-classique", action="store_true",
                    help="utilise classify (distances sur toute la carte) au lieu de classify_fenetre")
    ap.add_argument("--save", default=None, help="dossier de sortie (défaut : sorties_test_vslam/<env>)")
    ap.add_argument("--no-video", action="store_true")
    args = ap.parse_args()
    CLASSIFY_DEF["seuil_occ"], CLASSIFY_DEF["seuil_free"] = args.seuil_occ, args.seuil_free   # la vue de dessus suit les mêmes seuils
    follow_kw = {k: v for k, v in dict(v_max=args.v_max, lookahead=args.lookahead,
                                       a_frein=args.a_frein, v_min=args.v_min,
                                       a_lat=args.a_lat, a_front=args.a_front, marge_front=args.marge_front,
                                       a_lat_max=args.a_lat_max or None, k_yaw=args.k_yaw, yaw_cos2=args.yaw_cos2 or None,
                                       courbure_lisse=args.courbure_lisse or None).items() if v is not None}
    if args.rayon_follow is not None:
        follow_kw["rayon"] = None if args.rayon_follow == "aucun" else args.rayon_follow
    follow_eff = {**FOLLOW_DEF, **follow_kw}
    follow_eff.pop("grid", None)
    use_true = args.pose == "vraie"

    dt_img, dt_vid, dt = 1.0 / args.img_par_s, 1.0 / args.fps, args.dt_phys
    if args.save is None:
        args.save = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sorties_test_vslam", args.env)
    os.makedirs(args.save, exist_ok=True)

    ctx, window = make_hidden_context()
    scene = sim.Scene(ctx, maze_seed=args.seed)
    mode = sim.ENV_NAMES[args.env]
    scene.building_mode = mode
    if mode == sim.ENV_FOREST:
        scene.forest_seed = args.seed
    if mode == sim.ENV_CITY:
        scene.city_seed = args.seed   # ville fixe, la graine tire départ et cible
    pos, yaw = scene.spawn(mode)
    if mode == sim.ENV_FOREST:
        scene.ensure_forest()
    goal_w = np.array(scene.target(), float)
    drone = sim.Drone(scene, position=np.array(pos, float), yaw=float(yaw))
    drone.cams_on_chassis = True              # la VIO suppose des caméras solidaires de l'IMU
    drone.update_cameras()
    cams = drone.cameras[:2]
    cap = StereoCapture(ctx, scene, cams, W=args.W, H=args.H)

    # --- extrinsèques caméra/corps (calibration) et initialisation de la VIO sur la pose vraie ---
    R_wb = drone.flight.rotation.copy()
    p_b = np.array(drone.position, float)
    R_wc = np.column_stack([cams[0].right, -cams[0].up, cams[0].forward]).astype(float)
    R_bc = R_wb.T @ R_wc
    p_bcL = R_wb.T @ (np.array(cams[0].position, float) - p_b)
    p_bcR = R_wb.T @ (np.array(cams[1].position, float) - p_b)
    K = cams[0].intrinsics(args.W, args.H)
    front = FrontEnd(K, n_max=args.n_points, min_dist=args.dist_min, klt=make_klt(args))
    vio = VIO(K, R_bc, p_bcL, p_bcR, R_wb, p_b, drone.velocity, fenetre=args.fenetre,
              kf_parallaxe=args.kf_parallaxe, iters=args.vio_iters, front=front, schur=args.schur,
              sigma_g=max(args.gyro_bruit, 2e-3), sigma_a=max(args.acc_bruit, 2e-2))
    imu = SimIMU(drone, gyro_noise=args.gyro_bruit, acc_noise=args.acc_bruit, seed=args.seed, rel_noise=args.bruit_imu)

    def est_pose():
        """Position et lacet utilisés par la carte et la commande."""
        if use_true:
            return np.array(drone.position, float), float(drone.yaw)
        return vio.p.copy(), vio_yaw_deg(vio)

    p0 = np.array(pos, float)
    lo = np.minimum(p0, goal_w) - args.marge_carte
    hi = np.maximum(p0, goal_w) + args.marge_carte
    vm = VoxelMap(np.array([lo[0], p0[1] - 2.0, lo[2]]), np.array([hi[0] - lo[0], 10.0, hi[2] - lo[2]]), args.voxel)

    print(f"\n=== test_vslam : env={args.env} seed={args.seed}, {args.W}x{args.H}, {args.img_par_s:g} img/s, "
          f"v_max {follow_eff['v_max']:g} m/s, pose {'VRAIE (VIO observatrice)' if use_true else 'VIO'}")
    print(f"  VIO : KLT {args.klt}, {args.n_points} points, fenêtre {args.fenetre} images clés, "
          f"IMU {1 / dt:g} Hz {f'bruit relatif {args.bruit_imu:g}' if args.bruit_imu else 'parfait' if not (args.gyro_bruit or args.acc_bruit) else f'bruit gyro {args.gyro_bruit:g} acc {args.acc_bruit:g}'}")
    print("  follow : " + ", ".join(f"{k}={v:g}" if isinstance(v, (int, float)) else f"{k}={v}" for k, v in follow_eff.items()))
    print(f"  départ {np.round(p0, 1)}, cible {np.round(goal_w, 1)}, distance {np.linalg.norm(goal_w - p0):.1f} m")

    writer = top = chase_fbo = None
    if not args.no_video:
        top = TopViewVIO(vm, args, scene, goal_w, p0)
        chase = sim.ChaseCamera(drone, distance=1.3, height=0.5)
        chase_fbo = ctx.framebuffer(ctx.renderbuffer((PANEL_W, PANEL_H)), ctx.depth_renderbuffer((PANEL_W, PANEL_H)))
        video_path = os.path.join(args.save, "test_vslam.mp4")
        writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (PANEL_W + TOP_W, TOP_H))
        if not writer.isOpened():
            raise SystemExit(f"Impossible d'ouvrir {video_path} en écriture")
    half_fov_h = math.degrees(math.atan(math.tan(math.radians(sim.CAMERA_FOVY) / 2) * args.W / args.H))

    path_w, libre, dist_obs, grid_f = None, None, None, None
    camp, t_img = None, np.zeros(len(STEPS))
    traj = [drone.position.copy()]
    traj_est = [vio.p.copy()]
    times, rows, vol_rows, vio_rows = [], [], [], []
    n_contacts, in_contact, min_clear = 0, False, np.inf
    t_immobile = 0.0
    t_no_path = None
    cmd_v, cmd_yaw = np.zeros(3), 0.0
    status = "durée max atteinte"
    next_img, next_vid, t_sim, it = 0.0, 0.0, 0.0, 0
    t_video = 0.0
    err_p, err_yaw = 0.0, 0.0
    acc, extra = dict(follow=0.0, phys=0.0, video=0.0), []
    t_wall = time.perf_counter()

    while t_sim <= args.duree:
        # --- perception (VIO + carte) + planification ---
        if t_sim >= next_img - 1e-9:
            next_img += dt_img
            t = np.zeros(len(STEPS))
            t0 = time.perf_counter()
            scene.target_items = lambda: []
            frame = cap.grab(cams, mode, t_sim)
            del scene.target_items
            t[0] = time.perf_counter() - t0
            P_w, C, ptsL, Z, n_coins = perceive_vio(frame, args, t, vio, use_true)
            pos_e, yaw_e = est_pose()
            t0 = time.perf_counter()
            vm.integrate(P_w, C, max_range=args.portee, max_par_frame=args.max_par_frame)
            t[5] = time.perf_counter() - t0
            path_w, info = plan(vm, pos_e, goal_w, args)
            libre, dist_obs, grid_f = info.get("libre"), info.get("dist"), info.get("grid")
            t[6], t[7] = info["t_classify"], info["t_astar"]
            times.append(t)
            extra.append(acc)
            acc = dict(follow=0.0, phys=0.0, video=0.0)
            t_img = t
            # erreur de la VIO (pose de la dernière image, pas la pose propagée)
            R_true, p_true = drone.flight.rotation, np.array(drone.position, float)
            err_p = float(np.linalg.norm(vio.p - p_true))
            err_rot = float(np.degrees(np.linalg.norm(so3_log(R_true.T @ vio.R))))
            err_yaw = yaw_err_deg(vio_yaw_deg(vio), drone.yaw)
            st = getattr(vio, "stats", {})
            vio_rows.append([it, f"{t_sim:.3f}", *np.round(p_true, 4), *np.round(vio.p, 4),
                             round(drone.yaw, 3), round(vio_yaw_deg(vio), 3), f"{err_p:.4f}", f"{err_yaw:.3f}", f"{err_rot:.3f}",
                             *np.round(drone.velocity, 3), *np.round(vio.v, 3),
                             st.get("suivis", 0), st.get("total", 0), st.get("stereo", 0), st.get("points3d", 0),
                             st.get("obs", 0), st.get("images", 0), st.get("cle", 0),
                             f"{1000 * st.get('t_front', 0):.1f}", f"{1000 * st.get('t_back', 0):.1f}"])
            d_goal = float(np.linalg.norm(goal_w - pos_e))
            rows.append([it, f"{t_sim:.3f}", n_coins, len(ptsL), *[f"{1000 * x:.2f}" for x in t],
                         f"{1000 * t.sum():.2f}", f"{d_goal:.2f}"])
            print(f"  it {it:3d}  t {t_sim:5.1f} s  pts {len(ptsL):5d}  total {1000 * t.sum():4.0f} ms "
                  f"(VIO {1000 * t[2]:3.0f})  v {np.linalg.norm(drone.velocity):4.2f} m/s  cible {d_goal:5.1f} m  "
                  f"VIO : klt {st.get('klt', 0):3d}/{st.get('prev', 0):3d} suivis {st.get('suivis', 0):3d} 3D {st.get('points3d', 0):3d}{' clé' if st.get('cle') else '    '} "
                  f"err {100 * err_p:5.1f} cm {err_yaw:+5.2f}°  contacts {n_contacts}"
                  + ("" if path_w is not None else "  PAS DE CHEMIN"))
            if writer is not None:
                camp = camera_panel(frame.images[0], ptsL, Z, args.portee)
                put_lines(camp, [f"image traitee {it}  t = {t_sim:5.2f} s",
                                 f"apparies {len(ptsL)}  (couleur = profondeur 0-{args.portee:g} m)",
                                 f"VIO : {st.get('suivis', 0)} points suivis, {st.get('points3d', 0)} points 3D"])
                s = PANEL_W / args.W
                for u, v in front.pts:
                    cv2.circle(camp, (int(u * s), int(v * s)), 3, (255, 0, 255), 1)
            it += 1
            if d_goal < 1.0:
                status = "cible atteinte (selon la pose " + ("vraie)" if use_true else "VIO)")
                break
            if path_w is None:
                if t_no_path is None:
                    t_no_path = t_sim
                if t_sim - t_no_path >= args.arret_sans_chemin:
                    status = f"pas de chemin depuis {args.arret_sans_chemin:g} s" if args.arret_sans_chemin > 0 else "pas de chemin"
                    break
            else:
                t_no_path = None

        # --- image vidéo ---
        if writer is not None and t_sim >= next_vid - 1e-9:
            next_vid += dt_vid
            tv = time.perf_counter()
            pos_e, yaw_e = est_pose()
            chase_fbo.use()
            chase_fbo.clear(0.55, 0.7, 0.9)
            scene.render(chase, PANEL_W / PANEL_H, extra=drone.render_items() + scene.target_items(), building_mode=mode)
            chp = np.frombuffer(chase_fbo.read(components=3), np.uint8).reshape(PANEL_H, PANEL_W, 3)[::-1].copy()
            ctx.screen.use()
            v = drone.velocity
            put_lines(chp, [f"vue de suivi  t = {t_sim:5.2f} s",
                            f"vitesse {np.linalg.norm(v):4.2f} m/s  (commande {np.linalg.norm(cmd_v):4.2f})",
                            f"lacet {drone.yaw:5.1f} deg  tangage {drone.pitch:5.1f}  roulis {drone.roll:5.1f}",
                            f"erreur VIO {100 * np.linalg.norm(vio.p - drone.position):5.1f} cm, lacet {yaw_err_deg(vio_yaw_deg(vio), drone.yaw):+5.2f} deg",
                            f"contacts {n_contacts}" + ("  CONTACT" if drone.contact else "")])
            lines = [(n, f"{1000 * x:.0f} ms") for n, x in zip(STEPS, t_img)] + [("TOTAL", f"{1000 * t_img.sum():.0f} ms")]
            put_lines(chp, lines, org=(8, PANEL_H - 22 * len(lines)), scale=0.5, col2=190)
            topi, _ = top.draw_vio(traj, traj_est, path_w, drone.position, pos_e, yaw_e, t_sim, half_fov_h, args.portee, err_p)
            img = np.zeros((TOP_H, PANEL_W + TOP_W, 3), np.uint8)
            img[:PANEL_H, :PANEL_W] = camp
            img[PANEL_H:2 * PANEL_H, :PANEL_W] = chp
            img[:, PANEL_W:] = topi
            writer.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            t_video += time.perf_counter() - tv
            acc["video"] += time.perf_counter() - tv
            vol_rows.append([f"{t_sim:.3f}", *np.round(drone.position, 3), *np.round(vio.p, 3),
                             *np.round(drone.velocity, 3), *np.round(cmd_v, 3), round(cmd_yaw, 2), int(drone.contact)])

        # --- commande (pose estimée) + physique + IMU ---
        tf = time.perf_counter()
        pos_e, yaw_e = est_pose()
        cmd_v, cmd_yaw = follow(path_w, libre, dist_obs, pos_e, yaw_e, goal_w, grid=grid_f, **follow_kw)
        tp = time.perf_counter()
        hit = drone.update(dt, cmd_velocity=cmd_v, cmd_yaw_rate=cmd_yaw)
        w_imu, f_imu = imu.sample(dt)
        if args.bruit_imu:   # covariance de préintégration cohérente avec le bruit injecté
            vio.sg, vio.sa = max(imu.sg, 2e-3), max(imu.sa, 2e-2)
        vio.imu(dt, w_imu, f_imu)
        acc["follow"] += tp - tf
        acc["phys"] += time.perf_counter() - tp
        if hit and not in_contact:
            n_contacts += 1
            if n_contacts >= args.max_contacts:
                status = f"{n_contacts} contacts"
                break
        in_contact = hit
        if np.linalg.norm(drone.velocity) < args.v_immobile and t_sim > 1.0:
            t_immobile += dt
        else:
            t_immobile = 0.0
        if args.arret_immobile > 0 and t_immobile >= args.arret_immobile:
            status = f"immobile depuis {args.arret_immobile:g} s"
            break
        if scene.forest is not None:
            min_clear = min(min_clear, clearance_trees(scene, drone.position) - args.rayon)
        elif scene.city is not None and mode == sim.ENV_CITY:
            min_clear = min(min_clear, scene.city.clearance(drone.position) - args.rayon)
        traj.append(drone.position.copy())
        traj_est.append(vio.p.copy())
        t_sim += dt

    if writer is not None:
        writer.release()
    glfw.destroy_window(window)
    glfw.terminate()

    # --- bilan vol ------------------------------------------------------------
    times, traj, traj_est = np.array(times), np.array(traj), np.array(traj_est)
    T = times[1:] if len(times) > 1 else times
    tot = T.sum(axis=1)
    dist_parc = float(np.sum(np.linalg.norm(np.diff(traj, axis=0), axis=1)))
    print(f"\n  Bilan : {status} à t = {t_sim:.1f} s simulées ({len(times)} images traitées)")
    print(f"  distance parcourue {dist_parc:.1f} m, distance finale (vraie) à la cible {np.linalg.norm(goal_w - traj[-1]):.1f} m")
    print(f"  contacts avec un obstacle : {n_contacts}")
    if np.isfinite(min_clear):
        print(f"  distance minimale bord du drone (rayon {args.rayon} m) - tronc : {100 * min_clear:.0f} cm")

    # --- bilan VIO --------------------------------------------------------------
    V = np.array([[float(x) for x in r[1:11]] for r in vio_rows])     # t, p vraie (3), p est (3), lacets, err_p
    tv_, Pt, Pe = V[:, 0], V[:, 1:4], V[:, 4:7]
    E = np.linalg.norm(Pe - Pt, axis=1)
    Ey = np.array([float(r[11]) for r in vio_rows])
    Er = np.array([float(r[12]) for r in vio_rows])
    rpe = []
    k_1s = max(1, int(round(args.img_par_s)))
    for i in range(len(Pt) - k_1s):                   # erreur relative sur 1 s (dérive locale)
        rpe.append(np.linalg.norm((Pe[i + k_1s] - Pe[i]) - (Pt[i + k_1s] - Pt[i])))
    rpe = np.array(rpe) if rpe else np.zeros(1)
    vio_lines = [f"  VIO ({'observatrice' if use_true else 'utilisée pour voler'}) :",
                 f"    ATE (RMSE position) {100 * np.sqrt(np.mean(E ** 2)):.1f} cm, max {100 * E.max():.1f} cm, "
                 f"finale {100 * E[-1]:.1f} cm ({100 * E[-1] / max(dist_parc, 1e-6):.2f} % de la distance parcourue)",
                 f"    RPE sur 1 s : médiane {100 * np.median(rpe):.1f} cm, max {100 * rpe.max():.1f} cm",
                 f"    lacet : erreur max {np.abs(Ey).max():.2f}°, finale {Ey[-1]:+.2f}° ; rotation 3D max {Er.max():.2f}°",
                 f"    images clés {int(sum(int(r[25]) for r in vio_rows))}/{len(vio_rows)}, points suivis médiane "
                 f"{np.median([r[19] for r in vio_rows]):.0f}, points 3D médiane {np.median([r[22] for r in vio_rows]):.0f}"]
    print(chr(10).join(vio_lines))

    # --- récap des temps --------------------------------------------------------
    Ex = extra[1:1 + len(T)]
    Ex = {k: np.array([e[k] for e in Ex]) for k in ("follow", "phys", "video")} if len(Ex) == len(T) else {}
    cols = [(n, c) for n, c in zip(STEPS, T.T)] + [("TOTAL vision", T.sum(axis=1))]
    lab = {"follow": "follow (cmd), tous les pas", "phys": "physique MuJoCo + IMU, tous les pas", "video": "vidéo (rendu + mp4)"}
    cols += [(lab[k], v) for k, v in Ex.items()]
    tot_all = T.sum(axis=1) + sum(Ex.values()) if Ex else T.sum(axis=1)
    cols.append(("TOTAL réel / image", tot_all))
    ref = np.median(tot_all)
    vt = np.array([[float(r[-2]), float(r[-1])] for r in vio_rows[1:]]) if len(vio_rows) > 1 else np.zeros((1, 2))
    lines = [f"  Récap des temps par image traitée, première image exclue (ms) :",
             f"    {'étape':<36s} {'médiane':>9s} {'moyenne':>9s} {'max':>9s} {'% total méd.':>13s}"]
    for n, c in cols:
        lines.append(f"    {n:<36s} {1000 * np.median(c):9.1f} {1000 * c.mean():9.1f} {1000 * c.max():9.1f} {100 * np.median(c) / ref:12.0f} %")
    lines.append(f"    dont VIO front-end (KLT, RANSAC, stéréo) médiane {np.median(vt[:, 0]):.1f} ms, "
                 f"back-end (optimisation) médiane {np.median(vt[:, 1]):.1f} ms")
    lines.append(f"  (statut : {status}, {len(times)} images, {t_sim:.1f} s simulées ; temps réel écoulé {time.perf_counter() - t_wall:.0f} s)")
    print(chr(10).join(lines))
    with open(os.path.join(args.save, "recap_temps.txt"), "w", encoding="utf8") as f:
        f.write(chr(10).join(lines + [""] + vio_lines) + chr(10))
    tot_v = T[:, 1:].sum(axis=1)
    print(f"  hors capture : médiane {1000 * np.median(tot_v):.0f} ms, max {1000 * tot_v.max():.0f} ms, "
          f"{100 * (tot_v <= dt_img).mean():.0f} % des images dans le budget ({1000 * dt_img:.0f} ms)")

    # --- fichiers -----------------------------------------------------------------
    with open(os.path.join(args.save, "temps.csv"), "w", newline="") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["image", "t_sim_s", "coins", "apparies", *[f"{s}_ms" for s in STEPS], "total_ms", "dist_cible_m"])
        w.writerows(rows)
    with open(os.path.join(args.save, "vol.csv"), "w", newline="") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["t_s", "x", "y", "z", "x_vio", "y_vio", "z_vio", "vx", "vy", "vz",
                    "cmd_avant", "cmd_droite", "cmd_haut", "cmd_lacet_deg_s", "contact"])
        w.writerows(vol_rows)
    with open(os.path.join(args.save, "vslam.csv"), "w", newline="") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["image", "t_s", "x", "y", "z", "x_vio", "y_vio", "z_vio", "lacet", "lacet_vio",
                    "err_pos_m", "err_lacet_deg", "err_rot_deg", "vx", "vy", "vz", "vx_vio", "vy_vio", "vz_vio",
                    "suivis", "points", "stereo", "points3d", "observations", "images_fenetre", "image_cle",
                    "front_ms", "back_ms"])
        w.writerows(vio_rows)
    fig, ax = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
    ax[0].plot(tv_, 100 * E, c="tab:purple"); ax[0].set_ylabel("erreur position (cm)"); ax[0].grid(alpha=0.3)
    ax[1].plot(tv_, Ey, c="tab:purple", label="lacet"); ax[1].plot(tv_, Er, c="tab:gray", lw=0.8, label="rotation 3D")
    ax[1].set_ylabel("erreur (°)"); ax[1].set_xlabel("t (s)"); ax[1].grid(alpha=0.3); ax[1].legend(fontsize=8)
    ax[0].set_title(f"Erreur de la VIO, {args.env} graine {args.seed} (pose {'vraie' if use_true else 'VIO'} pour voler)")
    fig.tight_layout(); fig.savefig(os.path.join(args.save, "erreur_vio.png"), dpi=120); plt.close(fig)
    print(f"  Sorties dans {args.save}/")


if __name__ == "__main__":
    main()
