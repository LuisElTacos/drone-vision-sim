"""test_vol.py : pipeline complète en vol MuJoCo réel (plus de téléportation), vidéo continue.

Même chaîne que test_full.py (capture -> coins -> appariement -> triangulation -> voxels ->
classify -> A*), lancée img-par-s fois par seconde simulée (12 par défaut). Entre deux images, la physique
MuJoCo avance par pas de --dt-phys et vision/follow.py transforme le dernier chemin A* en
commande (vitesse + vitesse de lacet) pour le contrôleur en cascade de mujoco_flight.py.

Simplification : le chemin calculé sur une image est appliqué tout de suite (on ne simule pas
la latence du calcul).

Sorties (dossier --save, par défaut sorties_test_vol/<env>/ à côté du script) :
  test_vol.mp4   vidéo à --fps images/s en temps simulé réel :
                 en haut à gauche, dernière image traitée + points appariés (3 Hz) ;
                 en bas à gauche, vue de suivi en continu ;
                 à droite, vue de dessus (carte, chemin A*, trajectoire, troncs réels)
  temps.csv      temps de chaque étape pour chaque image traitée (ms)
  vol.csv        état du drone à chaque image vidéo (position, vitesse, commande, contact)

    python test_vol.py                                   # forêt, graine 0
    python test_vol.py --env murs --duree 90
    python test_vol.py --v-max 2.5 --cams-stab           # plus rapide, caméras stabilisées
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

import scene as sim
from test14 import make_hidden_context
from test18 import plan, clearance_trees
from test_full import STEPS, PANEL_W, PANEL_H, TOP_W, TOP_H, TopView, perceive, camera_panel, put_lines
from vision.capture import StereoCapture
from vision.follow import follow
from vision.planner import classify
from vision.voxels import VoxelMap

# Valeurs par défaut lues directement dans vision/follow.py et vision/planner.py
_defaults = lambda f: {k: p.default for k, p in inspect.signature(f).parameters.items()
                       if p.default is not inspect.Parameter.empty}
FOLLOW_DEF = _defaults(follow)
CLASSIFY_DEF = _defaults(classify)


class TopViewAltitude(TopView):
    """Vue de dessus = couche de voxels à l'altitude du drone, avec les seuils de classify
    (la vue de test_full prend le max sur toute la tranche : un linteau au-dessus d'une ouverture la cache)."""

    def draw(self, traj, path_w, pos, yaw_deg, t_sim, half_fov_deg, rng):
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
        y, h = math.radians(yaw_deg), math.radians(half_fov_deg)
        a = (pos[0] + rng * math.cos(y - h), pos[2] + rng * math.sin(y - h))
        b = (pos[0] + rng * math.cos(y + h), pos[2] + rng * math.sin(y + h))
        self.fov_line.set_data([a[0], pos[0], b[0]], [a[1], pos[2], b[1]])
        self.title.set_text(f"Vue de dessus, t = {t_sim:5.2f} s, couche y = {pos[1]:.2f} m (altitude du drone)\n"
                            "(gris inconnu, vert libre, rouge occupé, seuils de classify)")
        self.fig.canvas.draw()
        img = np.asarray(self.fig.canvas.buffer_rgba())[:, :, :3]
        return cv2.resize(img, (TOP_W, TOP_H))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", choices=sorted(sim.ENV_NAMES), default="foret")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--img-par-s", type=float, default=12.0, help="images traitées par seconde simulée")
    ap.add_argument("--dt-phys", type=float, default=0.01, help="pas de commande (s) ; MuJoCo sous-échantillonne à 2 ms")
    ap.add_argument("--fps", type=float, default=30.0, help="images par seconde de la vidéo")
    ap.add_argument("--duree", type=float, default=60.0, help="durée simulée max (s)")
    ap.add_argument("--max-contacts", type=int, default=5, help="arrêt après ce nombre de contacts")
    ap.add_argument("--arret-immobile", type=float, default=2.0,
                    help="arrêt si la vitesse reste sous --v-immobile pendant cette durée (s) ; 0 désactive")
    ap.add_argument("--v-immobile", type=float, default=0.05, help="seuil de vitesse considérée nulle (m/s)")
    ap.add_argument("--cams-stab", action="store_true", help="caméras stabilisées (lacet seul) au lieu de fixées au châssis")
    # follow
    # follow : None = valeur par défaut de vision/follow.py
    ap.add_argument("--v-max", type=float, default=6.0)
    ap.add_argument("--lookahead", type=float, default=1.3)
    ap.add_argument("--a-frein", type=float, default=None)
    ap.add_argument("--v-min", type=float, default=None)
    ap.add_argument("--rayon-follow", choices=("avant", "vitesse", "aucun"), default=None, help="défaut : celui de follow")
    ap.add_argument("--arret-sans-chemin", type=float, default=2.0,
                    help="arrêt 'pas de chemin' seulement après cette durée (s simulées) sans chemin A* consécutif ; 0 = dès la première image sans chemin. Entre-temps follow commande vitesse nulle + rotation explore_rate")
    ap.add_argument("--a-lat-max", type=float, default=4.0, help="accélération latérale max (m/s2) : v <= sqrt(a/courbure) ; 0 = désactivé")
    ap.add_argument("--courbure-lisse", action=argparse.BooleanOptionalAction, default=True, help="courbure calculée sur le chemin lissé (moyenne glissante ~1 m)")
    ap.add_argument("--yaw-cos2", action="store_true", help="vitesse multipliée par cos(erreur de lacet)^2 au lieu de cos")
    ap.add_argument("--k-yaw", type=float, default=None, help="gain de lacet, défaut : follow")
    ap.add_argument("--a-lat", type=float, default=None, help="poids de v_prox (tous les obstacles), défaut : follow")
    ap.add_argument("--a-front", type=float, default=None, help="poids de v_front (obstacles en face), défaut : follow")
    ap.add_argument("--marge-front", type=float, default=None, help="marge avant le voxel touché par le rayon (m)")
    # vision / planification (comme test_full)
    ap.add_argument("--W", type=int, default=865)
    ap.add_argument("--H", type=int, default=650)
    ap.add_argument("--max-coins", type=int, default=3000)
    ap.add_argument("--seuil", type=float, default=1e-6)
    ap.add_argument("--grille", type=int, nargs=2, default=(12, 16))
    ap.add_argument("--matcher", choices=("lignes", "coins"), default="lignes")
    ap.add_argument("--zncc", type=float, default=0.8)
    ap.add_argument("--ratio", type=float, default=0.6)
    ap.add_argument("--voxel", type=float, default=0.25)
    ap.add_argument("--max-par-frame", type=int, default=1,
                    help="nb max de rayons (ou impacts) comptés par voxel sur une même image (1 à 255)")
    ap.add_argument("--portee", type=float, default=15.0)
    ap.add_argument("--marge-carte", type=float, default=16.0, help="marge autour du segment départ-cible (m)")
    ap.add_argument("--ymin", type=float, default=0.6)
    ap.add_argument("--ymax", type=float, default=4.4, help="altitude max de vol (m), plafond de la mission")
    ap.add_argument("--rayon", type=float, default=0.14, help="rayon horizontal du drone (m), englobe le châssis à tout lacet")
    ap.add_argument("--rayon-v", type=float, default=0.035, help="demi-hauteur du drone (m) ; gonflement cylindrique")
    ap.add_argument("--marge", type=float, default=0.3, help="marge dure autour du drone (m) ; l éloignement vient du coût d_secu de classify")
    ap.add_argument("--k-unk", type=float, default=CLASSIFY_DEF["k_unk"], help="défaut : celui de classify")
    ap.add_argument("--k-prox", type=float, default=8.0)
    ap.add_argument("--d-secu", type=float, default=1.0, help="distance de sécurité du coût de proximité (m)")
    ap.add_argument("--seuil-free", type=float, default=None, help="seuil libre de classify (défaut : classify)")
    ap.add_argument("--seuil-occ", type=float, default=None, help="seuil occupé de classify (défaut : classify)")
    ap.add_argument("--w", type=float, default=2.0)
    ap.add_argument("--save", default=None,
                    help="dossier de sortie (défaut : sorties_test_vol/<env> à côté de test_vol.py)")
    ap.add_argument("--no-video", action="store_true")
    args = ap.parse_args()
    # seuls les paramètres donnés en ligne de commande remplacent les défauts de follow
    follow_kw = {k: v for k, v in dict(v_max=args.v_max, lookahead=args.lookahead,
                                       a_frein=args.a_frein, v_min=args.v_min,
                                       a_lat=args.a_lat, a_front=args.a_front, marge_front=args.marge_front,
                                       a_lat_max=args.a_lat_max or None, k_yaw=args.k_yaw, yaw_cos2=args.yaw_cos2 or None, courbure_lisse=args.courbure_lisse or None).items() if v is not None}
    if args.rayon_follow is not None:
        follow_kw["rayon"] = None if args.rayon_follow == "aucun" else args.rayon_follow
    follow_eff = {**FOLLOW_DEF, **follow_kw}
    follow_eff.pop("grid", None)

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
    cams = drone.cameras[:2]
    cap = StereoCapture(ctx, scene, cams, W=args.W, H=args.H)

    p0 = np.array(pos, float)
    lo = np.minimum(p0, goal_w) - args.marge_carte
    hi = np.maximum(p0, goal_w) + args.marge_carte
    vm = VoxelMap(np.array([lo[0], p0[1] - 2.0, lo[2]]), np.array([hi[0] - lo[0], 10.0, hi[2] - lo[2]]), args.voxel)

    print(f"\n=== test_vol : env={args.env} seed={args.seed}, {args.W}x{args.H}, {args.img_par_s:g} img/s, "
          f"v_max {follow_eff['v_max']:g} m/s, caméras {'stabilisées' if args.cams_stab else 'fixées au châssis'}")
    print("  follow : " + ", ".join(f"{k}={v:g}" if isinstance(v, (int, float)) else f"{k}={v}" for k, v in follow_eff.items()))
    print(f"  départ {np.round(p0, 1)}, cible {np.round(goal_w, 1)}, distance {np.linalg.norm(goal_w - p0):.1f} m")

    writer = top = chase_fbo = None
    if not args.no_video:
        top = TopViewAltitude(vm, args, scene, goal_w, p0)
        chase = sim.ChaseCamera(drone, distance=1.3, height=0.5)
        chase_fbo = ctx.framebuffer(ctx.renderbuffer((PANEL_W, PANEL_H)), ctx.depth_renderbuffer((PANEL_W, PANEL_H)))
        video_path = os.path.join(args.save, "test_vol.mp4")
        writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (PANEL_W + TOP_W, TOP_H))
        if not writer.isOpened():
            raise SystemExit(f"Impossible d'ouvrir {video_path} en écriture")
    half_fov_h = math.degrees(math.atan(math.tan(math.radians(sim.CAMERA_FOVY) / 2) * args.W / args.H))

    path_w, libre, dist_obs, grid_f = None, None, None, None
    camp, t_img = None, np.zeros(len(STEPS))
    traj = [drone.position.copy()]
    times, rows, vol_rows = [], [], []
    n_contacts, in_contact, min_clear = 0, False, np.inf
    t_immobile = 0.0
    t_no_path = None
    cmd_v, cmd_yaw = np.zeros(3), 0.0
    status = "durée max atteinte"
    next_img, next_vid, t_sim, it = 0.0, 0.0, 0.0, 0
    t_video = 0.0
    acc, extra = dict(follow=0.0, phys=0.0, video=0.0), []     # temps hors vision cumulé entre deux images traitées
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
            P_w, C, ptsL, Z, n_coins = perceive(frame, args, t)
            t0 = time.perf_counter()
            vm.integrate(P_w, C, max_range=args.portee, max_par_frame=args.max_par_frame)
            t[4] = time.perf_counter() - t0
            path_w, info = plan(vm, drone.position, goal_w, args)
            libre, dist_obs, grid_f = info.get("libre"), info.get("dist"), info.get("grid")
            t[5], t[6] = info["t_classify"], info["t_astar"]
            times.append(t)
            extra.append(acc)
            acc = dict(follow=0.0, phys=0.0, video=0.0)
            t_img = t
            d_goal = float(np.linalg.norm(goal_w - drone.position))
            rows.append([it, f"{t_sim:.3f}", n_coins, len(ptsL), *[f"{1000 * x:.2f}" for x in t],
                         f"{1000 * t.sum():.2f}", f"{d_goal:.2f}"])
            print(f"  it {it:3d}  t {t_sim:5.1f} s  pts {len(ptsL):5d}  total {1000 * t.sum():4.0f} ms  "
                  f"v {np.linalg.norm(drone.velocity):4.2f} m/s  dist cible {d_goal:5.1f} m  contacts {n_contacts}"
                  + ("" if path_w is not None else "  PAS DE CHEMIN"))
            if writer is not None:
                camp = camera_panel(frame.images[0], ptsL, Z, args.portee)
                put_lines(camp, [f"image traitee {it}  t = {t_sim:5.2f} s",
                                 f"apparies {len(ptsL)}  (couleur = profondeur 0-{args.portee:g} m)"])
            it += 1
            if d_goal < 1.0:
                status = "cible atteinte"
                break
            if path_w is None:                       # pas de chemin : on attend --arret-sans-chemin s avant d'arrêter
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
            chase_fbo.use()
            chase_fbo.clear(0.55, 0.7, 0.9)
            scene.render(chase, PANEL_W / PANEL_H, extra=drone.render_items() + scene.target_items(), building_mode=mode)
            chp = np.frombuffer(chase_fbo.read(components=3), np.uint8).reshape(PANEL_H, PANEL_W, 3)[::-1].copy()
            ctx.screen.use()
            v = drone.velocity
            put_lines(chp, [f"vue de suivi  t = {t_sim:5.2f} s",
                            f"vitesse {np.linalg.norm(v):4.2f} m/s  (commande {np.linalg.norm(cmd_v):4.2f})",
                            f"lacet {drone.yaw:5.1f} deg  tangage {drone.pitch:5.1f}  roulis {drone.roll:5.1f}",
                            f"contacts {n_contacts}" + ("  CONTACT" if drone.contact else "")])
            lines = [(n, f"{1000 * x:.0f} ms") for n, x in zip(STEPS, t_img)] + [("TOTAL", f"{1000 * t_img.sum():.0f} ms")]
            put_lines(chp, lines, org=(8, PANEL_H - 22 * len(lines)), scale=0.5, col2=190)
            topi = top.draw(traj, path_w, drone.position, drone.yaw, t_sim, half_fov_h, args.portee)
            img = np.zeros((TOP_H, PANEL_W + TOP_W, 3), np.uint8)
            img[:PANEL_H, :PANEL_W] = camp
            img[PANEL_H:2 * PANEL_H, :PANEL_W] = chp
            img[:, PANEL_W:] = topi
            writer.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            t_video += time.perf_counter() - tv
            acc["video"] += time.perf_counter() - tv
            vol_rows.append([f"{t_sim:.3f}", *np.round(drone.position, 3), *np.round(drone.velocity, 3),
                             *np.round(cmd_v, 3), round(cmd_yaw, 2), int(drone.contact)])

        # --- commande + physique ---
        tf = time.perf_counter()
        cmd_v, cmd_yaw = follow(path_w, libre, dist_obs, drone.position, drone.yaw, goal_w, grid=grid_f, **follow_kw)
        tp = time.perf_counter()
        hit = drone.update(dt, cmd_velocity=cmd_v, cmd_yaw_rate=cmd_yaw)
        acc["follow"] += tp - tf
        acc["phys"] += time.perf_counter() - tp
        if hit and not in_contact:
            n_contacts += 1
            if n_contacts >= args.max_contacts:      # arrêt, vidéo et sorties enregistrées
                status = f"{n_contacts} contacts"
                break
        in_contact = hit
        # arrêt si le drone est immobile depuis --arret-immobile s (après la première seconde)
        if np.linalg.norm(drone.velocity) < args.v_immobile and t_sim > 1.0:
            t_immobile += dt
        else:
            t_immobile = 0.0
        if args.arret_immobile > 0 and t_immobile >= args.arret_immobile:
            status = f"immobile depuis {args.arret_immobile:g} s"
            break
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
    # --- récap des temps : pipeline vision (temps.csv) + hors vision, par image traitée ---
    E = extra[1:1 + len(T)]
    E = {k: np.array([e[k] for e in E]) for k in ("follow", "phys", "video")} if len(E) == len(T) else {}
    cols = [(n, c) for n, c in zip(STEPS, T.T)] + [("TOTAL vision", T.sum(axis=1))]
    lab = {"follow": "follow (cmd), tous les pas", "phys": "physique MuJoCo, tous les pas", "video": "vidéo (rendu + mp4)"}
    cols += [(lab[k], v) for k, v in E.items()]
    tot_all = T.sum(axis=1) + sum(E.values()) if E else T.sum(axis=1)
    cols.append(("TOTAL réel / image", tot_all))
    ref = np.median(tot_all)
    lines = [f"  Récap des temps par image traitée, première image exclue (ms) :",
             f"    {'étape':<32s} {'médiane':>9s} {'moyenne':>9s} {'max':>9s} {'% total méd.':>13s}"]
    for n, c in cols:
        lines.append(f"    {n:<32s} {1000 * np.median(c):9.1f} {1000 * c.mean():9.1f} {1000 * c.max():9.1f} {100 * np.median(c) / ref:12.0f} %")
    lines.append(f"  (statut : {status}, {len(times)} images, {t_sim:.1f} s simulées ; temps réel écoulé {time.perf_counter() - t_wall:.0f} s)")
    print(chr(10).join(lines))
    with open(os.path.join(args.save, "recap_temps.txt"), "w", encoding="utf8") as f:
        f.write(chr(10).join(lines) + chr(10))
    tot_v = T[:, 1:].sum(axis=1)             # sans la capture (rendu simulé, remplacé par la caméra en vrai)
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
