"""test14.py : pipeline vision, étapes 1 à 4, sur une capture du simulateur.

  1. capture      : paire stéréo + z-buffer (vérité terrain, évaluation seulement)
  2. coins        : Harris sur les deux images
  3. appariement  : coins gauche <-> droite le long des lignes (cas rectifié)
  4. triangulation: Z = f B / d, puis repère monde ; comparaison au z-buffer

Lancement (depuis le dossier Projet) :
    python test14.py                      # murs troués, graine 0
    python test14.py --env foret --seed 3
    python test14.py --avance 4 --lacet 20 --save sorties_test14

La profondeur estimée vient uniquement des images ; le z-buffer ne sert qu'à mesurer l'erreur.
"""

import argparse
import math
import os
import time

import glfw
import moderngl
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

import scene as sim
from vision.capture import StereoCapture
from vision.corners import harris_function, detect_corners
from vision.matching import match_stereo, to_gray
from vision.triangulation import triangulate

OUTLIER_PX = 2.0      # erreur de disparité au-delà de laquelle un appariement est jugé faux
EDGE_REL = 0.05       # coin « sur un bord de profondeur » si la profondeur GT varie de > 5 % autour


# ---------------------------------------------------------------------------
# 1. Capture
# ---------------------------------------------------------------------------

def match_along_rows(imgL, imgR, cL, min_zncc=0.8, ratio=0.8, grayL=None, grayR=None):
    """Adaptation de vision.matching.match_stereo (gris, [x_g,y,x_d,d,coût]) -> (ptsL, ptsR, scores)."""
    gL = grayL if grayL is not None else to_gray(imgL)
    gR = grayR if grayR is not None else to_gray(imgR)
    m = match_stereo(gL, gR, cL, max_cost=2.0 * (1.0 - min_zncc), ratio=ratio).astype(np.float64)
    if len(m) == 0:
        return np.empty((0, 2)), np.empty((0, 2)), np.empty(0)
    return m[:, 0:2], np.column_stack([m[:, 2], m[:, 1]]), 1.0 - m[:, 4] / 2.0


def make_hidden_context():
    """Contexte OpenGL 3.3 dans une fenêtre glfw cachée (comme scene.py, sans affichage)."""
    if not glfw.init():
        raise SystemExit("Impossible d'initialiser GLFW")
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, True)
    glfw.window_hint(glfw.VISIBLE, False)
    window = glfw.create_window(64, 64, "test14", None, None)
    if not window:
        glfw.terminate()
        raise SystemExit("Impossible de créer la fenêtre cachée")
    glfw.make_context_current(window)
    return moderngl.create_context(), window


def capture(args):
    ctx, window = make_hidden_context()
    scene = sim.Scene(ctx, maze_seed=args.seed)
    mode = sim.ENV_NAMES[args.env]
    scene.building_mode = mode
    pos, yaw = scene.spawn(mode)
    if mode == sim.ENV_FOREST:
        scene.ensure_forest()
    yaw = float(yaw) + args.lacet
    y = math.radians(yaw)
    pos = np.array(pos, dtype=float) + args.avance * np.array([math.cos(y), 0.0, math.sin(y)])
    drone = sim.Drone(scene, position=pos, yaw=yaw)
    cams = drone.cameras[:2]                      # paire stéréo : gauche, droite

    cap = StereoCapture(ctx, scene, cams, W=args.W, H=args.H)
    frame = cap.grab(cams, mode, 0.0)
    near, far = cams[0].near, 500.0               # mêmes plans que Camera.projection
    Z_gt = StereoCapture.linearize_depth(frame.depths[0].astype(np.float64), near, far)
    Z_gt[frame.depths[0] >= 1.0] = np.inf        # ciel : pas de surface
    glfw.destroy_window(window)
    glfw.terminate()
    return frame, Z_gt


# ---------------------------------------------------------------------------
# Évaluation
# ---------------------------------------------------------------------------

def gt_at(Z_gt, u, v):
    """Profondeur GT au pixel (u, v) et indicateur « bord de profondeur » (voisinage 3x3).

    Si le pixel tombe sur le ciel (coin sur une silhouette), on prend la surface la plus
    proche du voisinage 3x3 ; le point est alors marqué « bord ».
    """
    H, W = Z_gt.shape
    ui = np.clip(np.rint(u).astype(int), 1, W - 2)
    vi = np.clip(np.rint(v).astype(int), 1, H - 2)
    Zf = np.where(np.isfinite(Z_gt), Z_gt, np.nan)
    neigh = np.stack([Zf[vi + dy, ui + dx] for dy in (-1, 0, 1) for dx in (-1, 0, 1)], axis=1)
    z = Zf[vi, ui]
    neigh = np.where(np.isnan(neigh), np.inf, neigh)
    zmin, zmax = neigh.min(axis=1), neigh.max(axis=1)
    z = np.where(np.isfinite(z), z, zmin)               # ciel au pixel -> surface voisine
    with np.errstate(invalid="ignore"):
        edge = ~np.isfinite(zmax) | ((zmax - zmin) > EDGE_REL * zmin)   # ciel voisin -> bord
    z = np.where(np.isfinite(z), z, np.inf)
    return z, edge


def matches_any_surface(Z_gt, u, v, d_est, fB, r=3):
    """True si d_est est à OUTLIER_PX près de la disparité vraie d'un pixel du voisinage (2r+1)^2."""
    H, W = Z_gt.shape
    ui, vi = np.rint(u).astype(int), np.rint(v).astype(int)
    hit = np.zeros(len(u), dtype=bool)
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            z = Z_gt[np.clip(vi + dy, 0, H - 1), np.clip(ui + dx, 0, W - 1)]
            d_true = np.where(np.isfinite(z), fB / np.where(np.isfinite(z), z, 1.0), 0.0)
            hit |= np.abs(d_est - d_true) <= OUTLIER_PX
    return hit


def stats_line(name, x):
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return f"  {name:<28s} (aucune valeur)"
    return (f"  {name:<28s} médiane {np.median(x):8.3f}   moyenne {np.mean(x):8.3f}   "
            f"p90 {np.percentile(x, 90):8.3f}   max {np.max(x):8.3f}")


# ---------------------------------------------------------------------------
# Programme principal
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Pipeline vision étapes 1 à 4 + erreur vs z-buffer")
    ap.add_argument("--env", choices=sorted(sim.ENV_NAMES), default="murs")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--avance", type=float, default=0.0, help="avance le drone de N m depuis le spawn")
    ap.add_argument("--lacet", type=float, default=0.0, help="tourne le drone de N degrés")
    ap.add_argument("--W", type=int, default=1730)
    ap.add_argument("--H", type=int, default=1300)
    ap.add_argument("--max-coins", type=int, default=10000)
    ap.add_argument("--seuil", type=float, default=1e-6,
                    help="seuil Harris relatif au max global (bas : la grille répartit les coins)")
    ap.add_argument("--grille", type=int, nargs=2, default=(12, 16), help="cellules (lignes, colonnes)")
    ap.add_argument("--matcher", choices=("lignes", "coins"), default="lignes",
                    help="lignes : coin gauche cherché dans toute la ligne droite ; coins : coin à coin")
    ap.add_argument("--zncc", type=float, default=0.8, help="score ZNCC minimal")
    ap.add_argument("--ratio", type=float, default=0.6, help="test du ratio (plus bas = plus strict)")
    ap.add_argument("--save", default=None, help="dossier où enregistrer les figures PNG")
    ap.add_argument("--no-show", action="store_true", help="n'ouvre pas les fenêtres matplotlib")
    args = ap.parse_args()

    timings = {}

    # 1. Capture -----------------------------------------------------------
    t0 = time.perf_counter()
    frame, Z_gt = capture(args)
    timings["1 capture"] = time.perf_counter() - t0
    imgL, imgR = frame.images[0], frame.images[1]
    K = frame.K
    f = K[0, 0]
    B = float(np.linalg.norm(frame.Cs[1] - frame.Cs[0]))
    R, C = frame.Rs[0], frame.Cs[0]

    # 2. Coins de Harris ---------------------------------------------------
    t0 = time.perf_counter()
    RL = harris_function(to_gray(imgL))
    RR = harris_function(to_gray(imgR))
    kw = dict(rel_thresh=args.seuil, max_n=args.max_coins, grid_cells=tuple(args.grille))
    cL = detect_corners(RL, **kw)
    cR = detect_corners(RR, **kw)
    timings["2 coins"] = time.perf_counter() - t0

    # 3. Appariement -------------------------------------------------------
    t0 = time.perf_counter()
    if args.matcher == "lignes":
        ptsL, ptsR, scores = match_along_rows(imgL, imgR, cL, min_zncc=args.zncc, ratio=args.ratio)
    else:
        ptsL, ptsR, scores = match_along_rows(imgL, imgR, cL, min_zncc=args.zncc, ratio=args.ratio)
    timings["3 appariement"] = time.perf_counter() - t0

    # 4. Triangulation -----------------------------------------------------
    t0 = time.perf_counter()
    uL, vL, uR, vR = ptsL[:, 0], ptsL[:, 1], ptsR[:, 0], ptsR[:, 1]
    P_w, ok = triangulate(uL, vL, uR, vR, K, B, R, C)
    timings["4 triangulation"] = time.perf_counter() - t0

    # Évaluation (z-buffer) ------------------------------------------------
    d_est = uL - uR
    Z_est = np.where(ok, f * B / np.where(ok, d_est, 1.0), np.nan)
    z_true, edge = gt_at(Z_gt, uL, vL)
    has_gt = np.isfinite(z_true)
    d_true = np.where(has_gt, f * B / np.where(has_gt, z_true, 1.0), 0.0)   # ciel : disparité 0
    d_err = d_est - d_true
    # Faux appariement : la disparité ne correspond à aucune surface du voisinage 7x7.
    # (un coin sur une silhouette peut légitimement prendre la profondeur de l'avant-plan
    #  ou de l'arrière-plan : on ne le compte pas comme faux)
    outlier = ~matches_any_surface(Z_gt, uL, vL, d_est, f * B)
    good = ok & has_gt & ~outlier

    # Point 3D vrai : même pixel gauche, profondeur GT
    z_fin = np.where(has_gt, z_true, np.nan)
    X_t = (uL - K[0, 2]) * z_fin / f
    Y_t = (vL - K[1, 2]) * z_fin / f
    P_true = np.stack([X_t, Y_t, z_fin], axis=1) @ R.T + C
    err3d = np.linalg.norm(P_w - P_true, axis=1)
    dZ = Z_est - z_true

    # Résumé console -------------------------------------------------------
    print(f"\n=== test14 : env={args.env} seed={args.seed} avance={args.avance} lacet={args.lacet}")
    print(f"  K : f = {f:.1f} px, c = ({K[0, 2]:.0f}, {K[1, 2]:.0f}),  B = {B * 100:.1f} cm,"
          f"  pose gauche C = {np.round(C, 2)}")
    print(f"  coins     : gauche {len(cL)}, droite {len(cR)}")
    print(f"  appariés  : {len(ptsL)}  (ZNCC médiane {np.median(scores) if len(scores) else float('nan'):.3f})")
    print(f"  triangulés: {int(ok.sum())} (d > 0.5 px),  avec GT : {int((ok & has_gt).sum())}")
    n_eval = int((ok & has_gt).sum())
    if n_eval:
        n_out = int((ok & has_gt & outlier).sum())
        n_edge = int((ok & has_gt & edge).sum())
        print(f"  faux appariements (aucune surface du 7x7 à ±{OUTLIER_PX:.0f} px) : {n_out} / {n_eval} "
              f"({100 * n_out / n_eval:.1f} %)   coins sur bord de profondeur : {n_edge}")
        sel = good
        print("  Erreurs sur les appariements corrects :")
        print(stats_line("|Δd| (px)", np.abs(d_err[sel])))
        print(stats_line("|ΔZ| (m)", np.abs(dZ[sel])))
        print(stats_line("|ΔZ|/Z (%)", 100 * np.abs(dZ[sel]) / z_true[sel]))
        print(stats_line("erreur 3D (m)", err3d[sel]))
        sel2 = good & ~edge
        print("  Idem hors bords de profondeur :")
        print(stats_line("|Δd| (px)", np.abs(d_err[sel2])))
        print(stats_line("|ΔZ| (m)", np.abs(dZ[sel2])))
    print("  Temps : " + ", ".join(f"{k} {1000 * v:.0f} ms" for k, v in timings.items()))

    # Figures --------------------------------------------------------------
    figs = []
    H, W = imgL.shape[:2]
    zmax = np.nanpercentile(np.where(np.isfinite(Z_gt), Z_gt, np.nan), 98)

    # Fig 1 : capture
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    ax[0].imshow(imgL); ax[0].set_title("1. Caméra gauche")
    ax[1].imshow(imgR); ax[1].set_title("1. Caméra droite")
    im = ax[2].imshow(np.where(np.isfinite(Z_gt), Z_gt, np.nan), cmap="turbo", vmin=0, vmax=zmax)
    ax[2].set_title("Z-buffer gauche linéarisé (vérité terrain, m)")
    fig.colorbar(im, ax=ax[2], fraction=0.035)
    for a in ax: a.axis("off")
    fig.tight_layout(); figs.append(("1_capture", fig))

    # Fig 2 : coins de Harris
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    ax[0].imshow(np.sign(RL) * np.log1p(np.abs(RL) / (np.abs(RL).max() * 1e-4 + 1e-12)), cmap="gray")
    ax[0].set_title("2. Réponse de Harris gauche (log)")
    ax[1].imshow(imgL); ax[1].scatter(cL[:, 0], cL[:, 1], s=8, c="yellow", edgecolors="k", linewidths=0.3)
    ax[1].set_title(f"2. Coins gauche ({len(cL)})")
    ax[2].imshow(imgR); ax[2].scatter(cR[:, 0], cR[:, 1], s=8, c="yellow", edgecolors="k", linewidths=0.3)
    ax[2].set_title(f"2. Coins droite ({len(cR)})")
    for a in ax: a.axis("off")
    fig.tight_layout(); figs.append(("2_coins", fig))

    # Fig 3 : appariement (images côte à côte, segments verts = corrects, rouges = faux)
    fig, ax = plt.subplots(1, 1, figsize=(16, 6.2))
    ax.imshow(np.hstack([imgL, imgR]))
    if len(ptsL):
        segs = np.stack([ptsL, ptsR + [W, 0]], axis=1)
        cols = np.where((outlier & has_gt)[:, None], [[0.9, 0.1, 0.1, 0.9]], [[0.1, 0.85, 0.2, 0.6]])
        ax.add_collection(LineCollection(segs, colors=cols, linewidths=0.7))
        ax.scatter(ptsL[:, 0], ptsL[:, 1], s=6, c="cyan")
        ax.scatter(ptsR[:, 0] + W, ptsR[:, 1], s=6, c="cyan")
    n_bad = int((outlier & has_gt).sum())
    ax.set_title(f"3. Appariements : {len(ptsL)} (vert : correct, rouge : faux, {n_bad})")
    ax.axis("off")
    fig.tight_layout(); figs.append(("3_appariement", fig))

    # Fig 4 : triangulation et erreurs
    fig = plt.figure(figsize=(17, 9.5))
    a1 = fig.add_subplot(2, 3, 1)
    a1.imshow(imgL)
    sc = a1.scatter(uL[ok], vL[ok], c=Z_est[ok], s=12, cmap="turbo", vmin=0, vmax=zmax,
                    edgecolors="k", linewidths=0.3)
    fig.colorbar(sc, ax=a1, fraction=0.035)
    a1.set_title("4. Profondeur estimée Z = fB/d (m)"); a1.axis("off")

    a2 = fig.add_subplot(2, 3, 2)
    m = ok & has_gt
    a2.scatter(z_true[m & ~outlier & ~edge], Z_est[m & ~outlier & ~edge], s=8, label="correct")
    a2.scatter(z_true[m & ~outlier & edge], Z_est[m & ~outlier & edge], s=8, c="orange", label="bord prof.")
    a2.scatter(z_true[m & outlier], Z_est[m & outlier], s=10, c="red", marker="x", label="faux")
    lim = [0, max(1.0, zmax, float(np.nanmax(np.r_[z_true[m], Z_est[m], 0.0])) * 1.05)]
    a2.plot(lim, lim, "k--", lw=0.8)
    a2.set_xlim(lim); a2.set_ylim(lim)
    a2.set_xlabel("Z vrai (m)"); a2.set_ylabel("Z estimé (m)"); a2.set_title("Z estimé vs z-buffer")
    a2.legend(fontsize=8)

    a3 = fig.add_subplot(2, 3, 3)
    e = d_err[m]
    a3.hist(np.clip(e, -5, 5), bins=60, color="tab:blue")
    a3.axvline(-OUTLIER_PX, color="r", ls=":"); a3.axvline(OUTLIER_PX, color="r", ls=":")
    a3.set_xlabel("erreur de disparité d_est - d_vrai (px, bornée à ±5)")
    a3.set_title("Erreur de disparité")

    a4 = fig.add_subplot(2, 3, 4)
    s = good
    a4.scatter(z_true[s], np.abs(dZ[s]), s=8, label="|ΔZ| mesuré")
    zz = np.linspace(0.1, max(1.0, zmax), 200)
    sig = np.median(np.abs(d_err[s & ~edge])) if np.any(s & ~edge) else 0.5
    a4.plot(zz, zz ** 2 / (f * B) * sig, "k--", lw=1, label=f"Z²/(fB)·{sig:.2f} px")
    a4.set_xlabel("Z vrai (m)"); a4.set_ylabel("|ΔZ| (m)"); a4.set_title("Erreur de profondeur (appariements corrects)")
    a4.legend(fontsize=8)

    a5 = fig.add_subplot(2, 3, 5)
    a5.imshow(imgL)
    sc = a5.scatter(uL[s], vL[s], c=np.abs(dZ[s]) / z_true[s] * 100, s=12, cmap="magma_r",
                    vmin=0, vmax=10, edgecolors="k", linewidths=0.3)
    a5.scatter(uL[m & outlier], vL[m & outlier], s=20, c="red", marker="x")
    fig.colorbar(sc, ax=a5, fraction=0.035)
    a5.set_title("|ΔZ|/Z (%), croix rouges = faux"); a5.axis("off")

    a6 = fig.add_subplot(2, 3, 6, projection="3d")
    # Repère monde : Y vers le haut ; on affiche (x, z, y) pour avoir la hauteur verticale
    a6.scatter(P_true[m, 0], P_true[m, 2], P_true[m, 1], s=4, c="gray", alpha=0.5, label="vrai")
    a6.scatter(P_w[s, 0], P_w[s, 2], P_w[s, 1], s=5, c="tab:blue", label="estimé")
    a6.scatter(P_w[m & outlier, 0], P_w[m & outlier, 2], P_w[m & outlier, 1], s=8, c="red", marker="x")
    a6.scatter([C[0]], [C[2]], [C[1]], c="green", s=40, marker="^", label="caméra")
    a6.set_xlabel("x"); a6.set_ylabel("z"); a6.set_zlabel("y (haut)")
    a6.set_title("Points 3D (repère monde)"); a6.legend(fontsize=8)
    fig.tight_layout(); figs.append(("4_triangulation", fig))

    if args.save:
        os.makedirs(args.save, exist_ok=True)
        for name, fg in figs:
            fg.savefig(os.path.join(args.save, f"{name}.png"), dpi=110)
        print(f"  Figures enregistrées dans {args.save}/")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
