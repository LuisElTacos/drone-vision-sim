"""dense_capture.py : capture de séquences stéréo (réalisme par défaut) + profondeur vraie, pour tester la stéréo dense.

Le drone est téléporté (pas de MuJoCo) : départ de l'environnement, avance de --pas m par image vers l'avant,
lacet qui oscille de ±--osc degrés ; s'il y a un obstacle à moins de --d-stop m droit devant (profondeur vraie),
il tourne de 40° au lieu d'avancer. Images consécutives à 1/--img-par-s s d'écart (flou de mouvement réaliste).

Sortie : sorties/dense/seq/<env>_s<graine>.npz avec
  L, R      (N, H, W, 3) uint8   images gauche / droite
  Z         (N, H, W) float32    profondeur vraie de la caméra gauche (m, inf = ciel)
  Rs, Cs    (N, 3, 3), (N, 3)    pose caméra gauche -> monde
  K, B                           intrinsèques, base stéréo (m)

    python dense_capture.py --env ville --seed 0
"""
import argparse
import math
import os

import glfw
import numpy as np

import scene as sim
from test14 import make_hidden_context
from vision.capture import StereoCapture

ICI = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", choices=sorted(sim.ENV_NAMES), default="ville")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--pas", type=float, default=0.35)
    ap.add_argument("--osc", type=float, default=25.0)
    ap.add_argument("--d-stop", type=float, default=2.5)
    ap.add_argument("--img-par-s", type=float, default=8.0)
    ap.add_argument("--W", type=int, default=865)
    ap.add_argument("--H", type=int, default=650)
    args = ap.parse_args()

    ctx, window = make_hidden_context()
    scene = sim.Scene(ctx, maze_seed=args.seed)
    mode = sim.ENV_NAMES[args.env]
    scene.building_mode = mode
    if mode == sim.ENV_FOREST:
        scene.forest_seed = args.seed
    if mode == sim.ENV_CITY:
        scene.city_seed = args.seed
    pos, yaw0 = scene.spawn(mode)
    if mode == sim.ENV_FOREST:
        scene.ensure_forest()
    drone = sim.Drone(scene, position=np.array(pos, float), yaw=float(yaw0))
    drone.cams_on_chassis = False
    cams = drone.cameras[:2]
    cap = StereoCapture(ctx, scene, cams, W=args.W, H=args.H)
    scene.target_items = lambda: []
    near = cams[0].near

    Ls, Rs_img, Zs, Rs, Cs = [], [], [], [], []
    p = np.array(pos, float)
    base_yaw = float(yaw0)
    for i in range(args.n):
        drone.position = p.copy()
        drone.yaw = base_yaw + args.osc * math.sin(2 * math.pi * i / 12)
        drone.update_cameras()
        fr = cap.grab(cams, mode, i / args.img_par_s)
        Z = StereoCapture.linearize_depth(fr.depths[0].astype(np.float64), near, 500.0).astype(np.float32)
        Z[fr.depths[0] >= 1.0] = np.inf
        Ls.append(fr.images[0]); Rs_img.append(fr.images[1]); Zs.append(Z)
        Rs.append(fr.Rs[0]); Cs.append(fr.Cs[0]); K = fr.K
        B = float(np.linalg.norm(fr.Cs[1] - fr.Cs[0]))
        # avance ou tourne selon la profondeur vraie au centre (bande de la largeur du drone)
        H, W = Z.shape
        zc = np.nanmin(np.where(np.isfinite(Z[H // 3:2 * H // 3, W // 2 - 60:W // 2 + 60]),
                                Z[H // 3:2 * H // 3, W // 2 - 60:W // 2 + 60], 1e3))
        if zc < args.d_stop:
            base_yaw += 40.0
        else:
            p = p + args.pas * drone.forward
        print(f"  image {i:2d}  pos {np.round(p, 2)}  lacet {drone.yaw:6.1f}  Z centre min {zc:5.1f} m")

    out = os.path.join(ICI, "sorties", "dense", "seq")
    os.makedirs(out, exist_ok=True)
    f = os.path.join(out, f"{args.env}_s{args.seed}.npz")
    np.savez_compressed(f, L=np.array(Ls), R=np.array(Rs_img), Z=np.array(Zs), Rs=np.array(Rs), Cs=np.array(Cs),
                        K=K, B=B)
    print("->", f)
    glfw.destroy_window(window)
    glfw.terminate()


if __name__ == "__main__":
    main()
