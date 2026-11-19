"""Captures avant / après du rendu réaliste (realisme.py) et courte vidéo de démonstration.

Sorties dans sorties/realisme/ :
  <env>_avant.png, <env>_apres.png, <env>_comparaison.png   (caméra gauche du drone, 640 x 480)
  <env>_effets.png    planche : chaque option seule
  demo_realisme.mp4   vol court (déplacement cinématique), à gauche rendu actuel, à droite tout activé

Lancement : python demo_realisme.py [--envs murs,foret,ville] [--video-env foret] [--secondes 8]
"""

import argparse
import math
import os

import cv2
import glfw
import moderngl
import numpy as np

import realisme
import scene as sim

W, H = 640, 480
OUT = os.path.join(sim.HERE, "sorties", "realisme")


def hidden_context():
    if not glfw.init():
        raise SystemExit("Impossible d'initialiser GLFW")
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, True)
    glfw.window_hint(glfw.VISIBLE, False)
    window = glfw.create_window(64, 64, "demo_realisme", None, None)
    glfw.make_context_current(window)
    return moderngl.create_context(), window


def set_options(scene, on):
    for k in realisme.OPTIONS:
        scene.fx.options[k] = k in on


def grab(ctx, fbo, scene, cam, mode):
    fbo.use()
    ctx.viewport = (0, 0, W, H)
    fbo.clear(*realisme.SKY, 1.0)
    scene.render(cam, W / H, extra=scene.target_items(), building_mode=mode)
    img = np.frombuffer(fbo.read(components=3), np.uint8).reshape(H, W, 3)[::-1]
    return np.ascontiguousarray(img[:, :, ::-1])   # BGR pour cv2


def label(img, text):
    img = img.copy()
    cv2.rectangle(img, (0, 0), (len(text) * 11 + 14, 28), (0, 0, 0), -1)
    cv2.putText(img, text, (7, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def place(scene, drone, mode, offset=(0.0, 0.0, 0.0), yaw=None, pitch=0.0):
    pos, yaw0 = scene.spawn(mode)
    drone.yaw = yaw0 if yaw is None else yaw
    drone.position = np.array(pos, dtype="f8") + offset
    drone.camera_pitch = pitch
    drone.cams_on_chassis = False
    drone.update_cameras()


# point de vue de chaque environnement : (décalage depuis le départ, lacet (None : celui du départ), inclinaison)
VIEWS = {"murs": ((-4.0, 1.5, -5.5), 135.0, -20.0), "foret": ((0.0, 1.0, 0.0), None, -12.0),
         "ville": ((0.0, 1.5, 0.0), None, -10.0)}


def captures(ctx, scene, drone, env):
    mode = sim.ENV_NAMES[env]
    scene.building_mode = mode
    scene.spawn(mode)
    place(scene, drone, mode, *VIEWS[env])
    cam = drone.cameras[0]
    fbo = ctx.framebuffer(ctx.renderbuffer((W, H)), ctx.depth_renderbuffer((W, H)))

    def shot(on, step=2.0):
        set_options(scene, on)
        scene.fx._cams.clear()
        # deux images pour initialiser l'exposition et le flou (petit mouvement de lacet entre les deux)
        if on & {"expo", "flou"}:
            for k in range(6):
                scene.fx.time = k / 30.0
                drone.yaw += step
                drone.update_cameras()
                grab(ctx, fbo, scene, cam, mode)
            img = grab(ctx, fbo, scene, cam, mode)
            drone.yaw -= 6 * step
            drone.update_cameras()
            return img
        return grab(ctx, fbo, scene, cam, mode)

    avant = shot(set())
    apres = shot(set(realisme.OPTIONS), step=0.5)   # lacet lent : 15 deg/s
    cv2.imwrite(os.path.join(OUT, f"{env}_avant.png"), avant)
    cv2.imwrite(os.path.join(OUT, f"{env}_apres.png"), apres)
    cv2.imwrite(os.path.join(OUT, f"{env}_comparaison.png"),
                np.hstack([label(avant, "Rendu actuel"), np.zeros((H, 4, 3), np.uint8),
                           label(apres, "Tout active (lacet 15 deg/s)")]))
    tiles = [label(avant, "Aucun effet")]
    names = {"ombres": "Ombres", "ssao": "SSAO", "bruit": "Bruit capteur", "flou": "Flou (lacet 60 deg/s)",
             "expo": "Exposition auto", "distorsion": "Distorsion k1=-0.20", "jpeg": "JPEG q=60"}
    for k in realisme.OPTIONS:
        img = shot({k})
        if k in ("bruit", "jpeg"):   # zoom x2 du centre pour voir le grain et les blocs
            c = img[H // 3: 2 * H // 3, W // 3: 2 * W // 3]
            img = cv2.resize(c, (W, H), interpolation=cv2.INTER_NEAREST)
            names[k] += " (zoom x3)"
        tiles.append(label(img, names[k]))
    small = [cv2.resize(t, (W // 2, H // 2), interpolation=cv2.INTER_AREA) for t in tiles]
    rows = [np.hstack(small[i:i + 4]) for i in (0, 4)]
    cv2.imwrite(os.path.join(OUT, f"{env}_effets.png"), np.vstack(rows))
    fbo.release()
    print(env, "ok")


def video(ctx, scene, drone, env, seconds, fps=30):
    mode = sim.ENV_NAMES[env]
    scene.building_mode = mode
    pos0, yaw0 = scene.spawn(mode)
    fbo = ctx.framebuffer(ctx.renderbuffer((W, H)), ctx.depth_renderbuffer((W, H)))
    path = os.path.join(OUT, "demo_realisme.mp4")
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"avc1"), fps, (2 * W + 4, H))   # H.264
    if not writer.isOpened():
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (2 * W + 4, H))
    cam = drone.cameras[0]
    n = int(seconds * fps)
    frames_a, frames_b = [], []
    for want, store in ((set(), frames_a), (set(realisme.OPTIONS), frames_b)):
        set_options(scene, want)
        scene.fx._cams.clear()
        for i in range(n):
            t = i / fps
            scene.fx.time = t
            # trajectoire : avance à 3 m/s, lacet sinusoïdal (±35°, période 4 s), légère montée
            yaw = yaw0 + 35.0 * math.sin(2 * math.pi * t / 4.0)
            fwd0 = np.array([math.cos(math.radians(yaw0)), 0.0, math.sin(math.radians(yaw0))])
            drone.position = np.array(pos0, dtype="f8") + fwd0 * min(3.0 * t, 12.0) + (0, 0.15 * t, 0)
            drone.yaw = yaw
            drone.camera_pitch = -5.0
            drone.cams_on_chassis = False
            drone.update_cameras()
            store.append(grab(ctx, fbo, scene, cam, mode))
    sep = np.zeros((H, 4, 3), np.uint8)
    for a, b in zip(frames_a, frames_b):
        writer.write(np.hstack([label(a, "Rendu actuel"), sep, label(b, "Tout active")]))
    writer.release()
    fbo.release()
    print("video", path)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--envs", default="murs,foret,ville")
    ap.add_argument("--video-env", default="foret", help="environnement de la vidéo (vide : pas de vidéo)")
    ap.add_argument("--secondes", type=float, default=8.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    ctx, window = hidden_context()
    scene = sim.Scene(ctx, maze_seed=args.seed)
    drone = sim.Drone(scene, position=(0.0, 1.5, 0.0))
    for env in [e for e in args.envs.split(",") if e]:
        captures(ctx, scene, drone, env)
    if args.video_env:
        video(ctx, scene, drone, args.video_env, args.secondes)
    glfw.terminate()


if __name__ == "__main__":
    main()
