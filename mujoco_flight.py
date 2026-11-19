"""Dynamique de vol par MuJoCo (option hybride) : MuJoCo calcule le vol et les chocs, le rendu reste moderngl.

- Quadrotor à 4 moteurs à l'échelle de l'Avata 2 (377 g, boîte du modèle GLB), traînée de l'air.
- Scène de collision générée à partir des collisionneurs du simulateur (Box, Sphere, VerticalCylinder,
  Ground), sans aucun visuel ; reconstruite quand l'environnement change.
- Contrôleur en cascade : vitesse voulue -> poussée et attitude -> couples -> poussée de chaque moteur
  (contrôle géométrique sur SO(3)). Commande : vitesse voulue et vitesse de lacet.

Repères : le simulateur est Y-haut, MuJoCo Z-haut. (x, y, z)_sim -> (x, -z, y)_mujoco.
Repère du corps MuJoCo : x avant, y gauche, z haut.
"""

import math
import operator

import mujoco
import numpy as np
from numba import njit

from collision import Box, Ground, Sphere, VerticalCylinder

G = 9.81
MASS = 0.377               # masse de l'Avata 2 (kg)
THRUST_TO_WEIGHT = 3.0     # poussée max totale / poids
YAW_COEF = 0.015           # couple de lacet / poussée d'un rotor (m)
TIMESTEP = 0.002           # pas de physique (s), petit pour des contacts raides
TILT_MAX = math.radians(40.0)   # inclinaison max par défaut (réglable) ; elle limite aussi le freinage : a_max = g·tan(inclinaison)
ACC_MAX = 15.0             # accélération horizontale demandée max (m/s²)
KV, KI = 8.0, 0.8         # gains vitesse : proportionnel (1/s) et intégral (compense la traînée)
I_MAX = 2.0                # borne du terme intégral (m/s²)
KR, KW = 300.0, 30.0       # gains attitude (multipliés par l'inertie)
YAW_ERR_MAX = math.radians(60.0)

# Changement de repère MuJoCo -> simulateur
C_MJ_TO_SIM = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])


@njit(cache=True)
def _control_nb(R, v, w, v_des, psi_des, v_int, ki, kv, tilt_max, J, A_inv, f_max):
    """Contrôleur en cascade (voir MujocoFlight._control) ; met à jour v_int en place."""
    for i in range(3):
        v_int[i] = min(max(v_int[i] + ki * (v_des[i] - v[i]) * TIMESTEP, -I_MAX), I_MAX)
    a = kv * (v_des - v) + v_int
    a_h = math.sqrt(a[0] * a[0] + a[1] * a[1])
    if a_h > ACC_MAX:
        a[0] *= ACC_MAX / a_h
        a[1] *= ACC_MAX / a_h
    f = MASS * (a + np.array([0.0, 0.0, G]))
    f[2] = max(f[2], 0.2 * MASS * G)
    horiz = math.sqrt(f[0] * f[0] + f[1] * f[1])
    lim = f[2] * math.tan(tilt_max)
    if horiz > lim:
        f[0] *= lim / horiz
        f[1] *= lim / horiz
    b3 = f / math.sqrt(f[0] * f[0] + f[1] * f[1] + f[2] * f[2])
    c = np.array([math.cos(psi_des), math.sin(psi_des), 0.0])
    b2 = np.cross(b3, c)
    b2 = b2 / math.sqrt(b2[0] * b2[0] + b2[1] * b2[1] + b2[2] * b2[2])
    Rd = np.empty((3, 3))
    Rd[:, 0] = np.cross(b2, b3)
    Rd[:, 1] = b2
    Rd[:, 2] = b3
    E = 0.5 * (Rd.T @ R - R.T @ Rd)
    e_R = np.array([E[2, 1], E[0, 2], E[1, 0]])
    Jw = J @ w
    tau = -KR * (J @ e_R) - KW * Jw + np.cross(w, Jw)
    thrust = f[0] * R[0, 2] + f[1] * R[1, 2] + f[2] * R[2, 2]
    u = A_inv @ np.array([thrust, tau[0], tau[1], tau[2]])
    for i in range(4):
        u[i] = min(max(u[i], 0.0), f_max)
    return u


def to_mj(p):
    return np.array([p[0], -p[2], p[1]], dtype="f8")


def from_mj(m):
    return C_MJ_TO_SIM @ np.asarray(m, dtype="f8")


def _vec(v):
    return " ".join(f"{x:.6g}" for x in v)


def colliders_to_mjcf(colliders):
    """Géométries statiques MJCF équivalentes aux collisionneurs du simulateur."""
    out = []
    for c in colliders:
        if isinstance(c, Box):
            out.append(f'<geom type="box" pos="{_vec(to_mj(c.center))}" size="{_vec((c.half[0], c.half[2], c.half[1]))}"/>')
        elif isinstance(c, Sphere):
            out.append(f'<geom type="sphere" pos="{_vec(to_mj(c.center))}" size="{c.radius:.6g}"/>')
        elif isinstance(c, VerticalCylinder):
            cx, cz = c.center_xz
            pos = to_mj((cx, (c.y_min + c.y_max) / 2.0, cz))
            out.append(f'<geom type="cylinder" pos="{_vec(pos)}" size="{c.radius:.6g} {(c.y_max - c.y_min) / 2.0:.6g}"/>')
        elif isinstance(c, Ground):
            out.append(f'<geom type="plane" pos="0 0 {c.height:.6g}" size="0 0 1"/>')
    return "\n    ".join(out)


def drone_mjcf(half, colliders):
    """Modèle complet : quadrotor (boîte de collision = boîte du drone) + obstacles."""
    hx, hy, hz = half[2], half[0], half[1]          # avant, gauche, haut
    ax, ay = 0.8 * hx, 0.8 * hy                     # position des rotors
    f_max = MASS * G * THRUST_TO_WEIGHT / 4.0
    rotors = [(ax, -ay, -1), (-ax, -ay, 1), (-ax, ay, -1), (ax, ay, 1)]  # sens de rotation alternés
    sites = "\n      ".join(f'<site name="rotor{i}" pos="{x:.4f} {y:.4f} {hz:.4f}"/>' for i, (x, y, _) in enumerate(rotors))
    motors = "\n    ".join(
        f'<motor site="rotor{i}" gear="0 0 1 0 0 {s * YAW_COEF}" ctrlrange="0 {f_max:.4f}"/>'
        for i, (_, _, s) in enumerate(rotors))
    return f"""
<mujoco model="drone">
  <option timestep="{TIMESTEP}" density="1.225" viscosity="1.8e-5"/>
  <worldbody>
    {colliders_to_mjcf(colliders)}
    <body name="drone" pos="0 0 1">
      <freejoint/>
      <geom name="hull" type="box" size="{hx:.4f} {hy:.4f} {hz:.4f}" mass="{MASS}" friction="0.6"
            solref="0.004 1" solimp="0.95 0.99 0.001" margin="0.02"/>
      {sites}
    </body>
  </worldbody>
  <actuator>
    {motors}
  </actuator>
</mujoco>"""


class MujocoFlight:
    """État du drone calculé par MuJoCo, lu et écrit dans le repère du simulateur."""

    def __init__(self, half_extents, colliders, position, yaw_deg):
        self.half = np.asarray(half_extents, dtype="f8")
        self.scene_key = None
        self.scene_list = None
        self.model = self.data = None
        self.yaw_target = math.radians(-yaw_deg)     # lacet MuJoCo = -lacet simulateur
        self.time_tilted = 0.0
        self.v_int = np.zeros(3)
        self.t_acc = 0.0
        self.sync_scene(colliders)
        self.reset(position, yaw_deg)

    # --- scène -----------------------------------------------------------------

    def sync_scene(self, colliders):
        """Reconstruit le modèle si la liste des collisionneurs a changé (état du drone conservé).
        Comparaison par identité des objets (comme l'ancienne clé tuple(id(c))), mais faite en C avec
        map(operator.is_) au lieu d'un tuple de ~30 000 id() Python à chaque pas."""
        prev = self.scene_key
        if prev is not None and len(prev) == len(colliders) and (
                colliders is self.scene_list or all(map(operator.is_, prev, colliders))):
            self.scene_list = colliders
            return
        self.scene_key = list(colliders)
        self.scene_list = colliders        # garde les objets (donc leurs id restent uniques)
        old = None if self.data is None else (self.data.qpos.copy(), self.data.qvel.copy())
        self.model = mujoco.MjModel.from_xml_string(drone_mjcf(self.half, colliders))
        self.data = mujoco.MjData(self.model)
        self.bid = self.model.body("drone").id
        self._init_controller()
        if old is not None:
            self.data.qpos[:], self.data.qvel[:] = old
        mujoco.mj_forward(self.model, self.data)

    def _init_controller(self):
        self.kv, self.ki, self.tilt_max = KV, KI, TILT_MAX   # réglables en direct depuis le panneau
        m = self.model
        Ri = np.zeros(9)
        mujoco.mju_quat2Mat(Ri, m.body_iquat[self.bid])
        Ri = Ri.reshape(3, 3)
        self.J = Ri @ np.diag(m.body_inertia[self.bid]) @ Ri.T   # inertie en repère corps
        com = m.body_ipos[self.bid]
        A = np.zeros((4, 4))   # [poussée, couple x, couple y, couple z] = A @ poussées des rotors
        for i in range(4):
            r = m.site_pos[m.site(f"rotor{i}").id] - com
            A[:, i] = [1.0, r[1], -r[0], m.actuator_gear[i, 5]]
        self.A_inv = np.linalg.inv(A)
        self.f_max = m.actuator_ctrlrange[0, 1]

    # --- état ------------------------------------------------------------------

    def reset(self, position, yaw_deg, velocity=(0.0, 0.0, 0.0)):
        """Place le drone à plat (téléportation, changement d'environnement)."""
        psi = math.radians(-yaw_deg)
        self.yaw_target = psi
        self.data.qpos[:3] = to_mj(position)
        self.data.qpos[3:7] = (math.cos(psi / 2), 0.0, 0.0, math.sin(psi / 2))
        self.data.qvel[:] = 0.0
        self.data.qvel[:3] = to_mj(velocity)
        self.time_tilted = 0.0
        self.v_int = np.zeros(3)
        mujoco.mj_forward(self.model, self.data)

    @property
    def position(self):
        return from_mj(self.data.qpos[:3])

    @property
    def velocity(self):
        return from_mj(self.data.qvel[:3])

    @property
    def rotation(self):
        """Axes du corps en repère simulateur : colonnes (avant, gauche, haut)."""
        return C_MJ_TO_SIM @ self.data.xmat[self.bid].reshape(3, 3)

    @property
    def yaw_deg(self):
        R = self.data.xmat[self.bid].reshape(3, 3)
        return math.degrees(-math.atan2(R[1, 0], R[0, 0])) % 360.0

    @property
    def contact(self):
        # seul le drone est mobile : tout contact le concerne ; dist < 0 = interpénétration réelle
        # (les contacts de la marge de 2 cm, sans toucher, ne comptent pas)
        n = self.data.ncon
        return n > 0 and bool((self.data.contact.dist[:n] < 0.0).any())

    # --- contrôle --------------------------------------------------------------

    def _control(self, v_des, psi_des):
        R = np.ascontiguousarray(self.data.xmat[self.bid].reshape(3, 3))
        return _control_nb(R, self.data.qvel[:3].copy(), self.data.qvel[3:6].copy(), np.asarray(v_des, float),
                           float(psi_des), self.v_int, float(self.ki), float(self.kv), float(self.tilt_max),
                           self.J, self.A_inv, float(self.f_max))

    def step(self, dt, v_world, yaw_rate_deg):
        """Avance de dt secondes. v_world : vitesse voulue en repère simulateur (m/s)."""
        v_des = to_mj(v_world)
        R = self.data.xmat[self.bid].reshape(3, 3)
        psi = math.atan2(R[1, 0], R[0, 0])
        self.yaw_target -= math.radians(yaw_rate_deg) * dt
        # Le lacet visé ne s'éloigne pas trop du lacet réel (pas d'emballement si le drone est bloqué)
        err = (self.yaw_target - psi + math.pi) % (2 * math.pi) - math.pi
        self.yaw_target = psi + float(np.clip(err, -YAW_ERR_MAX, YAW_ERR_MAX))

        hit = False
        # Accumulateur : le temps simulé suit le temps réel quel que soit le nombre d'images par seconde
        self.t_acc += dt
        n_steps = int(self.t_acc / TIMESTEP)
        self.t_acc -= n_steps * TIMESTEP
        for _ in range(n_steps):
            self.data.ctrl[:] = self._control(v_des, self.yaw_target)
            mujoco.mj_step(self.model, self.data)
            if self.contact:
                self.v_int[:] = 0.0   # bloqué contre un obstacle : pas d'accumulation de l'intégrale
                hit = True

        # Drone renversé au sol depuis plus d'une seconde : on le redresse sur place
        up_z = self.data.xmat[self.bid][8]
        self.time_tilted = self.time_tilted + dt if (up_z < 0.5 and self.contact) else 0.0
        if self.time_tilted > 1.0:
            self.reset(self.position + np.array([0.0, 0.2, 0.0]), self.yaw_deg)
        return hit
