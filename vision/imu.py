"""IMU simulé et préintégration (style VINS-Fusion / Forster et al. 2017).

Repères
  monde : celui du simulateur (Y haut), gravité g_w = (0, -G, 0)
  corps : axes du châssis (avant, gauche, haut), R_wb = drone.flight.rotation (corps -> monde)

IMU simulé (SimIMU) : un échantillon par pas de commande (--dt-phys), calculé sur l'état MuJoCo
  gyro  w  = Log(R_0^T R_1) / dt                      vitesse angulaire moyenne (repère corps)
  accéléro f = R_0^T ((v_1 - v_0) / dt - g_w)          force spécifique moyenne (repère corps)
Avec ces définitions, la préintégration ci-dessous redonne exactement la rotation et la vitesse
vraies (la position à la discrétisation près). Bruit blanc et biais en marche aléatoire sont
prévus mais à 0 par défaut (IMU parfait).
"""

import numpy as np

from mujoco_flight import G

G_W = np.array([0.0, -G, 0.0])


# ---------------------------------------------------------------------------
# SO(3)
# ---------------------------------------------------------------------------

def skew(v):
    """(..., 3) -> (..., 3, 3) matrice du produit vectoriel."""
    v = np.asarray(v, float)
    S = np.zeros(v.shape[:-1] + (3, 3))
    S[..., 0, 1], S[..., 0, 2] = -v[..., 2], v[..., 1]
    S[..., 1, 0], S[..., 1, 2] = v[..., 2], -v[..., 0]
    S[..., 2, 0], S[..., 2, 1] = -v[..., 1], v[..., 0]
    return S


def so3_exp(w):
    """Rodrigues, (..., 3) -> (..., 3, 3)."""
    w = np.asarray(w, float)
    th = np.linalg.norm(w, axis=-1)[..., None, None]
    K = skew(w)
    small = th < 1e-8
    ths = np.where(small, 1.0, th)
    A = np.where(small, 1.0 - th ** 2 / 6, np.sin(ths) / ths)
    B = np.where(small, 0.5 - th ** 2 / 24, (1 - np.cos(ths)) / ths ** 2)
    return np.eye(3) + A * K + B * (K @ K)


def so3_log(R):
    """(..., 3, 3) -> (..., 3)."""
    R = np.asarray(R, float)
    c = np.clip((np.trace(R, axis1=-2, axis2=-1) - 1) / 2, -1.0, 1.0)
    th = np.arccos(c)
    v = np.stack([R[..., 2, 1] - R[..., 1, 2], R[..., 0, 2] - R[..., 2, 0], R[..., 1, 0] - R[..., 0, 1]], -1)
    s = np.sin(th)
    k = np.where(th < 1e-6, 0.5 + th ** 2 / 12, th / (2 * np.where(th < 1e-6, 1.0, s)))
    return k[..., None] * v


def right_jacobian(w):
    """Jacobienne à droite de SO(3), (3,) -> (3, 3)."""
    th = np.linalg.norm(w)
    K = skew(w)
    if th < 1e-6:
        return np.eye(3) - 0.5 * K
    return np.eye(3) - (1 - np.cos(th)) / th ** 2 * K + (th - np.sin(th)) / th ** 3 * (K @ K)


def orthonormalize(R):
    U, _, Vt = np.linalg.svd(R)
    return U @ Vt


# ---------------------------------------------------------------------------
# IMU simulé
# ---------------------------------------------------------------------------

class SimIMU:
    """Mesures IMU tirées de l'état MuJoCo du drone, une par appel à sample()."""

    def __init__(self, drone, gyro_noise=0.0, acc_noise=0.0, gyro_walk=0.0, acc_walk=0.0, seed=0, rel_noise=0.0):
        """Bruits en densité spectrale : gyro (rad/s/sqrt(Hz)), accéléro (m/s²/sqrt(Hz)),
        marches aléatoires des biais (rad/s²/sqrt(Hz), m/s³/sqrt(Hz)). Tout à 0 : IMU parfait.
        rel_noise : bruit blanc gaussien relatif (0.05 = 5 %) ; écart-type par axe = rel_noise * amplitude
        RMS du signal (gyro et accéléro séparément). Remplace gyro_noise/acc_noise s'il est non nul ;
        les densités équivalentes courantes sont lues dans self.sg / self.sa."""
        self.drone = drone
        self.rel, self.ms_w, self.ms_f = rel_noise, None, None
        self.sg, self.sa, self.wg, self.wa = gyro_noise, acc_noise, gyro_walk, acc_walk
        self.rng = np.random.default_rng(seed)
        self.bg, self.ba = np.zeros(3), np.zeros(3)
        self.R0 = drone.flight.rotation.copy()
        self.v0 = np.array(drone.velocity, float)

    def sample(self, dt):
        """À appeler juste après drone.update(dt). Retourne (gyro, acc) en repère corps."""
        R1 = self.drone.flight.rotation.copy()
        v1 = np.array(self.drone.velocity, float)
        w = so3_log(self.R0.T @ R1) / dt
        f = self.R0.T @ ((v1 - self.v0) / dt - G_W)
        self.R0, self.v0 = R1, v1
        if self.wg or self.wa:
            self.bg += self.wg * np.sqrt(dt) * self.rng.standard_normal(3)
            self.ba += self.wa * np.sqrt(dt) * self.rng.standard_normal(3)
        if self.rel:
            # bruit relatif : écart-type = rel * RMS courant (moyenne glissante) de la norme du signal vrai
            a = 0.02
            self.ms_w = (1 - a) * self.ms_w + a * float(w @ w) if self.ms_w is not None else float(w @ w)
            self.ms_f = (1 - a) * self.ms_f + a * float(f @ f) if self.ms_f is not None else float(f @ f)
            std_w = self.rel * np.sqrt(self.ms_w / 3)   # par axe
            std_f = self.rel * np.sqrt(self.ms_f / 3)
            self.sg, self.sa = std_w * np.sqrt(dt), std_f * np.sqrt(dt)   # densités équivalentes
        if self.sg or self.sa:
            w = w + self.sg / np.sqrt(dt) * self.rng.standard_normal(3)
            f = f + self.sa / np.sqrt(dt) * self.rng.standard_normal(3)
        return w + self.bg, f + self.ba


# ---------------------------------------------------------------------------
# Préintégration (biais supposés nuls / connus, pas de correction au premier ordre)
# ---------------------------------------------------------------------------

class Preintegration:
    """ΔR, Δv, Δp entre deux images et covariance 9x9 (ordre : rotation, vitesse, position).

    sigma_g, sigma_a : densités de bruit utilisées pour la covariance. Même avec un IMU parfait on
    garde un plancher (erreur de discrétisation), sinon le poids IMU serait infini.
    """

    def __init__(self, sigma_g=2e-3, sigma_a=2e-2):
        self.sg2, self.sa2 = sigma_g ** 2, sigma_a ** 2
        self.dR, self.dv, self.dp = np.eye(3), np.zeros(3), np.zeros(3)
        self.dt = 0.0
        self.cov = np.zeros((9, 9))
        self.samples = []

    def add(self, dt, w, f):
        self.samples.append((dt, w, f))
        dR, dv = self.dR, self.dv
        dRk = so3_exp(w * dt)
        A = np.eye(9)
        A[0:3, 0:3] = dRk.T
        A[3:6, 0:3] = -dR @ skew(f) * dt
        A[6:9, 0:3] = -0.5 * dR @ skew(f) * dt ** 2
        A[6:9, 3:6] = np.eye(3) * dt
        Bg = np.zeros((9, 3)); Bg[0:3] = right_jacobian(w * dt) * dt
        Ba = np.zeros((9, 3)); Ba[3:6] = dR * dt; Ba[6:9] = 0.5 * dR * dt ** 2
        self.cov = A @ self.cov @ A.T + Bg @ Bg.T * (self.sg2 / dt) + Ba @ Ba.T * (self.sa2 / dt)
        self.dp = self.dp + dv * dt + 0.5 * dR @ f * dt ** 2
        self.dv = dv + dR @ f * dt
        self.dR = orthonormalize(dR @ dRk)
        self.dt += dt

    def extend(self, other):
        """Concatène une autre préintégration (suppression d'une image intermédiaire)."""
        for s in other.samples:
            self.add(*s)

    def predict(self, R, p, v):
        """État au bout de l'intervalle à partir de (R, p, v) au début."""
        T = self.dt
        return (orthonormalize(R @ self.dR),
                p + v * T + 0.5 * G_W * T ** 2 + R @ self.dp,
                v + G_W * T + R @ self.dv)

    def residual(self, Ri, pi, vi, Rj, pj, vj):
        """Résidu 9 (rotation, vitesse, position) et jacobiennes 9x9 par rapport à
        (θi, pi, vi) et (θj, pj, vj) ; perturbations R <- R Exp(δθ), p <- p + δp, v <- v + δv."""
        T = self.dt
        dv_w = vj - vi - G_W * T
        dp_w = pj - pi - vi * T - 0.5 * G_W * T ** 2
        rR = so3_log(self.dR.T @ Ri.T @ Rj)
        rv = Ri.T @ dv_w - self.dv
        rp = Ri.T @ dp_w - self.dp
        Jr_inv = np.eye(3) + 0.5 * skew(rR)
        Ji, Jj = np.zeros((9, 9)), np.zeros((9, 9))
        Ji[0:3, 0:3] = -Jr_inv @ Rj.T @ Ri
        Jj[0:3, 0:3] = Jr_inv
        Ji[3:6, 0:3] = skew(Ri.T @ dv_w)
        Ji[3:6, 6:9] = -Ri.T
        Jj[3:6, 6:9] = Ri.T
        Ji[6:9, 0:3] = skew(Ri.T @ dp_w)
        Ji[6:9, 3:6] = -Ri.T
        Ji[6:9, 6:9] = -Ri.T * T
        Jj[6:9, 3:6] = Ri.T
        return np.concatenate([rR, rv, rp]), Ji, Jj

    def sqrt_info(self):
        """W tel que ||W r||² = r^T Cov^-1 r (mis en cache tant qu'aucun échantillon n'est ajouté)."""
        n = len(self.samples)
        if getattr(self, "_W", None) is None or self._W[0] != n:
            C = self.cov + np.eye(9) * 1e-12
            self._W = (n, np.linalg.cholesky(np.linalg.inv(C)).T)
        return self._W[1]
