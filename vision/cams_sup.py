"""Caméras supplémentaires seules (ex. avant inclinée vers le haut / vers le bas), fixées au châssis.

Elles ne servent qu'à la carte. Une caméra seule ne donne pas de profondeur sur une image : on suit ses
coins (KLT) d'une image clé à l'image courante, et quand le drone s'est assez déplacé (base >= base_min),
on triangule chaque point à partir des deux poses (méthode du point milieu entre les deux rayons).
La pose de la caméra vient de la pose du corps (VIO, ou vraie avec --pose vraie) et de ses extrinsèques
corps/caméra, mesurées une fois à l'initialisation comme pour la paire frontale.
La VIO reste sur la paire frontale (caméras 0 et 1).
"""

import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from vision.corners import harris_function, detect_corners
from vision.klt import KLTCV
from vision.matching import to_gray


def trianguler_2_vues(p1, p2, K, R1, C1, R2, C2, ang_min_deg=3.0, d_max=40.0, err_max=2.0):
    """Point milieu entre les rayons des deux vues. p1, p2 : (N, 2) pixels ; R : caméra -> monde.
    Renvoie (P_w (N, 3), valide (N,)) : profondeur > 0 dans les deux vues, parallaxe >= ang_min_deg,
    distance <= d_max, erreur de reprojection <= err_max px dans les deux images."""
    Ki = np.linalg.inv(K)
    h1 = np.column_stack([p1, np.ones(len(p1))]) @ Ki.T
    h2 = np.column_stack([p2, np.ones(len(p2))]) @ Ki.T
    d1 = h1 @ R1.T
    d2 = h2 @ R2.T
    d1 /= np.linalg.norm(d1, axis=1, keepdims=True)
    d2 /= np.linalg.norm(d2, axis=1, keepdims=True)
    b = C2 - C1
    a12 = np.einsum("ij,ij->i", d1, d2)
    den = 1.0 - a12 ** 2
    bd1 = d1 @ b
    bd2 = d2 @ b
    with np.errstate(divide="ignore", invalid="ignore"):
        s = (bd1 - a12 * bd2) / den            # C1 + s d1 ~ C2 + t d2
        t = (a12 * bd1 - bd2) / den
    P = 0.5 * ((C1 + s[:, None] * d1) + (C2 + t[:, None] * d2))
    ok = np.isfinite(P).all(1) & (s > 0) & (t > 0)
    ok &= a12 < np.cos(np.radians(ang_min_deg))
    ok &= np.linalg.norm(P - C2, axis=1) <= d_max
    for p, R, C in ((p1, R1, C1), (p2, R2, C2)):
        Pc = (P - C) @ R                        # monde -> caméra
        with np.errstate(divide="ignore", invalid="ignore"):
            proj = Pc[:, :2] / Pc[:, 2:3] @ K[:2, :2].T + K[:2, 2]
        ok &= (Pc[:, 2] > 0) & (np.linalg.norm(proj - p, axis=1) <= err_max)
    return P, ok


class _Mono:
    def __init__(self, R_bc, p_bc):
        self.R_bc, self.p_bc = R_bc, p_bc
        self.kf = None            # (pts image clé (N, 2), R, C)
        self.prev = None          # (pyramide précédente, pts suivis (N, 2), masque vivant (N,))


class PairesSup:
    """Caméras sup seules : cams[2], cams[3], ... (nom gardé pour les scripts de test)."""

    def __init__(self, drone, cams, base_min=0.3, vivants_min=0.3):
        R_wb = drone.flight.rotation.copy()
        p_b = np.array(drone.position, float)
        self.base_min, self.vivants_min = base_min, vivants_min
        self.klt = KLTCV(max_res=0.15)
        self._pool, self._prep = None, None
        self.cams = []
        for c in cams[2:]:
            R_wc = np.column_stack([c.right, -c.up, c.forward]).astype(float)
            self.cams.append(_Mono(R_wb.T @ R_wc, R_wb.T @ (np.array(c.position, float) - p_b)))

    def __len__(self):
        return len(self.cams)

    def _coins(self, g, args):
        RhL = harris_function(g, version=args.harris)
        c = detect_corners(RhL, rel_thresh=args.seuil, max_n=args.max_coins, grid_cells=tuple(args.grille), cache=False)
        return np.asarray(c[:, :2], np.float64)

    def _preparer_cam(self, m, img, args):
        """Partie sans pose (OpenCV, libère le GIL) : niveaux de gris, pyramide, suivi KLT depuis l'image
        précédente, et coins de l'image (utilisés seulement si elle devient image clé)."""
        g = to_gray(img, version="numpy")   # pas de noyau numba parallèle hors du fil principal (couche workqueue)
        pyr = self.klt.pyramid(g)
        if m.kf is None:
            return pyr, None, None, self._coins(g, args)
        pyr0, p_prev, vivant = m.prev
        p_cur = p_prev.copy()
        idx = np.flatnonzero(vivant)
        if len(idx):
            p1, ok = self.klt.track_fb(pyr0, pyr, p_prev[idx], p_prev[idx])
            p_cur[idx] = p1
            vivant = vivant.copy()
            vivant[idx] = ok
        return pyr, p_cur, vivant, self._coins(g, args)

    def preparer(self, frame, args):
        """Lance en tâche de fond la partie sans pose de chaque caméra (à appeler juste après la capture,
        avant la VIO de la paire frontale) ; perceive() attend ces résultats. Résultats identiques au calcul en série."""
        if self._pool is None:
            self._pool = ThreadPoolExecutor(max_workers=max(len(self.cams), 1))
        self._prep = [self._pool.submit(self._preparer_cam, m, frame.images[2 + k], args) for k, m in enumerate(self.cams)]

    def perceive(self, frame, args, vio, use_true, t):
        """Renvoie [(P_w, C, n_triangulés), ...] par caméra. P_w est vide tant que la base depuis l'image clé
        est < base_min. Temps : attente de la partie sans pose (preparer) dans t[3], triangulation dans t[4]."""
        t0 = time.perf_counter()
        if self._prep is None:
            self.preparer(frame, args)
        prep = [f.result() for f in self._prep]
        self._prep = None
        t1 = time.perf_counter()
        t[3] += t1 - t0
        out = []
        K = frame.K
        for k, (m, (pyr, p_cur, vivant, pts)) in enumerate(zip(self.cams, prep)):
            i = 2 + k
            if use_true:
                R, C = frame.Rs[i], frame.Cs[i]
            else:
                R, C = vio.R @ m.R_bc, vio.p + vio.R @ m.p_bc
            P_out = np.empty((0, 3))
            nouvelle_cle = True
            if m.kf is not None:
                pts_kf, R0, C0 = m.kf
                nouvelle_cle = vivant.mean() < self.vivants_min if len(vivant) else True
                if np.linalg.norm(C - C0) >= self.base_min and vivant.any():
                    P, okt = trianguler_2_vues(pts_kf[vivant], p_cur[vivant], K, R0, C0, R, C)
                    P_out = P[okt]
                    nouvelle_cle = True
            if nouvelle_cle:
                m.kf, m.prev = (pts, R, C), (pyr, pts, np.ones(len(pts), bool))
            else:
                m.prev = (pyr, p_cur, vivant)
            out.append((P_out, C, len(P_out)))
        t[4] += time.perf_counter() - t1
        return out
