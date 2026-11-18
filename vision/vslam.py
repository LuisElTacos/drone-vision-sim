"""Squelette à trous du front-end VIO (FrontEnd). À compléter, puis comparer à vslam.py (référence).
Ordre conseillé : A (RANSAC F) -> B (suivi) -> C (renouvellement) -> D (stéréo) -> E (assemblage).
Chaque brique se teste seule : A sur des points synthétiques, B/C/D sur une paire d'images du simulateur.
"""

import time
import cv2
import numpy as np
from vision.corners import harris_function, detect_corners
from vision.klt import KLT
from vision.matching import _match_corners, _window_norms
from vision.imu import Preintegration, so3_exp, skew, orthonormalize
from scipy.optimize import least_squares
from numba import njit

# ---- noyaux numba du back-end (mêmes formules que la version numpy d'origine) ----------------------------
@njit(cache=True)
def _reproj_nb(Rs, ps, P, F, Li, C, UV, p_bc, R_bc, f, cx, cy, sig):
    n = len(F)
    r = np.empty((n, 2))
    Pb = np.empty((n, 3))
    Pc = np.empty((n, 3))
    Zs = np.empty(n)
    valid = np.empty(n, np.bool_)
    for k in range(n):
        Rf = Rs[F[k]]
        d = P[Li[k]] - ps[F[k]]
        for i in range(3):
            Pb[k, i] = Rf[0, i] * d[0] + Rf[1, i] * d[1] + Rf[2, i] * d[2]
        q = Pb[k] - p_bc[C[k]]
        for j in range(3):
            Pc[k, j] = q[0] * R_bc[0, j] + q[1] * R_bc[1, j] + q[2] * R_bc[2, j]
        Z = Pc[k, 2]
        valid[k] = Z > 0.05
        Zs[k] = Z if Z > 0.05 else 0.05
        r[k, 0] = (f * Pc[k, 0] / Zs[k] + cx - UV[k, 0]) / sig
        r[k, 1] = (f * Pc[k, 1] / Zs[k] + cy - UV[k, 1]) / sig
    return r, Pb, Pc, Zs, valid


@njit(cache=True)
def _linearize_nb(Rs, F, Li, r, Pb, Pc, Z, valid, R_bc, f, sig, huber, nF, nL):
    """Accumulation des blocs normaux de reprojection : Hll (nL,3,3), gl (nL,3), Hpp6 (nF-1,6,6), gp6 (nF-1,6), Hpl6 (nF-1,6,nL,3)."""
    Hll = np.zeros((nL, 3, 3))
    gl = np.zeros((nL, 3))
    Hpp6 = np.zeros((nF - 1, 6, 6))
    gp6 = np.zeros((nF - 1, 6))
    Hpl6 = np.zeros((nF - 1, 6, nL, 3))
    RbT = np.ascontiguousarray(R_bc.T)
    for k in range(len(F)):
        e = np.sqrt(r[k, 0] ** 2 + r[k, 1] ** 2)
        w = 1.0 if e < huber else huber / max(e, 1e-9)
        if not valid[k]:
            w = 0.0
        fz = f / Z[k] / sig
        Jp = np.zeros((2, 3))
        Jp[0, 0] = fz
        Jp[0, 2] = -fz * Pc[k, 0] / Z[k]
        Jp[1, 1] = fz
        Jp[1, 2] = -fz * Pc[k, 1] / Z[k]
        JR = Jp @ RbT
        A = JR @ np.ascontiguousarray(Rs[F[k]].T)
        sk = np.zeros((3, 3))
        sk[0, 1] = -Pb[k, 2]; sk[0, 2] = Pb[k, 1]
        sk[1, 0] = Pb[k, 2];  sk[1, 2] = -Pb[k, 0]
        sk[2, 0] = -Pb[k, 1]; sk[2, 1] = Pb[k, 0]
        Jth = JR @ sk
        Jx = np.empty((2, 6))
        Jx[:, 0:3] = Jth
        Jx[:, 3:6] = -A
        wr = w * r[k]
        l = Li[k]
        Hll[l] += w * (A.T @ A)
        gl[l] += A.T @ wr
        if F[k] > 0:
            fi = F[k] - 1
            Hpp6[fi] += w * (Jx.T @ Jx)
            gp6[fi] += Jx.T @ wr
            Hpl6[fi, :, l, :] += w * (Jx.T @ A)
    return Hll, gl, Hpp6, gp6, Hpl6


# ---- variante creuse du complément de Schur (schur="creux") -----------------------------------------------
# Hpl n'est non nul que pour les couples (image, point) réellement observés : au lieu de la matrice dense
# (9(nF-1), nL, 3), on garde un bloc 6x3 par couple, et on somme Hpl_l Hll_l^-1 Hpl_l^T point par point.
def _paires(F, Li, nF, nL):
    """Couples (image fi = F-1, point l) observés avec F > 0, triés par point. Retourne pf (npair,), pl (npair,),
    lptr (nL+1,) (couples du point l : pf[lptr[l]:lptr[l+1]]) et pidx (nobs,) (couple de chaque observation, -1 si F = 0)."""
    m = F > 0
    key = Li.astype(np.int64) * nF + F.astype(np.int64)
    uk, inv = np.unique(key[m], return_inverse=True)
    pl, pf = uk // nF, uk % nF - 1
    pidx = np.full(len(F), -1, np.int64)
    pidx[m] = inv
    lptr = np.zeros(nL + 1, np.int64)
    np.add.at(lptr, pl + 1, 1)
    return pf.astype(np.int64), pl.astype(np.int64), np.cumsum(lptr), pidx


@njit(cache=True)
def _linearize_sp_nb(Rs, F, Li, pidx, npair, r, Pb, Pc, Z, valid, R_bc, f, sig, huber, nF, nL):
    """Même calcul que _linearize_nb, mais Hpl rangé par couple (image, point) : Hplp (npair,6,3).
    Petits produits écrits en boucles scalaires (le @ de numba appelle BLAS, coûteux sur des 2x3)."""
    Hll = np.zeros((nL, 3, 3))
    gl = np.zeros((nL, 3))
    Hpp6 = np.zeros((nF - 1, 6, 6))
    gp6 = np.zeros((nF - 1, 6))
    Hplp = np.zeros((npair, 6, 3))
    Jp = np.zeros((2, 3))
    JR = np.empty((2, 3))
    A = np.empty((2, 3))
    Jx = np.empty((2, 6))
    for k in range(len(F)):
        e = np.sqrt(r[k, 0] ** 2 + r[k, 1] ** 2)
        w = 1.0 if e < huber else huber / max(e, 1e-9)
        if not valid[k]:
            w = 0.0
        fz = f / Z[k] / sig
        Jp[0, 0] = fz
        Jp[0, 2] = -fz * Pc[k, 0] / Z[k]
        Jp[1, 1] = fz
        Jp[1, 2] = -fz * Pc[k, 1] / Z[k]
        Rf = Rs[F[k]]
        for a in range(2):
            for b in range(3):                      # JR = Jp R_bc^T
                JR[a, b] = Jp[a, 0] * R_bc[b, 0] + Jp[a, 1] * R_bc[b, 1] + Jp[a, 2] * R_bc[b, 2]
            for b in range(3):                      # A = JR R^T
                A[a, b] = JR[a, 0] * Rf[b, 0] + JR[a, 1] * Rf[b, 1] + JR[a, 2] * Rf[b, 2]
            # Jth = JR [Pb]x
            x, y, z = Pb[k, 0], Pb[k, 1], Pb[k, 2]
            Jx[a, 0] = JR[a, 1] * z - JR[a, 2] * y
            Jx[a, 1] = -JR[a, 0] * z + JR[a, 2] * x
            Jx[a, 2] = JR[a, 0] * y - JR[a, 1] * x
            Jx[a, 3] = -A[a, 0]
            Jx[a, 4] = -A[a, 1]
            Jx[a, 5] = -A[a, 2]
        wr0, wr1 = w * r[k, 0], w * r[k, 1]
        l = Li[k]
        for i in range(3):
            gl[l, i] += A[0, i] * wr0 + A[1, i] * wr1
            for j in range(3):
                Hll[l, i, j] += w * (A[0, i] * A[0, j] + A[1, i] * A[1, j])
        if F[k] > 0:
            fi = F[k] - 1
            p = pidx[k]
            for i in range(6):
                gp6[fi, i] += Jx[0, i] * wr0 + Jx[1, i] * wr1
                for j in range(6):
                    Hpp6[fi, i, j] += w * (Jx[0, i] * Jx[0, j] + Jx[1, i] * Jx[1, j])
                for j in range(3):
                    Hplp[p, i, j] += w * (Jx[0, i] * A[0, j] + Jx[1, i] * A[1, j])
    return Hll, gl, Hpp6, gp6, Hplp


@njit(cache=True)
def _schur_sp_nb(Hll_inv, Hplp, gl, pf, lptr, npo):
    """Somme sur les points de Hpl_l Hll_l^-1 Hpl_l^T (npo,npo) et de Hpl_l Hll_l^-1 gl_l (npo,)
    (seules les lignes 9fi..9fi+6 d'une image fi sont non nulles)."""
    SS = np.zeros((npo, npo))
    bb = np.zeros(npo)
    nL = len(lptr) - 1
    B = np.empty((6, 3))
    for l in range(nL):
        a0, a1 = lptr[l], lptr[l + 1]
        for a in range(a0, a1):
            for i in range(6):                      # B = Hpl_a Hll^-1
                for j in range(3):
                    B[i, j] = Hplp[a, i, 0] * Hll_inv[l, 0, j] + Hplp[a, i, 1] * Hll_inv[l, 1, j] + Hplp[a, i, 2] * Hll_inv[l, 2, j]
            sa = 9 * pf[a]
            for i in range(6):
                bb[sa + i] += B[i, 0] * gl[l, 0] + B[i, 1] * gl[l, 1] + B[i, 2] * gl[l, 2]
            for b in range(a0, a1):
                sb = 9 * pf[b]
                for i in range(6):
                    for j in range(6):
                        SS[sa + i, sb + j] += B[i, 0] * Hplp[b, j, 0] + B[i, 1] * Hplp[b, j, 1] + B[i, 2] * Hplp[b, j, 2]
    return SS, bb


@njit(cache=True)
def _backsub_sp_nb(Hll_inv, Hplp, gl, pf, lptr, dx):
    """dl_l = Hll_l^-1 (-gl_l - Hpl_l^T dx)."""
    nL = len(lptr) - 1
    dl = np.empty((nL, 3))
    t = np.empty(3)
    for l in range(nL):
        for j in range(3):
            t[j] = -gl[l, j]
        for a in range(lptr[l], lptr[l + 1]):
            sa = 9 * pf[a]
            for j in range(3):
                for i in range(6):
                    t[j] -= Hplp[a, i, j] * dx[sa + i]
        for i in range(3):
            dl[l, i] = Hll_inv[l, i, 0] * t[0] + Hll_inv[l, i, 1] * t[1] + Hll_inv[l, i, 2] * t[2]
    return dl



# ---- A. Matrice fondamentale + RANSAC ---------------------------------------
def normalize(p):
    """Hartley : translater au centroïde, échelle pour une distance moyenne = sqrt(2).
    Retourne (points normalisés (N,2), T (3,3))."""
    c = np.mean(p, axis=0)
    d = p - c
    s = np.sqrt(2) / np.mean(np.sqrt((d**2).sum(axis=1)))
    T = np.array([[s, 0, -s * c[0]], [0, s, -s * c[1]], [0, 0, 1]])
    return d * s, T


def eight_point(q0, q1):
    """F (3,3) de rang 2 telle que x1^T F x0 = 0, à partir de >= 8 points normalisés.
    Indice : une ligne de A par point (9 termes), SVD, dernier vecteur singulier, puis forcer S[2]=0.
    """
    A = np.column_stack(
        [
            q1[:, 0:1] * q0,
            q1[:, 0:1],
            q1[:, 1:2] * q0,
            q1[:, 1:2],
            q0,
            np.ones((len(q0), 1)),
        ]
    )
    _, _, Vt = np.linalg.svd(A)
    F = Vt[-1].reshape(3, 3)
    U, S, Vt2 = np.linalg.svd(F)
    S[2] = 0
    return U @ np.diag(S) @ Vt2


def sampson(F, p0, p1):
    """Distance de Sampson au carré (N,) : (x1^T F x0)^2 / ((Fx0)_1^2+(Fx0)_2^2+(F^T x1)_1^2+(F^T x1)_2^2)."""
    x1 = np.column_stack([p1, np.ones(len(p0))])
    x0 = np.column_stack([p0, np.ones(len(p0))])
    Fx0 = x0 @ F.T  # (N,3) ligne i = F @ x0[i]
    Ftx1 = x1 @ F  # (N,3) ligne i = F.T @ x1[i]
    num = (x1 * Fx0).sum(axis=1) ** 2
    den = Fx0[:, 0] ** 2 + Fx0[:, 1] ** 2 + Ftx1[:, 0] ** 2 + Ftx1[:, 1] ** 2
    return num / np.maximum(den, 1e-12)


@njit(cache=True)
def _sampson_best_nb(F, p0, p1, thr2):
    """Pour chaque hypothèse F[k], inliers = distance de Sampson^2 < thr2 ; retourne le masque de la première
    hypothèse au plus grand consensus (comme inl[inl.sum(1).argmax()])."""
    K, n = F.shape[0], p0.shape[0]
    best, kbest = -1, 0
    for k in range(K):
        f = F[k]
        c = 0
        for m in range(n):
            x0, y0, x1, y1 = p0[m, 0], p0[m, 1], p1[m, 0], p1[m, 1]
            a0 = f[0, 0] * x0 + f[0, 1] * y0 + f[0, 2]
            a1 = f[1, 0] * x0 + f[1, 1] * y0 + f[1, 2]
            a2 = f[2, 0] * x0 + f[2, 1] * y0 + f[2, 2]
            b0 = f[0, 0] * x1 + f[1, 0] * y1 + f[2, 0]
            b1 = f[0, 1] * x1 + f[1, 1] * y1 + f[2, 1]
            num = (x1 * a0 + y1 * a1 + a2) ** 2
            den = max(a0 * a0 + a1 * a1 + b0 * b0 + b1 * b1, 1e-12)
            if num / den < thr2:
                c += 1
        if c > best:
            best, kbest = c, k
    f = F[kbest]
    out = np.empty(n, np.bool_)
    for m in range(n):
        x0, y0, x1, y1 = p0[m, 0], p0[m, 1], p1[m, 0], p1[m, 1]
        a0 = f[0, 0] * x0 + f[0, 1] * y0 + f[0, 2]
        a1 = f[1, 0] * x0 + f[1, 1] * y0 + f[1, 2]
        a2 = f[2, 0] * x0 + f[2, 1] * y0 + f[2, 2]
        b0 = f[0, 0] * x1 + f[1, 0] * y1 + f[2, 0]
        b1 = f[0, 1] * x1 + f[1, 1] * y1 + f[2, 1]
        out[m] = (x1 * a0 + y1 * a1 + a2) ** 2 / max(a0 * a0 + a1 * a1 + b0 * b0 + b1 * b1, 1e-12) < thr2
    return out


def fundamental_ransac(p0, p1, threshold=1.0, iters=200, rng=None, version="numba"):
    """version : "einsum" (version d'origine) ou "numba" (même test de Sampson, sans tableaux (iters, n, 3))."""
    n = len(p0)
    if n < 12:
        return np.ones(n, bool)
    rng = np.random.default_rng(0) if rng is None else rng
    q0, T0 = normalize(p0)
    q1, T1 = normalize(p1)
    A = np.column_stack(
        [q1[:, 0:1] * q0, q1[:, 0:1], q1[:, 1:2] * q0, q1[:, 1:2], q0, np.ones((n, 1))]
    )  # (n,9)
    idx = np.argsort(rng.random((iters, n)), axis=1)[
        :, :8
    ]  # (iters,8) : 8 indices distincts par hypothèse
    _, _, Vt = np.linalg.svd(A[idx])  # A[idx] : (iters,8,9) ; Vt : (iters,9,9)
    F = Vt[:, -1].reshape(iters, 3, 3)  # dernière ligne de chaque Vt
    U, S, Vt2 = np.linalg.svd(F)  # SVD par lots, S : (iters,3)
    S[:, 2] = 0
    F = U @ (S[:, :, None] * Vt2)  # = U diag(S) Vt2 par hypothèse
    F = T1.T[None] @ F @ T0[None]  # dénormalisation, (iters,3,3)
    if version == "numba":
        return _sampson_best_nb(np.ascontiguousarray(F), np.ascontiguousarray(p0, dtype=float),
                                np.ascontiguousarray(p1, dtype=float), float(threshold) ** 2)
    x0 = np.column_stack([p0, np.ones(n)])
    x1 = np.column_stack([p1, np.ones(n)])
    Fx0 = np.einsum("kij,nj->kni", F, x0)  # (iters,n,3) = F_k @ x0_n
    Ftx1 = np.einsum("kji,nj->kni", F, x1)  # (iters,n,3) = F_k.T @ x1_n
    num = np.einsum("ni,kni->kn", x1, Fx0) ** 2  # (iters,n)
    den = Fx0[..., 0] ** 2 + Fx0[..., 1] ** 2 + Ftx1[..., 0] ** 2 + Ftx1[..., 1] ** 2
    inl = num / np.maximum(den, 1e-12) < threshold**2  # (iters,n)
    return inl[inl.sum(1).argmax()]  # hypothèse au plus grand consensus


# ---- B-E. Front-end ----------------------------------------------------------
class FrontEnd:
    def __init__(
        self,
        K,
        n_max=150,
        min_dist=20,
        klt=None,
        zncc=0.8,
        ratio=0.8,
        d_max=160,
        harris_seuil=1e-5,
        grille=(6, 8),
    ):
        self.K, self.Kinv = K, np.linalg.inv(K)
        self.n_max, self.min_dist = n_max, min_dist
        self.klt = klt or KLT()
        self.zncc, self.ratio, self.d_max = zncc, ratio, d_max
        self.harris_seuil, self.grille = harris_seuil, grille
        self.prev_pyr = None
        self.pts = np.empty((0, 2))
        self.ids = np.empty(0, np.int64)
        self.next_id = 0
        self.norms = None

    def predict(self, R_cc):
        """B1. Prédiction des points dans l'image t par l'homographie de rotation H = K R_cc K^-1
        (le gyro donne R_cc, rotation caméra t-1 -> t). Retourne (N,2). Si R_cc est None : self.pts.
        """
        # TODO : coordonnées homogènes, multiplier par H, diviser par la 3e composante
        if R_cc is None:
            return self.pts
        x = np.column_stack([self.pts, np.ones(len(self.pts))])
        x2 = x @ (self.K @ R_cc @ self.Kinv).T
        return x2[:, :2] / x2[:, 2:3]

    def track(self, pyr, R_cc):
        """B2. KLT t-1 -> t avec self.klt.track_fb(prev_pyr, pyr, pts, pred) -> (p1, ok), puis RANSAC F sur
        les ok. Met à jour self.pts et self.ids (garder seulement les survivants)."""
        pred = self.predict(R_cc)
        p1, ok = self.klt.track_fb(self.prev_pyr, pyr, self.pts, pred)
        i = np.nonzero(ok)[0]
        bons = fundamental_ransac(self.pts[i], p1[i])
        ok[i[~bons]] = False
        self.pts = p1[ok]
        self.ids = self.ids[ok]

    def renew(self, grayL, harris=None):
        """C. Si len(pts) < n_max : coins de Harris (detect_corners), triés par score, hors bordure klt.win,
        acceptés seulement s'ils sont à > min_dist de tous les points déjà présents (masque uint8 + cv2.circle).
        Nouveaux ids = next_id..., next_id += n."""
        if len(self.pts) > self.n_max:
            return None
        corners = detect_corners(
            harris if harris is not None else harris_function(grayL),
            rel_thresh=self.harris_seuil,
            max_n=4 * self.n_max,
            border=self.klt.win,
            grid_cells=self.grille,
        )
        c = corners[np.argsort(-corners[:, 2])]
        mask = np.full((grayL.shape), 255, np.uint8)
        for x, y in self.pts:
            cv2.circle(mask, (int(round(x)), int(round(y))), self.min_dist, 0, -1)
        new = []
        for x, y, _ in c:
            if len(new) + len(self.pts) >= self.n_max:
                break
            xi, yi = int(x), int(y)
            if mask[yi, xi]:
                new.append((x, y))
                cv2.circle(mask, (xi, yi), self.min_dist, 0, -1)
        if new:
            new = np.array(new, float)
            self.pts = np.vstack([self.pts, new])
            self.ids = np.concatenate(
                [self.ids, np.arange(self.next_id, self.next_id + len(new))]
            )
            self.next_id += len(new)

    def stereo(self, grayL, grayR):
        """D. Pour chaque point suivi, chercher son correspondant sur la même ligne dans grayR
        (_match_corners, ZNCC, ratio, contrôle gauche-droite). uvR = (u - d, v) ou nan si pas d'appariement.
        """
        uvR = np.full_like(self.pts, np.nan)
        self.norms = None  # (normes gauche, droite, half), réutilisables par l'appariement de la même paire d'images
        if len(self.pts):
            L = np.ascontiguousarray(grayL, np.float32)
            R = np.ascontiguousarray(grayR, np.float32)
            half = 5
            yc = np.round(self.pts[:, 1]).astype(np.int64)
            self.norms = (_window_norms(L, half), _window_norms(R, half), half)
            out, keep = _match_corners(
                L,
                R,
                self.norms[0],
                self.norms[1],
                np.ascontiguousarray(self.pts[:, 0], np.float64),
                yc,
                1,
                self.d_max,
                half,
                self.ratio,
                2.0 * (1.0 - self.zncc),
                True,
            )
            keep = np.asarray(keep, bool)
            d = out[keep, 3]
            uvR[keep, 0] = self.pts[keep, 0] - d
            uvR[keep, 1] = self.pts[keep, 1]
        return uvR

    def process(self, grayL, grayR, R_cc=None, harris=None):
        """E. pyr = self.klt.pyramid(grayL) ; si prev_pyr existe : track ; renew ; prev_pyr = pyr ; stereo.
        Retourne ids, uvL, uvR."""
        pyr = self.klt.pyramid(grayL)
        if self.prev_pyr is not None and len(self.pts):
            self.track(pyr, R_cc)
        self.renew(grayL, harris)
        self.prev_pyr = pyr
        uvR = self.stereo(grayL, grayR)
        return self.ids.copy(), self.pts.copy(), uvR


class _Frame:
    __slots__ = ("t", "R", "p", "v", "kf", "ids", "uvL", "uvR", "preint")

    def __init__(self, t, R, p, v, ids, uvL, uvR, preint):
        self.t, self.R, self.p, self.v = t, R, p, v
        self.ids, self.uvL, self.uvR, self.preint = ids, uvL, uvR, preint
        self.kf = False


class VIO:
    def __init__(
        self,
        K,
        R_bc,
        p_bcL,
        p_bcR,
        R0,
        p0,
        v0,
        t0=0.0,
        fenetre=10,
        kf_parallaxe=75,
        kf_suivi=0.1,
        kf_dt_max=1.0,
        sigma_px=1.0,
        huber=2.0,
        iters=5,
        prof_max=25.0,
        sigma_g=2e-3,
        sigma_a=2e-2,
        front=None,
        schur="creux",
        obs="numpy",
    ):
        # schur : "dense" (version d'origine, Hpl dense (9(nF-1), nL, 3)) ou "creux" (un bloc 6x3 par couple
        # (image, point) observé, complément de Schur en numba ; mêmes équations, arrondis flottants près)
        self.schur = schur
        self.obs = obs  # "python" (version d'origine de _observations) ou "numpy" (vectorisée, mêmes sorties)
        self.K, self.R_bc = K, R_bc
        self.p_bc = np.stack([p_bcL, p_bcR])
        self.f, self.cx, self.cy = K[0, 0], K[0, 2], K[1, 2]
        self.N, self.kf_par, self.kf_suivi, self.kf_dt = (
            fenetre,
            kf_parallaxe,
            kf_suivi,
            kf_dt_max,
        )
        self.sig, self.huber, self.iters, self.prof_max = (
            sigma_px,
            huber,
            iters,
            prof_max,
        )
        self.sg, self.sa = sigma_g, sigma_a
        self.front = front or FrontEnd(K)
        self.frames = []
        self.lm = {}  # id -> point 3D monde
        self.bad = set()  # ids rejetés
        self.state0 = (
            orthonormalize(np.asarray(R0, float)),
            np.array(p0, float),
            np.array(v0, float),
            t0,
        )
        self.pre = Preintegration(self.sg, self.sa)
        self.R, self.p, self.v = self.state0[:3]

    def _init_landmarks(self, fr):
        # TODO étape 2 : pour chaque point de fr apparié en stéréo et absent de self.lm / self.bad :
        # d = uL - uR (ignorer d < 1 px) ; Z = f * baseline / d (ignorer Z > prof_max) ;
        # point caméra = Z * ((u-cx)/f, (v-cy)/f, 1) ; point monde = fr.R @ (R_bc @ Pc + p_bc[0]) + fr.p ;
        # stocker dans self.lm[id].
        ok = np.isfinite(
            fr.uvR[:, 0]
        )  # enlever les nan qui correspondent aux points non appariés
        for i in np.nonzero(ok)[0]:
            point_id = int(fr.ids[i])
            if point_id in self.lm or point_id in self.bad:
                continue
            d = fr.uvL[i, 0] - fr.uvR[i, 0]
            if d < 1.0:
                continue
            Z = self.f * np.linalg.norm(self.p_bc[1] - self.p_bc[0]) / d
            if Z > self.prof_max:
                continue
            Pc = Z * np.array(
                [
                    (fr.uvL[i, 0] - self.cx) / self.f,
                    (fr.uvL[i, 1] - self.cy) / self.f,
                    1.0,
                ]
            )
            self.lm[point_id] = fr.R @ (self.R_bc @ Pc + self.p_bc[0]) + fr.p

    # ---- Squelette du back-end (étapes 1 à 7) : à compléter -----------------------------------

    def imu(self, dt, w, f):
        """Un échantillon IMU (gyro w, accéléromètre f) : l'ajouter à self.pre, puis propager l'état courant
        (R, p, v) à partir de la dernière image (ou de l'état initial)."""
        self.pre.add(dt, w, f)
        R0, p0, v0 = (
            (self.frames[-1].R, self.frames[-1].p, self.frames[-1].v)
            if self.frames
            else self.state0[:3]
        )
        self.R, self.p, self.v = self.pre.predict(R0, p0, v0)

    def cam_pose(self, R=None, p=None, cam=0):
        """ETAPE 1. Rotation caméra -> monde (R @ R_bc) et centre optique de la caméra cam (p + R @ p_bc[cam])."""
        R = self.R if R is None else R
        p = self.p if p is None else p
        return R @ self.R_bc, p + R @ self.p_bc[cam]

    def _reproj(self, Rs, ps, P, F, Li, C, UV):
        """ETAPE 3. Observation k : image F[k], point Li[k] (P : points 3D (nL,3)), caméra C[k] (0/1), pixel UV[k].
        Chaîne : monde -> corps (R^T (P - p)) -> caméra (R_bc^T (Pb - p_bc[C])) -> pixel (f X/Z + cx, f Y/Z + cy).
        Retourne r = (uv_prédit - UV)/sigma_px, plus ce dont on a besoin ensuite (Pb, Pc, Z borné, masque Z > 0.05).
        """
        c = np.ascontiguousarray
        return _reproj_nb(c(Rs, dtype=float), c(ps, dtype=float), c(P, dtype=float), c(F, dtype=np.int64), c(Li, dtype=np.int64),
                          c(C, dtype=np.int64), c(UV, dtype=float), c(np.asarray(self.p_bc, float)), c(np.asarray(self.R_bc, float)),
                          float(self.f), float(self.cx), float(self.cy), float(self.sig))

    def _observations(self):
        """Tableaux des observations des points vus dans >= 2 images de la fenêtre."""
        count = {}
        for f in self.frames:
            for lid in f.ids.tolist():
                if lid in self.lm:
                    count[lid] = count.get(lid, 0) + 1
        act = [l for l, c in count.items() if c >= 2]
        lidx = {l: k for k, l in enumerate(act)}
        F, Li, C, UV = [], [], [], []
        for fi, f in enumerate(self.frames):
            for j, lid in enumerate(f.ids.tolist()):
                k = lidx.get(lid)
                if k is None:
                    continue
                F.append(fi)
                Li.append(k)
                C.append(0)
                UV.append(f.uvL[j])
                if np.isfinite(f.uvR[j, 0]):
                    F.append(fi)
                    Li.append(k)
                    C.append(1)
                    UV.append(f.uvR[j])
        return (
            act,
            np.array(F, int),
            np.array(Li, int),
            np.array(C, int),
            np.array(UV, float).reshape(-1, 2),
        )

    def _observations_np(self):
        """Même résultat que _observations (même ordre), en numpy : act dans l'ordre de première apparition,
        observations dans l'ordre (image, point), gauche puis droite si appariée."""
        nf = len(self.frames)
        ids = np.concatenate([f.ids for f in self.frames]) if nf else np.empty(0, np.int64)
        if len(ids) == 0 or not self.lm:
            return [], np.empty(0, int), np.empty(0, int), np.empty(0, int), np.empty((0, 2))
        fidx = np.repeat(np.arange(nf), [len(f.ids) for f in self.frames])
        uvL = np.concatenate([f.uvL for f in self.frames]).reshape(-1, 2)
        uvR = np.concatenate([f.uvR for f in self.frames]).reshape(-1, 2)
        in_lm = np.isin(ids, np.fromiter(self.lm.keys(), np.int64, len(self.lm)))
        u, first, cnt = np.unique(ids[in_lm], return_index=True, return_counts=True)
        sel = cnt >= 2
        act_arr = u[sel][np.argsort(first[sel], kind="stable")]       # ordre de première apparition
        if len(act_arr) == 0:
            return [], np.empty(0, int), np.empty(0, int), np.empty(0, int), np.empty((0, 2))
        ordre = np.argsort(act_arr, kind="stable")
        u_act = act_arr[ordre]                                          # trié, pour searchsorted
        pos = np.minimum(np.searchsorted(u_act, ids), len(u_act) - 1)
        rows = np.nonzero(u_act[pos] == ids)[0]
        k = np.empty(len(ids), np.int64)
        k[rows] = ordre[pos[rows]]                                      # indice dans act
        st = np.isfinite(uvR[rows, 0])
        n_rep = 1 + st.astype(np.int64)
        r2 = np.repeat(rows, n_rep)
        C = np.zeros(len(r2), int)
        C[np.cumsum(n_rep)[st] - 1] = 1
        UV = np.where(C[:, None] == 1, uvR[r2], uvL[r2])
        return act_arr.tolist(), fidx[r2].astype(int), k[r2].astype(int), C, UV.astype(float)

    def _cost(self, Rs, ps, vs, P, obs):
        """ETAPE 5b. Somme des Huber(||r||) sur les observations valides + somme des 0.5 * ||W r_imu||^2 sur
        les paires d'images consécutives (pre.residual(...), pre.sqrt_info())."""
        F, Li, C, UV = obs
        r, _, _, _, valid = self._reproj(Rs, ps, P, F, Li, C, UV)
        e = np.linalg.norm(r, axis=1)
        c = np.where(
            e < self.huber, 0.5 * e**2, self.huber * (e - 0.5 * self.huber)
        )  # avec un carré pur, une observation aberrante (e = 20) coûterait 200 et dominerait toutes les autres ;
        c = c[valid].sum()

        for i in range(1, len(Rs)):
            pre = self.frames[i].preint
            ri, _, _ = pre.residual(
                Rs[i - 1], ps[i - 1], vs[i - 1], Rs[i], ps[i], vs[i]
            )
            c += 0.5 * np.sum((pre.sqrt_info() @ ri) ** 2)
        return c

    def _optimize(self):
        nF = len(self.frames)
        act, F, Li, C, UV = self._observations_np() if self.obs == "numpy" else self._observations()
        if nF < 2:
            return 0.0, 0.0, len(F)
        obs = (F, Li, C, UV)
        Rs = np.array([f.R for f in self.frames])
        ps = np.array([f.p for f in self.frames])
        vs = np.array([f.v for f in self.frames])
        P = np.array([self.lm[l] for l in act]).reshape(-1, 3)
        nL, npo = len(act), 9 * (nF - 1)
        Ws = [None] + [self.frames[i].preint.sqrt_info() for i in range(1, nF)]
        cost = cost0 = self._cost(Rs, ps, vs, P, obs)
        lam = 1e-4
        creux = self.schur == "creux"
        if creux:
            pf, pl, lptr, pidx = _paires(F, Li, nF, nL)
            Hplp = np.zeros((0, 6, 3))
        for _ in range(self.iters):
            # --- linéarisation ---
            Hpp = np.zeros((npo, npo))
            gp = np.zeros(npo)
            Hll = np.zeros((nL, 3, 3))
            gl = np.zeros((nL, 3))
            if not creux:
                Hpl = np.zeros((npo, nL, 3))
            if len(F):
                r, Pb, Pc, Z, valid = self._reproj(Rs, ps, P, F, Li, C, UV)
                args_nb = (np.ascontiguousarray(Rs, dtype=float), np.ascontiguousarray(F, dtype=np.int64), np.ascontiguousarray(Li, dtype=np.int64))
                args_nb2 = (r, Pb, Pc, Z, valid, np.ascontiguousarray(self.R_bc, dtype=float), float(self.f), float(self.sig), float(self.huber), nF, nL)
                if creux:
                    Hll, gl, Hpp6, gp6, Hplp = _linearize_sp_nb(*args_nb, pidx, len(pf), *args_nb2)
                else:
                    Hll, gl, Hpp6, gp6, Hpl6 = _linearize_nb(*args_nb, *args_nb2)
                for k in range(nF - 1):
                    sk = 9 * k
                    Hpp[sk : sk + 6, sk : sk + 6] += Hpp6[k]
                    gp[sk : sk + 6] += gp6[k]
                    if not creux:
                        Hpl[sk : sk + 6] += Hpl6[k]
            # IMU
            for i in range(1, nF):
                ri, Ji, Jj = self.frames[i].preint.residual(
                    Rs[i - 1], ps[i - 1], vs[i - 1], Rs[i], ps[i], vs[i]
                )
                W = Ws[i]
                ri, Ji, Jj = W @ ri, W @ Ji, W @ Jj
                sj = 9 * (i - 1)
                Hpp[sj : sj + 9, sj : sj + 9] += Jj.T @ Jj
                gp[sj : sj + 9] += Jj.T @ ri
                if i >= 2:
                    si = 9 * (i - 2)
                    Hpp[si : si + 9, si : si + 9] += Ji.T @ Ji
                    Hpp[si : si + 9, sj : sj + 9] += Ji.T @ Jj
                    Hpp[sj : sj + 9, si : si + 9] += Jj.T @ Ji
                    gp[si : si + 9] += Ji.T @ ri
            if not creux:
                Hpl2 = Hpl.reshape(npo, 3 * nL)
            # --- LM avec complément de Schur ---
            improved = False
            for _ in range(6):
                Hll_d = Hll + lam * (
                    np.einsum("nii->ni", Hll)[:, :, None] * np.eye(3) + 1e-6 * np.eye(3)
                )
                Hll_inv = np.linalg.inv(Hll_d)
                Hpp_d = Hpp + lam * np.diag(np.diag(Hpp)) + 1e-9 * np.eye(npo)
                if creux:
                    SS, bb = _schur_sp_nb(Hll_inv, Hplp, gl, pf, lptr, npo)
                    S = Hpp_d - SS
                    bp = -gp + bb
                else:
                    HplHi = np.einsum("pli,lij->plj", Hpl, Hll_inv).reshape(npo, 3 * nL)
                    S = Hpp_d - HplHi @ Hpl2.T
                    bp = -gp + HplHi @ gl.ravel()
                try:
                    dx = np.linalg.solve(S, bp)
                except np.linalg.LinAlgError:
                    lam *= 10
                    continue
                if creux:
                    dl = _backsub_sp_nb(Hll_inv, Hplp, gl, pf, lptr, dx)
                else:
                    dl = np.einsum(
                        "lij,lj->li", Hll_inv, -gl - (Hpl2.T @ dx).reshape(nL, 3)
                    )
                d = dx.reshape(nF - 1, 9)
                Rn = Rs.copy()
                pn = ps.copy()
                vn = vs.copy()
                Rn[1:] = Rs[1:] @ so3_exp(d[:, 0:3])
                pn[1:] += d[:, 3:6]
                vn[1:] += d[:, 6:9]
                Pn = P + dl
                c = self._cost(Rn, pn, vn, Pn, obs)
                if c < cost:
                    Rs, ps, vs, P, cost = Rn, pn, vn, Pn, c
                    lam = max(lam / 10, 1e-7)
                    improved = True
                    break
                lam *= 10
            if not improved or cost0 - cost < 1e-6 * cost0:
                break
        for i, f in enumerate(self.frames):
            f.R, f.p, f.v = orthonormalize(Rs[i]), ps[i], vs[i]
        for k, l in enumerate(act):
            self.lm[l] = P[k]
        self._last_obs = (act, obs, Rs, ps, P)
        return float(cost0), float(cost), len(F)

    def _reject_outliers(self, seuil_px=3.0):
        if not hasattr(self, "_last_obs"):
            return
        act, (F, Li, C, UV), Rs, ps, P = self._last_obs
        del self._last_obs
        if not len(F):
            return
        r, *_, valid = self._reproj(Rs, ps, P, F, Li, C, UV)
        e = np.linalg.norm(r, axis=1) * self.sig
        e[~valid] = np.inf
        worst = np.zeros(len(act))
        np.maximum.at(worst, Li, e)
        for k in np.nonzero(worst > seuil_px)[0]:
            lid = act[k]
            self.lm.pop(lid, None)
            self.bad.add(lid)

    def _is_keyframe(self, fr):
        """ETAPE 7b. Image clé si : trop de temps depuis la dernière clé, ou trop peu de points communs avec elle,
        ou parallaxe moyenne (déplacement moyen des points communs, en px) > kf_par."""
        kf = next(f for f in reversed(self.frames[:-1]) if f.kf)
        if fr.t - kf.t >= self.kf_dt:
            return True
        common, ia, ib = np.intersect1d(kf.ids, fr.ids, return_indices=True)
        if len(common) < self.kf_suivi * max(len(kf.ids), 1) or len(common) < 20:
            return True
        par = np.linalg.norm(kf.uvL[ia] - fr.uvL[ib], axis=1).mean()
        return par > self.kf_par

    def _cleanup_landmarks(self):
        """ETAPE 7c. Supprimer de self.lm les points qui n'apparaissent plus dans aucune image de la fenêtre."""
        seen = set()
        for f in self.frames:
            seen.update(f.ids.tolist())
        for lid in [l for l in self.lm if l not in seen]:
            del self.lm[lid]

    def add_frame(self, t, grayL, grayR, corners=None, harris=None):
        """ETAPE 7d (assemblage). 1. R_cc (gyro) -> self.front.process ; 2. créer la _Frame (la première = état
        initial vrai, clé) ou fusionner la précédente si non clé, avec la préintégration de self.pre ;
        3. réinitialiser self.pre ; 4. _init_landmarks ; 5. _optimize ; 6. _reject_outliers ; 7. décider si fr est clé ;
        8. retirer la plus vieille image si la fenêtre déborde ; 9. _cleanup_landmarks ; 10. mettre à jour l'état
        courant. Retourne (R, p)."""
        t0 = time.perf_counter()
        R_cc = None
        if self.frames:
            R_cc = self.R_bc.T @ self.pre.dR.T @ self.R_bc  # caméra t-1 -> t
        ids, uvL, uvR = self.front.process(grayL, grayR, R_cc, harris)
        t1 = time.perf_counter()

        if not self.frames:
            R, p, v = self.state0[:3]
            fr = _Frame(t, R, p, v, ids, uvL, uvR, None)
            fr.kf = True
        else:
            pre = self.pre
            if (
                len(self.frames) >= 2 and not self.frames[-1].kf
            ):  # on retire la dernière image non clé
                old = self.frames.pop()
                merged = Preintegration(self.sg, self.sa)
                merged.extend(old.preint)
                merged.extend(pre)
                pre = merged
            R, p, v = self.R, self.p, self.v
            fr = _Frame(t, R, p, v, ids, uvL, uvR, pre)
        self.frames.append(fr)
        self.pre = Preintegration(self.sg, self.sa)

        self._init_landmarks(fr)
        cost0, cost1, n_obs = self._optimize()
        self._reject_outliers()
        if len(self.frames) > 1:
            fr.kf = self._is_keyframe(fr)
        while len(self.frames) > self.N + 1:
            self.frames.pop(0)
            self.frames[0].preint = None
        self._cleanup_landmarks()
        last = self.frames[-1]
        self.R, self.p, self.v = last.R, last.p, last.v
        t2 = time.perf_counter()
        return self.R, self.p
