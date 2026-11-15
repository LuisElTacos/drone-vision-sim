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


def fundamental_ransac(p0, p1, threshold=1.0, iters=200, rng=None):
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

    def renew(self, grayL):
        """C. Si len(pts) < n_max : coins de Harris (detect_corners), triés par score, hors bordure klt.win,
        acceptés seulement s'ils sont à > min_dist de tous les points déjà présents (masque uint8 + cv2.circle).
        Nouveaux ids = next_id..., next_id += n."""
        if len(self.pts) > self.n_max:
            return None
        corners = detect_corners(
            harris_function(grayL),
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
        if len(self.pts):
            L = np.ascontiguousarray(grayL, np.float32)
            R = np.ascontiguousarray(grayR, np.float32)
            half = 5
            yc = np.round(self.pts[:, 1]).astype(np.int64)
            out, keep = _match_corners(
                L,
                R,
                _window_norms(L, half),
                _window_norms(R, half),
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

    def process(self, grayL, grayR, R_cc=None):
        """E. pyr = self.klt.pyramid(grayL) ; si prev_pyr existe : track ; renew ; prev_pyr = pyr ; stereo.
        Retourne ids, uvL, uvR."""
        pyr = self.klt.pyramid(grayL)
        if self.prev_pyr is not None and len(self.pts):
            self.track(pyr, R_cc)
        self.renew(grayL)
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
    ):
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
        Rf = Rs[F]  # rotations maintenant indexées par F
        pf = ps[F]  # positions des cam indexées par F
        Pw = P[Li]  # points 3D des observations
        d = Pw - pf
        Pb = np.matmul(Rf.transpose(0, 2, 1), d[:, :, None])[:, :, 0]
        Pc = (Pb - self.p_bc[C]) @ self.R_bc
        Z = Pc[:, 2]
        Zs = np.where(Z > 0.05, Z, 0.05)
        uv = np.column_stack(
            [self.f * Pc[:, 0] / Zs + self.cx, self.f * Pc[:, 1] / Zs + self.cy]
        )
        r = (uv - UV) / self.sig
        return r, Pb, Pc, Zs, Z > 0.05

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
        act, F, Li, C, UV = self._observations()
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
        for _ in range(self.iters):
            # --- linéarisation ---
            Hpp = np.zeros((npo, npo))
            gp = np.zeros(npo)
            Hll = np.zeros((nL, 3, 3))
            gl = np.zeros((nL, 3))
            Hpl = np.zeros((npo, nL, 3))
            if len(F):
                r, Pb, Pc, Z, valid = self._reproj(Rs, ps, P, F, Li, C, UV)
                e = np.linalg.norm(r, axis=1)
                w = (
                    np.where(e < self.huber, 1.0, self.huber / np.maximum(e, 1e-9))
                    * valid
                )
                fz = self.f / Z / self.sig
                Jp = np.zeros((len(F), 2, 3))
                Jp[:, 0, 0] = fz
                Jp[:, 0, 2] = -fz * Pc[:, 0] / Z
                Jp[:, 1, 1] = fz
                Jp[:, 1, 2] = -fz * Pc[:, 1] / Z
                JpRcb = Jp @ self.R_bc.T  # dr/dPb
                A = np.einsum("nij,nkj->nik", JpRcb, Rs[F])  # dr/dPw = Jp R_bc^T R_f^T
                Jth = JpRcb @ skew(Pb)
                Jx = np.concatenate([Jth, -A], axis=2)  # (n, 2, 6) : θ, p de l'image
                wr = w[:, None] * r
                # landmarks
                np.add.at(Hll, Li, w[:, None, None] * np.einsum("nki,nkj->nij", A, A))
                np.add.at(gl, Li, np.einsum("nki,nk->ni", A, wr))
                m = F > 0
                if m.any():
                    Fi, Lm = F[m] - 1, Li[m]
                    Hxx = w[m, None, None] * np.einsum("nki,nkj->nij", Jx[m], Jx[m])
                    gx = np.einsum("nki,nk->ni", Jx[m], wr[m])
                    Hxl = w[m, None, None] * np.einsum("nki,nkj->nij", Jx[m], A[m])
                    Hpp6 = np.zeros((nF - 1, 6, 6))
                    np.add.at(Hpp6, Fi, Hxx)
                    gp6 = np.zeros((nF - 1, 6))
                    np.add.at(gp6, Fi, gx)
                    Hpl6 = np.zeros((nF - 1, 6, nL, 3))
                    np.add.at(Hpl6, (Fi, slice(None), Lm), Hxl)
                    for k in range(nF - 1):
                        s = 9 * k
                        Hpp[s : s + 6, s : s + 6] += Hpp6[k]
                        gp[s : s + 6] += gp6[k]
                        Hpl[s : s + 6] += Hpl6[k]
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
            Hpl2 = Hpl.reshape(npo, 3 * nL)
            # --- LM avec complément de Schur ---
            improved = False
            for _ in range(6):
                Hll_d = Hll + lam * (
                    np.einsum("nii->ni", Hll)[:, :, None] * np.eye(3) + 1e-6 * np.eye(3)
                )
                Hll_inv = np.linalg.inv(Hll_d)
                Hpp_d = Hpp + lam * np.diag(np.diag(Hpp)) + 1e-9 * np.eye(npo)
                HplHi = np.einsum("pli,lij->plj", Hpl, Hll_inv).reshape(npo, 3 * nL)
                S = Hpp_d - HplHi @ Hpl2.T
                bp = -gp + HplHi @ gl.ravel()
                try:
                    dx = np.linalg.solve(S, bp)
                except np.linalg.LinAlgError:
                    lam *= 10
                    continue
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

    def add_frame(self, t, grayL, grayR, corners=None):
        """ETAPE 7d (assemblage). 1. R_cc (gyro) -> self.front.process ; 2. créer la _Frame (la première = état
        initial vrai, clé) ou fusionner la précédente si non clé, avec la préintégration de self.pre ;
        3. réinitialiser self.pre ; 4. _init_landmarks ; 5. _optimize ; 6. _reject_outliers ; 7. décider si fr est clé ;
        8. retirer la plus vieille image si la fenêtre déborde ; 9. _cleanup_landmarks ; 10. mettre à jour l'état
        courant. Retourne (R, p)."""
        t0 = time.perf_counter()
        R_cc = None
        if self.frames:
            R_cc = self.R_bc.T @ self.pre.dR.T @ self.R_bc  # caméra t-1 -> t
        ids, uvL, uvR = self.front.process(grayL, grayR, R_cc)
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
