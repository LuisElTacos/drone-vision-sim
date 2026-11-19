"""stereo_dense.py : stéréo dense OpenCV (StereoSGBM) pour remplir la carte de voxels.

Les coins + KLT restent pour la VIO ; la dense ajoute des points sur les surfaces peu texturées
mais bordées de contours verticaux (poteaux, troncs) que les coins ratent.

Mesures (dense_eval.py, séquences de dense_capture.py, réalisme actif) : SGBM retrouve ~85 % des
poteaux / troncs contre quelques % pour les coins, mais son bruit (~0,5-1 px de disparité) épaissit
les surfaces lointaines vers la caméra. D'où, par défaut : points denses jusqu'à portee_dense = 8 m
seulement (coins au-delà) et un voxel n'est gardé que s'il reçoit >= min_pts points sur l'image.
"""
import cv2
import numpy as np

_SGBM = {}


def disparite_sgbm(imgL, imgR, demi=True, ndisp=128, bs=5, uniq=10):
    """Disparité (H, W) float32 en pixels de la pleine résolution, NaN = pas d'estimation.
    demi=True : calcul sur les images réduites de moitié (4x moins de pixels), disparité remise à l'échelle."""
    gL = cv2.cvtColor(imgL, cv2.COLOR_RGB2GRAY) if imgL.ndim == 3 else imgL
    gR = cv2.cvtColor(imgR, cv2.COLOR_RGB2GRAY) if imgR.ndim == 3 else imgR
    H, W = gL.shape
    if demi:
        gL = cv2.resize(gL, (W // 2, H // 2), interpolation=cv2.INTER_AREA)
        gR = cv2.resize(gR, (W // 2, H // 2), interpolation=cv2.INTER_AREA)
        ndisp = max(16, ndisp // 2 // 16 * 16)
    key = (ndisp, bs, uniq)
    if key not in _SGBM:
        _SGBM[key] = cv2.StereoSGBM_create(minDisparity=0, numDisparities=ndisp, blockSize=bs,
                                           P1=8 * bs * bs, P2=32 * bs * bs, disp12MaxDiff=1,
                                           uniquenessRatio=uniq, speckleWindowSize=100, speckleRange=2,
                                           mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
    d = _SGBM[key].compute(gL, gR).astype(np.float32) / 16.0
    d[d <= 0] = np.nan
    if demi:
        d = 2.0 * cv2.resize(d, (W, H), interpolation=cv2.INTER_NEAREST)
    return d


def points_denses(D, K, B, R, C, portee, origin, voxel, pas=2, min_pts=3, d_min=0.5):
    """Points monde (N, 3) tirés de la disparité D, un par voxel touché (point moyen).
    pas : sous-échantillonnage des pixels ; min_pts : nb min de pixels échantillonnés par voxel (filtre du bruit)."""
    H, W = D.shape
    vv, uu = np.mgrid[0:H:pas, 0:W:pas]
    d = D[vv, uu]
    with np.errstate(invalid="ignore", divide="ignore"):
        ok = np.isfinite(d) & (d > d_min)
        f, cx, cy = K[0, 0], K[0, 2], K[1, 2]
        z = np.where(ok, f * B / np.where(ok, d, 1.0), np.inf)
    ok &= z <= portee
    u, v, z = uu[ok], vv[ok], z[ok]
    if len(z) == 0:
        return np.empty((0, 3))
    P = np.stack([(u - cx) * z / f, (v - cy) * z / f, z], 1) @ np.asarray(R).T + np.asarray(C)
    idx = np.floor((P - np.asarray(origin)) / voxel).astype(np.int64)
    idx -= idx.min(0)
    n1, n2 = idx[:, 1].max() + 1, idx[:, 2].max() + 1
    cle = (idx[:, 0] * n1 + idx[:, 1]) * n2 + idx[:, 2]       # indice linéaire : np.unique 1D, bien plus rapide que axis=0
    u_cle, inv, cnt = np.unique(cle, return_inverse=True, return_counts=True)
    keep = cnt >= min_pts
    S = np.stack([np.bincount(inv, weights=P[:, k], minlength=len(u_cle)) for k in range(3)], 1)
    return S[keep] / cnt[keep, None]
