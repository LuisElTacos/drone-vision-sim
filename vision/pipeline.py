"""Perception pour les 12 caméras (6 paires stéréo, voir scene.STEREO_PAIRS).

Chaque paire est traitée comme la paire frontale d'avant : coins de Harris dans l'image gauche ->
appariement le long des lignes (match_stereo, via match_along_rows) -> triangulation
dans le repère de la caméra gauche puis passage en repère monde. Les points de toutes les paires
s'intègrent ensuite dans la même VoxelMap avec VoxelMap.integrate_multi.
"""

import time

import numpy as np

from vision.corners import harris_function, detect_corners
from vision.matching import match_stereo, to_gray
from vision.triangulation import triangulate


def match_along_rows(imgL, imgR, cL, min_zncc=0.8, ratio=0.8, grayL=None, grayR=None, norms=None):
    """Adaptation de vision.matching.match_stereo (gris, [x_g,y,x_d,d,coût]) -> (ptsL, ptsR, scores),
    identique à test14.match_along_rows."""
    gL = grayL if grayL is not None else to_gray(imgL)
    gR = grayR if grayR is not None else to_gray(imgR)
    m = match_stereo(gL, gR, cL, max_cost=2.0 * (1.0 - min_zncc), ratio=ratio, norms=norms).astype(np.float64)
    if len(m) == 0:
        return np.empty((0, 2)), np.empty((0, 2)), np.empty(0)
    return m[:, 0:2], np.column_stack([m[:, 2], m[:, 1]]), 1.0 - m[:, 4] / 2.0


def perceive_pair(imgL, imgR, K, B, R, C, corner_kw=None, match_kw=None):
    """Une paire stéréo rectifiée.

    corner_kw : arguments de detect_corners (rel_thresh, max_n, grid_cells, ...)
    match_kw  : arguments de match_along_rows (min_zncc, ratio, ...)
    Retourne P_w (N, 3) points monde valides, ptsL (N, 2) pixels gauches, Z (N,) profondeurs (m),
    n_coins, et les temps (coins, appariement, triangulation) en s.
    """
    corner_kw = corner_kw or {}
    match_kw = match_kw or {}
    t0 = time.perf_counter()
    gL, gR = to_gray(imgL), to_gray(imgR)
    cL = detect_corners(harris_function(gL), **corner_kw)
    t1 = time.perf_counter()
    ptsL, ptsR, _ = match_along_rows(imgL, imgR, cL, grayL=gL, grayR=gR, **match_kw)
    t2 = time.perf_counter()
    P_w, ok = triangulate(ptsL[:, 0], ptsL[:, 1], ptsR[:, 0], ptsR[:, 1], K, B, R, C)
    t3 = time.perf_counter()
    ptsL = ptsL[ok]
    disp = ptsL[:, 0] - ptsR[ok][:, 0]
    Z = K[0, 0] * B / np.maximum(disp, 1e-6)
    return P_w[ok], ptsL, Z, len(cL), (t1 - t0, t2 - t1, t3 - t2)


def perceive_all(frame, corner_kw=None, match_kw=None):
    """Toutes les paires d'une Frame (caméras 2k gauche, 2k+1 droite).

    Retourne une liste (une entrée par paire) de dicts P_w, C, ptsL, Z, n_coins, et les temps
    cumulés sur les paires (coins, appariement, triangulation) en s.
    """
    out, t = [], np.zeros(3)
    for k in range(frame.n_pairs):
        imgL, imgR, B, R, C = frame.pair(k)
        P_w, ptsL, Z, n_coins, tk = perceive_pair(imgL, imgR, frame.K, B, R, C, corner_kw, match_kw)
        out.append(dict(P_w=P_w, C=C, ptsL=ptsL, Z=Z, n_coins=n_coins))
        t += tk
    return out, t
