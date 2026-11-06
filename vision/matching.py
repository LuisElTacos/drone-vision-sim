"""Étape 3 : appariement stéréo des coins de Harris (cas rectifié).

Les deux caméras sont parallèles et décalées le long de leur axe x : la ligne épipolaire
d'un pixel (uL, vL) dans l'image droite est la même ligne vL. Un coin gauche ne peut donc
être apparié qu'à un coin droit situé sur (presque) la même ligne, et plus à gauche
(disparité d = uL - uR >= 0).

Chaîne :
  1. candidats : coins droits avec |vR - vL| <= row_tol et 0 <= uL - uR <= max_disp
  2. score : ZNCC (corrélation croisée normalisée centrée) entre fenêtres autour des coins
  3. filtres : score minimal, test du ratio (ambiguïté), cohérence gauche-droite
  4. raffinement sous-pixel de uR : recherche le long de la ligne vL dans l'image droite
     puis ajustement d'une parabole sur le score ZNCC

Sortie : ptsL, ptsR (N, 2) float64, colonnes [u, v], avec vR = vL (cas rectifié),
et scores (N,) ; directement utilisables par triangulate(uL, vL, uR, vR, ...).
"""

import numpy as np
import cv2
from numba import njit, prange
from numpy.lib.stride_tricks import sliding_window_view


def to_gray(img):
    """Image RGB uint8 (H, W, 3) -> niveaux de gris float32 dans [0, 1]."""
    if img.ndim == 2:
        return img.astype(np.float32) / (255.0 if img.dtype == np.uint8 else 1.0)
    rgb = img[..., :3].astype(np.float32) / 255.0
    return rgb @ np.array([0.299, 0.587, 0.114], dtype=np.float32)


def extract_patches(gray, xs, ys, half):
    """Fenêtres (2*half+1)^2 centrées sur (xs, ys) entiers, aplaties : (N, (2h+1)^2).

    L'image est complétée par réflexion pour les points proches du bord.
    """
    padded = np.pad(gray, half, mode="reflect")
    win = sliding_window_view(padded, (2 * half + 1, 2 * half + 1))
    xs = np.asarray(xs, dtype=np.int64)
    ys = np.asarray(ys, dtype=np.int64)
    return win[ys, xs].reshape(len(xs), -1)


def normalize_patches(P, min_std=1e-3):
    """Centre et normalise chaque fenêtre (pour la ZNCC). Renvoie (P_norm, texture_ok).

    texture_ok = False pour une fenêtre presque uniforme (écart-type < min_std) :
    sa ZNCC n'a pas de sens, elle est mise à 0.
    """
    P = P - P.mean(axis=1, keepdims=True)
    std = P.std(axis=1)
    ok = std > min_std
    norm = np.linalg.norm(P, axis=1)
    P = np.where(ok[:, None], P / np.where(ok, norm, 1.0)[:, None], 0.0)
    return P, ok


def zncc_matrix(PL, PR):
    """ZNCC entre toutes les fenêtres gauche et droite (déjà normalisées) : (NL, NR) dans [-1, 1]."""
    return PL @ PR.T


def refine_subpixel(grayL, grayR, uL, vL, uR0, half, search=2):
    """Raffine uR (entier) le long de la ligne vL de l'image droite.

    On évalue la ZNCC de la fenêtre gauche contre les fenêtres droites centrées en
    uR0 - search ... uR0 + search (même ligne vL), on garde le meilleur entier puis on
    ajuste une parabole sur ses deux voisins : décalage = (s- - s+) / (2 (s- - 2 s0 + s+)).
    Renvoie (uR raffiné float64, score au meilleur entier).
    """
    H, W = grayR.shape
    n = len(uL)
    if n == 0:
        return np.empty(0), np.empty(0)
    offs = np.arange(-search - 1, search + 2)               # une colonne de marge pour la parabole
    PL, _ = normalize_patches(extract_patches(grayL, uL, vL, half))
    cols = np.clip(uR0[:, None] + offs[None, :], 0, W - 1)  # (n, K)
    rows = np.repeat(vL[:, None], len(offs), axis=1)
    PR, _ = normalize_patches(extract_patches(grayR, cols.ravel(), rows.ravel(), half))
    S = np.einsum("nk,nmk->nm", PL, PR.reshape(n, len(offs), -1))   # (n, K)

    inner = S[:, 1:-1]                                      # positions -search..+search
    k = np.argmax(inner, axis=1) + 1                        # indice dans S
    idx = np.arange(n)
    s_m, s_0, s_p = S[idx, k - 1], S[idx, k], S[idx, k + 1]
    denom = s_m - 2.0 * s_0 + s_p
    delta = np.where(denom < -1e-9, 0.5 * (s_m - s_p) / np.where(denom < -1e-9, denom, -1.0), 0.0)
    delta = np.clip(delta, -0.5, 0.5)
    uR = uR0 + offs[k] + delta
    return uR.astype(np.float64), s_0


def match_stereo(imgL, imgR, cornersL, cornersR, half_win=5, max_disp=160, row_tol=1,
                 min_zncc=0.8, ratio=0.8, lr_check=True, subpixel=True):
    """Apparie les coins de Harris gauche et droite d'une paire rectifiée.

    imgL, imgR : (H, W, 3) uint8 RGB ou (H, W) gris.
    cornersL, cornersR : sorties de detect_corners, (N, 3) [x, y, score] (x = colonne, y = ligne).
    half_win : demi-taille de la fenêtre de corrélation (5 -> 11x11).
    max_disp : disparité max en pixels (= profondeur min f*B/max_disp, ~0,47 m ici).
    row_tol : écart de ligne toléré entre les deux coins (pixels).
    min_zncc : score minimal accepté.
    ratio : test d'ambiguïté, on garde si (1 - s1) < ratio * (1 - s2), s2 = 2e meilleur score.
    lr_check : le meilleur coin gauche du coin droit choisi doit être le coin de départ.
    subpixel : raffine uR le long de la ligne vL (sinon uR = colonne entière du coin droit).

    Renvoie ptsL (N, 2), ptsR (N, 2) float64 [u, v] et scores (N,) float64.
    """
    empty = (np.empty((0, 2)), np.empty((0, 2)), np.empty(0))
    if len(cornersL) == 0 or len(cornersR) == 0:
        return empty

    grayL, grayR = to_gray(imgL), to_gray(imgR)
    xL = np.rint(cornersL[:, 0]).astype(np.int64)
    yL = np.rint(cornersL[:, 1]).astype(np.int64)
    xR = np.rint(cornersR[:, 0]).astype(np.int64)
    yR = np.rint(cornersR[:, 1]).astype(np.int64)

    PL, okL = normalize_patches(extract_patches(grayL, xL, yL, half_win))
    PR, okR = normalize_patches(extract_patches(grayR, xR, yR, half_win))
    S = zncc_matrix(PL, PR)                                   # (NL, NR)

    # Contrainte épipolaire (même ligne) et disparité positive bornée
    d = xL[:, None] - xR[None, :]
    allowed = (np.abs(yL[:, None] - yR[None, :]) <= row_tol) & (d >= 0) & (d <= max_disp)
    allowed &= okL[:, None] & okR[None, :]
    S = np.where(allowed, S, -np.inf)

    # Meilleur et deuxième meilleur candidat pour chaque coin gauche
    order = np.argsort(-S, axis=1)
    rows = np.arange(len(xL))
    j1 = order[:, 0]
    s1 = S[rows, j1]
    s2 = S[rows, order[:, 1]] if S.shape[1] > 1 else np.full(len(xL), -np.inf)

    keep = np.isfinite(s1) & (s1 >= min_zncc)
    # Test du ratio : un 2e candidat presque aussi bon = motif répété, appariement ambigu
    keep &= ~np.isfinite(s2) | ((1.0 - s1) < ratio * (1.0 - s2))
    if lr_check:
        best_left_of_right = np.argmax(S, axis=0)             # (NR,)
        keep &= best_left_of_right[j1] == rows

    i = np.nonzero(keep)[0]
    j = j1[i]
    if len(i) == 0:
        return empty

    uL = xL[i].astype(np.float64)
    vL = yL[i].astype(np.float64)
    if subpixel:
        uR, score = refine_subpixel(grayL, grayR, xL[i], yL[i], xR[j], half_win)
    else:
        uR, score = xR[j].astype(np.float64), s1[i]

    ptsL = np.column_stack([uL, vL])
    ptsR = np.column_stack([uR, vL])                          # cas rectifié : vR = vL
    return ptsL, ptsR, np.asarray(score, dtype=np.float64)


def _row_zncc(PL_norm, grayR, xs0, ys, ndisp, half):
    """ZNCC de chaque fenêtre gauche contre les fenêtres droites de sa ligne.

    Fenêtre droite k centrée en colonne xs0 - k (k = 0..ndisp-1, disparité k).
    Renvoie S (n, ndisp), -inf hors image ou fenêtre uniforme.
    """
    H, W = grayR.shape
    n = len(xs0)
    k = np.arange(ndisp)
    cols = xs0[:, None] - k[None, :]
    inside = (cols >= 0) & (cols < W)
    cols_c = np.clip(cols, 0, W - 1)
    rows = np.repeat(ys[:, None], ndisp, axis=1)
    PR, okR = normalize_patches(extract_patches(grayR, cols_c.ravel(), rows.ravel(), half))
    S = np.einsum("nk,nmk->nm", PL_norm, PR.reshape(n, ndisp, -1))
    return np.where(inside & okR.reshape(n, ndisp), S, -np.inf)


def match_along_rows_ref(imgL, imgR, cornersL, half_win=5, max_disp=160, min_zncc=0.8,
                     ratio=0.6, lr_check=True, subpixel=True, chunk=256):
    """Variante : chaque coin gauche est cherché dans TOUTE sa ligne de l'image droite.

    Pas besoin que Harris trouve le même coin à droite (utile en forêt : feuillage, coins
    instables d'une image à l'autre). Pour chaque coin gauche (uL, vL) :
      - ZNCC contre les fenêtres droites (uL - d, vL), d = 0..max_disp
      - meilleur score s1 en d1 ; s2 = meilleur score hors de d1 ± 2 (pic secondaire)
      - test du ratio (1 - s1) < ratio (1 - s2) et s1 >= min_zncc
      - cohérence gauche-droite : en repartant de la fenêtre droite trouvée et en cherchant
        dans la ligne gauche, on doit retomber sur uL (± 1 px)
      - raffinement sous-pixel par parabole sur S(d1 - 1), S(d1), S(d1 + 1)
    Renvoie ptsL, ptsR (N, 2) [u, v] et scores (N,), comme match_stereo.
    """
    empty = (np.empty((0, 2)), np.empty((0, 2)), np.empty(0))
    if len(cornersL) == 0:
        return empty
    grayL, grayR = to_gray(imgL), to_gray(imgR)
    H, W = grayL.shape
    xL = np.rint(cornersL[:, 0]).astype(np.int64)
    yL = np.rint(cornersL[:, 1]).astype(np.int64)
    ndisp = max_disp + 1
    PL, okL = normalize_patches(extract_patches(grayL, xL, yL, half_win))

    out_uR = np.full(len(xL), np.nan)
    out_s = np.full(len(xL), -np.inf)
    keep = np.zeros(len(xL), dtype=bool)
    for a in range(0, len(xL), chunk):
        sl = slice(a, a + chunk)
        n = len(xL[sl])
        S = _row_zncc(PL[sl], grayR, xL[sl], yL[sl], ndisp, half_win)
        idx = np.arange(n)
        d1 = np.argmax(S, axis=1)
        s1 = S[idx, d1]
        masked = S.copy()
        near_peak = np.abs(np.arange(ndisp)[None, :] - d1[:, None]) <= 2
        masked[near_peak] = -np.inf
        s2 = masked.max(axis=1)
        ok = okL[sl] & np.isfinite(s1) & (s1 >= min_zncc)
        ok &= ~np.isfinite(s2) | ((1.0 - s1) < ratio * (1.0 - s2))

        if lr_check:
            # fenêtre droite trouvée -> recherche dans la ligne gauche (colonnes xR + d)
            xR = xL[sl] - d1
            # (images retournées gauche-droite pour réutiliser _row_zncc : colonne x -> W-1-x)
            PRb, _ = normalize_patches(extract_patches(grayR[:, ::-1], (W - 1) - xR, yL[sl], half_win))
            Sb = _row_zncc(PRb, grayL[:, ::-1], (W - 1) - xR, yL[sl], ndisp, half_win)
            back = xR + np.argmax(Sb, axis=1)          # colonne gauche retrouvée
            ok &= np.abs(back - xL[sl]) <= 1

        # parabole sur la disparité (S décroît de part et d'autre du pic)
        dm = np.clip(d1 - 1, 0, ndisp - 1)
        dp = np.clip(d1 + 1, 0, ndisp - 1)
        s_m, s_p = S[idx, dm], S[idx, dp]
        delta = np.zeros(n)
        if subpixel:
            good = np.isfinite(s_m) & np.isfinite(s_p) & (d1 > 0) & (d1 < ndisp - 1)
            denom = np.where(good, s_m - 2.0 * s1 + s_p, -1.0)
            delta = np.where(good & (denom < -1e-9), 0.5 * (s_m - s_p) / np.minimum(denom, -1e-9), 0.0)
            delta = np.clip(delta, -0.5, 0.5)
        disp = d1 + delta
        out_uR[sl] = xL[sl] - disp
        out_s[sl] = s1
        keep[sl] = ok

    i = np.nonzero(keep)[0]
    if len(i) == 0:
        return empty
    ptsL = np.column_stack([xL[i], yL[i]]).astype(np.float64)
    ptsR = np.column_stack([out_uR[i], yL[i]]).astype(np.float64)
    return ptsL, ptsR, out_s[i]


# ---------------------------------------------------------------------------
# match_along_rows optimisée (numba, images intégrales). L'ancienne version reste disponible
# sous le nom match_along_rows_ref (même interface, résultats quasi identiques).
# ---------------------------------------------------------------------------
def window_stats(gray, half, min_std=1e-3):
    """Pour chaque pixel : image complétée par réflexion, norme de la fenêtre centrée, texture ok.

    Sommes des fenêtres par images intégrales (cv2.integral2, en float64).
    """
    w = 2 * half + 1
    n = w * w
    g = np.ascontiguousarray(gray, dtype=np.float32)
    pad32 = cv2.copyMakeBorder(g, half, half, half, half, cv2.BORDER_REFLECT_101)  # = np.pad 'reflect'
    S, SQ = cv2.integral2(pad32, sdepth=cv2.CV_64F, sqdepth=cv2.CV_64F)
    norm, ok = _norm_from_integrals(S, SQ, g.shape[0], g.shape[1], w, min_std)
    return pad32, norm, ok


@njit(cache=True, parallel=True)
def _norm_from_integrals(S, SQ, H, W, w, min_std):
    n = w * w
    thr = min_std * np.sqrt(n)
    norm = np.empty((H, W))
    ok = np.empty((H, W), dtype=np.bool_)
    for y in prange(H):
        for x in range(W):
            s1 = S[y + w, x + w] - S[y, x + w] - S[y + w, x] + S[y, x]
            s2 = SQ[y + w, x + w] - SQ[y, x + w] - SQ[y + w, x] + SQ[y, x]
            v = np.sqrt(max(s2 - s1 * s1 / n, 0.0))
            norm[y, x] = v
            ok[y, x] = v > thr
    return norm, ok


@njit(cache=True, inline="always")
def _dot(PLn, pad, y, x, half):
    # fenêtre centrée en (y, x) de l'image d'origine = pad[y:y+2h+1, x:x+2h+1]
    s = 0.0
    w = 2 * half + 1
    for a in range(w):
        row = pad[y + a]
        base = a * w
        for b in range(w):
            s += PLn[base + b] * row[x + b]
    return s


@njit(cache=True, parallel=True)
def _match_kernel(padL, padR, normL, okL_img, normR, okR_img, xL, yL, half, ndisp,
                  min_zncc, ratio, lr_check, subpixel):
    n = len(xL)
    H, W = normR.shape
    w = 2 * half + 1
    npx = w * w
    out_uR = np.full(n, np.nan)
    out_s = np.full(n, -np.inf)
    keep = np.zeros(n, dtype=np.bool_)
    for i in prange(n):
        x0, y0 = xL[i], yL[i]
        if not okL_img[y0, x0]:
            continue
        # fenêtre gauche centrée et normalisée
        PL = np.empty(npx)
        m = 0.0
        for a in range(w):
            for b in range(w):
                v = padL[y0 + a, x0 + b]
                PL[a * w + b] = v
                m += v
        m /= npx
        nn = 0.0
        for t in range(npx):
            PL[t] -= m
            nn += PL[t] * PL[t]
        nn = np.sqrt(nn)
        for t in range(npx):
            PL[t] /= nn

        S = np.full(ndisp, -np.inf)
        d1 = 0
        s1 = -np.inf
        for k in range(ndisp):
            x = x0 - k
            if x < 0:
                break
            if okR_img[y0, x]:
                v = _dot(PL, padR, y0, x, half) / normR[y0, x]
                S[k] = v
                if v > s1:
                    s1 = v
                    d1 = k
        if not (s1 >= min_zncc):
            continue
        s2 = -np.inf
        for k in range(ndisp):
            if abs(k - d1) > 2 and S[k] > s2:
                s2 = S[k]
        if np.isfinite(s2) and not ((1.0 - s1) < ratio * (1.0 - s2)):
            continue

        if lr_check:
            xR = x0 - d1
            PR = np.empty(npx)
            m = 0.0
            for a in range(w):
                for b in range(w):
                    v = padR[y0 + a, xR + b]
                    PR[a * w + b] = v
                    m += v
            m /= npx
            nn = 0.0
            for t in range(npx):
                PR[t] -= m
                nn += PR[t] * PR[t]
            nn = np.sqrt(nn)
            for t in range(npx):
                PR[t] /= nn
            best = -np.inf
            kb = 0
            for k in range(ndisp):
                x = xR + k
                if x >= W:
                    break
                if okL_img[y0, x]:
                    v = _dot(PR, padL, y0, x, half) / normL[y0, x]
                    if v > best:
                        best = v
                        kb = k
            if abs(xR + kb - x0) > 1:
                continue

        delta = 0.0
        if subpixel and d1 > 0 and d1 < ndisp - 1:
            s_m, s_p = S[d1 - 1], S[d1 + 1]
            if np.isfinite(s_m) and np.isfinite(s_p):
                den = s_m - 2.0 * s1 + s_p
                if den < -1e-9:
                    delta = 0.5 * (s_m - s_p) / den
                    delta = min(max(delta, -0.5), 0.5)
        out_uR[i] = x0 - (d1 + delta)
        out_s[i] = s1
        keep[i] = True
    return out_uR, out_s, keep


def match_along_rows(imgL, imgR, cornersL, half_win=5, max_disp=160, min_zncc=0.8,
                     ratio=0.6, lr_check=True, subpixel=True, grayL=None, grayR=None):
    """Même interface que vision.matching.match_along_rows (+ grayL/grayR déjà calculés)."""
    empty = (np.empty((0, 2)), np.empty((0, 2)), np.empty(0))
    if len(cornersL) == 0:
        return empty
    if grayL is None:
        grayL = to_gray(imgL)
    if grayR is None:
        grayR = to_gray(imgR)
    xL = np.rint(cornersL[:, 0]).astype(np.int64)
    yL = np.rint(cornersL[:, 1]).astype(np.int64)
    padL, normL, okL = window_stats(grayL, half_win)
    padR, normR, okR = window_stats(grayR, half_win)
    uR, s, keep = _match_kernel(padL, padR, normL, okL, normR, okR, xL, yL, half_win,
                                max_disp + 1, min_zncc, ratio, lr_check, subpixel)
    i = np.nonzero(keep)[0]
    if len(i) == 0:
        return empty
    ptsL = np.column_stack([xL[i], yL[i]]).astype(np.float64)
    ptsR = np.column_stack([uR[i], yL[i]]).astype(np.float64)
    return ptsL, ptsR, s[i]
