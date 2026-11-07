import numpy as np
import cv2
from numba import njit, prange


def _normalize_patch(patch: np.ndarray, eps: float = 1e-5) -> tuple[np.ndarray | None, float]:
    p_flat = patch.astype(np.float32).ravel()
    p_centered = p_flat - np.mean(p_flat)
    norm = float(np.linalg.norm(p_centered))
    if norm < eps:
        return None, 0.0
    return (p_centered / norm), norm


def _window_norms(img, half):
    """Norme de (fenêtre - moyenne) centrée en chaque pixel, par images intégrales (0 sur les bords).
    Remplace le calcul de norm_r fenêtre par fenêtre de la version d'origine."""
    S, SQ = cv2.integral2(np.ascontiguousarray(img, dtype=np.float32), sdepth=cv2.CV_64F, sqdepth=cv2.CV_64F)
    return _norms_kernel(S, SQ, img.shape[0], img.shape[1], 2 * half + 1)


@njit(cache=True)
def _norms_kernel(S, SQ, H, W, w):
    n = w * w
    h = w // 2
    out = np.zeros((H, W))
    for y in range(h, H - h):
        for x in range(h, W - h):
            s1 = S[y + h + 1, x + h + 1] - S[y - h, x + h + 1] - S[y + h + 1, x - h] + S[y - h, x - h]
            s2 = SQ[y + h + 1, x + h + 1] - SQ[y - h, x + h + 1] - SQ[y + h + 1, x - h] + SQ[y - h, x - h]
            out[y, x] = np.sqrt(max(s2 - s1 * s1 / n, 0.0))
    return out


@njit(cache=True)
def _patch_norm(img, y, x, half):
    """Équivalent numba de _normalize_patch : (patch centré / norme, norme) de la fenêtre centrée en (y, x)."""
    w = 2 * half + 1
    p = np.empty(w * w)
    m = 0.0
    for a in range(w):
        for b in range(w):
            v = img[y - half + a, x - half + b]
            p[a * w + b] = v
            m += v
    m /= w * w
    nn = 0.0
    for t in range(w * w):
        p[t] -= m
        nn += p[t] * p[t]
    nn = np.sqrt(nn)
    if nn >= 1e-5:
        for t in range(w * w):
            p[t] /= nn
    return p, nn


@njit(cache=True)
def _zncc(p, img, norms, y, x, half):
    """ZNCC entre le patch normalisé p et la fenêtre centrée en (y, x) ; -1 si la fenêtre est uniforme (coût 4)."""
    nr = norms[y, x]
    if nr <= 1e-5:
        return -1.0
    w = 2 * half + 1
    s = 0.0
    for a in range(w):
        for b in range(w):
            s += p[a * w + b] * img[y - half + a, x - half + b]
    return s / nr


@njit(cache=True, parallel=True)
def _match_corners(left, right, norm_l, norm_r, xg_f_all, y_coords, d_min, d_max, half, ratio, max_cost, cross_check):
    """Corps de la boucle d'origine (un coin par itération, coins en parallèle)"""
    H, W = left.shape
    n = len(xg_f_all)
    out = np.zeros((n, 5))
    keep = np.zeros(n, dtype=np.bool_)
    for idx in prange(n):
        y = y_coords[idx]
        if y - half < 0 or y + half >= H:
            continue

        xg_f = xg_f_all[idx]
        xg = int(np.round(xg_f))

        if xg - half < 0 or xg + half >= W:
            continue

        p_l_norm, norm_val = _patch_norm(left, y, xg, half)
        if norm_val < 1e-5:
            continue  # Patch trop uniforme

        d_upper = min(d_max, xg - half)
        if d_upper < d_min + 2:
            continue

        nd = d_upper - d_min + 1
        costs = np.empty(nd)
        for k in range(nd):
            d = d_min + k                      # d_vals[k]
            zncc = _zncc(p_l_norm, right, norm_r, y, xg - d, half)
            costs[k] = 4.0 if zncc == -1.0 else 2.0 - 2.0 * zncc

        k_star = int(np.argmin(costs))
        c_star = costs[k_star]
        d_star = d_min + k_star

        if k_star == 0 or k_star == nd - 1:
            continue

        if c_star > max_cost:
            continue

        c_second = np.inf
        for k in range(nd):
            if abs((d_min + k) - d_star) > 2 and costs[k] < c_second:   # mask_second
                c_second = costs[k]
        if not np.isfinite(c_second):
            continue
        if c_star > ratio * c_second:
            continue

        if cross_check:
            xd_star = xg - d_star
            # On recherche depuis xd_star vers la gauche sur left[y]
            d_rev_upper = min(d_max, (W - 1 - half) - xd_star)
            if d_rev_upper < d_min:
                continue

            p_r_norm, _ = _patch_norm(right, y, xd_star, half)
            nr = d_rev_upper - d_min + 1
            costs_rev = np.empty(nr)
            for k in range(nr):
                zncc = _zncc(p_r_norm, left, norm_l, y, xd_star + d_min + k, half)   # xl_cand
                costs_rev[k] = 4.0 if zncc == -1.0 else 2.0 - 2.0 * zncc

            d_rev_star = d_min + int(np.argmin(costs_rev))
            if abs(d_rev_star - d_star) > 1:
                continue

        c_prev = costs[k_star - 1]
        c_next = costs[k_star + 1]
        denom = 2.0 * (c_prev - 2.0 * c_star + c_next)

        if abs(denom) > 1e-6:
            delta = (c_prev - c_next) / denom
            delta = min(max(delta, -0.5), 0.5)
        else:
            delta = 0.0

        d_sub = d_star + delta
        xd_sub = xg_f - d_sub

        out[idx, 0] = xg_f
        out[idx, 1] = y
        out[idx, 2] = xd_sub
        out[idx, 3] = d_sub
        out[idx, 4] = c_star
        keep[idx] = True
    return out, keep


def match_stereo(left: np.ndarray, right: np.ndarray, corners_l: np.ndarray,d_min: int = 1, d_max: int = 120,
    half: int = 5, ratio: float = 0.8, max_cost: float = 0.5, cross_check: bool = True):
    if len(corners_l) == 0:
        return np.empty((0, 5), dtype=np.float32)

    left = np.ascontiguousarray(left, dtype=np.float32)
    right = np.ascontiguousarray(right, dtype=np.float32)
    norm_l = _window_norms(left, half)      # normes des fenêtres, calculées une fois pour toute l'image
    norm_r = _window_norms(right, half)

    y_coords = np.round(corners_l[:, 1]).astype(np.int64)
    xg_f_all = np.ascontiguousarray(corners_l[:, 0], dtype=np.float64)

    out, keep = _match_corners(left, right, norm_l, norm_r, xg_f_all, y_coords, d_min, d_max, half, ratio, max_cost, cross_check)

    ordre = np.argsort(y_coords, kind="stable")
    ordre = ordre[keep[ordre]]
    if len(ordre) == 0:
        return np.empty((0, 5), dtype=np.float32)

    return out[ordre].astype(np.float32)

if __name__ == "__main__":
    import time
    from scipy.ndimage import shift

    H, W = 480, 640
    np.random.seed(42)

    # Création d'une image gauche avec texture aléatoire lissée
    base = np.random.uniform(0.0, 1.0, size=(H, W)).astype(np.float32)
    from scipy.ndimage import gaussian_filter
    left_img = gaussian_filter(base, sigma=2.0)

    # Image droite décalée de d_vrai = 18.3 pixels vers la gauche
    d_vrai = 18.3
    right_img = shift(left_img, shift=(0, -d_vrai), order=3, mode="reflect")

    # Simulation de coins artificiels réguliers
    corners_test = np.array([
        [150.0, 100.0, 100.0],
        [250.0, 200.0, 120.0],
        [400.0, 300.0, 150.0],
    ], dtype=np.float32)

    t0 = time.perf_counter()
    res = match_stereo(left_img, right_img, corners_test, d_min=1, d_max=60)
    t1 = time.perf_counter()

    print(f"Temps appariement : {(t1 - t0)*1000:.2f} ms")
    print(f"Appariements trouvés : {len(res)} / {len(corners_test)}")
    if len(res) > 0:
        print("Résultats [x_g, y, x_d, d, cout] :")
        print(res)
        erreurs = np.abs(res[:, 3] - d_vrai)
        print(f"Erreur moyenne sous-pixel sur d : {np.mean(erreurs):.3f} px (attendu < 0.2 px)")
        assert np.all(erreurs < 0.2), "Erreur de disparité sous-pixel trop élevée"
        print("✓ Test de matching validé.")
    