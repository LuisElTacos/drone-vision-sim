import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def _normalize_patch(patch: np.ndarray, eps: float = 1e-5) -> tuple[np.ndarray | None, float]:
    p_flat = patch.astype(np.float32).ravel()
    p_centered = p_flat - np.mean(p_flat)
    norm = float(np.linalg.norm(p_centered))
    if norm < eps:
        return None, 0.0
    return (p_centered / norm), norm

def match_stereo(left: np.ndarray, right: np.ndarray, corners_l: np.ndarray,d_min: int = 1, d_max: int = 120,
    half: int = 5, ratio: float = 0.8, max_cost: float = 0.5, cross_check: bool = True):
    H, W = left.shape
    patch_size = 2 * half + 1

    if len(corners_l) == 0:
        return np.empty((0, 5), dtype=np.float32)

    matches = []

    y_coords = np.round(corners_l[:, 1]).astype(int)
    unique_ys = np.unique(y_coords)

    for y in unique_ys:
        if y - half < 0 or y + half >= H:
            continue

        right_band = right[y - half : y + half + 1, :]
        windows_r = sliding_window_view(right_band, (patch_size, patch_size))[0]
        n_windows_r = windows_r.shape[0]

        flat_r = windows_r.reshape(n_windows_r, -1).astype(np.float32)
        mean_r = np.mean(flat_r, axis=1, keepdims=True)
        diff_r = flat_r - mean_r
        norm_r = np.linalg.norm(diff_r, axis=1, keepdims=True)
        valid_r = (norm_r[:, 0] > 1e-5)

        normed_r = np.zeros_like(diff_r)
        normed_r[valid_r] = diff_r[valid_r] / norm_r[valid_r]


        idx_corners = np.where(y_coords == y)[0]

        for idx in idx_corners:
            xg_f = corners_l[idx, 0]
            xg = int(round(xg_f))

            if xg - half < 0 or xg + half >= W:
                continue


            patch_l = left[y - half : y + half + 1, xg - half : xg + half + 1]
            p_l_norm, norm_val = _normalize_patch(patch_l)
            if p_l_norm is None:
                continue  # Patch trop uniforme

            d_upper = min(d_max, xg - half)
            if d_upper < d_min + 2:
                continue

            d_vals = np.arange(d_min, d_upper + 1, dtype=int)
            idx_r = xg - d_vals - half


            sub_normed_r = normed_r[idx_r]
            sub_valid_r = valid_r[idx_r]

            zncc = sub_normed_r @ p_l_norm
            costs = 2.0 - 2.0 * zncc
            costs[~sub_valid_r] = 4.0  


            k_star = int(np.argmin(costs))
            c_star = float(costs[k_star])
            d_star = int(d_vals[k_star])


            if k_star == 0 or k_star == len(d_vals) - 1:
                continue

            if c_star > max_cost:
                continue

            dist_to_star = np.abs(d_vals - d_star)
            mask_second = dist_to_star > 2
            if not np.any(mask_second):
                continue

            c_second = float(np.min(costs[mask_second]))
            if c_star > ratio * c_second:
                continue

            if cross_check:
                xd_star = xg - d_star
                # On recherche depuis xd_star vers la gauche sur left[y]
                d_rev_upper = min(d_max, (W - 1 - half) - xd_star)
                if d_rev_upper < d_min:
                    continue

                d_rev_vals = np.arange(d_min, d_rev_upper + 1, dtype=int)
                xl_cand = xd_star + d_rev_vals

                cand_patches_l = np.stack(
                    [left[y - half : y + half + 1, x_c - half : x_c + half + 1].ravel() for x_c in xl_cand]
                ).astype(np.float32)

                cand_patches_l -= np.mean(cand_patches_l, axis=1, keepdims=True)
                norms_l = np.linalg.norm(cand_patches_l, axis=1, keepdims=True)
                valid_rev_l = (norms_l[:, 0] > 1e-5)
                cand_patches_l[valid_rev_l] /= norms_l[valid_rev_l]

                p_r_norm = sub_normed_r[k_star]
                costs_rev = 2.0 - 2.0 * (cand_patches_l @ p_r_norm)
                costs_rev[~valid_rev_l] = 4.0

                d_rev_star = d_rev_vals[np.argmin(costs_rev)]
                if abs(d_rev_star - d_star) > 1:
                    continue

            c_prev = float(costs[k_star - 1])
            c_next = float(costs[k_star + 1])
            denom = 2.0 * (c_prev - 2.0 * c_star + c_next)

            if abs(denom) > 1e-6:
                delta = (c_prev - c_next) / denom
                delta = float(np.clip(delta, -0.5, 0.5))
            else:
                delta = 0.0

            d_sub = float(d_star + delta)
            xd_sub = float(xg_f - d_sub)

            matches.append([float(xg_f), float(y), xd_sub, d_sub, c_star])

    if len(matches) == 0:
        return np.empty((0, 5), dtype=np.float32)

    return np.array(matches, dtype=np.float32)

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
    