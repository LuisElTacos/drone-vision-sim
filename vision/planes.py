from dataclasses import dataclass
import numpy as np


@dataclass
class Plane:
  n: np.ndarray  # normale unitaire, orientée vers le drone
  c: float  # n . P = c
  inliers: np.ndarray  # indices des points dans le nuage initial pts
  kind: str  # 'sol', 'mur', 'autre'
  origin: np.ndarray  # centre de gravité des inliers
  u: np.ndarray  # vecteur horizontal de la base 2D
  v: np.ndarray  # vecteur vertical de la base 2D


def fit_plane(pts):
    """ Renvoie le plan (n, c) qui approxime les points pts (N, 3) au sens des moindres carrés, avec n normalisé et c = n.T @ p0 pour un point p0 du plan. """
    if len(pts) < 3:
        return None, None
    p0 = np.mean(pts, axis=0)
    pts_centered = pts - p0
    _, _, vh = np.linalg.svd(pts_centered)
    n = vh[-1]
    norm_n= np.linalg.norm(n)
    if norm_n < 1e-10:
        return None, None
    n /= norm_n
    c = np.dot(n, p0)
    return n, c

def ransac_plane(pts, sigma, eps=0.05, iters=300, rng=None, C=None):
    N=len(pts)

    if N < 3:
        return None, None, np.empty(0, dtype=np.int64)

    if rng is None:
        rng = np.random.default_rng()

    best_inliers = np.empty(0, dtype=np.int64)
    best_n, best_c=None, None
    for _ in range(iters):
        tirage= rng.choice(N, size=3, replace=False)
        n=np.cross(pts[tirage[1]]-pts[tirage[0]], pts[tirage[2]]-pts[tirage[0]])
        if np.linalg.norm(n)<1e-6:
            continue
        n/=np.linalg.norm(n)
        c=np.dot(n, pts[tirage[0]])
        dists = np.abs(pts @ n - c)
        best_inliers_2=np.where(dists< eps + 2*sigma)[0]
        if len(best_inliers_2)>len(best_inliers):
            best_inliers=best_inliers_2
            best_n=n
            best_c=c
    if best_n is None or len(best_inliers) < 3:
        return None, None, np.empty(0, dtype=np.int64)

    best_n, best_c = fit_plane(pts[best_inliers])
    if best_n is None: return None, None, np.empty(0, dtype=np.int64)

    dists_refined = np.abs(pts @ best_n - best_c)
    final_inliers = np.flatnonzero(dists_refined < (eps + 2.0 * sigma))

    if C is not None and len(final_inliers) > 0:
        p_ref = pts[final_inliers[0]]
        if np.dot(best_n, C - p_ref) < 0:
            best_n = -best_n
            best_c = -best_c
    return best_n, best_c, final_inliers


def compute_plane_basis_and_kind(n, inlier_pts):
    """Calcule l'étiquetage (sol, mur, autre), le centre de gravité et la base 2D (u, v)."""

    origin = np.mean(inlier_pts, axis=0)
    up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    cos_up = abs(float(np.dot(n, up)))

    # 1. Étiquetage
    if cos_up > 0.9:
    # Plan horizontal : sol si proche de Y = 0 (ex: < 0.5m)
        kind = 'sol' if abs(origin[1]) < 0.5 else 'autre'
    elif cos_up < 0.2:
        kind = 'mur'
    else:
        kind = 'autre'

    # 2. Base 2D du plan (u horizontal le long du mur, v vertical)
    u_dir = np.cross(up, n)
    norm_u = np.linalg.norm(u_dir)
    if norm_u < 1e-4:
    # Cas particulier : plan horizontal (sol), on utilise l'axe X monde
        ref_x = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        u_dir = np.cross(n, ref_x)
        norm_u = np.linalg.norm(u_dir)

    u = u_dir / norm_u if norm_u >= 1e-4 else np.array([1.0, 0.0, 0.0])
    v = np.cross(n, u)
    norm_v = np.linalg.norm(v)
    v = v / norm_v if norm_v >= 1e-4 else np.array([0.0, 1.0, 0.0])

    return kind, origin, u, v

from scipy.ndimage import binary_dilation, label


def keep_largest_cluster_2d(pts_inliers, u, v, origin, cell_size=0.25):
    """ Conserve uniquement les inliers de la plus grande composante pour chaque plan. 
    Permet d'éviter que deux plans parallèles proches soient confondus. """

    K = len(pts_inliers)
    if K < 3:
        return np.arange(K, dtype=np.int64)

    # 1. Coordonnées 2D locales (u_coord le long de u, v_coord le long de v)
    pts_centered = pts_inliers - origin
    u_coords = pts_centered @ u
    v_coords = pts_centered @ v

    u_min, u_max = u_coords.min(), u_coords.max()
    v_min, v_max = v_coords.min(), v_coords.max()

    nu = int(np.ceil((u_max - u_min) / cell_size)) + 1
    nv = int(np.ceil((v_max - v_min) / cell_size)) + 1

    # grille 2D et remplissage
    col_indices = np.clip(
        np.floor((u_coords - u_min) / cell_size).astype(int), 0, nu - 1
    )
    row_indices = np.clip(
        np.floor((v_coords - v_min) / cell_size).astype(int), 0, nv - 1
    )

    grid = np.zeros((nv, nu), dtype=bool)
    grid[row_indices, col_indices] = True

    # relie les coins de Harris voisins d'un même panneau
    structure = np.ones((3, 3), dtype=bool)
    grid_dilated = binary_dilation(grid, structure=structure)

    # Étiquetage des régions connexes
    labeled_grid, num_features = label(grid_dilated)
    if num_features <= 1:
        return np.arange(K, dtype=np.int64)

    #Décompte des points par label
    point_labels = labeled_grid[row_indices, col_indices]

    # On ignore le label 0 (les pixels vides) et on trouve le label le plus fréquent parmi les inliers
    valid_labels = point_labels[point_labels > 0]
    if len(valid_labels) == 0:
        return np.arange(K, dtype=np.int64)

    counts = np.bincount(valid_labels)
    best_label = int(np.argmax(counts))

    # Masque des indices retenus parmi les inliers
    kept_cluster_indices = np.flatnonzero(point_labels == best_label)
    return kept_cluster_indices


def extract_planes(pts, sigma=None, C=None, min_inliers=40, max_planes=6, eps=0.05, max_iter=300, rng=None):
    """Extrait jusqu'à max_planes plans par RANSAC.

    pts   : (N, 3) coordonnées monde des points
    sigma : (N,) incertitude de profondeur par point
    C     : (3,) position caméra monde (pour orienter n vers le drone)
    """
    if sigma is None:
        sigma=0.0
    N = len(pts)
    if N < min_inliers:
        return []

    if rng is None:
        rng = np.random.default_rng()

    planes = []
    active_indices = np.arange(N, dtype=np.int64)

    while len(planes) < max_planes and len(active_indices) >= min_inliers:
        pts_sub = pts[active_indices]
        sigma_sub = sigma[active_indices] if hasattr(sigma, '__len__') else sigma

        # RANSAC sur les points restants
        n, c, local_inliers = ransac_plane(pts_sub, sigma_sub, eps=eps, iters=max_iter, rng=rng, C=C)

        # Condition d'arrêt : aucun plan significatif trouvé
        if n is None or len(local_inliers) < min_inliers:
            break

        pts_inliers = pts_sub[local_inliers]
        kind, origin, u, v = compute_plane_basis_and_kind(n, pts_inliers)

        kept_sub_idx = keep_largest_cluster_2d(pts_inliers, u, v, origin)

        if len(kept_sub_idx) < min_inliers:
            mask_remaining = np.ones(len(active_indices), dtype=bool)
            mask_remaining[local_inliers] = False
            active_indices = active_indices[mask_remaining]
            continue

        final_local_inliers = local_inliers[kept_sub_idx]
        global_inliers = active_indices[final_local_inliers]

        kind, origin, u, v = compute_plane_basis_and_kind(n, pts[global_inliers])

        plane_obj = Plane(n=n, c=c, inliers=global_inliers, kind=kind, origin=origin, u=u, v=v)
        planes.append(plane_obj)

        # Retrait des inliers pour la prochaine itération
        mask_remaining = np.ones(len(active_indices), dtype=bool)
        mask_remaining[final_local_inliers] = False
        active_indices = active_indices[mask_remaining]

    return planes
