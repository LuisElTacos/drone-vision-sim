import numpy as np
from scipy.ndimage import gaussian_filter, sobel, maximum_filter
import cv2 as cv2


def _ksize(sigma):
    return 2 * int(4.0 * sigma + 0.5) + 1          # troncature de scipy (truncate = 4)


def harris_function(img, sigma_d=1.0, sigma_i=1.5, k=0.04, version="ours"):
    """ Calcule la fonction de Harris en chaque pixel de l'image img.
    version="ours" : notre formule (gaussienne sigma_d, Sobel 3x3, fenêtre gaussienne sigma_i) ;
    version="opencv" : même chose avec cv2.cornerHarris (fenêtre en boîte), plus rapide ; échelle différente (seuil relatif)."""
    if version == "opencv":
        return harris_opencv(img, sigma_d, sigma_i, k)
    img = np.ascontiguousarray(img, dtype=np.float32) ## nécessaire pour utiliser les fonctions de cv2
    kd, ki = _ksize(sigma_d), _ksize(sigma_i)
    g = cv2.GaussianBlur(img, (kd, kd), sigma_d, borderType=cv2.BORDER_REFLECT) ## réduction du bruit avant calcul des dérivées
    Ix = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3, borderType=cv2.BORDER_REFLECT)
    Iy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3, borderType=cv2.BORDER_REFLECT)
    Sxx = cv2.GaussianBlur(Ix * Ix, (ki, ki), sigma_i, borderType=cv2.BORDER_REFLECT)    
    Sxy = cv2.GaussianBlur(Ix * Iy, (ki, ki), sigma_i, borderType=cv2.BORDER_REFLECT)
    Syy = cv2.GaussianBlur(Iy * Iy, (ki, ki), sigma_i, borderType=cv2.BORDER_REFLECT)
    trace=Sxx+Syy
    return Sxx*Syy-Sxy*Sxy-k*trace**2  

def harris_opencv(img, sigma_d=1.0, sigma_i=1.5, k=0.04):
    """Version OpenCV : lissage gaussien sigma_d, puis cv2.cornerHarris (Sobel 3x3, somme en boîte blockSize x blockSize).
    blockSize = 2*ceil(1.5*sigma_i)+1 = 7 pour sigma_i = 1.5 (boîte de largeur proche de celle de la gaussienne de sigma_i)."""
    img = np.ascontiguousarray(img, dtype=np.float32)
    kd = _ksize(sigma_d)
    g = cv2.GaussianBlur(img, (kd, kd), sigma_d, borderType=cv2.BORDER_REFLECT)
    bs = 2 * int(np.ceil(1.5 * sigma_i)) + 1
    return cv2.cornerHarris(g, bs, 3, k, borderType=cv2.BORDER_REFLECT)


# Candidats du dernier appel (même carte R) : un 2e appel sur la même carte avec un seuil et une bordure
# plus stricts (ex. FrontEnd.renew après la détection de test_vslam) les filtre au lieu de tout recalculer.
_cache_cand = None


def detect_corners(R, rel_thresh=0.01, nms=5, max_n=500, border=6, grid_cells=(6,8), cache=True):
    global _cache_cand
    M=float(R.max())
    N,P=R.shape
    if M <= 0.0:
        return np.empty((0, 3), dtype=np.float32)
    c = _cache_cand
    if cache and c is not None and c[0] is R and c[1] == nms and rel_thresh >= c[2] and border >= c[3] and c[4] == M:
        # mêmes candidats, dans le même ordre (ligne par ligne), que le calcul complet ci-dessous
        y_idx, x_idx = c[5], c[6]
        keep = (R[y_idx, x_idx] > (rel_thresh * M)) & (y_idx >= border) & (y_idx < N - border) & (x_idx >= border) & (x_idx < P - border)
        y_idx, x_idx = y_idx[keep], x_idx[keep]
    else:
        mask_thresh = R > (rel_thresh * M) ## garde uniquement les R au-dessus d'un certain seuil
        mask_nms = R == cv2.dilate(R, np.ones((nms, nms), np.uint8))  ## Supprime les non-maxima locaux
        mask_border = np.zeros((N, P), dtype=bool) ## Supprime les points trop près des bords
        mask_border[border : N - border, border : P - border] = True
        candidates=mask_thresh & mask_nms & mask_border
        y_idx, x_idx=np.nonzero(candidates)
        if cache:
            _cache_cand = (R, nms, rel_thresh, border, M, y_idx, x_idx)

    if len(x_idx) == 0:
        return np.empty((0, 3), dtype=np.float32)
    
    scores = R[y_idx, x_idx]
    n_lignes, n_cols = grid_cells
    pts_par_cell = int(np.ceil(max_n / (n_lignes * n_cols)))

    cell_n = N / n_lignes ## Hauteur de la case 
    cell_p = P / n_cols ## Largeur de la case

    lignes_bins = (y_idx / cell_n).astype(int) ## donne la rangée de la cellule de chaque pixel conservé
    col_bins = (x_idx / cell_p).astype(int) ## donne la colonne de la cellule de chaque pixel conservé
    cell_ids = lignes_bins * n_cols + col_bins ## donne l'identifiant de la cellule de chaque pixel conservé

    selected=[]
    order = np.lexsort((-scores, cell_ids))            # cellule croissante, puis score décroissant
    cell_ids_tries = cell_ids[order]                   # numéros de cellule dans cet ordre
    first = np.searchsorted(cell_ids_tries, cell_ids_tries)  # indice du 1er coin de sa cellule
    rank = np.arange(len(order)) - first               # rang du coin dans sa cellule
    selected = order[rank < pts_par_cell]              # les pts_par_cell meilleurs de chaque cellule

    selected = np.array(selected, dtype=np.int64)

    if len(selected)>max_n:
        top_global=np.argsort(-scores[selected])[:max_n]
        selected=selected[top_global]

    return np.column_stack(
        [
            x_idx[selected].astype(np.float32),
            y_idx[selected].astype(np.float32),
            scores[selected].astype(np.float32),
        ]
    )
