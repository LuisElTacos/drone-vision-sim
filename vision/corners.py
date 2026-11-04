import numpy as np
from scipy.ndimage import gaussian_filter, sobel, maximum_filter
import cv2 as cv2


def harris_function(img, sigma_d=1.0, sigma_i=1.5, k=0.04):
    g=gaussian_filter(img, sigma_d, mode='reflect')
    Ix=sobel(g, axis=1, mode='reflect')
    Iy=sobel(g, axis=0, mode='reflect')
    Ixx=Ix*Ix
    Ixy=Ix*Iy
    Iyy=Iy*Iy
    Sxx=gaussian_filter(Ixx, sigma_i, mode='reflect')
    Sxy=gaussian_filter(Ixy, sigma_i, mode='reflect')
    Syy=gaussian_filter(Iyy, sigma_i, mode='reflect')
    det=Sxx*Syy-Sxy*Sxy
    trace=Sxx+Syy
    R=det-k*trace**2
    return R

def detect_corners(R, rel_thresh=0.01, nms=5, max_n=500, border=6, grid_cells=(6,8)):
    M=float(R.max())
    N,P=R.shape
    if M <= 0.0:
        return np.empty((0, 3), dtype=np.float32)
    mask_thresh = R > (rel_thresh * M) ## garde uniquement les R au-dessus d'un certain seuil
    mask_nms = R == maximum_filter(R, size=nms, mode="constant", cval=-np.inf) ## Supprime les nom-maximum locaux
    mask_border = np.zeros((N, P), dtype=bool) ## Supprime les points trop près des bords
    mask_border[border : N - border, border : P - border] = True
    candidates=mask_thresh & mask_nms & mask_border
    y_idx, x_idx=np.nonzero(candidates)

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
    for cell_id in np.unique(cell_ids):
        in_cell = np.where(cell_ids == cell_id)[0]
        top_in_cell = in_cell[np.argsort(-scores[in_cell])[:pts_par_cell]]
        selected.extend(top_in_cell)

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
