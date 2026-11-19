import numpy as np

def triangulate(uL, vL, uR, vR, K, B, R, C, d_min=0.5):
    """ Renvoie les coordonnées 3D monde des points à partir de leurs coordonnées pixel dans les images gauche et droite, ainsi qu'un masque de validité.
    Args:
        uL, vL, uR, vR : coordonnées pixel (gauche/droite) des points correspondants
        K : matrice intrinsèques (3, 3)
        B : distance entre les caméras (m)
        R : rotation caméra->monde (3, 3)
        C : position caméra dans le monde (3,)
        d_min : distance minimale pour considérer un point valide
    """
    f, cx, cy = K[0, 0], K[0, 2], K[1, 2]
    d = uL - uR
    valid = d > d_min
    Z = np.where(valid, f * B / np.where(valid, d, 1.0), np.nan)
    X = (uL - cx) * Z / f
    Y = (vL - cy) * Z / f
    pts_cam = np.stack([X, Y, Z], axis=1)
    return pts_cam @ R.T + C, valid