import numpy as np

def triangulate(uL, vL, uR, vR, K, B, R, C, d_min=0.5):
    f, cx, cy = K[0, 0], K[0, 2], K[1, 2]
    d = uL - uR
    valid = d > d_min
    Z = np.where(valid, f * B / np.where(valid, d, 1.0), np.nan)
    X = (uL - cx) * Z / f
    Y = (vL - cy) * Z / f
    pts_cam = np.stack([X, Y, Z], axis=1)
    return pts_cam @ R.T + C, valid