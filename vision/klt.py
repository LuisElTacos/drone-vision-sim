import cv2
import numpy as np

def build_pyramid(gray, levels):
    """Liste [niveau 0 (plein), 1 (1/2), ...] d'images float32 + gradients (Scharr / 32)."""
    pyr = []
    img = np.ascontiguousarray(gray, np.float32)
    for lv in range(levels):
        if lv > 0:
            img = cv2.pyrDown(img)
        gx = cv2.Scharr(img, cv2.CV_32F, 1, 0) / 32.0
        gy = cv2.Scharr(img, cv2.CV_32F, 0, 1) / 32.0
        pyr.append((img, gx, gy))
    return pyr

def _sample(img, X, Y):
    """Interpolation bilinéaire de img aux positions (N, M) X, Y (bord répliqué)."""
    return cv2.remap(img, X.astype(np.float32), Y.astype(np.float32), cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REPLICATE)

class KLT:
    def __init__(self, win=15, levels=3, iters=10, eps=0.03, min_eig=1e-6, max_res=0.15):
        self.win, self.levels, self.iters, self.eps = win, levels, iters, eps
        self.min_eig, self.max_res = min_eig, max_res
        h = win // 2
        self.ox = np.array([i for j in range(-h, h+1) for i in range(-h, h+1)], dtype=np.float32)
        self.oy = np.array([j for j in range(-h, h+1) for i in range(-h, h+1)], dtype=np.float32)
        
    def pyramid(self, gray):
        return build_pyramid(gray, self.levels)
        
    def track(self, pyrI, pyrJ, p0, p_init=None):
        n = len(p0)
        if n == 0:
            return np.empty((0, 2)), np.zeros(0, bool)
        p_init = p0 if p_init is None else np.asarray(p_init, np.float64).reshape(-1, 2)
        L = len(pyrI)
        d = (p_init - p0) / 2 ** (L - 1)
        ok = np.ones(n, bool)
        for level in range(L-1,-1,-1):
            s = 2.0 ** level
            I, Ix, Iy = pyrI[level]
            J = pyrJ[level][0]
            center = p0/s
            X = center[:,0:1]+ self.ox[None]
            Y = center[:,1:2] + self.oy[None]
            T, gx, gy = _sample(I, X, Y), _sample(Ix, X, Y), _sample(Iy, X, Y)
            a, b, c = (gx * gx).sum(1), (gx * gy).sum(1), (gy * gy).sum(1)
            det = a*c-b*b
            lam_min = 0.5 * (a + c - np.sqrt((a - c) ** 2 + 4 * b * b)) / len(self.ox) #valeur propre minimale
            ok &= lam_min > self.min_eig
            active = ok.copy()
            for _ in range(self.iters):
                if not active.any(): break
                J_w = _sample(J, X + d[:, 0:1], Y + d[:, 1:2])
                e = T - J_w
                bx, by = (e * gx).sum(1), (e * gy).sum(1)
                dx = (c * bx - b * by) / det
                dy = (a * by - b * bx) / det
                step = np.where(active[:, None], np.column_stack([dx, dy]), 0.0)
                d += step
                active &= np.hypot(step[:, 0], step[:, 1]) > self.eps
            if level > 0:
                d *= 2.0
            else:
                res = np.abs(T - _sample(J, X + d[:, 0:1], Y + d[:, 1:2])).mean(1)
        p1 = p0 + d
        H, W = pyrJ[0][0].shape
        ok &= (res < self.max_res) & np.isfinite(p1).all(1)
        ok &= (p1[:, 0] >= 0) & (p1[:, 0] <= W - 1) & (p1[:, 1] >= 0) & (p1[:, 1] <= H - 1)
        return p1, ok
            
    def track_fb(self, pyrI, pyrJ, p0, p_init=None, fb_max=1.0):
            """Suivi aller (I -> J) puis retour (J -> I) : rejet si l'écart retour > fb_max px."""
            p1, ok = self.track(pyrI, pyrJ, p0, p_init)
            if ok.any():
                pb, okb = self.track(pyrJ, pyrI, p1[ok], p0[ok])
                fb = np.full(len(p0), np.inf)
                fb[ok] = np.where(okb, np.linalg.norm(pb - p0[ok], axis=1), np.inf)
                ok &= fb < fb_max
            return p1, ok