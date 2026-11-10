import numpy as np
from numba import njit

@njit(cache=True)
def _integrate(L, seen_free, seen_occ, stamp, pts, c, t, origin, vs, val_free, val_occ, max_par_frame):
    """ Intègre les points pts dans la carte de voxels L, en marquant les voxels traversés comme libres et les voxels contenant les points comme occupés.
    Args:
        L : carte de voxels à mettre à jour
        seen_free : tableau avec les voxels libres vus
        seen_occ : tableau avec les voxels occupés vus
        stamp : base du compteur de mise à jour (numéro d'image * 256) ; seen_* vaut stamp + nb de rayons déjà comptés
        pts : points 3D monde à intégrer (N, 3)
        c : centre de la caméra (3,)
        t : échantillons le long des rayons (1D)
        origin : origine du repère de la carte de voxels (3,)
        vs : taille d'un voxel (float)
        val_free : valeur à ajouter aux voxels libres
        val_occ : valeur à ajouter aux voxels occupés
        max_par_frame : nb max de rayons (ou impacts) comptés par voxel sur une même image (0 < max_par_frame < 256) ; au-delà, les rayons suivants sont ignorés
    """
    Nx, Ny, Nz = L.shape
    n = pts.shape[0]
    pidx = np.empty((n, 3), np.int64)
    for r in range(n):
        for a in range(3):
            pidx[r, a] = np.int64(np.floor((pts[r, a] - origin[a]) / vs))
    # 1. voxels traversés (hors voxel du point), chacun une seule fois
    for r in range(n):
        d0 = pts[r, 0] - c[0]
        d1 = pts[r, 1] - c[1]
        d2 = pts[r, 2] - c[2]
        for s in range(len(t)):
            i = np.int64(np.floor((c[0] + t[s] * d0 - origin[0]) / vs))
            j = np.int64(np.floor((c[1] + t[s] * d1 - origin[1]) / vs))
            k = np.int64(np.floor((c[2] + t[s] * d2 - origin[2]) / vs))
            if i < 0 or j < 0 or k < 0 or i >= Nx or j >= Ny or k >= Nz:
                continue
            if i == pidx[r, 0] and j == pidx[r, 1] and k == pidx[r, 2]:
                continue
            v = seen_free[i, j, k]
            if v < stamp:                      # premier rayon de cette image dans ce voxel
                v = stamp
            if v - stamp < max_par_frame:           # au plus max_par_frame rayons comptés par voxel et par image
                seen_free[i, j, k] = v + 1
                if L[i, j, k] <= 0.0:          # un voxel qui a reçu un impact (L > 0) reste occupé : un rayon libre ne l'efface pas
                    L[i, j, k] += val_free
    # 2. voxels des points
    for r in range(n):
        i, j, k = pidx[r, 0], pidx[r, 1], pidx[r, 2]
        if i < 0 or j < 0 or k < 0 or i >= Nx or j >= Ny or k >= Nz:
            continue
        v = seen_occ[i, j, k]
        if v < stamp:
            v = stamp
        if v - stamp < max_par_frame:
            seen_occ[i, j, k] = v + 1
            L[i, j, k] += val_occ
            
class VoxelMap:
    def __init__(self, origin, size_m, voxel_size):
        self.shape = np.ceil(np.asarray(size_m) / voxel_size).astype(int)
        self.map = np.zeros(self.shape,dtype=np.float32)
        self.origin=origin
        self.voxel_size=voxel_size
        self._seen_free = np.zeros(self.shape, np.int32)
        self._seen_occ = np.zeros(self.shape, np.int32)
        self._stamp = 0
        
    def integrate(self, points_world, cam_center, max_range=15.0,val_free=-0.4,val_occ=0.85, step_frac=0.5, max_par_frame=2):
        cam_center = np.asarray(cam_center, float)
        dist = np.linalg.norm(points_world - cam_center, axis=1)
        pts = np.ascontiguousarray(points_world[dist <= max_range], dtype=np.float64)
        if len(pts) == 0:
            return
        length = np.linalg.norm(pts - cam_center, axis=1).max()
        n_ech = int(length / (self.voxel_size * step_frac)) + 1
        t = np.linspace(0, 1, n_ech)
        max_par_frame = int(min(max(max_par_frame, 1), 255))   # le compteur par voxel tient sur 8 bits (stamp * 256)
        self._stamp += 1
        _integrate(self.map, self._seen_free, self._seen_occ, self._stamp * 256, pts, cam_center, t, np.asarray(self.origin, np.float64), float(self.voxel_size), np.float32(val_free), np.float32(val_occ), int(max_par_frame))
        np.clip(self.map, -2.0, 3.5, out=self.map)
          
    def world_to_idx(self,p):
        return np.floor((p - self.origin) / self.voxel_size).astype(int) 
    
    def idx_to_world(self,idx):
        return idx*self.voxel_size+self.origin+self.voxel_size/2 #centre du voxel
         
        
               
    