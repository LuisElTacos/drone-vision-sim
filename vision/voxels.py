import numpy as np
from numba import njit

@njit(cache=True)
def _integrate(L, seen_free, seen_occ, stamp, pts, c, t, origin, vs, val_free, val_occ, max_par_frame, occ_lock):
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
        occ_lock : un rayon s'arrête sur un voxel occupé de façon sûre (L > occ_lock) au lieu de le traverser, et son point est ignoré
    """
    Nx, Ny, Nz = L.shape
    n = pts.shape[0]
    pidx = np.empty((n, 3), np.int64)
    for r in range(n):
        for a in range(3):
            pidx[r, a] = np.int64(np.floor((pts[r, a] - origin[a]) / vs))
    blocked_ray = np.zeros(n, np.bool_)
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
            if L[i, j, k] > occ_lock:
                blocked_ray[r] = True    # le rayon traverse une surface sûre : point incohérent avec la carte
                break
            v = seen_free[i, j, k]
            if v < stamp:                      # premier rayon de cette image dans ce voxel
                v = stamp
            if v - stamp < max_par_frame:           # au plus max_par_frame rayons comptés par voxel et par image
                seen_free[i, j, k] = v + 1
                if L[i, j, k] <= 0.0:          # un voxel qui a reçu un impact (L > 0) reste occupé : un rayon libre ne l'efface pas
                    L[i, j, k] += val_free
    # 2. voxels des points
    for r in range(n):
        if blocked_ray[r]:
            continue
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
        self.map = np.zeros(self.shape,dtype=np.float32) ## grille de voxels de taille voxel_size
        self.origin=origin
        self.voxel_size=voxel_size
        self._seen_free = np.zeros(self.shape, np.int32)
        self._seen_occ = np.zeros(self.shape, np.int32)
        self._stamp = 0
        
    def integrate(self, points_world, cam_center, max_range=15.0,val_free=-0.4,val_occ=0.85, step_frac=0.5, max_par_frame=2,
                  occ_lock=2.0):
        self.integrate_multi([points_world], [cam_center], max_range=max_range, val_free=val_free, val_occ=val_occ,
                             step_frac=step_frac, max_par_frame=max_par_frame, occ_lock=occ_lock)

    def integrate_multi(self, points_list, centers, max_range=15.0, val_free=-0.4, val_occ=0.85, step_frac=0.5,
                        max_par_frame=2, occ_lock=2.0):
        """Plusieurs paires stéréo vues au même instant : points_list[p] (N_p, 3) vus depuis centers[p].

        Un seul tampon pour toutes les paires : dans une image, chaque voxel reçoit au plus max_par_frame
        mises à jour « libre » et « occupé », même s'il est vu par plusieurs paires (comme integrate
        pour une seule paire)."""
        max_par_frame = int(min(max(max_par_frame, 1), 255))   # le compteur par voxel tient sur 8 bits (stamp * 256)
        self._stamp += 1
        origin = np.asarray(self.origin, np.float64)
        for points_world, cam_center in zip(points_list, centers):
            cam_center = np.asarray(cam_center, float)
            if len(points_world) == 0:
                continue
            dist = np.linalg.norm(points_world - cam_center, axis=1)
            pts = np.ascontiguousarray(points_world[dist <= max_range], dtype=np.float64)
            if len(pts) == 0:
                continue
            length = np.linalg.norm(pts - cam_center, axis=1).max()
            n_ech = int(length / (self.voxel_size * step_frac)) + 1
            t = np.linspace(0, 1, n_ech)
            _integrate(self.map, self._seen_free, self._seen_occ, self._stamp * 256, pts, cam_center, t, origin,
                       float(self.voxel_size), np.float32(val_free), np.float32(val_occ), int(max_par_frame),
                       np.float32(occ_lock))
        np.clip(self.map, -2.0, 3.5, out=self.map)

    def world_to_idx(self,p):
        """ Donne les indices du voxel contenant p"""
        return np.floor((p - self.origin) / self.voxel_size).astype(int) 
    
    def idx_to_world(self,idx):
        """ Donne les coordonées dans le monde du voxel """
        return idx*self.voxel_size+self.origin+self.voxel_size/2 #centre du voxel
         
        
               
    