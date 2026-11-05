import numpy as np

class VoxelMap:
    def __init__(self, origin, size_m, voxel_size):
        self.shape = np.ceil(np.asarray(size_m) / voxel_size).astype(int)
        self.map = np.zeros(self.shape,dtype=np.float32)
        self.origin=origin
        self.voxel_size=voxel_size
        
    def integrate(self, points_world, cam_center, max_range=15.0,val_free=-0.4,val_occ=0.85):
        cam_center = np.asarray(cam_center, float)
        dist = np.linalg.norm(points_world - cam_center, axis=1)
        points_world = points_world[dist <= max_range]
        if len(points_world) == 0:
            return
        free, occ = self.raycast(points_world, cam_center)
        self.map[free[:, 0], free[:, 1], free[:, 2]] += val_free
        self.map[occ[:, 0], occ[:, 1], occ[:, 2]] += val_occ
        np.clip(self.map, -2.0, 3.5, out=self.map)
               
               
    def world_to_idx(self,p):
        return np.floor((p - self.origin) / self.voxel_size).astype(int) 
    
    def idx_to_world(self,idx):
        return idx*self.voxel_size+self.origin+self.voxel_size/2 #centre du voxel
    
    def raycast(self,points_world, cam_center, step_frac=0.5):
        d = points_world - cam_center                      # (N, 3)  un vecteur par rayon
        length = np.linalg.norm(d, axis=1).max()                       # rayon le plus long (m)
        n_echantillonage = int(length / (self.voxel_size * step_frac)) + 1
        t = np.linspace(0, 1, n_echantillonage)[None, :, None]  # (1, n, 1)  n positions le long du rayon
        pts = cam_center + t * d[:, None, :]            # (N, n, 3)
        idx = self.world_to_idx(pts) # (N, n, 3) indices
        ok = ((idx >= 0) & (idx < self.shape)).all(axis=-1)
        idx_P = self.world_to_idx(points_world)
        not_P = (idx != idx_P[:, None, :]).any(axis=-1)
        free = np.unique(idx[ok & not_P], axis=0)
        ok_P = ((idx_P >= 0) & (idx_P < self.shape)).all(axis=-1)
        occ = np.unique(idx_P[ok_P], axis=0)
        return free, occ
        
        
               
    