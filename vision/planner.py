import numpy as np
import heapq as h
import math
from scipy.ndimage import distance_transform_edt

def classify(vmap, y_min, y_max, r_total, k_unk=0.35, k_prox=3.0, seuil_occ=0.0, seuil_free=0.0, d_secu=0.8, details=False):    
    vs = vmap.voxel_size

    # 1. tranche d'altitude : indices de couches j_min .. j_max (axe 1 = y vertical)
    j_min = max(0, int(np.ceil((y_min - vmap.origin[1]) / vs)))
    j_max = min(vmap.shape[1] - 1, int(np.floor((y_max - vmap.origin[1]) / vs)))
    L = vmap.map[:, j_min:j_max + 1, :]

    # 2. états
    occ = L > seuil_occ
    unknown = (~occ) & (L >= -seuil_free)            # ni occupé (L > seuil_occ), ni libre (L < -seuil_free) : avec 0/0, seul L == 0 (jamais observé)

    # 3. distance (m) de chaque voxel au voxel occupé le plus proche
    if occ.any():
        dist = distance_transform_edt(~occ) * vs
    else:
        dist = np.full(L.shape, np.inf)

    # 4. sorties
    blocked = dist < r_total                     # obstacle gonflé
    prox = np.clip((d_secu - dist) / d_secu, 0, 1) ** 2
    cost_extra = (k_unk * unknown + k_prox * prox).astype(np.float32)
    if details:                                  # pour follow : état franchement libre et distance aux obstacles
        libre = L < -seuil_free
        return blocked, cost_extra, j_min, libre, dist
    return blocked, cost_extra, j_min

def neighbours(n, shape):
    nx, ny, nz = shape
    i, j, k = n
    m = []
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            for dk in (-1, 0, 1):
                if (di, dj, dk) == (0, 0, 0):
                    continue
                a, b, c = i + di, j + dj, k + dk
                if 0 <= a < nx and 0 <= b < ny and 0 <= c < nz:
                    m.append((a, b, c))
    return m

def cout_arc(n, m, cost_extra, voxel_size):
    longueur = voxel_size * math.dist(n, m)      
    return longueur * (1 + cost_extra[m])       #facteur près des murs/sols et si pas sur qu'il soit libre

def heuristic(n, voxel_size, goal_idx):
    return voxel_size * math.dist(n, goal_idx)

def astar(blocked, cost_extra, start_idx, goal_idx, voxel_size, w=2.0):
    shape = blocked.shape
    open_heap = []
    hs=heuristic(start_idx, voxel_size, goal_idx)
    h.heappush(open_heap, (w *hs , hs, start_idx))
    g = np.full(blocked.shape, np.inf) 
    g[start_idx]=0
    closed = {}
    parent = {}
    while open_heap != []:
        n = h.heappop(open_heap)[2]                     # plus petit f
        if n in closed:              
            continue
        if n == goal_idx : 
            path = [n]
            while n in parent:  
                n = parent[n]
                path.append(n)
            path.reverse()          # start -> ... -> goal
            return path
        closed[n]=0
        for m in neighbours(n, shape) :
            if blocked[m] or m in closed: continue
            c = cout_arc(n, m, cost_extra, voxel_size)
            if g[n] + c < g[m] :
                g[m] = g[n] + c
                parent[m] = n
                hm = heuristic(m, voxel_size, goal_idx)
                h.heappush(open_heap, (g[m] + w*hm, hm, m))
    return None