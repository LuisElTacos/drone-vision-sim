import numpy as np
import heapq as h
import math
from scipy.ndimage import distance_transform_edt, maximum_filter1d

def classify(vmap, y_min, y_max, r_total, k_unk=0.3, k_prox=3.0, seuil_occ=2.0, seuil_free=0.2, d_secu=1.2, details=False, r_v=None, d_unk=0.5):    
    vs = vmap.voxel_size

    # 1. tranche d'altitude : indices de couches j_min .. j_max (axe 1 = y vertical)
    j_min = max(0, int(np.ceil((y_min - vmap.origin[1]) / vs)))
    j_max = min(vmap.shape[1] - 1, int(np.floor((y_max - vmap.origin[1]) / vs)))
    L = vmap.map[:, j_min:j_max + 1, :]

    # 2. états
    occ = L > seuil_occ
    unknown = (~occ) & (L > -seuil_free)             # ni occupé, ni franchement libre

    # 3. distance (m) de chaque voxel au voxel occupé le plus proche
    if occ.any():
        dist = distance_transform_edt(~occ) * vs
    else:
        dist = np.full(L.shape, np.inf)

    # 4. sorties
    # Distances de centre à centre : la surface réelle peut être jusqu'à vs/2 plus près que le centre
    # du voxel occupé, d'où le gonflement r + vs/2 (sinon un trou d'un voxel laisse passer le drone).
    if r_v is None:
        blocked = dist < r_total + vs / 2        # gonflement sphérique
    elif occ.any():
        # gonflement cylindrique : rayon horizontal r_total (le drone tourne en lacet), demi-hauteur r_v
        nv = max(1, int(np.floor((r_v + vs / 2) / vs)))   # au moins une couche (inclinaison du drone)
        occ_v = maximum_filter1d(occ.astype(np.uint8), size=2 * nv + 1, axis=1) > 0 if nv > 0 else occ
        dist_h = distance_transform_edt(~occ_v, sampling=(vs, 1e6, vs))   # distance horizontale, couche par couche
        blocked = dist_h < r_total + vs / 2
    else:
        blocked = np.zeros(L.shape, bool)
    # L'inconnu collé à un obstacle n'est pas traversable : un trou d'un voxel dans un mur débouche
    # sur de l'inconnu, alors qu'une vraie ouverture est vue libre de part en part.
    blocked |= unknown & (dist < d_unk)
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