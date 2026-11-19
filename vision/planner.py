import numpy as np
import heapq as h
import math
from scipy.ndimage import distance_transform_edt

def classify_fenetre(vmap, pos, y_min, y_max, r_total, rayon=8.0, k_unk=0.35, k_prox=3.0,
                     seuil_occ=0.0, seuil_free=0.0, d_secu=0.8, details=False):
    vs = vmap.voxel_size
    j_min = max(0, int(np.ceil((y_min - vmap.origin[1]) / vs)))
    j_max = min(vmap.shape[1] - 1, int(np.floor((y_max - vmap.origin[1]) / vs)))
    L = vmap.map[:, j_min:j_max + 1, :]
    nx, _, nz = L.shape

    occ = L > seuil_occ
    unknown = (~occ) & (L >= -seuil_free)

    # fenêtre horizontale autour du drone (indices x et z), plus une marge pour ne pas perdre
    # les obstacles juste hors fenêtre dans le calcul des distances
    ix = int(np.clip((pos[0] - vmap.origin[0]) / vs, 0, nx - 1))
    iz = int(np.clip((pos[2] - vmap.origin[2]) / vs, 0, nz - 1))
    h = int(np.ceil(rayon / vs))
    pad = int(np.ceil(max(r_total, d_secu) / vs)) + 1
    xi0, xi1 = max(0, ix - h), min(nx, ix + h + 1)          # fenêtre utile
    zi0, zi1 = max(0, iz - h), min(nz, iz + h + 1)
    x0, x1 = max(0, xi0 - pad), min(nx, xi1 + pad)          # fenêtre de calcul (avec marge)
    z0, z1 = max(0, zi0 - pad), min(nz, zi1 + pad)

    # distances : seulement dans la fenêtre de calcul
    occ_w = occ[x0:x1, :, z0:z1]
    if occ_w.any():
        dist_w = distance_transform_edt(~occ_w) * vs
    else:
        dist_w = np.full(occ_w.shape, np.inf)
    sx = slice(xi0 - x0, xi1 - x0)
    sz = slice(zi0 - z0, zi1 - z0)
    dist = np.full(L.shape, np.inf, np.float32)             # hors fenêtre : distance inconnue = infinie
    dist[xi0:xi1, :, zi0:zi1] = dist_w[sx, :, sz]

    # sorties : hors fenêtre, obstacles non gonflés et pas de coût de proximité
    blocked = occ | (dist < r_total)
    cost_extra = (k_unk * unknown).astype(np.float32)
    prox_w = np.clip((d_secu - dist[xi0:xi1, :, zi0:zi1]) / d_secu, 0, 1) ** 2
    cost_extra[xi0:xi1, :, zi0:zi1] += (k_prox * prox_w).astype(np.float32)
    if details:
        libre = L < -seuil_free
        return blocked, cost_extra, j_min, libre, dist
    return blocked, cost_extra, j_min

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

def astar(blocked, cost_extra, start_idx, goal_idx, voxel_size, w=2.0, version="numba", cost_vert=None):
    """version="numba" : même algorithme (même ordre du tas (f, h, indice), mêmes arrondis), compilé ;
    version="python" : version d'origine.
    cost_vert : None (défaut, inchangé) ou tableau de la forme de blocked : coût ajouté, par mètre, aux arcs qui
    montent ou descendent (dj != 0) vers ce voxel (ex. k_unk_v * inconnu) ; inf = arc vertical interdit."""
    if version == "numba":
        return _astar_numba(blocked, cost_extra, start_idx, goal_idx, voxel_size, w, cost_vert)
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
            if cost_vert is not None and m[1] != n[1]:
                c = c + voxel_size * math.dist(n, m) * float(cost_vert[m])
            if g[n] + c < g[m] :
                g[m] = g[n] + c
                parent[m] = n
                hm = heuristic(m, voxel_size, goal_idx)
                h.heappush(open_heap, (g[m] + w*hm, hm, m))
    return None

# ---- A* compilé (numba) : mêmes clés de tas (f, h, i, j, k) que la version Python, donc même ordre de sortie ----
from numba import njit as _njit


@_njit(cache=True, inline="always")
def _key_lt(F, Hh, I, a, b):
    """Clé a < clé b, ordre lexicographique (f, h, i, j, k) comme les tuples Python."""
    if F[a] != F[b]:
        return F[a] < F[b]
    if Hh[a] != Hh[b]:
        return Hh[a] < Hh[b]
    for t in range(3):
        if I[a, t] != I[b, t]:
            return I[a, t] < I[b, t]
    return False


@_njit(cache=True)
def _astar_nb(blocked, cost_extra, s0, s1, s2, g0, g1, g2, vs, w, c32, cost_vert, use_v):
    nx, ny, nz = blocked.shape
    cap = 1024
    F = np.empty(cap); Hh = np.empty(cap); I = np.empty((cap, 3), np.int64)
    size = 0
    g = np.full((nx, ny, nz), np.inf)
    closed = np.zeros((nx, ny, nz), np.bool_)
    par = np.full((nx, ny, nz), -1, np.int64)
    hs = vs * np.sqrt(float((s0 - g0) ** 2 + (s1 - g1) ** 2 + (s2 - g2) ** 2))
    F[0] = w * hs; Hh[0] = hs; I[0, 0] = s0; I[0, 1] = s1; I[0, 2] = s2
    size = 1
    g[s0, s1, s2] = 0.0
    found = False
    while size > 0:
        # pop (tas binaire min)
        a, b, c = I[0, 0], I[0, 1], I[0, 2]
        size -= 1
        if size > 0:
            F[0] = F[size]; Hh[0] = Hh[size]; I[0, 0] = I[size, 0]; I[0, 1] = I[size, 1]; I[0, 2] = I[size, 2]
            p = 0
            while True:
                l, r, m = 2 * p + 1, 2 * p + 2, p
                if l < size and _key_lt(F, Hh, I, l, m):
                    m = l
                if r < size and _key_lt(F, Hh, I, r, m):
                    m = r
                if m == p:
                    break
                F[p], F[m] = F[m], F[p]; Hh[p], Hh[m] = Hh[m], Hh[p]
                for t in range(3):
                    I[p, t], I[m, t] = I[m, t], I[p, t]
                p = m
        if closed[a, b, c]:
            continue
        if a == g0 and b == g1 and c == g2:
            found = True
            break
        closed[a, b, c] = True
        gn = g[a, b, c]
        for di in range(-1, 2):
            for dj in range(-1, 2):
                for dk in range(-1, 2):
                    if di == 0 and dj == 0 and dk == 0:
                        continue
                    x, y, z = a + di, b + dj, c + dk
                    if x < 0 or x >= nx or y < 0 or y >= ny or z < 0 or z >= nz:
                        continue
                    if blocked[x, y, z] or closed[x, y, z]:
                        continue
                    longueur = vs * np.sqrt(float(di * di + dj * dj + dk * dk))
                    if c32:   # float Python * float32 -> float32 (règles numpy 2), comme cout_arc
                        cc = float(np.float32(longueur) * (np.float32(1.0) + np.float32(cost_extra[x, y, z])))
                    else:
                        cc = longueur * (1.0 + float(cost_extra[x, y, z]))
                    if use_v and dj != 0:
                        cc = cc + longueur * float(cost_vert[x, y, z])
                    if gn + cc < g[x, y, z]:
                        g[x, y, z] = gn + cc
                        par[x, y, z] = (a * ny + b) * nz + c
                        hm = vs * np.sqrt(float((x - g0) ** 2 + (y - g1) ** 2 + (z - g2) ** 2))
                        if size == cap:
                            cap *= 2
                            F2 = np.empty(cap); H2 = np.empty(cap); I2 = np.empty((cap, 3), np.int64)
                            F2[:size] = F[:size]; H2[:size] = Hh[:size]; I2[:size] = I[:size]
                            F, Hh, I = F2, H2, I2
                        F[size] = g[x, y, z] + w * hm; Hh[size] = hm
                        I[size, 0] = x; I[size, 1] = y; I[size, 2] = z
                        q = size
                        size += 1
                        while q > 0:
                            pq = (q - 1) // 2
                            if _key_lt(F, Hh, I, q, pq):
                                F[q], F[pq] = F[pq], F[q]; Hh[q], Hh[pq] = Hh[pq], Hh[q]
                                for t in range(3):
                                    I[q, t], I[pq, t] = I[pq, t], I[q, t]
                                q = pq
                            else:
                                break
    if not found:
        return np.empty((0, 3), np.int64)
    n = 1
    v = (g0 * ny + g1) * nz + g2
    while par.flat[v] >= 0:
        v = par.flat[v]
        n += 1
    out = np.empty((n, 3), np.int64)
    v = (g0 * ny + g1) * nz + g2
    for t in range(n - 1, -1, -1):
        out[t, 0] = v // (ny * nz); out[t, 1] = (v // nz) % ny; out[t, 2] = v % nz
        if t > 0:
            v = par.flat[v]
    return out


def _astar_numba(blocked, cost_extra, start_idx, goal_idx, voxel_size, w, cost_vert=None):
    c32 = isinstance(voxel_size * 1.0 * (1 + cost_extra.flat[0]), np.float32)
    s = [int(v) for v in start_idx]; t = [int(v) for v in goal_idx]
    out = _astar_nb(np.ascontiguousarray(blocked), np.ascontiguousarray(cost_extra), s[0], s[1], s[2], t[0], t[1], t[2],
                    float(voxel_size), float(w), c32,
                    np.zeros((1, 1, 1)) if cost_vert is None else np.ascontiguousarray(cost_vert, np.float64),
                    cost_vert is not None)
    if len(out) == 0:
        return None
    return [tuple(int(v) for v in r) for r in out]
