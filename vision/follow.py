import math
import numpy as np


def follow(path_w, libre, dist, pos, yaw_deg, goal_w, v_max=4.5, lookahead=1.3, a_frein=3,
           v_up_max=4, k_yaw=5.0, yaw_rate_max=180.0, marge=0.8, d_secu=1.5, v_min=0.2, v_prox_min=0.3,
           explore_rate=30.0, rayon="avant", grid=None, marge_front=1.0, a_lat=0.6, a_front=0.5, a_lat_max=0.0, yaw_cos2=False, courbure_lisse=False):
    """Commande de suivi du chemin A* pour le contrôleur en cascade MuJoCo (appelée à chaque pas de physique).

    Entrées
      path_w      : (N, 3) points du chemin A* en repère monde (m), départ -> cible, ou None si pas de chemin
      libre       : (N,) booléens, libre[k] = True ssi le voxel du point k est franchement libre
      dist        : (N,) floats, distance (m) du point k à l'obstacle le plus proche
      pos         : (3,) position actuelle du drone (m)
      yaw_deg     : lacet actuel (degrés), avant = (cos yaw, 0, sin yaw)
      goal_w      : (3,) position de la cible (m)
      v_max       : vitesse max le long du chemin (m/s)
      lookahead   : distance du point visé le long du chemin (m), > taille d'un voxel
      a_frein     : décélération utilisée pour les limites de vitesse (m/s²)
      v_up_max    : vitesse verticale max (m/s)
      k_yaw       : gain proportionnel sur l'erreur de lacet (1/s)
      yaw_rate_max: vitesse de lacet max (deg/s)
      marge       : distance gardée avant le premier point non libre (m)
      d_secu      : en dessous de cette distance à un obstacle, on ralentit (m)
      v_min       : vitesse plancher, évite de rester bloqué quand rien devant n'est encore connu (m/s)
      v_prox_min  : vitesse minimale permise par la proximité des obstacles (m/s), absolue
      explore_rate: vitesse de lacet quand il n'y a pas de chemin (deg/s)
      rayon       : "avant" (rayon selon l'axe avant du drone, borne v_avant seule), "vitesse" (rayon vers
                    le point visé, borne la norme) ou None (pas de rayon)
      grid        : (blocked, j_min, vm) renvoyé par plan dans info["grid"] ; None désactive le rayon
      marge_front : distance gardée avant le premier voxel bloqué touché par le rayon (m) ; blocked est déjà gonflé
      a_lat       : poids de v_prox dans [0, 1] (0 : ignoré, 1 : pleine limite), tous les obstacles autour
      a_front     : poids de v_front dans [0, 1], seulement les obstacles en face

    Sortie
      cmd_velocity : (3,) [v_avant, v_droite, v_haut] en repère drone (m/s), à passer à drone.update
      cmd_yaw_rate : vitesse de lacet voulue (deg/s), > 0 fait augmenter yaw

    Principe
      1. point le plus proche de pos sur le chemin (indice i)
      2. point visé = point à `lookahead` m le long du chemin depuis i
      3. vitesse = min(v_max, v_sur, v_prox, v_cible) :
           v_sur   = sqrt(2 a (d_libre - marge)), d_libre = longueur libre connue devant le long du chemin
                     (pas de limite si le chemin est libre jusqu'au bout)
           v_prox  = max(v_prox_min, v_max * clip(dist[i] / d_secu, 0, 1)), ralentit près des obstacles
           v_cible = sqrt(2 a d_cible), s'arrête sur la cible
      4. lacet visé = direction horizontale vers le point visé ; yaw_rate = k_yaw * erreur, borné
      5. vitesse multipliée par max(cos(erreur de lacet), 0) : le drone regarde là où il va
      6. passage repère monde -> repère drone (avant, droite, haut)
      7. pas de chemin : vitesse nulle, rotation lente sur place pour explorer
    """
    
    if path_w is None or len(path_w) < 2:
        return np.zeros(3), explore_rate

    pos = np.asarray(pos, float)
    yaw = math.radians(yaw_deg)

    # 1. point le plus proche sur le chemin
    d = np.linalg.norm(path_w - pos, axis=1)
    i = int(np.argmin(d))

    # 2. point visé, par l'abscisse curviligne depuis i
    seg = np.linalg.norm(np.diff(path_w[i:], axis=0), axis=1)   # longueurs des segments après i
    longueur = np.concatenate(([0.0], np.cumsum(seg)))               # longueur[k] = longueur de i à i+k
    k = int(np.searchsorted(longueur, lookahead))                    # premier point au-delà du lookahead
    if k >= len(longueur):
        target = path_w[-1]
    else:
        s = (lookahead - longueur[k - 1]) / max(seg[k - 1], 1e-9)
        target = path_w[i + k - 1] + s * (path_w[i + k] - path_w[i + k - 1])

    # 3. norme de la vitesse
    lib = np.array(libre, bool)
    lib[-1] = True                                              # le dernier point est la cible, jamais un obstacle
    bad = np.flatnonzero(~lib[i + 1:])                          # points non libres après i
    n = bad[0] if len(bad) else len(seg)                        # nb de segments sûrs
    d_libre = longueur[n]
    if len(bad):                                                # un point non libre devant : on garde la marge
        v_sur = math.sqrt(2.0 * a_frein * max(d_libre - marge, 0.0))
    else:                                                       # libre jusqu'au bout du chemin : v_cible gère l'arrêt
        v_sur = math.inf
    v_prox = max(v_prox_min, v_max * float(np.clip(dist[i] / d_secu, 0.0, 1.0)))   # plancher absolu 
    v_prox = v_max - a_lat * (v_max - v_prox)                  # poids : interpolation entre v_max et v_prox
    v = min(v_max, max(v_sur, v_min), v_prox)
    if a_lat_max > 0:                                           # option : v <= sqrt(a_lat_max / courbure max) sur les ~3 m suivants
        v = min(v, max(v_min, math.sqrt(a_lat_max / max(_kappa_max(path_w[i:], 3.0, lisse=courbure_lisse), 1e-6))))

    # 4. lacet
    dvec = target - pos
    dh = math.hypot(dvec[0], dvec[2])
    psi_des = math.atan2(dvec[2], dvec[0]) if dh > 0.05 else yaw   # mouvement vertical : on garde le cap
    err = (psi_des - yaw + math.pi) % (2.0 * math.pi) - math.pi
    yaw_rate = float(np.clip(k_yaw * math.degrees(err), -yaw_rate_max, yaw_rate_max))

    # 5. vitesse en repère monde, réduite si le drone ne regarde pas où il va
    L = float(np.linalg.norm(dvec))
    portee = v_max ** 2 / (2.0 * a_frein) + marge_front      # au-delà, un obstacle ne limite pas encore la vitesse
    if rayon == "vitesse" and grid is not None and L > 1e-6:   # rayon vers le point visé
        d_front = distance_libre(grid, pos, dvec / L, portee)
        v_front = math.sqrt(2.0 * a_frein * max(d_front - marge_front, 0.0))
        v = min(v, v_max - a_front * (v_max - v_front))
    vel = v * dvec / L if L > 1e-6 else np.zeros(3)
    vel[1] = float(np.clip(vel[1], -v_up_max, v_up_max))
    c = max(math.cos(err), 0.0)
    vel *= c * c if yaw_cos2 else c

    # 6. repère drone
    fwd = np.array([math.cos(yaw), 0.0, math.sin(yaw)])
    right = np.array([-math.sin(yaw), 0.0, math.cos(yaw)])       # = cross(fwd, haut)
    cmd = np.array([vel @ fwd, vel @ right, vel[1]])
    if rayon == "avant" and grid is not None:                   # rayon selon l'axe avant : borne v_avant seule
        d_front = distance_libre(grid, pos, fwd, portee)
        v_front = math.sqrt(2.0 * a_frein * max(d_front - marge_front, 0.0))
        cmd[0] = min(cmd[0], v_max - a_front * (v_max - v_front))
    return cmd, yaw_rate


def distance_libre(grid, p0, u, portee):
    """Distance le long de p0 + s*u (s de 0 à portee, pas d'un demi-voxel) jusqu'au premier voxel bloqué
    de la grille gonflée. Les voxels bloqués du début sont sautés (drone collé à un bord, sinon il ne
    repartirait jamais). Hors de la tranche d'altitude = bloqué ; hors carte horizontalement = libre.
    Renvoie portee si rien n'est touché."""
    blocked, j_min, vm = grid
    vs = vm.voxel_size
    s = np.arange(0.0, portee, vs / 2)
    idx = np.floor((p0 + s[:, None] * u - vm.origin) / vs).astype(int)
    idx[:, 1] -= j_min
    nx, ny, nz = blocked.shape
    hors_xz = (idx[:, 0] < 0) | (idx[:, 0] >= nx) | (idx[:, 2] < 0) | (idx[:, 2] >= nz)
    hors_y = (idx[:, 1] < 0) | (idx[:, 1] >= ny)
    b = np.ones(len(s), bool)
    ok = ~hors_xz & ~hors_y
    b[ok] = blocked[idx[ok, 0], idx[ok, 1], idx[ok, 2]]
    b[hors_xz & ~hors_y] = False
    libres = np.flatnonzero(~b)
    if len(libres) == 0:
        return portee
    b[:libres[0]] = False                                       # saute les voxels bloqués du début
    touche = np.flatnonzero(b)
    return float(s[touche[0]]) if len(touche) else portee


def _kappa_max(pts, horizon, pas=0.75, lisse=False):
    """Courbure max (1/m) du chemin sur `horizon` m : cercle par 3 points espacés de `pas` m le long du chemin (3D)."""
    if lisse and len(pts) > 5:                       # moyenne glissante sur ~1 m (5 points de 0.25 m), extrémités gardées
        pad = np.concatenate((np.repeat(pts[:1], 2, axis=0), pts, np.repeat(pts[-1:], 2, axis=0)))
        pts = np.stack([np.convolve(pad[:, d], np.ones(5) / 5, mode="valid") for d in range(3)], axis=1)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate(([0.0], np.cumsum(seg)))
    if s[-1] < 2 * pas:
        return 0.0
    kmax = 0.0
    for s0 in np.arange(0.0, min(horizon, s[-1] - 2 * pas) + 1e-9, pas / 2):
        P = [np.array([np.interp(si, s, pts[:, d]) for d in range(3)]) for si in (s0, s0 + pas, s0 + 2 * pas)]
        a, b, c = np.linalg.norm(P[1] - P[0]), np.linalg.norm(P[2] - P[1]), np.linalg.norm(P[2] - P[0])
        area2 = np.linalg.norm(np.cross(P[1] - P[0], P[2] - P[0]))   # 2 * aire
        if a * b * c > 1e-9:
            kmax = max(kmax, 2.0 * area2 / (a * b * c))
    return kmax
