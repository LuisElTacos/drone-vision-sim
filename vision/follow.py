import math
import numpy as np


def follow(path_w, libre, dist, pos, yaw_deg, goal_w, v_max=5, lookahead=1.3, a_frein=2.5,
           v_up_max=2, k_yaw=3.0, yaw_rate_max=90.0, marge=1.2, d_secu=1.5, v_min=0.3, v_prox_min=0.5,
           explore_rate=30.0, omni=False):
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
      omni        : True avec les 12 caméras (vue à 360°) : le drone peut avancer dans n'importe quelle
                    direction, la vitesse n'est plus réduite par l'erreur de lacet (étape 5)

    Sortie
      cmd_velocity : (3,) [v_avant, v_droite, v_haut] en repère drone (m/s), à passer à drone.update
      cmd_yaw_rate : vitesse de lacet voulue (deg/s), > 0 fait augmenter yaw

    Principe
      1. point le plus proche de pos sur le chemin (indice i)
      2. point visé = point à `lookahead` m le long du chemin depuis i
      3. vitesse = min(v_max, v_sur, v_prox, v_cible) :
           v_sur   = sqrt(2 a (d_libre - marge)), d_libre = longueur libre connue devant le long du chemin
           v_prox  = max(v_prox_min, v_max * clip(dist[i] / d_secu, 0, 1)), ralentit près des obstacles
           v_cible = sqrt(2 a d_cible), s'arrête sur la cible
      4. lacet visé = direction horizontale vers le point visé ; yaw_rate = k_yaw * erreur, borné
      5. vitesse multipliée par max(cos(erreur de lacet), 0) : le drone regarde là où il va (sauf omni)
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
    cum = np.concatenate(([0.0], np.cumsum(seg)))               # cum[k] = longueur de i à i+k
    k = int(np.searchsorted(cum, lookahead))                    # premier point au-delà du lookahead
    if k >= len(cum):
        target = path_w[-1]
    else:
        s = (lookahead - cum[k - 1]) / max(seg[k - 1], 1e-9)
        target = path_w[i + k - 1] + s * (path_w[i + k] - path_w[i + k - 1])

    # 3. norme de la vitesse
    bad = np.flatnonzero(~libre[i + 1:])                        # points non libres après i
    n = bad[0] if len(bad) else len(seg)                        # nb de segments sûrs
    d_libre = cum[n]
    v_sur = math.sqrt(2.0 * a_frein * max(d_libre - marge, 0.0))
    v_prox = max(v_prox_min, v_max * float(np.clip(dist[i] / d_secu, 0.0, 1.0)))   # plancher absolu
    d_cible = float(np.linalg.norm(np.asarray(goal_w, float) - pos))
    v_cible = math.sqrt(2.0 * a_frein * d_cible)
    v = min(v_max, max(v_sur, v_min), v_prox, v_cible)

    # 4. lacet
    dvec = target - pos
    dh = math.hypot(dvec[0], dvec[2])
    psi_des = math.atan2(dvec[2], dvec[0]) if dh > 0.05 else yaw   # mouvement vertical : on garde le cap
    err = (psi_des - yaw + math.pi) % (2.0 * math.pi) - math.pi
    yaw_rate = float(np.clip(k_yaw * math.degrees(err), -yaw_rate_max, yaw_rate_max))

    # 5. vitesse en repère monde, réduite si le drone ne regarde pas où il va
    L = float(np.linalg.norm(dvec))
    vel = v * dvec / L if L > 1e-6 else np.zeros(3)
    vel[1] = float(np.clip(vel[1], -v_up_max, v_up_max))
    if not omni:
        vel *= max(math.cos(err), 0.0)

    # 6. repère drone
    fwd = np.array([math.cos(yaw), 0.0, math.sin(yaw)])
    right = np.array([-math.sin(yaw), 0.0, math.cos(yaw)])       # = cross(fwd, haut)
    return np.array([vel @ fwd, vel @ right, vel[1]]), yaw_rate
