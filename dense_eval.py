"""dense_eval.py : compare coins (pipeline actuelle), StereoSGBM et un réseau stéréo pré-entraîné
sur les séquences de dense_capture.py (profondeur vraie = z-buffer).

Mesures par méthode (cartes : tranche d'altitude ymin-ymax seulement, sol exclu) (sorties/dense/resultats.csv) :
  densite      part des pixels à moins de --portee m qui ont une profondeur
  err_med      erreur relative médiane |Z - Zvrai| / Zvrai sur les pixels estimés (<= portee)
  bons         part des pixels estimés avec erreur < 10 %
  fin_couv     part des pixels « objet fin » (poteau, tronc) avec une profondeur à moins de 10 % près
  fin_obj      part des objets fins (composantes connexes) qui ont au moins 5 pixels bons
  vox_rappel   carte de voxels de la séquence : part des voxels occupés vrais retrouvés (à 1 voxel près)
  vox_fin      idem sur les voxels des objets fins
  vox_fin0     idem sans tolérance
  fin_libre    part des voxels des objets fins que la méthode croit libres (le drone passerait au travers)
  vox_faux     part des voxels occupés de la méthode sans voxel occupé vrai à 1 voxel près
  libre_perdu  part de l'espace vu libre par la vérité (à plus d'un voxel d'un obstacle) que la méthode marque occupé
  ms           temps par image (ms)

Objet fin : suite horizontale de pixels de même profondeur (saut < 8 %) bordée des deux côtés par un fond
plus loin de 30 %, de largeur physique < --largeur-fin m, et haute d'au moins 20 pixels.

    python dense_eval.py                          # toutes les séquences de sorties/dense/seq
    python dense_eval.py --methodes coins sgbm raft_rt
"""
import argparse
import csv
import glob
import os
import sys
import time

import cv2
import numpy as np
from numba import njit
from scipy import ndimage

from test14 import match_along_rows
from vision.corners import harris_function, detect_corners
from vision.matching import to_gray
from vision.voxels import VoxelMap

ICI = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ICI, "sorties", "dense")


# ---------------------------------------------------------------------------
# Méthodes : (L, R uint8 RGB, K, B) -> disparité (H, W) float32, NaN = pas d'estimation
# ---------------------------------------------------------------------------

def m_coins(L, R, K, B, max_coins=800):
    gL, gR = to_gray(L), to_gray(R)
    cL = detect_corners(harris_function(gL), rel_thresh=1e-6, max_n=max_coins, grid_cells=(12, 16))
    ptsL, ptsR, _ = match_along_rows(L, R, cL, min_zncc=0.8, ratio=0.6, grayL=gL, grayR=gR)
    D = np.full(L.shape[:2], np.nan, np.float32)
    if len(ptsL):
        u = np.rint(ptsL[:, 0]).astype(int); v = np.rint(ptsL[:, 1]).astype(int)
        D[v, u] = ptsL[:, 0] - ptsR[:, 0]
    return D


_SGBM = {}


def m_sgbm(L, R, K, B, bs=5, ndisp=128, uniq=10, p2=32):
    key = (bs, ndisp, uniq, p2)
    if key not in _SGBM:
        _SGBM[key] = cv2.StereoSGBM_create(minDisparity=0, numDisparities=ndisp, blockSize=bs,
                                           P1=8 * bs * bs, P2=p2 * bs * bs, disp12MaxDiff=1,
                                           uniquenessRatio=uniq, speckleWindowSize=100, speckleRange=2,
                                           mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
    gL = cv2.cvtColor(L, cv2.COLOR_RGB2GRAY); gR = cv2.cvtColor(R, cv2.COLOR_RGB2GRAY)
    d = _SGBM[key].compute(gL, gR).astype(np.float32) / 16.0
    d[d <= 0] = np.nan
    return d


def m_sgbm_demi(L, R, K, B):
    """SGBM sur images à demi-résolution (4x moins de pixels), disparité remise à l'échelle."""
    h, w = L.shape[:2]
    Ls = cv2.resize(L, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
    Rs = cv2.resize(R, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
    d = m_sgbm(Ls, Rs, K, B, bs=5, ndisp=64)
    return 2.0 * cv2.resize(d, (w, h), interpolation=cv2.INTER_NEAREST)


METHODES = {"coins": m_coins, "coins3000": lambda *a: m_coins(*a, max_coins=3000),
            "sgbm": m_sgbm, "sgbm_demi": m_sgbm_demi,
            # réglages essayés sur murs / forêt seulement (la ville reste un jeu de test)
            "sgbm_b7u15": lambda *a: m_sgbm(*a, bs=7, uniq=15),
            "sgbm_b5u25": lambda *a: m_sgbm(*a, bs=5, uniq=25),
            "sgbm_b9u15p64": lambda *a: m_sgbm(*a, bs=9, uniq=15, p2=64)}


def ajoute_reseaux():
    """Réseaux pré-entraînés (dense_reseaux.py), s'ils sont installés."""
    try:
        import dense_reseaux
    except Exception as e:      # pas installé
        print("réseaux indisponibles :", e)
        return
    METHODES.update(dense_reseaux.methodes())


# ---------------------------------------------------------------------------
# Vérité terrain : objets fins
# ---------------------------------------------------------------------------

@njit(cache=True)
def _fins_lignes(Z, f, largeur_max, portee, saut, fond):
    H, W = Z.shape
    out = np.zeros((H, W), np.bool_)
    for y in range(H):
        x = 0
        while x < W:
            z0 = Z[y, x]
            if not (z0 <= portee):
                x += 1
                continue
            x1 = x
            while x1 + 1 < W and np.isfinite(Z[y, x1 + 1]) and abs(Z[y, x1 + 1] - Z[y, x1]) < saut * Z[y, x1]:
                x1 += 1
            # run [x, x1]
            if x > 0 and x1 < W - 1:
                zm = Z[y, (x + x1) // 2]
                zg, zd = Z[y, x - 1], Z[y, x1 + 1]
                if zg > fond * zm and zd > fond * zm and (x1 - x + 1) * zm / f < largeur_max:
                    for k in range(x, x1 + 1):
                        out[y, k] = True
            x = x1 + 1
    return out


def masque_fin(Z, f, largeur_max=0.5, portee=15.0):
    m = _fins_lignes(Z.astype(np.float64), f, largeur_max, portee, 0.08, 1.3)
    m = ndimage.binary_opening(m, structure=np.ones((20, 1), bool))   # garde ce qui est haut d'au moins 20 px
    return m


# ---------------------------------------------------------------------------
# Évaluation
# ---------------------------------------------------------------------------

def points_monde(Zm, K, Rw, Cw, portee, pas=2, masque=None):
    H, W = Zm.shape
    vv, uu = np.mgrid[0:H:pas, 0:W:pas]
    z = Zm[vv, uu]
    ok = np.isfinite(z) & (z > 0.1) & (z <= portee)
    if masque is not None:
        ok &= masque[vv, uu]
    u, v, z = uu[ok], vv[ok], z[ok]
    f, cx, cy = K[0, 0], K[0, 2], K[1, 2]
    P = np.stack([(u - cx) * z / f, (v - cy) * z / f, z], 1)
    return P @ Rw.T + Cw


def un_par_voxel(P, origin, vs, min_pts=1):
    """Un point (moyen) par voxel touché : même occupation, beaucoup moins de rayons à lancer.
    min_pts > 1 : ne garde que les voxels touchés par au moins min_pts points (filtre des points isolés)."""
    if len(P) == 0:
        return P
    idx = np.floor((P - origin) / vs).astype(np.int64)
    _, inv, cnt = np.unique(idx, axis=0, return_inverse=True, return_counts=True)
    inv = inv.ravel()
    S = np.zeros((len(cnt), 3))
    np.add.at(S, inv, P)
    keep = cnt >= min_pts
    return S[keep] / cnt[keep, None]


def nouvelle_carte(Cs, portee, vs):
    lo = Cs.min(0) - portee - 1; hi = Cs.max(0) + portee + 1
    origin = np.array([lo[0], Cs[:, 1].min() - 2.0, lo[2]])
    return VoxelMap(origin, np.array([hi[0] - lo[0], 10.0, hi[2] - lo[2]]), vs)


def eval_sequence(f_seq, methodes, args):
    d = np.load(f_seq)
    Ls, Rs_, Zs, Rws, Cws, K, B = d["L"], d["R"], d["Z"], d["Rs"], d["Cs"], d["K"], float(d["B"])
    fx = K[0, 0]
    nom = os.path.splitext(os.path.basename(f_seq))[0]
    n = len(Ls)
    fins = [masque_fin(Z, fx, args.largeur_fin, args.portee) for Z in Zs]

    # carte vraie + voxels des objets fins
    gt = nouvelle_carte(Cws, args.portee, args.voxel)
    fin_vox = np.zeros(gt.shape, bool)
    for i in range(n):
        P = points_monde(Zs[i], K, Rws[i], Cws[i], args.portee)
        gt.integrate(un_par_voxel(P, gt.origin, gt.voxel_size), Cws[i], max_range=args.portee, max_par_frame=1)
        Pf = points_monde(Zs[i], K, Rws[i], Cws[i], args.portee, pas=1, masque=fins[i])
        if len(Pf):
            idx = gt.world_to_idx(Pf)
            ok = np.all((idx >= 0) & (idx < gt.shape), 1)
            fin_vox[tuple(idx[ok].T)] = True
    occ_gt = gt.map > 0
    fin_vox &= occ_gt
    # carte vraie à portée + 3 m : référence des faux positifs (un point à 16 m estimé à 14,9 m n'est pas un faux obstacle)
    gt_l = nouvelle_carte(Cws, args.portee, args.voxel)
    for i in range(n):
        P = points_monde(Zs[i], K, Rws[i], Cws[i], args.portee + 3.0)
        gt_l.integrate(un_par_voxel(P, gt_l.origin, gt_l.voxel_size), Cws[i], max_range=args.portee + 3.0, max_par_frame=1)
    # tranche d'altitude de vol (sol exclu), comme classify
    vs = gt.voxel_size
    j0 = max(0, int(np.ceil((args.ymin - gt.origin[1]) / vs)))
    j1 = min(gt.shape[1] - 1, int(np.floor((args.ymax - gt.origin[1]) / vs)))
    bande = np.zeros(gt.shape, bool); bande[:, j0:j1 + 1] = True
    occ_gt &= bande; fin_vox &= bande
    occ_gt_l = (gt_l.map > 0) & bande
    # espace vu libre par la vérité, à plus d'un voxel d'un obstacle vrai : un voxel occupé là bloque un passage réel
    libre_gt = (gt_l.map < 0) & bande & ~ndimage.binary_dilation(gt_l.map > 0, structure=np.ones((3, 3, 3), bool))

    res, cartes, exemples = {}, {"vrai": gt.map.copy()}, {}
    cache_coins = {}
    dil = lambda m: ndimage.binary_dilation(m, structure=np.ones((3, 3, 3), bool))
    for nm in methodes:
        fn = METHODES[nm]
        fn(Ls[0], Rs_[0], K, B)             # chauffe (numba, cudnn)
        dense = not nm.startswith("coins")
        # dense : deux cartes, tous les voxels touchés (min 1 pixel) et voxels soutenus par >= min_pts pixels sur l'image
        # variante -> (portée des points denses, min pixels par voxel, ajoute les coins jusqu'à la portée)
        pc = f"{args.portee_proche:g}m"
        variantes = {nm: (args.portee, 1, False)} if not dense else {
            nm: (args.portee, 1, False),
            f"{nm}_f{args.min_pts}": (args.portee, args.min_pts, False),
            f"{nm}_{pc}+coins": (args.portee_proche, args.min_pts, True)}
        vms = {v: nouvelle_carte(Cws, args.portee, args.voxel) for v in variantes}
        acc = dict(n_proche=0, n_est=0, n_bon=0, errs=[], n_fin=0, n_fin_bon=0, obj=0, obj_ok=0, t=0.0)
        exemples[nm] = {}
        for i in range(n):
            t0 = time.perf_counter()
            D = fn(Ls[i], Rs_[i], K, B)
            acc["t"] += time.perf_counter() - t0
            with np.errstate(invalid="ignore", divide="ignore"):
                Zm = np.where(D > 0.5, fx * B / D, np.nan).astype(np.float32)
            Zg = Zs[i]
            proche = np.isfinite(Zg) & (Zg <= args.portee)
            est = proche & np.isfinite(Zm)
            with np.errstate(invalid="ignore"):
                err = np.abs(Zm - Zg) / Zg
            bon = est & (err < 0.10)
            acc["n_proche"] += proche.sum(); acc["n_est"] += est.sum(); acc["n_bon"] += bon.sum()
            acc["errs"].append(err[est])
            fm = fins[i]
            acc["n_fin"] += fm.sum(); acc["n_fin_bon"] += (bon & fm).sum()
            lab, nl = ndimage.label(fm)
            if nl:
                nb = ndimage.sum(bon, lab, index=np.arange(1, nl + 1))
                acc["obj"] += nl; acc["obj_ok"] += int((nb >= 5).sum())
            for v, (por, mp, avec_coins) in variantes.items():
                vm = vms[v]
                P = un_par_voxel(points_monde(Zm, K, Rws[i], Cws[i], por, pas=2 if dense else 1), vm.origin, vm.voxel_size, mp)
                if avec_coins:
                    if i not in cache_coins:
                        Dc = m_coins(Ls[i], Rs_[i], K, B)
                        with np.errstate(invalid="ignore", divide="ignore"):
                            cache_coins[i] = np.where(Dc > 0.5, fx * B / Dc, np.nan)
                    P = np.vstack([P, points_monde(cache_coins[i], K, Rws[i], Cws[i], args.portee, pas=1)])
                vm.integrate(P, Cws[i], max_range=args.portee, max_par_frame=1)
            if i in args.exemples:
                exemples[nm][i] = Zm
        errs = np.concatenate(acc["errs"]) if acc["errs"] else np.empty(0)
        for v, vm in vms.items():
            occ = (vm.map > args.seuil_occ) & bande
            occ_d = dil(occ)
            res[v] = dict(
                densite=acc["n_est"] / max(acc["n_proche"], 1),
                err_med=float(np.median(errs)) if len(errs) else np.nan,
                bons=acc["n_bon"] / max(acc["n_est"], 1),
                fin_couv=acc["n_fin_bon"] / max(acc["n_fin"], 1),
                fin_obj=acc["obj_ok"] / max(acc["obj"], 1),
                vox_rappel=(occ_d & occ_gt).sum() / max(occ_gt.sum(), 1),
                vox_fin=(occ_d & fin_vox).sum() / max(fin_vox.sum(), 1),
                vox_fin0=(occ & fin_vox).sum() / max(fin_vox.sum(), 1),
                fin_libre=((vm.map < -args.seuil_free) & fin_vox & ~occ_d).sum() / max(fin_vox.sum(), 1),
                vox_faux=(occ & ~dil(occ_gt_l)).sum() / max(occ.sum(), 1),
                libre_perdu=(occ & libre_gt).sum() / max(libre_gt.sum(), 1),
                ms=1000 * acc["t"] / n)
            cartes[v] = vm.map.copy()
            r = res[v]
            print(f"  {nom:10s} {v:18s} dens {r['densite']:.3f}  err_med {r['err_med']:.3f}  bons {r['bons']:.3f}  "
                  f"fin_couv {r['fin_couv']:.3f}  fin_obj {r['fin_obj']:.2f}  vox_rappel {r['vox_rappel']:.3f}  "
                  f"vox_fin {r['vox_fin']:.3f}/{r['vox_fin0']:.3f}  fin_libre {r['fin_libre']:.3f}  vox_faux {r['vox_faux']:.3f}  libre_perdu {r['libre_perdu']:.3f}  {r['ms']:6.1f} ms", flush=True)
    info = dict(L=Ls, Z=Zs, fins=fins, fin_vox=fin_vox, gt=gt, Cs=Cws, nom=nom)
    return res, cartes, exemples, info


# ---------------------------------------------------------------------------
# Images
# ---------------------------------------------------------------------------

def couleur_prof(Z, portee):
    z = np.clip(np.nan_to_num(Z, nan=np.inf), 0, portee)
    im = cv2.applyColorMap((255 * (1 - z / portee)).astype(np.uint8), cv2.COLORMAP_TURBO)
    im[~(np.isfinite(Z) & (Z <= portee))] = 30
    return im


def titre(im, txt):
    im = im.copy()
    cv2.rectangle(im, (0, 0), (im.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(im, txt, (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    return im


def image_comparee(info, exemples, methodes, i, portee, f_out):
    L = info["L"][i][:, :, ::-1]
    fm = info["fins"][i]
    g = L.copy()
    cnt, _ = cv2.findContours(fm.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(g, cnt, -1, (255, 0, 255), 2)
    tuiles = [titre(g, "gauche (+ objets fins)"), titre(couleur_prof(info["Z"][i], portee), "profondeur vraie")]
    for nm in methodes:
        Zm = exemples[nm][i]
        if nm.startswith("coins"):
            im = (0.45 * L).astype(np.uint8)
            vv, uu = np.nonzero(np.isfinite(Zm) & (Zm <= portee))
            col = couleur_prof(Zm, portee)
            for u, v in zip(uu, vv):
                cv2.circle(im, (int(u), int(v)), 3, tuple(int(c) for c in col[v, u]), -1)
            cv2.drawContours(im, cnt, -1, (255, 0, 255), 1)
        else:
            im = couleur_prof(Zm, portee)
            cv2.drawContours(im, cnt, -1, (255, 0, 255), 1)
        tuiles.append(titre(im, nm))
    while len(tuiles) % 3:
        tuiles.append(np.zeros_like(tuiles[0]))
    lignes = [np.hstack(tuiles[k:k + 3]) for k in range(0, len(tuiles), 3)]
    out = np.vstack(lignes)
    out = cv2.resize(out, None, fx=0.6, fy=0.6, interpolation=cv2.INTER_AREA)
    cv2.imwrite(f_out, out)


def vue_dessus(cartes, info, methodes, args, f_out):
    gt = info["gt"]
    vs = gt.voxel_size
    j0 = max(0, int(np.ceil((args.ymin - gt.origin[1]) / vs)))
    j1 = min(gt.shape[1] - 1, int(np.floor((args.ymax - gt.origin[1]) / vs)))
    fin2d = info["fin_vox"][:, j0:j1 + 1].any(1)
    tuiles = []
    for nm in ["vrai"] + list(methodes):
        M = cartes[nm][:, j0:j1 + 1]
        s_o, s_f = (0.0, 0.0) if nm == "vrai" else (args.seuil_occ, args.seuil_free)
        occ = (M > s_o).any(1); libre = (M < -s_f).all(1) & ~occ
        im = np.full(occ.shape + (3,), 90, np.uint8)        # gris : inconnu
        im[libre] = (235, 235, 235); im[occ] = (40, 40, 40)
        im[fin2d & occ] = (255, 0, 255)                     # objet fin vu
        im[fin2d & ~occ] = (0, 200, 255)                    # objet fin manqué (orange)
        im = np.transpose(im, (1, 0, 2))[::-1]               # x horizontal, z vertical
        im = cv2.resize(im, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)
        for C in info["Cs"]:
            p = ((C[0] - gt.origin[0]) / vs * 4, im.shape[0] - (C[2] - gt.origin[2]) / vs * 4)
            cv2.circle(im, (int(p[0]), int(p[1])), 3, (0, 160, 0), -1)
        tuiles.append(titre(im, nm))
    cv2.imwrite(f_out, np.hstack(tuiles))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", nargs="*", default=None)
    ap.add_argument("--methodes", nargs="*", default=None)
    ap.add_argument("--portee", type=float, default=15.0)
    ap.add_argument("--voxel", type=float, default=0.25)
    ap.add_argument("--min-pts", type=int, default=3, help="dense : points (pas de 2 px) min par voxel et par image")
    ap.add_argument("--seuil-occ", type=float, default=0.0, help="occupé si L > seuil (test_vslam : 1.2)")
    ap.add_argument("--seuil-free", type=float, default=0.0, help="libre si L < -seuil (test_vslam : 0.6)")
    ap.add_argument("--portee-proche", type=float, default=8.0, help="variante mixte : dense jusqu'à cette portée, coins au-delà")
    ap.add_argument("--largeur-fin", type=float, default=0.5)
    ap.add_argument("--ymin", type=float, default=0.6)
    ap.add_argument("--ymax", type=float, default=4.4)
    ap.add_argument("--exemples", type=int, nargs="*", default=[0, 15])
    args = ap.parse_args()
    ajoute_reseaux()
    methodes = args.methodes or list(METHODES)
    seqs = args.seq or sorted(glob.glob(os.path.join(OUT, "seq", "*.npz")))
    os.makedirs(os.path.join(OUT, "images"), exist_ok=True)
    lignes = []
    for f in seqs:
        res, cartes, ex, info = eval_sequence(f, methodes, args)
        for nm, r in res.items():
            lignes.append(dict(seq=info["nom"], methode=nm, **{k: round(float(v), 4) for k, v in r.items()}))
        for i in args.exemples:
            image_comparee(info, ex, methodes, i, args.portee, os.path.join(OUT, "images", f"{info['nom']}_img{i}.png"))
        vue_dessus(cartes, info, list(res), args, os.path.join(OUT, "images", f"{info['nom']}_carte.png"))
    with open(os.path.join(OUT, "resultats.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(lignes[0]), delimiter=";")
        w.writeheader(); w.writerows(lignes)
    # moyenne par environnement
    print("\nmoyennes par environnement")
    envs = sorted({l["seq"].split("_")[0] for l in lignes})
    cles = [k for k in lignes[0] if k not in ("seq", "methode")]
    variantes = list(dict.fromkeys(l["methode"] for l in lignes))
    for e in envs:
        for nm in variantes:
            ls = [l for l in lignes if l["methode"] == nm and l["seq"].startswith(e + "_")]
            print(f"  {e:6s} {nm:18s} " + "  ".join(f"{k} {np.mean([l[k] for l in ls]):.3f}" for k in cles))


if __name__ == "__main__":
    main()
