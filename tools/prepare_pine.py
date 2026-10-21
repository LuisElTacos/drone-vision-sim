"""Prépare le pin de la forêt à partir du modèle téléchargé « fir_spruce.fbx »
(zip tree-3d-model-fir-spruce-pine : source/fir_spruce.fbx + textures/).

Le modèle est composé de trois sortes d'objets :
- « Circle.002 » : le fût (cylindre, matériau Bark) -> gardé ;
- « st.001 »     : toutes les branches (bâtons, matériau Bark) -> supprimé entièrement ;
- « Plane.0xx »  : les cartes de feuillage (matériau Branches, découpe par transparence),
  qui commencent vers le tiers de la hauteur -> gardées.

Étape 1 (Blender, sans interface) : import du FBX, suppression des branches, export GLB.
Étape 2 (Python) : lecture du GLB, mise à l'échelle (hauteur 1), sauvegarde en .npz,
textures réduites : écorce 8K -> 2048, feuillage (preview 360 px avec alpha) copié.

Sortie : Assets/models/pine/pine.npz, bark.jpg, needles.png

Usage : python tools/prepare_pine.py <dossier dézippé> [chemin de blender.exe]
"""

import os
import subprocess
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from gltf_loader import load_glb  # noqa: E402

DEFAULT_BLENDER = r"C:\Program Files\Blender Foundation\Blender 4.5\blender.exe"
DROP_OBJECTS = ("st.001",)  # objets « branches » à supprimer

BLENDER_SCRIPT = r'''
import bpy, sys
src, dst, drop = sys.argv[sys.argv.index("--") + 1:][:3]
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.fbx(filepath=src)
for name in drop.split(","):
    obj = bpy.data.objects.get(name)
    if obj is not None:
        bpy.data.objects.remove(obj, do_unlink=True)
        print("supprimé :", name)
bpy.ops.export_scene.gltf(filepath=dst, export_format="GLB", export_image_format="NONE",
                          export_yup=True, export_apply=True)
'''


def main(src_dir, blender=DEFAULT_BLENDER):
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    dst = os.path.join(ROOT, "Assets", "models", "pine")
    os.makedirs(dst, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        script = os.path.join(tmp, "strip.py")
        glb = os.path.join(tmp, "fir.glb")
        with open(script, "w", encoding="utf-8") as f:
            f.write(BLENDER_SCRIPT)
        run = subprocess.run([blender, "-b", "--python", script, "--",
                              os.path.join(src_dir, "source", "fir_spruce.fbx"), glb, ",".join(DROP_OBJECTS)],
                             capture_output=True, text=True, errors="replace")
        if not os.path.exists(glb):
            raise RuntimeError("Blender n'a pas exporté le GLB :\n" + run.stdout[-3000:] + run.stderr[-3000:])
        parts = load_glb(glb)

    # Une primitive par objet : le fût est le seul objet vertical, haut et centré ;
    # toutes les autres primitives sont des cartes de feuillage.
    trunk, foliage = [], []
    for verts, idx, color, img in parts:
        tri = verts[idx]
        r = np.hypot(tri[:, 0], tri[:, 2])
        (trunk if r.max() < 1.5 and np.ptp(tri[:, 1]) > 10.0 else foliage).append(tri)
    trunk, foliage = np.vstack(trunk), np.vstack(foliage)

    y0 = trunk[:, 1].min()
    height = max(trunk[:, 1].max(), foliage[:, 1].max()) - y0
    for a in (trunk, foliage):
        a[:, 1] -= y0
        a[:, :3] /= height  # pin de hauteur 1 : l'échelle est choisie dans forest_env.py

    foliage_from = foliage[:, 1].min()
    np.savez_compressed(os.path.join(dst, "pine.npz"), trunk=trunk.astype("f4"), foliage=foliage.astype("f4"),
                        foliage_from=foliage_from, source_height=height)

    bark = Image.open(os.path.join(src_dir, "textures", "wgvqdap_8K_Albedo.jpg")).convert("RGB")
    bark.resize((2048, 2048), Image.LANCZOS).save(os.path.join(dst, "bark.jpg"), quality=90)
    Image.open(os.path.join(src_dir, "textures", "qgpuG2_Preview.png")).convert("RGBA").save(
        os.path.join(dst, "needles.png"))
    print(f"fût : {len(trunk) // 3} triangles, feuillage : {len(foliage) // 3} triangles, "
          f"feuillage à partir de {foliage_from:.2f} de la hauteur -> {dst}")


if __name__ == "__main__":
    main(sys.argv[1], *(sys.argv[2:3]))
