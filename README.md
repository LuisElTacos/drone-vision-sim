# drone-vision-sim

Simulation de drone en Python (moderngl) pour le projet du cours CSC_51073 *Analyse d'Image et Vision par Ordinateur* (IP Paris). Le drone porte deux caméras frontales (stéréo) et doit percevoir les obstacles et les ouvertures avec des méthodes du cours (coins de Harris, géométrie épipolaire, RANSAC, triangulation) pour rejoindre une cible.

Détails des choix, de l'avancement et des prochaines étapes : [PROJET.md](PROJET.md).

## Installation

```bash
pip install -r requirements.txt
```

Le modèle du drone (`Assets/models/DJI_Avata2_Sketchfab.glb`, DJI Avata 2 issu de Sketchfab) et les modèles de la forêt (`Assets/models/pine/`, `Assets/models/nature/`) sont inclus dans le dépôt.

## Lancer

```bash
python scene.py
```

Options : `--env murs|foret|batiment|vide` choisit l'environnement de départ (par défaut `murs`), `--seed N` fixe la graine des murs troués et de la forêt.

## Environnements

- **Murs troués** (`walls_env.py`) : murs successifs (une texture différente par mur : enduit, bois, tôle, pierre, parpaings) percés de passages, de culs-de-sac (poches fermées) et d'ouvertures trop étroites. Un chemin vers la cible (sphère verte) est garanti, un plafond invisible empêche de passer par-dessus. Généré à partir d'une graine ; vérité terrain des ouvertures via `get_openings()`. Bouton « Nouveaux murs » dans le panneau.
- **Forêt** (`forest_env.py`) : forêt dense de pins à tronc droit (environ 165 pins de 11 à 14 m sur 50 × 50 m, fût dégagé sur le tiers inférieur) avec rochers, fougères, herbes, champignons et fleurs, dessinée en instanciation. Le pin vient du projet C++ de 2A, préparé par `tools/prepare_pine.py` (branches retirées, feuillage bas coupé) dans `Assets/models/pine/` ; le décor vient du pack [Stylized Nature MegaKit](https://quaternius.itch.io/stylized-nature-megakit) de Quaternius (CC0), dans `Assets/models/nature/`. Collisions : cylindre par tronc, deux cylindres pour le feuillage, boîte par rocher. Bouton « Nouvelle forêt » dans le panneau.
- **Bâtiment** (`building.py`) : bâtiment industriel procédural pour le vol intérieur.
- **Vide** : sol seul.

## Commandes

| Touche | Action |
|---|---|
| Z/Q/S/D (W/A/S/D en QWERTY) | avancer, gauche, reculer, droite |
| Espace / Ctrl | monter / descendre |
| Shift | aller plus vite |
| Souris (après clic dans une vue) | lacet du drone, inclinaison des caméras |
| Échap | libérer la souris, puis quitter |
| Tab | caméra de suivi |

Le panneau de gauche permet de choisir les vues (caméra gauche, caméra droite, caméra fixe, suivi), de régler le vol et d'afficher ou non le bâtiment procédural.

## Fichiers

| Fichier | Rôle |
|---|---|
| `scene.py` | point d'entrée : scène, drone, caméras, boucle principale |
| `building.py` | bâtiment procédural et ses collisions |
| `walls_env.py` | environnement « murs troués » |
| `forest_env.py` | environnement « forêt » (rendu instancié) |
| `collision.py` | collisions du drone (sol, sphères, boîtes) |
| `gltf_loader.py` | chargeur GLB |
| `ui_panel.py` | panneau latéral ImGui |
| `Docs/Approche_vision_drone.md` | approche vision détaillée |
