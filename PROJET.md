# Projet CSC_51073 : simulation de drone et vision stéréo

Projet final du cours *Analyse d'Image et Vision par Ordinateur* (CSC_51073, IP Paris). Rendu le 16/12 : rapport, slides, vidéo, code.

Ce fichier résume les choix, l'état d'avancement et l'organisation. Mis à jour le 14/10/2026.

## Objectif

Un drone simulé, équipé de deux caméras, perçoit les obstacles avec des méthodes du cours et se dirige vers une cible :
- détecter les obstacles, leur distance et leur forme ;
- mesurer les ouvertures (ex. un trou carré à distance d) et décider si le drone peut passer ;
- savoir si une ouverture mène plus loin vers la cible (ou si c'est la seule sortie) ;
- construire une carte 3D et atteindre la cible avec un algorithme déterministe.

Hors vision : la génération d'obstacles et le moteur physique ne sont pas le cœur du sujet.

## Choix faits

| Sujet | Choix | Raison |
|---|---|---|
| Langage et rendu | **Python + moderngl + glfw** (pas C++ + pybind11) | Le rendu se fait sur le GPU, Python n'est pas le goulot ; les images arrivent en numpy, directement exploitables. Le C++ ajoutait CMake, compilation Windows et débogage entre deux langages pour peu de gain. |
| Caméras | **Deux caméras frontales (stéréo)**, gauche / droite, séparées de 18 cm | Une seule image ne permet pas de séparer taille et distance (petit trou proche = grand trou loin). La stéréo donne la profondeur à chaque frame, même à l'arrêt ou en allant droit vers la cible. Les vrais drones (DJI, Skydio) utilisent des paires stéréo. |
| Méthode vision | **Coins de Harris + appariement le long des lignes épipolaires + RANSAC + triangulation** | Utilise directement les cours 2 à 4. Les coins sont adaptés à l'appariement (contrairement aux contours Canny) et tombent sur les arêtes d'obstacles et les bords de trous. |
| Carte | **Carte 3D de voxels libre / occupé / inconnu**, type DJI | Le planificateur cherche un chemin dans le libre ; l'exploration des frontières libre / inconnu dit si un passage mène quelque part. |
| Navigation | Algorithme déterministe (A* sur la carte, contrôleur vers la cible) | — |
| Cible | Coordonnée connue dans le monde | La vision sert à percevoir les obstacles, pas à trouver la cible. |
| Lumière comme indice | **Abandonné** | Peu fiable, pas utilisé par les vraies solutions. |
| Physique | **Hybride : MuJoCo pour la dynamique de vol et les chocs, rendu moderngl gardé** (`mujoco_flight.py`). L'ancien modèle cinématique a été retiré le 21/10 | Vraie dynamique (4 moteurs, inertie, gravité, traînée, chocs) sans perdre shaders, instanciation, panneau ni vérité terrain. Une migration complète du rendu vers MuJoCo coûtait 8 à 12 jours sans gain pour la vision (étude du 21/10). |

## Ce qui est fait

- Scène moderngl : **sol quadrillé par défaut avec toggle**, **bâtiment industriel procédural (`building.py`) avec vol intérieur et 25 ouvertures**, sphère et mur de test percé.
- **Bâtiment industriel procédural (`building.py`)** :
  - Décomposition exacte en 90 boîtes solides (`Box` de collision SAT) : **aucun obstacle invisible, le drone peut voler à l'intérieur en toute fluidité**.
  - Deux étages : grand hall hangar (hauteur 5.8 m), atelier/laboratoire adjacent, étage mezzanine (Y = 3.1 m), toiture industrielle.
  - **25 ouvertures modélisées avec vérités terrain** :
    - 16 ouvertures franchissables de tailles variées (grand portail 3.0x2.7m, portes piétonnes, fenêtres larges, fenêtres hautes).
    - 1 fenêtre étroite de précision (0.50x0.70m, largeur drone 0.21m).
    - 1 fente d'aération infranchissable (0.22x0.22m, déclenche la collision physique).
    - 2 niches aveugles (recesses avec fond plein à faible profondeur pour tester la distinction passage/niche par stéréo).
    - 1 trémie de mezzanine (2.0x2.0m) pour voler entre le rez-de-chaussée et l'étage.
    - 1 puits de lumière zénithal (2.0x2.0m) en toiture pour insertion/extraction aérienne.
- **Shader architectural procédural GLSL 330** :
  - Murs en parpaings de béton industriels avec joints de mortier, appareil décalé (running bond), grain et variations subtiles de teinte.
  - Cadres et linteaux métalliques en acier foncé (avec rivets simulés) offrant un **fort contraste local optimal pour le détecteur de coins de Harris**.
  - Zébrures d'avertissement de sécurité (jaune/noir) au-dessus du portail principal.
  - Dalle béton intérieure lissée avec joints de dilatation et réflexion spéculaire.
  - Éclairage équilibré (soleil direct + ambiance hémisphérique ciel/sol + illumination de rebond intérieure) assurant une visibilité claire à l'intérieur.
- **Modèle 3D du drone** DJI Avata 2 chargé depuis le GLB, à l'échelle réelle (21 × 6 × 18 cm).
- **Vol par MuJoCo** (`mujoco_flight.py`) : quadrotor à 4 moteurs à l'échelle de l'Avata 2 (377 g, boîte du GLB, poussée max 3 × poids), traînée de l'air. Contrôleur en cascade : vitesse (PI) -> poussée et attitude (inclinaison max 30°) -> couples (contrôle géométrique sur SO(3)) -> poussée de chaque moteur. Même commande qu'avant (vitesse en repère drone + vitesse de lacet). La scène de collision MuJoCo est générée depuis les collisionneurs (`Box`, `Sphere`, `VerticalCylinder`, `Ground`) et reconstruite au changement d'environnement. Un choc peut faire basculer le drone ; renversé au sol plus d'1 s, il est redressé sur place. Caméras **fixées au châssis** par défaut (elles suivent tangage et roulis : la paire reste rectifiée, mais le passage en repère monde doit utiliser la rotation 3D complète), ou stabilisées (case du panneau).
- Pilotage clavier et manette (support GLFW gamepad avec détection automatique). Contacts MuJoCo raidis (pas de 2 ms, `solref` 4 ms, marge 2 cm) et caméras 3 cm en retrait de la face avant : elles ne passent pas dans les murs. Le drone n'est pas rendu dans ses propres caméras.
- **Deux caméras** sur la face avant du drone, parallèles (paire déjà rectifiée), fovy 60°, inclinables.
- **Panneau latéral** (Dear ImGui) : choix des vues, bascule manette, sliders de réglage en direct (vitesse max en m/s et km/h, sensibilité lacet et caméra), sélecteur de bâtiment (Procédural / Aucun), **boutons de téléportation rapide** (Portail hangar, Grand hall, Mezzanine, Atelier, Toit), option de quadrillage du sol, télémétrie complète.
- **Forêt** : slider « Densité (arbres/100 m²) » (1 à 10, défaut 7) dans le panneau ; la forêt est régénérée au relâchement du slider, avec la même graine (`regen_forest_density` dans scene.py). Le bouton « Nouvelle forêt » garde la densité choisie. Au-delà de ~9, le placement sature (distance minimale entre troncs 3,2 m).
- Slides des cours 1 à 4 récupérées depuis Moodle (`Docs/Slides`).
- Analyse de l'approche vision : `Docs/Approche_vision_drone.md`.

## Ce qu'on veut faire (prochaines étapes)

1. Récupérer les images des deux caméras en numpy (rendu hors écran) et calculer `K` (déjà disponible : `Camera.intrinsics`).
2. Harris + appariement épipolaire + RANSAC + triangulation sur le bâtiment procédural et le mur de test ; comparaison au z-buffer (vérité terrain).
3. Détection du mur (RANSAC plan) et des ouvertures de passage → largeur, hauteur, distance ; règle de passage.
4. Profondeur derrière le trou (passage ou niche).
5. Carte de voxels, frontières, A* vers la cible.
6. Option : caméras légèrement convergentes et estimation de `F` par 8 points + RANSAC (Lab 4) ; suivi des coins entre frames pour estimer le déplacement du drone.

## Organisation des fichiers

```
Projet/
├── PROJET.md               ce fichier
├── building.py             générateur de bâtiment procédural, collision exacte et shader architectural
├── scene.py                point d'entrée : scène, drone, caméras, panneau, boucle principale
├── collision.py            formes de collision des environnements (boîtes, sphères, cylindres, sol), converties pour MuJoCo
├── mujoco_flight.py        vol par MuJoCo : quadrotor, scène de collision, contrôleur en cascade
├── gltf_loader.py          chargeur GLB (géométrie, normales, UVs, textures, couleurs)
├── ui_panel.py             panneau latéral ImGui (choix des vues, réglages vol, téléportation, infos)
├── requirements.txt        moderngl, glfw, numpy, pillow, imgui-bundle, mujoco
├── Assets/
│   ├── models/             DJI_Avata2_Sketchfab.glb (seul asset versionné)
│   ├── source/             DJI_Avata2_Sketchfab.zip (local, non versionné)
│   └── textures/
│       └── drone/          textures du modèle de drone (local, non versionné)
└── Docs/
    ├── Approche_vision_drone.md   approche vision détaillée
    └── Slides/             PDF des cours 1 à 4
```

Réglages en haut de `scene.py` : `STEREO_BASELINE`, `CAMERA_FOVY`, `SPLIT`, `WINDOW_SIZE`, `WORLD_CAMERA_POSITION`, `WORLD_CAMERA_TARGET`.

## Lancer

```bash
pip install -r requirements.txt
python scene.py
```

Commandes : ZQSD pour se déplacer, Espace / Ctrl pour monter / descendre, Shift pour accélérer. Clic dans une vue pour piloter à la souris (lacet et inclinaison des caméras), Échap pour libérer la souris, Tab pour la caméra de suivi.

Environnement testé : Windows, Python 3.14, moderngl 5.13.0, glfw 2.10.2, numpy 2.5.4.
