# Guide d'utilisation — AstroFrame

Guide pratique pour installer, configurer et exécuter AstroFrame. La
spécification de la solution est dans [Architecture.md](Architecture.md) et la
référence du code dans [API.md](API.md).

## Sommaire

1. [Installation](#installation)
2. [Interface web (Gradio)](#interface-web-gradio)
3. [Calibration](#calibration)
4. [Validation et entraînement de la détection](#validation-et-entraînement-de-la-détection)
5. [Ligne de commande](#ligne-de-commande)
6. [Configuration (config.yaml)](#configuration-configyaml)
7. [Workflow vidéo](#workflow-vidéo)
8. [Limitations et remarques](#limitations-et-remarques)

---

## Installation

Nécessite Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows : .venv\Scripts\activate
pip install -e ".[dev]"
```

Sans environnement virtuel (`pip install -r requirements.txt` fonctionne, mais
sur Debian/Ubuntu 24+ le venv est obligatoire — PEP 668).

## Interface web (Gradio)

Le point d'entrée le plus simple est `main.py` à la racine du dépôt : il
démarre le frontend (Gradio) et le backend (moteur de traitement dans `core/`)
dans le même processus et ouvre le navigateur automatiquement.

```bash
python main.py
```

S'ouvre sur `http://127.0.0.1:7860`. Pour configurer, changer le port ou
obtenir un lien public :

```bash
python main.py --config config.yaml --port 7861 --share
python main.py --no-browser        # n'ouvre pas le navigateur (utile sur les serveurs)
```

Le même serveur est disponible via la CLI installée, équivalent à :

```bash
astroframe serve                   # équivalent à python main.py
astroframe serve --config config.yaml --port 7861 --share
```

> `--share` crée une URL publique temporaire (via le tunnel Gradio) — ne
> l'utilisez pas avec du matériel sensible.

L'interface a deux onglets :

### Onglet Image

- **Entrée** — charger la photo/frame (format d'image arbitraire).
- **Stabilisée** — disque centré, avec les disques détectés dessinés :
  **vert** = astre le plus grand, **jaune** = compagnons d'éclipse (ex. la
  Lune entrant dans le Soleil), **rouge** = reflets de l'objectif.
- **Traitée** — CLAHE + débruitage + netteté + **polissage par astre**
  (chaque astre rehaussé individuellement et recomposé sans couture ; fond =
  moyenne du fond original ; reflets supprimés).
- **Zoom** — agrandissement centré sur la couronne/le limbe.
- **Paramètres** — limite de recadrage CLAHE, force du débruitage, netteté,
  zoom et échelle de la couronne conservée par le polissage, avec des valeurs
  initiales issues du `config.yaml`.
- **Évaluation automatique** — étoiles (0–5) calculées à partir du bruit, du
  contraste, de la taille du disque et de la couleur de la couronne.
- **Évaluation manuelle + apprentissage** — glissez le nombre d'étoiles que le
  résultat mérite et cliquez sur *Enregistrer l'évaluation manuelle* :
  l'exécution est enregistrée et, la prochaine fois avec le même profil de
  caméra, les sliders **s'ajustent automatiquement** (correction légère pour
  les bonnes évaluations, forte pour les mauvaises). L'onglet *Journal
  d'apprentissage* montre l'historique et la raison de chaque ajustement.

### Onglet Vidéo

1. **Charger la vidéo** (`.mp4/.avi/.mov`). À ce moment, les **métadonnées**
   sont lues — ffprobe (si installé ; sinon seulement OpenCV :
   résolution/fps/frames) pour la vidéo, EXIF (PIL) pour les images — et
   affichées dans le panneau **Ratio / qualité / suggestions** (résolution,
   aspect ratio, fps, codec, bitrate, ISO, exposition, caméra). Les **sliders
   sont préremplis** avec les suggestions d'optimisation **et avec ce que l'IA
   a appris** (évaluations précédentes du même profil), mais restent
   modifiables.
2. **Traiter la vidéo** — pendant que la pipeline tourne :
   - **Gauche (en direct)** — la frame originale en temps réel avec les
     disques détectés : **vert** = astre le plus grand, **jaune** = compagnons
     d'éclipse, **rouge** = reflets de l'objectif.
   - **Droite (résultat final)** — à des frames bien espacées, le résultat
     avec toutes les corrections (stabilisée + CLAHE + débruitage + netteté +
     polissage par astre).
   - Barre d'état avec la frame actuelle et la progression, et **évaluation
     automatique** du résultat final.
3. **Exportation facultative** — cochez *"Exporter la vidéo traitée (.mp4, sans
   audio)"* pour écrire la vidéo complète à la fin (même passe, sans sauter de
   frames).
4. **Évaluation manuelle + apprentissage** — comme dans l'onglet Image, donnez
   des étoiles au résultat vidéo ; l'ajustement s'applique aux prochains
   chargements du même type de vidéo et apparaît dans le journal
   d'apprentissage.

### Base d'apprentissage (où c'est stocké)

Les exécutions (paramètres utilisés, métriques et évaluations) sont stockées
dans un fichier SQLite à `~/.astroframe/feedback.db`. Vous pouvez changer
l'emplacement avec la variable d'environnement `ASTROFRAME_FEEDBACK_DB` (par
exemple pour partager l'apprentissage entre plusieurs machines).

## Calibration

AstroFrame inclut une **interface dédiée à la calibration** : elle charge les
photos et vidéos du dossier [samples/](../../samples/README.md) et permet de
**dessiner les astres à la main** et de valider la détection automatique par
rapport au ground truth sur **toutes** les échantillons.

```bash
python calibrate.py                          # fenêtre desktop native (tkinter)
python calibrate.py --ui gradio              # interface navigateur
python calibrate.py --samples samples
astroframe calibrate --samples samples       # équivalent (CLI installée)
```

### Ce qui entre dans la calibration

- **Images** (jpg/png/bmp/tif/webp) — chacune est un élément.
- **Vidéos** (mp4/avi/mov/mkv/m4v) — chacune contribue 8 frames échantillonnées
  de façon équidistante et déterministe (reproductible dans la validation).
- Le dossier est scanné récursivement : organisez-le par sujet comme vous
  voulez (éclipse, lune, soleil, planètes — sous-dossiers dans
  `samples/images/` et `samples/videos/`).

### Workflow

Le flux se déroule en **deux passes** :

1. **1re passe — manuelle (détection désactivée par défaut) :**
   1. **Choisir l'échantillon** — la liste du panneau affiche tous les
      éléments (`IMG …` pour les images, `VID … #frame` pour les frames).
   2. **Dessiner les astres** — sur le canvas :
      - **cliquer sur un espace vide** → crée un cercle (ou une ellipse, selon
        le sélecteur) à ce point ;
      - **glisser l'intérieur** de la forme sélectionnée → déplace le centre ;
      - **glisser la poignée droite** → ajuste le rayon horizontal ; **poignée
        du haut** → rayon vertical (ellipse) ; les curseurs Rayon X/Rayon Y
        font le même réglage fin en temps réel ;
      - **molette** → zoom sur le curseur ; glisser avec le bouton
        droit/milieu → déplacer ; **Suppr** supprime la forme sélectionnée,
        les flèches la déplacent de 1 px (Maj = 10 px).
   3. **Enregistrer (Ctrl+S)** — écrit le ground truth de l'élément dans
      `samples/calibration.json` (fichier local, ignoré par git).
2. **2e passe — validation (activez « Détection automatique au chargement ») :**
   4. **Les échantillons sans ground truth** sont remplis automatiquement par
      la détection ; ceux enregistrés s'ouvrent exactement comme vous les
      avez laissés. **Ajustez** ce qu'il faut (mêmes gestes) et
      réenregistrez.
   5. **Valider tous les échantillons** — lance la détection automatique sur
      tout et compare avec le ground truth manuel : par échantillon et
      globalement elle retourne rappel, précision, IoU moyen, erreur du
      centre (px) et du rayon (%), faux négatifs/positifs, un **score de
      calibration (0–100)** et des **suggestions de paramètres** (ex. baisser
      `min_radius` si les petits disques échouent, monter `param2` s'il y a
      des fausses détections).
   6. Les curseurs de paramètres relancent la détection au relâchement
      (détection activée), pour affiner `param2`/rayons sans quitter
      l'échantillon.

> Les ellipses sont enregistrées comme objets (avec `ry` dans le JSON) ; la
> validation utilise l'IoU par masque quand il y a des ellipses et le rayon
> géométrique pour les erreurs.

### À quoi sert la calibration

Les cercles manuels sont la "bonne réponse" que le système compare avec la
détection automatique. Avec un dossier varié (éclipses, Lune, Soleil, planètes
— disques grands et petits, contraste haut et bas), la validation montre où la
détection échoue et quoi ajuster dans le `config.yaml` avant de traiter le
matériel réel.

## Validation et entraînement de la détection

`validator.py` utilise ce même ground truth pour **affiner la détection par
forme** : il parcourt les échantillons un par un, montre ce que
`find_all_disks` a trouvé (disque principal + compagnons d'éclipse) sur
l'image, et apprend à distinguer les bonnes détections des fausses.

```bash
python validator.py                          # fenêtre desktop (tkinter)
python validator.py --check                  # rapport sans interface
python validator.py --auto --series 3        # entraînement automatique (3 séries)
python validator.py --auto --iou 0.7         # IoU minimum exigé avec le guide
python validator.py --reset-state --check    # repartir de zéro et vérifier
```

### Comment ça marche

1. **Tour manuel** — sur chaque échantillon vous voyez la détection et le
   guide manuel (`calibration.json`) ; **Accepter/Rejeter** dit si la forme
   est correcte.
   - Avec un **aperçu à la détection** : la détection se dessine sur l'image
     en temps réel avant de demander le verdict.
2. **Entraînement automatique (`--auto`)** — sans fenêtre : chaque série
   re-détecte les échantillons et **s'auto-évalue** forme par forme contre le
   guide (IoU minimum configurable avec `--iou`). Chaque forme correcte
   **récompense** les paramètres qui l'ont trouvée ; chaque forme fausse ou
   manquée est **punie** (doublée pour les rejets obstinés). Le processus se
   termine avec 100 % du matériel traité.
3. **Poids entraînables (7)** — `param2`, `param1`, `dp`,
   `gaussian_kernel_size`, `gaussian_sigma`, `occluded_ratio`,
   `occluded_ring` : chacun a des deltas de récompense/punition, des bornes
   minimales et maximales et un historique d'application.
4. **Rapport final** — score de la détection, poids entraînés avec des
   **infobulles ⓘ** expliquant chaque paramètre, et le bouton **Enregistrer**
   exporte la configuration entraînée vers `trained_config.json` (dans le
   dossier des échantillons), prête pour le système réel.

### État

- La progression est conservée dans `validator_state.json` (dans le dossier
  des échantillons par défaut) : tours, séries, historique des
  poids/deltas et l'IoU minimum actuel.
- `--reset-state` efface tout (y compris l'historique) et repart ; seul, il
  ouvre ensuite l'interface ; combiné à `--check`/`--auto`, il s'exécute sans
  fenêtre.
- `--state fichier.json` change l'emplacement de l'état ; `--export
  sortie.json` change la destination du rapport enregistré.

## Ligne de commande

Les sous-commandes complètes (`astroframe --help`) :

| Commande | Description |
|---|---|
| `serve` | Démarre l'interface Gradio |
| `process` | Traite des photos en lot (`--input a.jpg b.jpg --output-dir dossier/`) |
| `video` | Traite une vidéo (`--mode stabilize\|enhance\|stack`) |
| `config-template` | Génère `config.yaml` avec les valeurs par défaut |
| `calibrate` | Ouvre l'interface de calibration (`--samples dossier/`) |

La validation/entraînement de la détection est un script séparé (voir
[Validation et entraînement de la détection](#validation-et-entraînement-de-la-détection)) :
`python validator.py [--check|--auto|--reset-state|--iou N]`.

### Photos en lot

```bash
astroframe process --input photo1.jpg photo2.jpg --output-dir outputs/ --config config.yaml
```

- Chaque fichier est traité indépendamment : si l'un est corrompu, le lot
  **continue** et le résumé sort à la fin (nombre d'échecs).
- Les sorties sont des PNG avec le suffixe `_processed.png`.

### Vidéo

```bash
astroframe video --input eclipse.mp4                                  # mode enhance (défaut)
astroframe video --input eclipse.mp4 --mode stabilize                 # centre seulement le disque
astroframe video --input eclipse.mp4 --mode stack --stack-n 20        # empile les 20 meilleures frames
astroframe video --input eclipse.mp4 --mode enhance --fast            # sans débruitage (rapide)
astroframe video --input eclipse.mp4 --output sortie.mp4              # nom du fichier de sortie
```

- **enhance / stabilize** — la vidéo est **stabilisée frame par frame** et
  ré-exportée en MP4 (`<nom>_stabilized.mp4` par défaut) avec une barre de
  progression. L'anti-tremblement temporel lisse le centroïde (EMA) et garde
  le dernier déplacement quand une frame n'a pas de détection.
- **stack** — sélectionne les N frames les plus nettes (lucky imaging),
  **centre chacune** et les combine (médiane par défaut) en un seul PNG.
- `--fast` omet l'étape la plus lente (le débruitage) et réduit beaucoup le
  temps de traitement sur les grandes vidéos.

## Configuration (config.yaml)

Générez le modèle et modifiez uniquement ce qui est nécessaire (le reste garde
les valeurs par défaut) :

```bash
astroframe config-template --output config.yaml
```

Tous les champs et types :

### `clahe`
| Champ | Type | Défaut | Description |
|---|---|---|---|
| `clip_limit` | float | `3.0` | Limite de recadrage du CLAHE (plus grand = plus de contraste) |
| `tile_grid_size` | int | `8` | Taille de la grille (réduite automatiquement si l'image est plus petite) |

### `denoise`
| Champ | Type | Défaut | Description |
|---|---|---|---|
| `h` | float | `5.0` | Force du débruitage (monter avec ISO élevé ~ σ du bruit) |
| `template_window_size` | int | `7` | Fenêtre de template du Non-Local Means |
| `search_window_size` | int | `21` | Fenêtre de recherche (plus petite = plus rapide) |

### `unsharp`
| Champ | Type | Défaut | Description |
|---|---|---|---|
| `sigma` | float | `2.0` | Écart-type du flou gaussien |
| `amount` | float | `0.5` | Intensité de la netteté |

### `stabilizer`
| Champ | Type | Défaut | Description |
|---|---|---|---|
| `min_radius` / `max_radius` | int | `30` / `400` | Limites de rayon du disque (ajustées à la résolution de la frame) |
| `dp`, `min_dist`, `param1`, `param2` | — | `1.2` / `100` / `50` / `30` | Paramètres du `HoughCircles` |
| `gaussian_kernel_size`, `gaussian_sigma` | — | `9` / `2.0` | Flou de pré-détection |
| `contour_fallback` | bool | `true` | Repli par contours quand Hough échoue |
| `auto_crop` | bool | `true` | Supprime les bords noirs de la translation (sans couper le disque) |
| `jitter_alpha` | float | `0.5` | Lissage EMA du centroïde (1 = pas de lissage) |

### `polish`
| Champ | Type | Défaut | Description |
|---|---|---|---|
| `enabled` | bool | `true` | Active/désactive le polissage |
| `corona_scale` | float | `1.6` | Ligne de coupe (× rayon de l'astre) : entre le bord et la ligne l'image est fondue dans le fond |
| `feather` | float | `0.02` | Lissage (fraction du rayon) du contour et des chevauchements entre astres |
| `background_fill` | bool | `true` | Fond = moyenne du fond original (hors ligne de coupe) |
| `black_background` | bool | `false` | `true` = fond noir pur au lieu de la moyenne |
| `brightness` | float | `0.15` | Luminosité supplémentaire ajoutée aux astres (0 = seulement étirement du contraste) |
| `remove_reflections` | bool | `true` | Supprime les cercles-fantômes (centre hors de l'astre le plus grand) |
| `reflection_min_radius` | int | `8` | Rayon minimal (px) d'un reflet à supprimer (plus petit = étoile/bruit) |

### `feedback` (apprentissage)
| Champ | Type | Défaut | Description |
|---|---|---|---|
| `enabled` | bool | `true` | Stocke les évaluations et applique l'ajustement appris aux sliders |
| `db_path` | str | `~/.astroframe/feedback.db` | Base SQLite avec l'historique des exécutions et ajustements |
| `learning_rate` | float | `0.3` | Fraction du delta appliquée par exécution |
| `user_weight` | float | `2.0` | Multiplicateur quand l'utilisateur évalue manuellement |
| `history_limit` | int | `12` | Exécutions récentes considérées par profil |

### `lucky`
| Champ | Type | Défaut | Description |
|---|---|---|---|
| `min_sharpness` | float\|null | `null` | Seuil de netteté fixe ; `null` = estimer depuis la vidéo |
| `sharpness_percentile` | float | `25.0` | Percentile utilisé dans l'estimation automatique |
| `gaussian_kernel_size`, `gaussian_sigma` | — | `5` / `1.5` | Flou avant le Laplacien |

### `stacking`
| Champ | Type | Défaut | Description |
|---|---|---|---|
| `n_best` | int | `10` | Nombre de frames à empiler (utilisé si `--stack-n` n'est pas donné) |
| `use_median` | bool | `true` | `true` = médiane (robuste), `false` = moyenne |

**Validation :** les clés inconnues et les types inattendus génèrent des
avertissements dans le journal (par exemple `clip_limit: "abc"`), mais ne
plantent jamais le démarrage.

## Workflow vidéo

1. **Capture** — enregistrer l'éclipse avec une caméra statique ; le
   tremblement lent est acceptable (la stabilisation absolue par le disque le
   compense).
2. **Présélection** (facultatif) : `astroframe video --input clip.mp4 --mode stack --stack-n 30`
   retourne un seul PNG avec la meilleure "instantané" possible.
3. **Stabilisation complète** : `astroframe video --input clip.mp4 --mode enhance`
   — centre constant et image améliorée. Pour les vidéos 1080p/4K, utilisez
   `--fast`.
4. **Post-exécution** : fusionner l'audio avec ffmpeg (voir limitations).

## Limitations et remarques

- **Audio** : l'exportateur utilise OpenCV et **ne copie pas l'audio** :
  ```bash
  ffmpeg -i original.mp4 -i traitee.mp4 -c copy -map 0:a -map 1:v sortie.mp4
  ```
- **Débruitage lent** : ~1 s/frame en 480p ; en 1080p il peut atteindre
  plusieurs secondes par frame. `--fast` ou réduire `search_window_size`.
- **Stacking haute résolution** : les frames au-dessus de 1080p empilées
  utilisent du float32 en mémoire (avertissement dans le journal) — réduisez
  `n_best` si cela dépasse le nécessaire.
- **Frames sans disque** : `center_and_stabilize` retourne la frame inchangée
  (avec un avertissement) ; en vidéo, `AntiJitterStabilizer` réutilise le
  dernier déplacement valide.
- **RIFE** (interpolation sur les sauts) est facultatif et exige PyTorch ; voir
  [API.md](API.md).
