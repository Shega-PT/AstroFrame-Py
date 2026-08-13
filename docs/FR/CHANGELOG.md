# Changelog

Toutes les modifications notables d'AstroFrame seront documentées dans ce
fichier.

Le format suit [Keep a Changelog](https://keepachangelog.com/fr/1.1.0/), et le
versionnage [SemVer](https://semver.org/).

## [0.5.0] - 2026-08-13

### Ajouté

- **Calibration par exemples** (nouveau paquet `astroframe/calibration/`) :
  - `scan_samples` scanne le dossier d'exemples (`samples/` par défaut)
    récursivement — les images (jpg/png/bmp/tif/webp) entrent telles quelles
    et les vidéos (mp4/avi/mov/mkv/m4v) contribuent **8 frames équidistantes
    et déterministes** (reproductibles dans la validation).
  - `CalibrationStore` stocke le **ground truth manuel** dans
    `samples/calibration.json` (JSON v1, clé = chemin relatif + `#frame`).
  - `circles_to_layers` / `layers_to_circles` convertissent les cercles en
    **calques RGBA** du `gr.ImageEditor` (glisser = déplacer, pinceau =
    ajouter, gomme = supprimer ; un cercle par composante connexe).
  - `validate_all` compare la détection automatique (`find_all_disks`) avec le
    ground truth sur tous les échantillons : correspondance gourmande par IoU
    (≥0,5), rappel/précision, IoU moyen, erreurs de centre (px) et de rayon
    (%) signées, score de calibration 0–100 (rappel 0,4 · précision 0,3 ·
    IoU 0,3) et **suggestions de paramètres** (ex. baisser `min_radius` si les
    petits disques échouent, monter `param2` avec les fausses détections).
- **Interface de calibration** (`astroframe/ui/calibration_app.py`) : menu
  déroulant d'échantillons + éditeur de cercles + boutons "Détection
  automatique", "Enregistrer les ajustements" et "Valider tous les
  échantillons" (tableau par échantillon + résumé global + suggestions).
- **Points d'entrée** : `calibrate.py` à la racine (miroir de `main.py`, avec
  `--samples/--config/--host/--port/--share/--no-browser`) et la sous-commande
  `astroframe calibrate` dans la CLI.
- `FrameReader.frame_at(index)` — lecture directe d'une frame par index
  (`CAP_PROP_POS_FRAMES`).
- `samples/README.md` réécrit avec la structure recommandée (images/vidéos,
  sous-dossiers par sujet : éclipse, lune, soleil, planètes).

### Documentation (multilingue)

- La documentation est devenue **trilingue** :
  - `docs/PT/` — `API.md`, `Arquitetura.md`, `USO.md` (déplacés, avec la
    section calibration) ; le `CHANGELOG.md` de la racine reste canonique (PT).
  - `docs/EN/` — `API.md`, `Architecture.md`, `Usage.md`, `CHANGELOG.md`
    traduits en anglais.
  - `docs/FR/` — `API.md`, `Architecture.md`, `Usage.md`, `CHANGELOG.md`
    traduits en français.
  - `README-EN.md` et `README-FR.md` à la racine (traductions du README) ; le
    `README.md` pointe maintenant vers `docs/PT/` et les versions EN/FR.

### Tests

- Suite élargie à **~260 tests avec 100 % de couverture** du paquet
  (`tests/test_calibration_{scan,store,circles,validate,app}.py` +
  `astroframe calibrate` dans la CLI + `frame_at`).

## [0.4.0] - 2026-08-13

### Ajouté

- **Compagnons d'éclipse (ex. la Lune entrant dans le Soleil)** :
  `find_all_disks` fait maintenant une **deuxième passe Hough avec `minDist`
  réduit** (1/4 du normal) pour trouver les cercles intérieurs à l'astre le
  plus grand, que la passe normale rejetterait. L'interface les dessine en
  **jaune** (astre le plus grand en vert, reflets de l'objectif en rouge),
  dans les onglets Image et Vidéo.
- **Filtre de cercles-fantômes par surface** (`_is_occluded_artifact`) : un
  cercle presque entièrement contenu dans l'astre le plus grand est écarté
  quand le contraste avec l'anneau autour est faible (les bords Soleil+Lune
  détectés comme un seul cercle) ; compare le chevauchement de **surface** (et
  pas seulement le centre), résistant au raffinement par centroïde.
- **Polissage par astre** (`core/polish.py` réécrit) : chaque astre reçoit son
  propre rehaussement (étirement local du contraste + luminosité, avec les
  silhouettes sombres et uniformes — ex. Lune en éclipse — préservées
  intactes) et l'image est **recomposée sans couture** par fusion de masques
  avec fondu (chevauchements = moyenne douce des rehaussements). La ligne de
  coupe (`corona_scale`) fond l'anneau dans le fond.
- **Fond = moyenne du fond original** (`polish.background_fill`, maintenant
  par défaut) au lieu de noir pur ; `polish.black_background` re-choisit le
  noir et `polish.brightness` contrôle la luminosité supplémentaire des astres.
- **Limite de disques** : `find_all_disks` retourne au maximum 5 disques
  (`_MAX_DISKS`).
- `find_all_disks` accepte les images en niveaux de gris `(H, W)`.

### Corrigé

- Reflets dessinés en rouge à l'intérieur de l'astre le plus grand (la
  séparation principal/compagnon/reflet est maintenant par le centre par
  rapport au rayon de l'astre le plus grand).
- Polissage effaçant les compagnons d'éclipse : seuls les cercles avec le
  centre **hors** de l'astre le plus grand sont supprimés comme reflets.

### Documentation

- `docs/PT/USO.md` et `docs/PT/API.md` mis à jour pour le polissage par astre,
  le nouveau `PolishConfig` et la détection en deux passes ; suite avec
  **221 tests et 100 % de couverture**.

## [0.3.0] - 2026-08-13

### Ajouté

- **Détection de multiples disques** (`find_all_disks` dans
  `core/stabilizer.py`) : au lieu de seulement le principal, le disque
  principal et ses **reflets** sont détectés (Hough + contours, avec fusion
  des doublons et préservation du plus lumineux à chaque centre). Le
  stabilisateur continue d'utiliser le principal et garde la dernière
  détection dans les frames sans disque (`last_detection`).
- **Polissage** (`core/polish.py`) : `polish_image()` applique de la
  luminosité au disque principal (en gardant la couronne floue), supprime les
  reflets et est utilisé dans l'aperçu/frame finale et dans la vidéo exportée.
- **Évaluation automatique** (`ai/score.py`) : `score_image()` calcule des
  étoiles (0–5) à partir du bruit, du contraste, de la taille du disque et de
  la couleur de la couronne ; l'interface montre le résultat dans
  "Évaluation automatique" (image **et** vidéo).
- **Base d'apprentissage par feedback** (`ai/feedback.py`) : chaque exécution
  est enregistrée (profil de caméra + paramètres + métriques + évaluation) ;
  l'utilisateur peut évaluer manuellement (0–5 étoiles) et le système
  **ajuste les sliders automatiquement** aux prochaines exécutions (plus doux
  avec les bonnes évaluations, plus fort avec les mauvaises ; débruitage
  supplémentaire pour le bruit, luminosité pour la couronne faible, etc.).
  Journal d'apprentissage avec l'historique et les raisons en SQLite (variable
  `ASTROFRAME_FEEDBACK_DB` pour l'emplacement).
- **Vidéos sans disque** : la pipeline stabilisation/aperçu saute le
  polissage et l'évaluation fonctionne sans détection (avant : échec).

### Corrigé

- Le polissage **effaçait un cercle au centre de l'image** : les cercles
  intérieurs quasi-concentriques au disque principal (ex. la silhouette de la
  Lune dans le Soleil) étaient détectés comme "reflets" et supprimés —
  `polish_image` ne supprime maintenant que les reflets dont le **centre est
  hors du disque principal**, et `find_all_disks` fusionne les cercles
  concentriques (tolérance de 12 % du rayon), évitant les doublons du même
  bord dans les deux sens (polissage et dessin en direct).

### Documentation

- `docs/PT/USO.md` : évaluation automatique/manuelle, journal d'apprentissage
  et section vidéo réécrite ; `docs/PT/API.md` avec `find_all_disks`,
  `polish_image`, `score_image` et le nouveau paquet `ai/`.

## [0.2.0] - 2026-08-12

### Ajouté

- **Nouvelle interface vidéo en direct** (onglet "Vidéo") : le panneau gauche
  montre la vidéo en temps réel pendant son traitement, avec le cercle (bounding
  box) du disque détecté ; le droit met à jour à des frames espacées le
  résultat final (stabilisée + CLAHE + débruitage + netteté). Exportation
  facultative de la vidéo traitée (.mp4, sans audio). `_best_frame_from_video`
  a été remplacé par ce flux complet.
- **Lecture des métadonnées** (nouveau paquet `meta/`, implémentation propre
  MIT) : vidéo via la cascade ffprobe → OpenCV (codec, bitrate, durée, fps,
  résolution) et image via PIL/EXIF (ISO, exposition, ouverture, distance
  focale, caméra, date) ; aucune nouvelle dépendance pip.
- **Suggestions automatiques de paramètres** (`meta/suggest.py`) : rayons du
  stabilisateur proportionnels à la résolution, `denoise.h` mis à l'échelle
  par l'ISO, réduction du débruitage dans les vidéos à bitrate très comprimé ;
  appliquées aux sliders au chargement de la vidéo (restent modifiables).
- Panneau "ratio/qualité" dans l'interface (résolution, aspect ratio, fps,
  codec, bitrate, ISO, exposition, caméra) + `gr.JSON` avec les métadonnées
  brutes.
- Interface réorganisée en onglets ("Image" / "Vidéo") ; le traitement
  d'image est passé à `process_image_input()` (fonction de module testable).
- Détection en demi-résolution couverte et suite élargie à **131 tests avec
  100 % de couverture** du paquet (y compris RIFE sans PyTorch, via un module
  `torch` factice dans les tests).

## [0.1.2] - 2026-08-12

### Ajouté

- L'interface Gradio accepte les vidéos (`.mp4/.avi/.mov`) : la frame la plus
  nette de la vidéo est sélectionnée automatiquement (lucky imaging) et
  traitée comme une image (`_best_frame_from_video` dans `ui.gradio_app`).
- Tests pour la sélection de la frame la plus nette à partir d'une vidéo
  synthétique.

## [0.1.1] - 2026-08-12

### Ajouté

- `main.py` à la racine : point d'entrée unique qui démarre le frontend
  (Gradio) et le backend (pipeline) ensemble, ouvrant le navigateur
  automatiquement (`python main.py [--config|--host|--port|--share|--no-browser]`).
- Paramètre `inbrowser` dans `ui.gradio_app.run()` (ouvre le navigateur par
  défaut).

### Documentation

- README : section installation avec un avertissement sur le PEP 668
  (Debian/Ubuntu) et `python main.py` comme première commande d'utilisation
  rapide.
- `docs/PT/USO.md` : interface web documentée avec `python main.py`.
- `docs/PT/API.md` : nouvelle signature de `run()` avec `inbrowser`.
- `.gitignore` : motifs génériques pour les vidéos (`*.mp4`, `*.MP4`,
  `*.MOV`, `*.mkv`).

## [0.1.0] - 2026-08-12

### Ajouté

- Pipeline complète : stabilisation géométrique (HoughCircles + contours),
  amélioration automatique (CLAHE/denoise/unsharp) et orchestration (`core/`).
- Vidéo : lecture frame par frame, lucky imaging avec seuil statistique et
  stacking (`video/`).
- Interfaces : Gradio (Avant/Après, sliders, zoom) et CLI
  (`astroframe serve|process|video|config-template`).
- Configuration externe via YAML (`astroframe config-template`), avec
  validation et avertissements.
- Stabilisation temporelle (EMA du centroïde) avec réutilisation du dernier
  déplacement dans les frames sans détection.
- Recadrage automatique après translation (sans bords noirs, sans couper le
  disque) et rayons de détection relatifs à la résolution de la frame.
- Détection en demi-résolution sur les grandes frames (≥1200 px).
- Mode `--fast` (omet le débruitage) pour les vidéos.
- Interpolation RIFE facultative (`astroframe[rife]`), avec import paresseux.
- Licence MIT, CI GitHub Actions (pytest 3.10/3.12 + ruff) et 43 tests.

### Corrigé

- Canaux RGB/BGR inversés dans l'interface Gradio (couleurs maintenant
  correctes).
- Crash CLAHE sur les images plus petites que la grille.
- Stacking sans alignement des frames (centre maintenant avant l'empilement).
- Lot de photos qui abandonnait à la première erreur (continue maintenant et
  résume le résultat).
- Clés/types invalides dans `config.yaml` acceptés en silence (avertissent
  maintenant).

### Connu

- La vidéo exportée n'inclut pas d'audio (utiliser ffmpeg pour fusionner la
  piste).