# Changelog

Toutes les modifications notables d'AstroFrame seront documentées dans ce
fichier.

Le format suit [Keep a Changelog](https://keepachangelog.com/fr/1.1.0/), et le
versionnage [SemVer](https://semver.org/).

## [0.8.0] - 2026-08-18

### Ajouté

- **Entraînement automatique avec CNN de détection** (`validator.py`) — en
  validation automatique (`--auto` avec `--cnn`), les patches des disques
  sont collectés à chaque série et la CNN `DiskFilter` est ré-entraînée
  entre les séries :
  - collecte de `cnn_positives` (cercles du guide) et `cnn_negatives`
    (formes rejetées + recadrages aléatoires déterministes exclus par IoU) ;
  - la série suivante juge avec le nouveau modèle (`--cnn-threshold`) ; le
    résultat est comparé au **champion** de la base — s'il est strictement
    meilleur il est promu (`disk_filter.npz` mis à jour) et la série suivante
    fait un warm-start des poids du champion ;
  - fenêtre d'entraînement auto avec case **Entraîner la CNN** et section CNN
    dans le rapport final ; `STATE_VERSION=2` (`cnn_series`, `cnn_positives`,
    `cnn_negatives`) avec compatibilité des fichiers v1 ;
  - nouvelles options CLI : `--epochs`, `--cnn-off`, `--cnn-threshold`.
- **Entraîneur de la CNN résiduelle** (`enhancer_trainer.py`) — outil
  autonome pour entraîner et valider l'amélioration d'image :
  - GUI côte à côte (sans-CNN vs avec-CNN) avec jugement **Valide/Rejeté**
    (Valide enregistre les paires entrée→sortie ; Rejeté entrée→entrée) ;
  - **Entraîner maintenant** entraîne la résiduelle avec les paires
    accumulées (warm-start du champion), compare par `mean_delta` et promeut
    si meilleure (`~/.astroframe/enhancer_cnn.npz`) ;
  - CLI avec `--check`, `--auto` (séries à dégradation synthétique),
    `--samples/--epochs/--seed/--export/--state/--reset-state/--config`.
- **Base d'apprentissage élargie** (`astroframe/ai/feedback.py`) — nouvelle
  table `logs` (historique d'événements par composant) et table `models`
  (artefacts NN avec métrique, champion et historique de séries).

### Sécurité

- L'IA d'entraînement reste désactivée par défaut (`--cnn-off` dans les
  séries automatiques, `ai.disk_filter`/`ai.cnn_enhance` toujours désactivés
  par défaut) ; les modèles manquants/corrompus dégradent silencieusement et
  le jugement ne vide jamais la liste détectée.

### Tests

- **649 tests, couverture 100 %** de `src/astroframe/`, `validator.py` et
  `enhancer_trainer.py` — nouvelles suites `tests/test_enhancer_trainer.py`
  (34) et `tests/test_enhancer_trainer_ui.py` (10), suites étendues
  `tests/test_validator.py` (69), `tests/test_validator_ui.py` (44) et
  `tests/test_feedback.py` (32) ; la fixture `_ai_isolado` du conftest isole
  base, modèles et chemins canoniques par test ; `ruff check` propre.

### Documentation

- Docs PT/EN/FR mises à jour : validation automatique avec CNN, entraîneur
  de la CNN résiduelle, tables `logs`/`models` et logique du champion.

### Infrastructure de données (`Logs/`)

- Nouvelle structure de dossiers à la racine du dépôt, remplaçant les
  anciens chemins sous `~/.astroframe/` et `samples/` :
  - `Logs/weights/` — modèles canoniques (`disk_filter.npz`,
    `enhancer_cnn.npz`, `lstm.npz`) avec `Logs/weights/staging/` pour les
    candidats de chaque ronde d'entraînement ;
  - `Logs/train/` — artefacts d'entraînement par défaut : `calibration.json`
    (ground truth global, avec repli sur `samples/calibration.json`),
    `validator_state.json`, `enhancer_state.json`, `trained_config.json` ;
  - `Logs/logs/ia/` — rapports de ronde des réseaux
    (`disk_filter_round_N.json`, `enhancer_round_N.json`) ;
  - `Logs/logs/system/` — journaux système rotatifs (1 Mio × 3) et
    `feedback.db` ;
  - nouveau module `src/astroframe/paths.py` (accesseurs + `migrate_legacy()`
    qui copie une fois les artefacts hérités de `~/.astroframe/` et
    `samples/calibration.json` sans supprimer la source ; la variable
    `ASTROFRAME_DATA_DIR` redirige la racine) et journalisation fichier dans
    chaque point d'entrée (`main`, `calibrate`, `validator`,
    `enhancer_trainer`, CLI) ; `.gitignore` ignore désormais `Logs/**` avec
    exceptions `.gitkeep`.

### Documentation (astro-centrée)

- Documentation et chaînes de code reformulées pour être **astro-centrées** :
  les protagonistes sont les astres (Soleil, Lune, planètes, comètes,
  étoiles) dans les astrophotographies et astrovidéos ; les phénomènes
  (éclipses, transits, occultations) n'apparaissent plus que comme exemples
  contextuels ; « compagnon d'éclipse » → « disque secondaire » ;
  description/mots-clés du `pyproject.toml`, README PT/EN/FR, `docs/PT|EN|FR/*`
  et `samples/README.md` mis à jour.

## [0.7.0] - 2026-08-17

### Ajouté

- **Auto-réglage** (nouveau module `astroframe.ai.tuner`) — optimise **tous
  les paramètres** de la pipeline contre les échantillons de `samples/` :
  - `ProxyEval` — évaluation rapide sur ~480 p (échelle de travail maximale
    0,5, jamais agrandie) : la détection réelle (Hough) est comparée au
    ground truth de `calibration.json` (**IoU moyen** entre disques détectés
    et attendus, avec **pénalités pour disques en trop/manquants**) et
    jusqu'à 3 frames notées en étoiles ; résultats **mis en cache par hash
    des paramètres effectifs** ;
  - `BoundedHillClimb` — montée de colline **déterministe** (graine fixe) et
    **bornée** : essais ±pas par paramètre, **momentum** (pas doublé après
    deux acceptations consécutives), **pas adaptatifs** (moitié en cas
    d'échec, plancher pas/8), **recuit facultatif** (acceptation de pires
    solutions avec probabilité exp(−Δ/T), température décroissante),
    patience (3 passes sans progrès) et **budget de temps** ; les paramètres
    coûteux (débruitage) ne sont essayés qu'une passe sur deux ; jamais hors
    des gammes sûres du registre ;
  - `run_autotune` orchestre tout et **enregistre le résultat dans la base
    d'apprentissage** (table `tuning` de la `FeedbackDB`, profil `tuning` par
    défaut) — appliqué automatiquement aux exécutions suivantes du même
    profil ; `export_trained_config` écrit la configuration optimisée (JSON
    versionné, défaut : `<samples>/trained_config.json`) ;
  - **pré-initialisation par les prédictions LSTM** du profil quand elles
    améliorent l'objectif du proxy (`_lstm_seed`).
- **Registre unifié des paramètres** (`astroframe.ai.params`) — source unique
  des **gammes sûres**, pas et deltas de toute la pipeline (34 paramètres ;
  les **17** des groupes détection + amélioration sont les cibles de
  l'auto-réglage) : bornes, pas, dtype, **parité impaire** des kernels
  gaussiens, coût d'évaluation et pénalités/récompenses du validator. Tout
  valeur apprise passe par `clamp_value` — le validator, le feedback et le
  tuner lisent le même registre et ne peuvent plus diverger.
- **LSTM en NumPy pur** (`astroframe.ai.lstm`, sans dépendance ; PyTorch
  facultatif via `torch_available()`) :
  - `LSTMTuner` — s'entraîne sur l'historique de feedback (une exécution =
    un pas de temps : notes par étoiles, 5 métriques, deltas) et **prédit le
    vecteur de deltas** de la prochaine exécution (point de départ de
    l'auto-réglage) ;
  - `TrajectoryPredictor` — prédit la position du disque au frame suivant
    pour l'anti-tremblement temporel : extrapolation linéaire (moindres
    carrés) + **raffinement LSTM facultatif** (cellule 2→8, `use_lstm`),
    entraîné hors ligne et de façon déterministe sur des **trajectoires
    synthétiques** (`train_trajectory_model`) ;
  - modèles `.npz` **versionnés** dans `~/.astroframe/lstm.npz` (fichier
    corrompu ou mauvaise version → repli silencieux).
- **CNN en NumPy pur** (`astroframe.ai.cnn`, im2col vectorisé, aucune
  dépendance) — petit réseau convolutif (conv 2D 3×3, ReLU, pooling moyen
  global, tête MLP), entraînement **hors ligne et déterministe** (graine
  fixe) avec early-stop, **gradients vérifiés par différences finies** :
  - `fit_residual` / `ResidualEnhancer` — modèle **résiduel** qui supprime
    bruit/smearing (`r = y − x`), appliqué en étape **post-unsharp** de
    `enhance_image` (`ai.cnn_enhance=true`) sur le canal **L du LAB** en
    tuiles **64×64 avec chevauchement**, couleurs préservées ; sans modèle,
    image intacte ;
  - `fit_classifier` / `DiskFilter` — classifieur **disque/bruit** qui note
    chaque candidat de `find_all_disks` (`confidence` = P(disque)) et peut
    **filtrer les faux positifs** (`ai.disk_filter > 0.0`) sans jamais vider
    la liste détectée ;
  - modèles : `~/.astroframe/enhancer_cnn.npz` et
    `~/.astroframe/disk_filter.npz`.
- **Feedback enrichi** — la base SQLite gagne la table **`tuning`**
  (`add_tuning`, `tuning_history`, `recent_tuning`, `reset_tuning`) ;
  `apply_learned` **additionne désormais les nudges par étoiles et les deltas
  d'auto-réglage**, toujours clamppés via le registre — la « mémoire » de
  l'IA entre exécutions ; sans rien d'appris, la configuration est retournée
  inchangée.
- **CLI** — sous-commande `astroframe autotune` (`--samples`, `--budget`,
  `--seed`, `--no-anneal`, `--params`, `--profile`, `--export`, `--reset` qui
  efface l'historique d'auto-réglage de la base) avec rapport en fin de
  recherche.
- **Interface Gradio** — nouvel onglet **Auto-tune** : dossier d'échantillons,
  budget en secondes, paramètres (multisélection), recuit, enregistrement
  dans la base ; affiche la progression, le **rapport** et la **configuration
  résultante**, avec un bouton pour effacer l'historique.
- **Configuration** — nouvelles sections **`[tuning]**
  (`enabled=false`, `budget_s=60.0`, `seed=42`, `anneal=true`,
  `params=null`, `proxy_scale=0.5`, `frames_per_sample=3`,
  `detection_weight=0.6`) et **`[ai]`** (`backend=numpy`,
  `lstm_trajectory=false`, `cnn_enhance=false`, `disk_filter=0.0`).

### Sécurité

- **Toute l'IA est désactivée par défaut** (`tuning.enabled=false` et `ai.*`
  à leurs valeurs par défaut) ; un modèle manquant ou corrompu **dégrade
  silencieusement** (repli sur le comportement d'origine) et **ne bloque
  jamais le pipeline**.

### Documentation

- Documentation française mise à jour : `docs/FR/API.md` (module
  `astroframe.ai` complet : registre, tuner, LSTM, CNN, feedback enrichi),
  `docs/FR/Architecture.md` (nouvelle section « Architecture IA »),
  `docs/FR/Usage.md` (CLI `autotune`, onglet Auto-tune, sections `[tuning]`
  et `[ai]`) et `README-FR.md` (fonctionnalités + paragraphe IA).

## [0.6.0] - 2026-08-14

### Ajouté

- **Validation et entraînement de la détection** (`validator.py` à la racine)
  — interface desktop native (tkinter) qui parcourt les échantillons de
  `samples/` un par un, montre la détection (principale + compagnons) sur
  l'image, et permet d'**accepter/rejeter** chaque forme par rapport au guide
  manuel (`calibration.json`) :
  - poids entraînables par forme : **7 paramètres** (`param2`, `param1`, `dp`,
    `gaussian_kernel_size`, `gaussian_sigma`, `occluded_ratio`,
    `occluded_ring`) avec deltas de récompense/punition, bornes et historique ;
  - **entraînement automatique** (`--auto`) : séries de re-détection avec
    auto-évaluation contre le guide (IoU minimum configurable), récompense/
    punition par forme détectée ou manquée, punition doublée pour les rejets
    obstinés, jusqu'à 100 % du matériel traité ;
  - **rapport final** avec score, poids entraînés et infobulles ⓘ par
    paramètre + bouton **Enregistrer** qui exporte la configuration entraînée
    vers `trained_config.json` (applicable au système réel) ;
  - **aperçu à la détection** — la détection se dessine sur l'image en temps
    réel avant de demander le verdict ;
  - **état persistant** dans `validator_state.json` (tours, séries, historique
    des poids et deltas) avec `--reset-state` pour repartir de zéro ;
  - mode `--check` (rapport sans interface) et interface avec curseurs d'IoU
    minimum, zoom/pan et dessin des disques.
- **Éditeur de calibration desktop** (`src/astroframe/ui/calibration_tk.py`) —
  fenêtre native (tkinter) dans `calibrate.py` (par défaut), remplaçant
  l'éditeur du navigateur :
  - clic crée un cercle/une ellipse, glisser déplace le centre, **poignées**
    ajustent les rayons horizontal/vertical, curseurs Rayon X/Rayon Y en
    temps réel, molette = zoom, bouton droit = déplacement, Suppr/flèches
    pour supprimer et déplacer (Maj = 10) ;
  - **deux passes** : 1re manuelle (détection désactivée) → ground truth ;
    2e avec **détection automatique au chargement** pour remplir/valider ;
  - les curseurs `param2`/rayon max relancent la détection au relâchement ;
  - **ellipses** prises en charge dans le ground truth (`ry` dans le JSON) et
    dans la validation (IoU par masque + rayon géométrique pour les erreurs) ;
  - `calibrate.py --ui gradio` conserve l'ancien éditeur navigateur.
- **Détection sans `min_radius`/`min_dist` explicites** — les rayons sont
  désormais dérivés automatiquement de l'image (résolution, diamètre
  principal) et la distance minimale est déduite de la détection ; la
  calibration ne suggère que les paramètres qui existent encore
  (`param2`/`param1`).
- **Couverture de tests à 100 %** sur tout le code (`validator.py` +
  `src/astroframe/`, ~435 tests) : tests d'interface avec Tk réel (fenêtre
  cachée), threads déterministes via `monkeypatch`, et infrastructure qui
  contourne l'avortement du GC de Python 3.12 pendant le bootstrap des
  threads avec couverture active (`gc.disable()` + collecte sûre sur le
  thread principal).

### Corrigé

- `RuntimeError: main thread is not in main loop` dans l'entraînement
  automatique — la valeur du curseur d'IoU était lue dans le thread de
  travail ; elle est maintenant capturée sur le thread principal avant de
  lancer la série.
- Avortement intermittent de la suite (`Fatal Python error: Aborted`) en
  exécutant la couverture sur des tests avec threads + Tk (GC pendant le
  bootstrap d'un thread de travail) — GC cyclique désactivé pour la session
  de tests.
- `_pan_start` non initialisé dans l'éditeur de validation (glisser sans clic
  préalable levait `AttributeError`).

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