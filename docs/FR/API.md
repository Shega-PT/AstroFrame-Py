# Référence de l'API — AstroFrame

Référence des modules et contrats du paquet `astroframe`. Pour une utilisation
pratique, voir [Usage.md](Usage.md).

**Conventions globales**

- Images : numpy `np.ndarray` **BGR** (convention OpenCV), `uint8`, `(H, W, 3)`.
  Les entrées en niveaux de gris `(H, W)` et RGBA `(H, W, 4)` (interprétées en
  RGBA) sont normalisées en BGR automatiquement.
- Configuration : `AstroFrameConfig` ; toute fonction de traitement l'accepte
  facultativement (utilise les valeurs par défaut si omis).

## `astroframe.config`

Dataclasses avec les paramètres (voir
[Usage.md](Usage.md#configuration-configyaml) pour le tableau champ par champ) :

- `CLAHEConfig`, `DenoiseConfig`, `UnsharpConfig`
- `StabilizerConfig`, `PolishConfig`, `FeedbackConfig`, `LuckyConfig`, `StackingConfig`
- `ScoreConfig`, `TuningConfig` (auto-réglage), `AIConfig` (réseaux LSTM/CNN)
- `AstroFrameConfig` — racine avec un champ par sous-configuration.

Méthodes de `AstroFrameConfig` :

```python
cfg.to_dict() -> dict
cfg.to_yaml(path) -> None
cfg.from_yaml(path) -> AstroFrameConfig   # classmethod
```

`from_yaml` valide les types et avertit (logging) des clés inconnues et des
types inattendus, sans échouer.

## `astroframe.core`

### `core.stabilizer`

```python
@dataclass(frozen=True)
class DiskDetection:
    cx: int
    cy: int          # centre détecté dans les coordonnées de l'image source
    radius: int      # rayon ajusté après recadrage/re-échelle

find_all_disks(image, config=None) -> list[DiskDetection]
find_disk_center(image, config=None) -> DiskDetection | None
center_and_stabilize(image, config=None) -> tuple[np.ndarray, DiskDetection | None]
class AntiJitterStabilizer(config=None, alpha=None): ...
```

- `find_all_disks` — **deux passes** de `HoughCircles` (la seconde avec
  `minDist` 1/4 du normal, pour les cercles intérieurs à l'astre le plus
  grand) + repli par contours ; jusqu'à **5 disques**, triés par rayon
  décroissant. Déduplication seulement des cercles du **même bord** (centres
  proches ET rayons quasi-égaux, tolérance 12 % du rayon), et rejet des
  **cercles-fantômes** : un candidat presque entièrement à l'intérieur d'un
  disque déjà accepté (≥90 % de la surface) avec un contraste faible face à
  l'anneau autour est écarté (`_is_occluded_artifact`). Accepte BGR ou niveaux
  de gris.
- `find_disk_center` — premier élément de `find_all_disks` ; HoughCircles +
  repli par contours + raffinement par centroïde d'intensité. Sur les frames
  ≥1200 px la détection tourne en demi-résolution.
- `center_and_stabilize` — translate la frame pour centrer le disque et coupe
  les bords noirs (`stabilizer.auto_crop`), retournant le rayon ajusté.
  Sans disque détecté, retourne l'image inchangée et `None`.
- `AntiJitterStabilizer.stabilize(frame) -> (frame, DiskDetection | None)` —
  état interne : EMA du centroïde (`jitter_alpha`) et réutilisation du dernier
  déplacement valide dans les frames sans détection (`last_detection` —
  propriété avec le dernier disque détecté, utilisée par la vidéo pour le
  polissage/aperçu).

### `core.polish`

```python
polish_image(image, detection, config=None) -> np.ndarray
```

- Polissage **par astre** : détecte tous les disques (`find_all_disks`),
  sépare les disques secondaires (centre à l'intérieur de l'astre le plus
  grand ; p. ex. la Lune devant le Soleil) des reflets de l'objectif (centre à l'extérieur), rehausse **chaque
  astre individuellement** (`_astro_boost` : étirement local du contraste +
  `polish.brightness` ;   les silhouettes sombres et uniformes — comme un disque
  secondaire concentrique (p. ex. la Lune devant le Soleil) — sont
  préservées intactes) et **recompose sans couture** par fusion
  de masques avec fondu (`_band_mask` + `_astro_region`) : la ligne de coupe
  `corona_scale × rayon` fond l'anneau dans le fond et les chevauchements entre
  astres sont la moyenne douce des rehaussements. Le fond est la **moyenne du
  fond original** (`background_fill`) ou noir pur (`black_background`) ; les
  reflets (rayon ≥ `reflection_min_radius` px) sont remplis avec le fond si
  `remove_reflections`. Sans détection, retourne l'image inchangée.

### `core.enhancer`

```python
clahe_enhance(image, config) -> np.ndarray
denoise(image, config) -> np.ndarray
unsharp_mask(image, config) -> np.ndarray
enhance_image(image, config=None, use_denoise=True) -> np.ndarray
```

- Ordre : CLAHE sur le canal L de LAB → `fastNlMeansDenoisingColored` →
  unsharp.
- `use_denoise=False` omet l'étape la plus lente (utilisée par `--fast`).

### `core.pipeline`

```python
@dataclass
class ProcessResult:
    original: np.ndarray
    stabilized: np.ndarray
    enhanced: np.ndarray       # stabilisée + CLAHE + denoise + unsharp + polissage
    enhanced_raw: np.ndarray   # la même, sans polissage (base de l'évaluation)
    detection: DiskDetection | None

process_image(image, config=None) -> ProcessResult
process_path(path, config=None) -> ProcessResult   # ValueError si illisible
```

## `astroframe.video`

### `video.reader`

```python
class FrameReader(path):
    .fps -> float
    .frame_count -> int          # 0 quand inconnu
    .size -> tuple[int, int]     # (largeur, hauteur)
    .close() / context manager
    itérable : frames BGR
    # ValueError si la vidéo ne s'ouvre pas
```

### `video.select` (lucky imaging)

```python
sharpness(frame, config=None) -> float        # variance du Laplacien
estimate_sharpness_threshold(scores, percentile=25.0) -> float
select_sharp_frames(frames, config=None, minimum=None) -> list[(idx, frame, score)]
```

- Ordre du seuil : `minimum` → `config.lucky.min_sharpness` → percentile
  estimé de la séquence elle-même.

### `video.stacking`

```python
stack_frames(frames, stacking=None) -> np.ndarray   # ValueError si vide ou shapes différents
select_best(frames, n_best, config=None) -> list[np.ndarray]
```

- `stack_frames` : médiane (`use_median=True`) ou moyenne ; avertit sur la
  mémoire au-dessus de 1080p avec beaucoup de frames.

## `astroframe.meta`

Lecture des métadonnées et suggestions de paramètres (implémentation propre,
MIT — inspirée de la même idée que MetadataExplorer, sans code copié).

### `meta.extractor`

```python
@dataclass(frozen=True)
class MediaMetadata:
    path: str | None
    kind: str                       # "image" | "video"
    width: int | None
    height: int | None
    aspect_ratio: float | None      # largeur / hauteur
    fps: float | None
    frame_count: int | None
    duration: float | None          # secondes
    codec: str | None
    bitrate: int | None             # bits/seconde
    format_name: str | None
    iso: int | None                 # sensibilité ISO (EXIF)
    exposure_time: float | None     # secondes
    focal_length: float | None      # mm
    aperture: float | None          # nombre f/
    camera_make: str | None
    camera_model: str | None
    captured_at: str | None         # date/heure EXIF
    raw: dict                       # tout ce qui a été lu (source→clé→valeur)

extract_metadata(path) -> MediaMetadata
```

- Vidéo : cascade **ffprobe** (si installé ; codec/bitrate/durée/format) →
  **OpenCV** (résolution/fps/frames — toujours disponible).
- Image : EXIF via PIL (ISO, exposition, ouverture, distance focale, caméra,
  date).
- `ValueError` si le chemin n'existe pas ; `kind="unknown"` avec ce qui peut
  être lu si ni ffprobe ni OpenCV n'ouvrent le fichier.
- `aspect_ratio` arrondi à 3 décimales (0.0 → `None`) ; le texte de
  présentation (ex. `5616×3744 · 3:2`) est `aspect_text` dans `extractor`
  (16:9, 3:2, 4:3, 1:1, carré ou changement décimal).

### `meta.suggest`

```python
suggest_config(meta: MediaMetadata) -> AstroFrameConfig
summary_fields(meta: MediaMetadata) -> dict[str, str]
```

- Heuristiques : rayons de détection proportionnels à la résolution
  (`min = 8 %` du demi-axe mineur, `max = 45 %`) ; `denoise.h` mis à l'échelle
  par l'ISO (`2 + ISO/1600*4`, limité à `[2, 15]`, utilisé par défaut si la
  config ne le définit pas) avec `unsharp` 0.4/0.6 ; dans les vidéos très
  compressées (< 0,1 bit/pixel) le débruitage est réduit ~30 % (moins de
  risque de "plastifier").
- `summary_fields` retourne le dictionnaire affiché dans le panneau "Ratio /
  qualité / suggestions" de l'interface.

## `astroframe.calibration`

Calibration de la détection contre des exemples (photos/vidéos), avec ground
truth manuel.

### `calibration.scan`

```python
@dataclass(frozen=True)
class SampleRef:
    kind: str            # "image" | "video"
    path: Path           # chemin absolu du fichier
    frame: int | None    # index de frame (None pour les images)
    key: str             # "chemin_relatif#frame" (clé stable dans le store)
    label: str           # "IMG chemin" / "VID chemin #frame" (interface)

scan_samples(root, frames_per_video=8) -> list[SampleRef]
sample_video_frames(frame_count, n=8) -> list[int]
load_frame(sample) -> np.ndarray          # BGR (image ou frame échantillonnée)
item_key(relpath, frame=None) -> str
item_label(kind, relpath, frame=None) -> str
```

- Scan **récursif** du dossier ; les images (jpg/jpeg/png/bmp/tif/tiff/webp)
  entrent telles quelles et les vidéos (mp4/avi/mov/mkv/m4v) contribuent N
  frames **équidistantes et déterministes** (milieux d'intervalles —
  reproductible dans la validation). Les vidéos illisibles sont ignorées avec
  un avertissement.
- `load_frame` lit l'image via `cv2.imread` ou la frame de la vidéo via
  `FrameReader.frame_at(index)` (nouveau — cherche `CAP_PROP_POS_FRAMES` ; les
  erreurs lèvent `ValueError`).

### `calibration.store`

```python
@dataclass
class CalibrationItem:
    path: str            # chemin relatif au dossier d'échantillons
    kind: str
    frame: int | None
    width: int
    height: int
    circles: list[DiskDetection] = []   # ground truth manuel

class CalibrationStore(path):           # JSON v1 (défaut : Logs/train/calibration.json ; repli : samples/calibration.json)
    .load() -> None                     # idempotent ; illisible/version -> vide
    .save() -> None
    .upsert_item(key, item) -> None     # écrit immédiatement
    .get_item(key) -> CalibrationItem | None
```

### `calibration.circles`

```python
circles_to_layers(image_rgb, circles) -> {"background": ..., "layers": [...]}
layers_to_circles(layers) -> list[DiskDetection]
```

- `circles_to_layers` construit la valeur du `gr.ImageEditor` : le fond +
  **un calque RGBA par cercle** (disque translucide + bord opaque). Les calques
  sont **glissables** dans l'interface → déplacer un cercle = glisser le
  calque ; peindre par-dessus ajoute ; la gomme supprime.
- `layers_to_circles` convertit ce que l'utilisateur a dessiné en cercles — un
  par **composante connexe** de chaque calque (deux peintures séparées sur le
  même calque = deux cercles) ; accepte des calques avec alpha ou RGB.

### `calibration.validate`

```python
circle_iou(a, b) -> float                                  # intersection/union 0–1
match_circles(manual, detected, iou_threshold=0.5)
    -> (pairs: list[(i, j)], unmatched_manual: set, unmatched_detected: set)

@dataclass
class ItemReport:        # par échantillon
    label, n_manual, n_detected, n_matched,
    n_false_negatives, n_false_positives,
    mean_iou, mean_center_error, mean_radius_error_pct   # None sans paires

@dataclass
class CalibrationReport: # agrégat
    items, total_* , recall, precision,
    mean_iou, mean_center_error, mean_radius_error_pct,
    score: float | None  # 0–100 = 0,4·rappel + 0,3·précision + 0,3·IoU

validate_item(label, manual, detected) -> ItemReport      # erreur de rayon signée (%)
validate_all([(label, manual, detected), ...]) -> CalibrationReport
suggest_parameters(report, config=None) -> list[str]      # suggestions
```

- Correspondance **gourmande par IoU décroissant** (seuil 0,5) : manuel↔détection.

## `astroframe.ui`

### `ui.gradio_app`

```python
build_app(config=None) -> gr.Blocks
run(config_path=None, host="127.0.0.1", port=7860, share=False, inbrowser=True) -> None

def inspect_video_upload(video_path, db=None, config=None) -> tuple[str, dict, dict, dict, dict, dict]
def process_video(video_path, export=False, denoise_h=None, ...) -> Generator[tuple]
def process_image_input(image, clip_limit=None, denoise_h=None, ..., db=None, config=None) -> tuple[np.ndarray, np.ndarray, np.ndarray, str, dict, str]
def manual_feedback(state, stars, db=None, config=None) -> tuple[str, str]
```

- L'UI convertit RGB→BGR à l'entrée et BGR→RGB aux sorties (fonctions
  `_to_pipeline` / `_from_pipeline`) ; les valeurs et la pipeline sont
  partagées avec la CLI.
- Deux onglets : **Image** (entrée, stabilisée, traitée, zoom, évaluation,
  sliders, évaluation manuelle + journal d'apprentissage) et **Vidéo**
  (upload, panneau de métadonnées, sliders préremplis, traitement en direct
  avec disques dessinés, évaluation automatique et manuelle, exportation
  facultative).
- `inspect_video_upload` appelle `meta.extractor` + `meta.suggest` +
  `apply_learned` (évaluations précédentes du même profil de caméra) et
  retourne, respectivement : le HTML du résumé (ratio/qualité/suggestions),
  les métadonnées brutes et les `update()` des sliders.
- `process_video` est un **générateur** (consommé par le `gr.Progress.track`
  de Gradio) ; chaque frame retourne :
  `(live_rgb, preview_rgb, out_video_path_ou_None, status, progress, rating_html,
  run_state, log_html)` — `live` est la frame originale en temps réel avec les
  disques détectés (`_draw_disks` : **vert** = astre le plus grand,   **jaune** =
  disques secondaires, **rouge** = reflets — séparés par `_split_disks`, qui
  utilise le centre du disque vs. le rayon de l'astre le plus grand), `preview`
  est le résultat final montré seulement à des frames espacées
  (`_preview_every`), les autres champs avec `None`/fraction au milieu de la
  passe. Sans disque détecté dans **aucune** frame, le résultat final sort sans
  polissage et l'évaluation est calculée sans détection (avertissement dans
  l'état). Si `export=True`, il écrit la vidéo complète polie (.mp4, codec
  `mp4v`, sans audio) et retourne le chemin sur la dernière frame.
- `process_image_input` retourne `(stabilisée, traitée, zoom, HTML de
  l'évaluation, état, journal d'apprentissage)` ; l'état (profil, évaluation,
  paramètres) alimente le `manual_feedback` qui enregistre l'évaluation en
  étoiles et rapporte l'ajustement appris dans le journal.
- `run()` accepte `inbrowser` pour ouvrir le navigateur automatiquement ; le
  point d'entrée unique équivalent est `python main.py` à la racine du dépôt.

### `ui.calibration_app`

```python
build_calibration_app(samples_dir="samples", config=None, store=None) -> gr.Blocks
run(samples_dir="samples", config_path=None, host="127.0.0.1", port=7860, share=False, inbrowser=True) -> None

def load_item_payload(key, samples_dir, config=None, store=None) -> (dict, str)
def auto_detect_payload(key, samples_dir, config=None) -> (dict, str)
def save_item_circles(editor_value, key, samples_dir, store=None) -> str
def validate_all_report(samples_dir, config=None, store=None) -> (rows, summary_html, suggestions_html)
```

- Disposition : menu déroulant d'échantillons + `gr.ImageEditor` (calques RGBA
  par cercle, pinceau/gomme) + boutons "Détection automatique" /
  "Enregistrer les ajustements" / "Valider tous les échantillons" → tableau par
  échantillon, résumé global (score 0–100, rappel, précision, IoU, erreurs) et
  suggestions de paramètres.
- `load_item_payload` donne la priorité au ground truth enregistré ; sans lui,
  il utilise la détection automatique comme point de départ.
  `save_item_circles` convertit les calques de l'éditeur en cercles et les
  enregistre dans le store. `validate_all_report` parcourt **tous** les
  échantillons (images + frames échantillonnées des vidéos).
- Point d'entrée équivalent : `python calibrate.py` à la racine du dépôt ou
  `astroframe calibrate` (CLI).

### `ui.cli`

```python
main(argv=None) -> int                 # point d'entrée du script `astroframe`
build_parser() -> argparse.ArgumentParser
process_images(paths, output_dir, config) -> tuple[int, int]   # (succès, échecs)
process_video(path, output, config, mode, stack_n, fast) -> str  # chemin de sortie
```

- Sous-commandes : `serve`, `process`, `video` (`--mode stabilize|enhance|stack`,
  `--fast`), `config-template`, `calibrate` (interface de calibration),
  `autotune` (auto-réglage — voir `ai.tuner`).
- `process_images` continue après les échecs individuels et lève `RuntimeError`
  si rien n'a été traité. `mode="stack"` centre les frames avant
  l'empilement. L'exportation vidéo ne copie pas l'audio (limité par OpenCV).

## `astroframe.ai` (IA : auto-réglage, LSTM/CNN, RIFE facultatif)

Toute l'IA est **désactivée par défaut** (`tuning.enabled=false`, `ai.*` aux
valeurs par défaut) ; un modèle manquant ou corrompu dégrade **silencieusement**
et ne bloque jamais le pipeline.

```python
class RifeInterpolator(repo, source="github", model_name="IFNet", device=None):
    .available() -> bool            # sans état : False si PyTorch n'est pas installé
    .interpolate(frame_a, frame_b, n_interp=1) -> list[np.ndarray]
```

- Nécessite `pip install -e ".[rife]"`. Accepte BGR ; retourne `n_interp`
  frames intermédiaires en BGR. L'interface du modèle dépend du dépôt RIFE
  utilisé (le `_infer` interne est le point à ajuster entre versions) ; sans
  PyTorch il lève `RuntimeError` avec des instructions.

### `ai.score` (évaluation automatique)

```python
@dataclass
class StarRating:
    stars: float            # 0.0–5.0 (poids des métriques = 1)
    score: float            # 0.0–1.0 non pondéré
    metrics: dict[str, float]  # noise | contrast | size | corona ; 0 (mauvais) à 1 (bon)
    explanation: str        # texte humain avec le pourquoi

score_image(image, detection=None, config=None) -> StarRating
package_rating(original, stabilized, detection, config=None) -> StarRating
score_from_stars(stars, metrics=None) -> StarRating   # pour tests/externalisation
```

- `noise` = variance du Laplacien (sans bruit → 1), `contrast` = rapport des
  percentiles 99/50 de la luminance, `size` = rayon du disque vs. frame,
  `corona` = luminosité moyenne de l'anneau de couronne (1–2× rayon) vs. le
  disque.
- `score_image` fonctionne **sans détection** (métriques de bruit/contraste
  seulement).

### `ai.feedback` (apprentissage par évaluation)

```python
@dataclass(frozen=True)
class ConfigNudge:
    clip_limit: dict       # {multiplicateur, offset}   ex. : {'m': 1.0, 'b': 0.0}
    denoise: dict          #                                ex. : {'h': {'m': 1.0, 'b': 1.5}}
    unsharp: dict          # {'amount': {...}, 'sigma': {'m': 0.7, 'b': 1.2}}
    polish: dict           # {'brightness': {...}}
    explanation: dict[str, str]  # texte par paramètre (pourquoi l'ajustement)
    judicial_override: bool      # True : la mauvaise évaluation impose la correction immédiatement
    factor: float          # échelle globale de la correction (feedback.learning_rate)

@dataclass(frozen=True)
class RunRecord:
    id: int; profile: str; kind: str; params: dict; metrics: dict
    stars: float; source: str; at: str; modified: dict | None

def profile_for(kind, width, height) -> str     # ex. : "video@5616x3744"
def format_profile(profile) -> str              # "5616×3744 · vídeo" (interface)
def record_run(db, kind, profile, config, params, rating, source="cli") -> RunRecord
def recent_nudges(db, profile, limit=5) -> list[RunRecord]
def apply_learned(config, profile, db=None) -> AstroFrameConfig
def _learning_db(config, db=None) -> FeedbackDB | None   # feedback.enabled ?
def _learning_log_html(profile, db) -> str               # historique (interface)
class FeedbackDB(path=None):               # SQLite avec retry de verrouillage (WAL)
    .history(profile, limit=50, base=None) -> list[RunRecord]
    .history_all(limit=32) -> list[RunRecord]   # toutes les exécutions (LSTM)
    .latest_ids(profile, limit=5) -> list[int]
    .nudges(profile_runs, nudge_params, factor) -> ConfigNudge  # règles
    .store_run(kind, profile, config, params, stars, source, metrics) -> RunRecord
    .apply_nudge(config, nudge) -> AstroFrameConfig
    .add_tuning(profile, base_params, deltas, report, source="autotune") -> int
    .tuning_history(profile, limit=12) -> list[dict]   # table tuning
    .recent_tuning(profile, limit=1) -> list[dict]     # derniers deltas d'auto-réglage
    .reset_tuning() -> int                             # efface la table tuning
```

- Base SQLite à `Logs/logs/system/feedback.db` (ou `$ASTROFRAME_FEEDBACK_DB`) ;
  créée au premier usage, avec retry sur base verrouillée et `history_limit`
  par profil. Deux tables : `runs` (exécutions + évaluations) et `tuning`
  (auto-réglages enregistrés par profil).
- `apply_learned` retourne la config originale s'il n'y a pas d'historique (ou
  si le `judicial_override`/`factor` est nul) ; il **additionne les nudges par
  étoiles et les deltas d'auto-réglage** (`recent_tuning`), toujours clamppés
  par le registre (`ai.params`) — la « mémoire » de l'IA entre exécutions ;
  règles : les bonnes évaluations cohérentes adoucissent l'ajustement
  (`user_weight`), les mauvaises évaluations appliquent un débruitage
  supplémentaire avec bruit (métriques >`1.0`), couronne faible augmente la
  luminosité du polissage, petit disque réduit les rayons du détecteur ; les
  valeurs sont limitées aux intervalles valides.
- `FeedbackDB.default_path() -> Path`, `.path -> Path`, `.close()`.

### `ai.params` (registre unifié des paramètres réglables)

Source unique de vérité des **gammes sûres**, pas d'optimisation et deltas
d'apprentissage de toute la pipeline — lus par le tuner, le validator et le
feedback, ils ne peuvent plus diverger.

```python
@dataclass(frozen=True)
class ParamSpec:
    path: str              # chemin dans la config ("stabilizer.param2")
    low: float             # gamme sûre (clamp : l'apprentissage ne sort jamais d'ici)
    high: float
    step: float            # pas initial du hill-climbing du tuner
    dtype: type            # int (arrondi) ou float
    odd: bool              # True : valeur impaire obligatoire (kernels gaussiens)
    group: str             # detect | geometry | enhance | stack | polish | score | meta
    costly: bool           # True : évaluation chère (débruitage)
    punish / reward: float # deltas du validator (groupe detect uniquement)
    .name -> str           # nom court (champ dans la section)

PARAM_SPECS: dict[str, ParamSpec]   # index par chemin complet
FEEDBACK_PARAMS: tuple[str, ...]    # les 5 paramètres visuels du feedback
INT_PARAMS: frozenset[str]          # paramètres à valeur entière

specs(group=None) -> list[ParamSpec]          # ordre de déclaration
spec(path) / spec_by_name(name) -> ParamSpec
bounds(path) -> tuple[float, float]
step(path) -> float
clamp_value(path, value) -> int | float       # clamp + arrondi + parité impaire
get_param(cfg, path) / set_param(cfg, path, value)
apply_deltas(cfg, deltas) -> AstroFrameConfig # copie clamppée (jamais la config d'origine)
deltas_dict(cfg, paths) -> dict[str, float]
default_punish_deltas() / default_reward_deltas() -> dict[str, float]
```

- 34 paramètres enregistrés ; les **17 du groupe détection + amélioration**
  sont les cibles de l'auto-réglage : `gaussian_kernel_size`, `gaussian_sigma`,
  `occluded_ratio`, `occluded_ring`, `param1`, `param2`, `dp`, `clip_limit`,
  `tile_grid_size`, `denoise.h`, `template_window_size`,
  `search_window_size`, `sigma`, `amount`, `sharpness_percentile`,
  `gaussian_kernel_size` (lucky, 3–15) et `gaussian_sigma` (lucky, 0,5–5).
- Tout valeur apprise passe par `clamp_value` — l'apprentissage ne sort
  jamais des gammes sûres, les entiers sont arrondis et les kernels forcés
  impairs.

### `ai.tuner` (auto-réglage)

```python
@dataclass
class TuneReport:
    objective: float          # 0–1 : détection·poids + étoiles/5 (ou étoiles seules sans ground truth)
    stars: float              # 0–5, moyenne des échantillons notés
    detection: float | None   # rappel·0,4 + précision·0,3 + IoU·0,3 ; None sans ground truth
    recall: float; precision: float; mean_iou: float
    elapsed_s: float; n_items: int; n_scored: int
    to_dict() -> dict

@dataclass
class TuneResult:
    config: AstroFrameConfig  # meilleure configuration trouvée
    deltas: dict[str, float]  # deltas par rapport à la base
    base: AstroFrameConfig    # configuration de départ
    report: TuneReport        # évaluation de la meilleure configuration
    evaluations: int
    lines: list[str]          # rapport lisible (paramètre · base → ajusté · delta · pas)
    to_dict() -> dict

class ProxyEval(samples_dir, work_scale=0.5, frames_per_sample=3,
                detection_weight=0.6, seed=42):
    .evaluate(config) -> TuneReport    # avec cache par hash des paramètres effectifs
    .clear_cache() -> None

class BoundedHillClimb(specs, budget_s=60.0, seed=42, anneal=True,
                       patience=3, improve_eps=1e-4):
    .optimize(evaluate, base, start_deltas=None) -> TuneResult

run_autotune(samples_dir, config=None, budget_s=60.0, seed=42, anneal=True,
             params_filter=None, export_path=None, profile="tuning", db=None,
             work_scale=0.5, frames_per_sample=3, detection_weight=0.6) -> TuneResult
export_trained_config(config, deltas, report, path) -> Path   # JSON versionné
tuning_table_lines(result) -> list[str]
```

- **Proxy** : chaque échantillon est réduit à ~480 p (échelle de travail
  maximale 0,5, jamais agrandi) ; la détection réelle (`find_all_disks`, Hough)
  est comparée au ground truth de `calibration.json` — **IoU moyen** entre les
  disques détectés et attendus, avec **pénalités pour disques en
  trop/manquants** (via précision/rappel) — et jusqu'à `frames_per_sample`
  frames sont notées en étoiles. Résultats mis en cache par hash des
  paramètres effectifs : seul ce qui change est réévalué.
- **Hill-climbing borné et déterministe** (seed) : par passe et par paramètre
  (ordre du registre), essai de +pas et −pas ; le **momentum** double le pas
  après deux acceptations consécutives dans la même direction, l'échec réduit
  le pas de moitié (plancher pas/8) ; `anneal` accepte des pires solutions
  avec probabilité exp(−Δ/T), T décroissant de 10 % par passe ; **patience**
  (3 passes sans progrès) et **budget de temps**. Les paramètres coûteux
  (débruitage) ne sont essayés qu'une passe sur deux. Jamais hors des gammes
  sûres du registre.
- `run_autotune` orchestre tout : pré-initialisation par les **prédictions
  LSTM** du profil (`_lstm_seed`, si elles améliorent l'objectif du proxy),
  optimisation, export facultatif et **enregistrement dans la base** (table
  `tuning`, profil `tuning` par défaut) — appliqué automatiquement aux
  exécutions suivantes via `apply_learned`.
- `export_trained_config` écrit un JSON versionné (kind
  `astroframe-tuned`, params effectifs, deltas, section `stabilizer`
  compatible avec l'export historique, rapport du proxy) — défaut CLI :
  `Logs/train/trained_config.json`.

### `ai.lstm` (LSTM NumPy pur)

```python
torch_available() -> bool        # PyTorch installé ? (accélération facultative)

class LSTMCell(n_in, n_hidden, rng=None):
    .forward(x_seq, h0=None) -> (h, cache)       # (T, n_in) → sorties (T, n_hidden)
    .forward_full(x_seq) -> np.ndarray
    .backward(x_seq, cache, dh_next=None) -> dict
    .save(path) -> Path
    .load(path) -> LSTMCell | None

run_features(run) -> np.ndarray                  # 9 dimensions (étoiles, 5 métriques, deltas)
target_deltas(run) -> np.ndarray
make_sequences(history, seq_len=8) -> (X, y)     # fenêtres glissantes

@dataclass
class FitHistory: epochs, final_loss, best_loss, best_epoch

class LSTMTuner(n_hidden=24, seed=42):
    .fit(history, epochs=200, lr=0.05, seq_len=8, val_fraction=0.2, patience=6) -> FitHistory
    .predict_next_delta(history, seq_len=8) -> dict[str, float]   # {} sans historique suffisant
    .save(path=None) -> Path        # Logs/weights/lstm.npz par défaut
    .load(path=None) -> LSTMTuner | None

class TrajectoryPredictor(maxlen=8, use_lstm=False, model_path=None):
    .push(cx, cy) / .clear() / __len__
    .predict() -> tuple[float, float] | None     # None sans historique suffisant

train_trajectory_model(trajectories, path=None, seed=42, epochs=60) -> Path
trajectory_model_path(path=None) -> Path
```

- Une seule **cellule LSTM 1 couche** (portes i/f/o/g concaténées, forward +
  backward avec backprop-through-time) implémentée à la main en **NumPy pur**
  — aucune dépendance ; `torch_available()` signale l'accélération optionnelle
  (`ai.backend="torch"`).
- `LSTMTuner` s'entraîne sur l'historique de feedback de la base (une
  exécution = un pas de temps : notes par étoiles, 5 métriques, deltas) et
  **prédit le vecteur de deltas** du prochain réglage (les 5 paramètres
  visuels de `FEEDBACK_PARAMS`) ; l'auto-réglage l'utilise comme point de
  départ. Sans données suffisantes ou sans convergence → `{}` (dégradation
  silencieuse).
- `TrajectoryPredictor` prédit la position du disque au frame suivant pour
  l'anti-tremblement temporel : **régression linéaire** (moindres carrés) sur
  l'historique récent des positions, avec **raffinement LSTM facultatif**
  (cellule 2→8, `use_lstm=True` + `ai.lstm_trajectory`) entraînée hors ligne
  sur des trajectoires synthétiques par `train_trajectory_model` (déterministe,
  seed fixe).
- Modèles `.npz` **versionnés** dans `Logs/weights/lstm.npz` : fichier
  corrompu ou mauvaise version → `None` (repli silencieux).

### `ai.cnn` (CNN NumPy pur)

```python
conv2d_forward(x, w, b, pad=1) -> np.ndarray            # (N, C, H, W) → (N, K, H, W)
conv2d_backward(grad_out, x, w) -> (dw, db, dx)         # gradients vérifiés par différences finies

@dataclass
class FitReport: epochs, final_loss, best_loss, best_epoch

class SmallCNN(mode="residual" | "classify", k=8, seed=42, n_in=1):
    .forward(x) / .predict_class(x) -> np.ndarray
    .backward_residual(...) / .backward_classify(...)
    .save(path) / .load(path) -> SmallCNN | None

fit_residual(pairs, model=None, epochs=40, lr=0.05, batch_size=8,
             val_fraction=0.2, seed=42) -> (SmallCNN, FitReport)
fit_classifier(positives, negatives, model=None, epochs=60, lr=0.05,
               batch_size=8, val_fraction=0.2, seed=42) -> (SmallCNN, FitReport)

class ResidualEnhancer(model=None, model_path=None):
    .available -> bool
    .apply(image_bgr) -> np.ndarray      # BGR → BGR ; image intacte sans modèle

class DiskFilter(model=None, model_path=None):
    .available -> bool
    .confidence(image, cx, cy, radius) -> float    # P(disque) ; 0,5 sans modèle
    .filter_disks(disks, image, threshold) -> list # ne vide jamais la liste
```

- Petit réseau convolutif en **NumPy pur** (im2col vectorisé, aucune
  dépendance) : corps partagé conv1 (1→8, 3×3, ReLU) → conv2 (8→8, 3×3, ReLU)
  + tête **résiduelle** (conv3 8→1) ou **classifieur** (moyenne globale →
  dense 8→2 softmax). Entraînement **hors ligne et déterministe** (seed) avec
  early-stop ; gradients vérifiés par **différences finies** (tests).
- `fit_residual` apprend le **résidu** `r = y − x` entre l'entrée et la cible
  (paires bruitées → propres) ; `ResidualEnhancer.apply` l'applique à l'étape
  **post-unsharp** de `enhance_image` (`ai.cnn_enhance=true`) : canal **L du
  LAB**, tuiles **64×64 avec chevauchement** (8 px), couleurs préservées.
  Sans modèle → image intacte (jamais pire).
- `fit_classifier` apprend disque vs bruit à partir des positifs (ground
  truth de `calibration.json`) et des négatifs (faux positifs) ;
  `DiskFilter` note chaque candidat de `find_all_disks`
  (`ai.disk_filter > 0.0` filtre ceux sous le seuil) et **ne vide jamais** la
  liste détectée — la détection ne régresse jamais.
- Modèles : `Logs/weights/enhancer_cnn.npz` et
  `Logs/weights/disk_filter.npz` (`.npz` versionnés ; corrompus ou mauvaise
  version → repli silencieux).