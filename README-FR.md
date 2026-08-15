# AstroFrame

Stabilisation géométrique et amélioration automatique de photos et vidéos d'éclipses solaires et lunaires.

![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue) ![License MIT](https://img.shields.io/badge/license-MIT-green)

## Fonctionnalités

- **Stabilisation par géométrie** — détecte le disque du Soleil/de la Lune (`cv2.HoughCircles` + repli par contours + raffinement par centroïde d'intensité) et réaligne chaque frame pour garder l'éclipse toujours au centre exact, sans bords noirs.
- **Amélioration automatique** — CLAHE dans l'espace LAB (sans brûler la luminosité), débruitage Non-Local Means (utile pour l'ISO élevé) et masquage de netteté (unsharp) pour mettre en évidence le limbe de la Lune.
- **Lucky imaging** — rejet des frames floues par variance du Laplacien, avec un seuil estimé statistiquement à partir de la vidéo elle-même.
- **Stacking** — combinaison (médiane ou moyenne) des N meilleures frames, alignées par centrage, pour réduire le bruit.
- **Anti-tremblement temporel** — lissage du centroïde (EMA) et réutilisation du dernier déplacement valide quand une frame n'a pas de détection.
- **Détection de disques multiples** — en plus du disque principal, les **reflets** sont détectés (Hough + contours) ; le polissage supprime les reflets et la vidéo en direct les montre en rouge.
- **Polissage et évaluation automatique** — `polish_image()` ajoute de la luminosité au disque en gardant la couronne ; `score_image()` attribue des **étoiles (0–5)** au résultat (bruit, contraste, taille et couleur de la couronne).
- **Calibration par exemples** — interface desktop native (`python calibrate.py`) qui charge les photos et vidéos de `samples/`, permet de **dessiner des cercles/ellipses à la main** (clic crée, glisser déplace, poignées redimensionnent) lors d'une 1re passe, activer la **détection automatique** à la 2e pour remplir/valider les échantillons restants, et compare tout au ground truth sur tous les échantillons (rappel, précision, IoU, erreurs + suggestions de paramètres).
- **Validation et entraînement de la détection** — `validator.py` (fenêtre desktop native) parcourt les échantillons, montre la détection avec zoom/pan, et apprend en **récompensant et punissant 7 paramètres du détecteur** forme par forme contre le guide manuel ; l'**entraînement automatique** (`--auto`) re-détecte en séries jusqu'à 100 % et exporte les **poids entraînés** pour le système réel.
- **Apprentissage par feedback** — chaque exécution est stockée en SQLite ; en plus de l'évaluation automatique, vous pouvez évaluer manuellement (0–5 étoiles) et AstroFrame **ajuste les sliders automatiquement** à la prochaine exécution avec le même profil de caméra, en montrant l'historique/journal dans l'interface elle-même.
- **Interface Gradio** — deux onglets : **Image** (Avant/Après, sliders, zoom sur la couronne/le limbe) et **Vidéo** (traitement en direct avec les disques détectés, aperçu final à frames espacées et exportation facultative). Au chargement d'une vidéo, les **métadonnées** sont lues (ffprobe/OpenCV/EXIF) et les **paramètres sont suggérés automatiquement** (ISO → débruitage, résolution → rayons du détecteur, bitrate → compression), tout en restant modifiables.
- **CLI** — lot de photos, vidéos (stabiliser/améliorer/stack), journaux et barre de progression.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate        # Windows : .venv\Scripts\activate
pip install -e ".[dev]"
```

Alternative simple : `pip install -r requirements.txt`. Nécessite Python 3.10+.

> [!IMPORTANT]
> Sur Debian/Ubuntu récents, `pip install` sur le Python système échoue avec
> `error: externally-managed-environment` (PEP 668). Utilisez toujours la
> virtualenv ci-dessus (`source .venv/bin/activate` avant d'installer), ou
> forcez avec `--break-system-packages` à vos risques et périls.

> [!TIP]
> Pour des métadonnées vidéo riches (codec, bitrate, durée), installez le
> `ffmpeg` du système — sans lui, AstroFrame utilise seulement OpenCV
> (résolution/fps/frames).

## Utilisation rapide

```bash
python main.py                             # interface web (Gradio) — frontend + backend ensemble
python calibrate.py                        # interface de calibration (samples/)
python validator.py                        # validation/entraînement de la détection (samples/)
astroframe serve                          # équivalent via la CLI installée
astroframe process --input photo1.jpg photo2.jpg --output-dir outputs/
astroframe video --input eclipse.mp4 --mode enhance
astroframe video --input eclipse.mp4 --mode stack --stack-n 20
astroframe video --input eclipse.mp4 --mode enhance --fast   # sans débruitage (plus rapide)
astroframe config-template                # génère un config.yaml modifiable
```

`main.py` est le point d'entrée unique : il démarre le serveur Gradio qui sert
le frontend dans le navigateur et traite les images dans le backend (le moteur
dans `core/` tourne dans le même processus, à chaque clic sur **Traiter**).
Options : `--config`, `--host`, `--port`, `--share` et `--no-browser`.

`validator.py` est la **validation/entraînement de la détection** (fenêtre
desktop native ; `--check` pour un rapport sans interface, `--auto` pour
l'entraînement automatique) : il compare la détection au guide manuel de
`calibration.json`, **récompense/punit les paramètres** forme par forme et se
termine par un rapport + poids entraînés exportables vers le système réel.

`calibrate.py` est l'**interface de calibration** : elle charge des images et
des frames de vidéo depuis `samples/`, permet d'ajuster les cercles à la main
(glisser = déplacer, pinceau = ajouter, gomme = supprimer) et **Valider tous**
compare la détection automatique avec le ground truth sur tous les échantillons.

## Documentation

- [docs/FR/API.md](docs/FR/API.md) — référence des modules `core/`, `video/`, `meta/`, `ai/`, `calibration/` et `config.py` (FR).
- [docs/FR/Usage.md](docs/FR/Usage.md) — guide pratique : CLI, configuration YAML champ par champ, interface, calibration et workflow vidéo (FR).
- [docs/FR/Architecture.md](docs/FR/Architecture.md) — spécification originale de la solution (référence, FR).
- [docs/FR/CHANGELOG.md](docs/FR/CHANGELOG.md) — changelog (FR).
- [docs/PT/](docs/PT/) — a mesma documentação em português (API, Arquitetura, USO).
- [docs/EN/](docs/EN/) — the same documentation in English (API, Architecture, Usage, CHANGELOG).
- [README.md](README.md) / [README-EN.md](README-EN.md) — ce README en portugais et en anglais.

## Limitations connues

- La vidéo exportée **ne contient pas d'audio** (`cv2.VideoWriter`) ; pour
  préserver le son, fusionnez la piste originale avec ffmpeg :
  `ffmpeg -i original.mp4 -i traitee.mp4 -c copy -map 0:a -map 1:v sortie.mp4`
- L'interpolation RIFE est facultative et exige PyTorch
  (`pip install -e ".[rife]"`) ; l'interface du modèle varie entre les
  versions des dépôts RIFE.
- Le débruitage est l'étape la plus lente (~1 s/frame en 480p) ; utilisez
  `--fast` sur les grandes vidéos.

## Développement

```bash
pytest                      # ~260 tests (pixel tests avec images synthétiques)
pytest --cov=astroframe     # couverture (100 % du paquet)
ruff check .                # lint
ruff format .               # formatage
```

CI (GitHub Actions) : pytest sur Python 3.10/3.12 + ruff, dans
`.github/workflows/ci.yml`.

## Structure

```
src/astroframe/
├── core/         stabilisateur géométrique, amélioration automatique et pipeline
├── video/        lecture des frames, lucky imaging et stacking
├── meta/         lecture des métadonnées (ffprobe/OpenCV/EXIF) et suggestions de paramètres
├── calibration/  scan des exemples, ground truth et validation de la détection
├── ui/           interface Gradio (Image/Vidéo + Calibration) et CLI
└── ai/           interpolation RIFE facultative (exige PyTorch)
```

## Licence

MIT — voir [LICENSE](LICENSE).