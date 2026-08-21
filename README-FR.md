# AstroFrame

Stabilisation géométrique et amélioration automatique d'astrophotographies et d'astrovidéos — photos et vidéos du Soleil, de la Lune, de planètes, de comètes et d'autres astres.

![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue) ![License MIT](https://img.shields.io/badge/license-MIT-green)

## Fonctionnalités

- **Stabilisation par géométrie** — détecte le disque de l'astre (`cv2.HoughCircles` + repli par contours + raffinement par centroïde d'intensité) et réaligne chaque frame pour garder l'astre toujours au centre exact, sans bords noirs.
- **Amélioration automatique** — CLAHE dans l'espace LAB (sans brûler la luminosité), débruitage Non-Local Means (utile pour l'ISO élevé) et masquage de netteté (unsharp) pour mettre en évidence le limbe de l'astre.
- **Lucky imaging** — rejet des frames floues par variance du Laplacien, avec un seuil estimé statistiquement à partir de la vidéo elle-même.
- **Stacking** — combinaison (médiane ou moyenne) des N meilleures frames, alignées par centrage, pour réduire le bruit.
- **Anti-tremblement temporel** — lissage du centroïde (EMA) et réutilisation du dernier déplacement valide quand une frame n'a pas de détection.
- **Détection de disques multiples** — en plus du disque principal, les **disques secondaires** (corps passant devant l'astre principal, p. ex. la Lune devant le Soleil) et les **reflets** de l'objectif sont détectés (Hough + contours) ; le polissage supprime les reflets et la vidéo en direct les montre en rouge.
- **Polissage et évaluation automatique** — `polish_image()` ajoute de la luminosité au disque en gardant la couronne/le limbe ; `score_image()` attribue des **étoiles (0–5)** au résultat (bruit, contraste, taille et couleur de la couronne).
- **Calibration par exemples** — interface desktop native (`python calibrate.py`) qui charge les photos et vidéos de `samples/`, permet de **dessiner des cercles/ellipses à la main** (clic crée, glisser déplace, poignées redimensionnent) lors d'une 1re passe, activer la **détection automatique** à la 2e pour remplir/valider les échantillons restants, et compare tout au ground truth sur tous les échantillons (rappel, précision, IoU, erreurs + suggestions de paramètres).
- **Validation et entraînement de la détection** — `validator.py` (fenêtre desktop native) parcourt les échantillons, montre la détection avec zoom/pan, et apprend en **récompensant et punissant les paramètres du détecteur** forme par forme contre le guide manuel ; chaque échantillon reçoit des **étoiles automatiques (0–5)** et un curseur d'**évaluation manuelle**.
- **Apprentissage par feedback** — chaque exécution est stockée en SQLite ; en plus de l'évaluation automatique, vous pouvez évaluer manuellement (0–5 étoiles) et AstroFrame **ajuste les sliders automatiquement** à la prochaine exécution avec le même profil de caméra, en montrant l'historique/journal dans l'interface elle-même.
- **Auto-réglage** — `astroframe autotune` **optimise tous les paramètres** de détection et d'amélioration contre les échantillons de `samples/` (évaluation par proxy : IoU des disques détectés vs. le ground truth de `calibration.json`, avec pénalités pour les disques en trop/manquants, + étoiles). Montée de colline **déterministe et bornée** (momentum, pas adaptatifs, recuit facultatif, budget de temps) qui ne sort jamais des gammes sûres du registre ; le résultat est enregistré par profil dans la base d'apprentissage et **appliqué automatiquement** aux exécutions suivantes.
- **IA légère LSTM/CNN (NumPy pur)** — une cellule LSTM implémentée à la main prédit le vecteur d'ajustement suivant (auto-réglage) et la trajectoire du disque (anti-tremblement) ; une petite CNN apprend un rehaussement résiduel (suppression du bruit/smearing) et un classifieur disque/bruit qui filtre les faux positifs de la détection. Modèles `.npz` versionnés dans `Logs/weights/`.
- **Contrôleur toujours actif** — thread daemon qui applique périodiquement les deltas appris à la configuration active en utilisant `LSTMTuner` (lorsque entraîné) ou `FallbackNet` (basé sur des règles).
- **Interface Gradio** — trois onglets : **Image** (Avant/Après, sliders groupés par fonction avec texte informatif, zoom sur la couronne/le limbe), **Vidéo** (traitement en direct avec les disques détectés, aperçu final à frames espacées et exportation facultative) et **Auto-tune** (réglage automatique des paramètres contre les échantillons, avec rapport et configuration résultante). Au chargement d'une vidéo, les **métadonnées** sont lues (ffprobe/OpenCV/EXIF) et les **paramètres sont suggérés automatiquement** (ISO → débruitage, résolution → rayons du détecteur, bitrate → compression), tout en restant modifiables.
- **CLI** — lot de photos, vidéos (stabiliser/améliorer/stack), auto-réglage (`autotune`), journaux et barre de progression.

**IA intégrée (v0.7)** — l'apprentissage par feedback est complété par un
**auto-réglage automatique** et de petites réseaux **LSTM/CNN** en NumPy pur
(le noyau reste NumPy ; PyTorch, obligatoire depuis la v0.9.0, alimente
uniquement l'interpolation RIFE). Toute l'IA est
**désactivée par défaut** (sections `[tuning]` et `[ai]` du config) : un modèle
manquant ou corrompu **dégrade silencieusement** et ne bloque jamais le
pipeline.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate        # Windows : .venv\Scripts\activate
pip install -e ".[dev]"
```

Alternative simple : `pip install -r requirements.txt`. Nécessite Python 3.10+.

> [!NOTE]
> PyTorch est obligatoire depuis la v0.9.0 (interpolation RIFE). Sous Linux,
> PyPI installe la build CUDA par défaut (~2,5 Go) ; pour la version CPU
> seule : `pip install torch --index-url https://download.pytorch.org/whl/cpu`
> avant l'installation du paquet (c'est ce qu'utilise le CI).

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
astroframe autotune --samples samples --budget 120           # auto-réglage des paramètres
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
- PyTorch est obligatoire depuis la v0.9.0 et alimente l'**interpolation RIFE**
  (`astroframe video --interp N` — lisse la vidéo avec N trames intermédiaires
  générées par IA entre chaque paire de trames) ; si le modèle ne charge pas,
  le CLI prévient et continue sans interpolation. L'interface du modèle varie
  entre les versions des dépôts RIFE.
- Le débruitage est l'étape la plus lente (~1 s/frame en 480p) ; utilisez
  `--fast` sur les grandes vidéos.

## Développement

```bash
pytest                      # 666 tests headless (fenêtres Tk/OpenCV fermées automatiquement)
pytest tests/test_e2e.py    # E2E : vrai CLI + pipeline complète + --check sans fenêtre
pytest --cov=astroframe     # couverture (100 % du paquet)
ruff check .                # lint
ruff format .               # formatage
```

CI (GitHub Actions) : pytest headless sur Python 3.10/3.12 + ruff, dans
`.github/workflows/ci.yml` — sans `xvfb-run` : sans `DISPLAY`, un `Xvfb`
virtuel est démarré automatiquement, et toute fenêtre ouverte par un test
est fermée à la fin de celui-ci (`--timeout=300` par test comme filet de
sécurité).

## Structure

```
src/astroframe/
├── core/         stabilisateur géométrique, amélioration automatique et pipeline
├── video/        lecture des frames, lucky imaging et stacking
├── meta/         lecture des métadonnées (ffprobe/OpenCV/EXIF) et suggestions de paramètres
├── calibration/  scan des exemples, ground truth et validation de la détection
├── ui/           interface Gradio (Image/Vidéo/Auto-tune + Calibration) et CLI
└── ai/           auto-réglage, LSTM/CNN (NumPy pur) et interpolation RIFE facultative (exige PyTorch)
```

## Licence

MIT — voir [LICENSE](LICENSE).