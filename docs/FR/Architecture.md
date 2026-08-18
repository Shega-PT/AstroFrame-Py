# Architecture de la solution

Le système se divise en trois étapes principales :

- Stabilisation par géométrie,
- Amélioration automatique de l'image,
- Interface minimale

---

# 1 Suivi et stabilisation vidéo (annulation du tremblement)

Au lieu de tenter de stabiliser le fond (sombre ou uniforme), l'algorithme
détecte le centroïde du Soleil/de la Lune dans chaque frame et déplace l'image
pour garder l'astre toujours au centre exact de la frame.

Détection de forme : utilise la transformée de Hough pour les cercles
(`cv2.HoughCircles`) ou la détection des plus grands contours
(`cv2.findContours`).

Recadrage automatique : si la caméra fait un saut rapide, l'algorithme calcule
le vecteur de déplacement du centre $(x, y)$ du Soleil par rapport au centre de
la frame et réaligne l'image.

Rejet des frames floues (Lucky Imaging) : les frames capturées pendant des
mouvements très rapides du caméscope sont floues à cause du motion blur. Python
peut calculer la variance du Laplacien de l'image (niveau de netteté) et
ignorer les frames les plus floues.

---

# 2 Traitement et amélioration automatiques (photos et vidéos)

Gamma et contraste adaptatifs (CLAHE) : l'algorithme Contrast Limited Adaptive
Histogram Equalization améliore les détails de la couronne solaire ou de la
transition d'éclairage sans brûler la luminosité.

Réduction du bruit : application du filtre Non-Local Means Denoising
(particulièrement utile pour les photos à ISO élevé).

Masquage de netteté (Unsharp Masking) : met en évidence le limbe de l'astre
(les bords exacts du disque).

---

# 3 Frontend minimal (Gradio ou Streamlit)

La façon la plus rapide de créer l'interface graphique en Python est d'utiliser
Gradio. Il tourne localement dans le navigateur avec des sélecteurs de
fichiers, des sliders et une visualisation côte à côte (Avant vs. Après).

Exemple de code de la pipeline (Python) : voici une implémentation fonctionnelle
de base utilisant OpenCV pour la logique de traitement et Gradio pour
l'interface.

```Python
import cv2
import numpy as np
import gradio as gr


def auto_enhance_frame(img):
    """
    Applique l'égalisation adaptative et la netteté axées sur l'astrophotographie.
    """
    # Convertir en niveaux de gris pour l'analyse des bords
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1. Filtre CLAHE pour rehausser le contraste de la couronne/du bord sans brûler
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

    # Appliquer sur le canal L de l'espace de couleur LAB pour préserver les couleurs originales
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l_enhanced = clahe.apply(l)
    updated_lab = cv2.merge((l_enhanced, a, b))
    enhanced_bgr = cv2.cvtColor(updated_lab, cv2.COLOR_LAB2BGR)

    # 2. Réduction de bruit douce
    denoised = cv2.fastNlMeansDenoisingColored(enhanced_bgr, None, 5, 5, 7, 21)

    # 3. Unsharp masking pour mettre en évidence le limbe de la Lune
    gaussian_3 = cv2.GaussianBlur(denoised, (0, 0), 2.0)
    unsharp = cv2.addWeighted(denoised, 1.5, gaussian_3, -0.5, 0)

    return unsharp


def center_and_stabilize(img):
    """
    Localise le Soleil/la Lune par géométrie et centre l'image.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    # Recherche de formes circulaires
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=100, param1=50, param2=30, minRadius=30, maxRadius=400
    )

    if circles is not None:
        circles = np.uint16(np.around(circles))
        # Prendre le plus grand cercle trouvé (Soleil)
        x, y, r = circles[0][0]

        # Calculer le déplacement vers le centre de l'image
        h, w = img.shape[:2]
        center_x, center_y = w // 2, h // 2
        dx = center_x - x
        dy = center_y - y

        # Matrice de translation
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        stabilized = cv2.warpAffine(img, M, (w, h))
        return stabilized

    return img


def process_image_pipeline(input_image):
    if input_image is None:
        return None
    # 1. Centrer sur le disque solaire
    stabilized = center_and_stabilize(input_image)
    # 2. Appliquer les améliorations automatiques
    result = auto_enhance_frame(stabilized)
    return result


# Interface minimale avec Gradio
with gr.Blocks(title="AstroFrame") as demo:
    gr.Markdown("# AstroFrame — stabilisation géométrique et amélioration automatique d'astrophotographies et d'astrovidéos")
    gr.Markdown("Stabilisation géométrique et amélioration automatique de photos et vidéos d'astres (Soleil, Lune, planètes).")

    with gr.Row():
        input_img = gr.Image(label="Photo/Frame originale")
        output_img = gr.Image(label="Résultat stabilisé et traité")

    btn = gr.Button("Traiter l'image", variant="primary")
    btn.click(fn=process_image_pipeline, inputs=input_img, outputs=output_img)

if __name__ == "__main__":
    demo.launch()
```

---

# 4 Architecture IA (version 0.7.0)

Depuis la version 0.7.0, la pipeline est augmentée par une couche d'IA
légère — auto-réglage, LSTM et CNN — qui **ne remplace aucun composant
existant** : elle ajuste des paramètres, filtre des détections et prédit des
trajectoires, toujours dans les gammes sûres du registre. Le principe
directeur est la **sécurité** : toute l'IA est désactivée par défaut, et un
modèle manquant ou corrompu dégrade silencieusement sans jamais bloquer le
pipeline.

## 4.1 Registre unifié des paramètres (`ai.params`)

Source unique de vérité des **gammes sûres**, pas d'optimisation et deltas
d'apprentissage de toute la pipeline : 34 paramètres enregistrés (détection,
géométrie, amélioration, stacking, polissage, évaluation, méta), chacun avec
ses bornes, son pas, son type (entier/flottant), la **parité impaire** des
kernels gaussiens, son coût d'évaluation (le débruitage est « coûteux ») et
ses pénalités/récompenses du validator. Les 17 paramètres des groupes
détection + amélioration sont les cibles de l'auto-réglage. **Tout valeur
apprise passe par `clamp_value`** — l'apprentissage ne sort jamais des
gammes sûres, les entiers sont arrondis et les kernels forcés impairs. Le
validator, le feedback et le tuner lisent tous le même registre : ils ne
peuvent plus diverger.

## 4.2 Auto-réglage (`ai.tuner`)

Deux briques, orchestrées par `run_autotune` :

- **`ProxyEval`** — évaluation rapide d'une configuration sur les
  échantillons de `samples/`. Chaque image/frame est réduite à ~480 p
  (échelle de travail maximale 0,5, jamais agrandie) ; la détection réelle
  (`find_all_disks`, Hough) est comparée au ground truth de
  `calibration.json` : **IoU moyen** entre disques détectés et attendus,
  avec pénalités pour les disques en trop/manquants (précision/rappel), et
  jusqu'à 3 frames notées en étoiles. L'objectif pondère détection (0,6) et
  étoiles (0,4). Les résultats sont **mis en cache par hash des paramètres
  effectifs** : seul ce qui change est réévalué.
- **`BoundedHillClimb`** — montée de colline **déterministe** (graine fixe)
  et **bornée** : par passe et par paramètre, essai de +pas et −pas ; le
  **momentum** double le pas après deux acceptations consécutives dans la même
  direction, l'échec réduit le pas de moitié (plancher pas/8) ; le **recuit
  facultatif** accepte des pires solutions avec probabilité exp(−Δ/T), la
  température décroissant de 10 % par passe — échapper aux minima locaux sans
  sortir des gammes sûres. La patience (3 passes sans progrès) et le **budget
  de temps** (`budget_s`) bornent la recherche ; les paramètres coûteux
  (débruitage) ne sont essayés qu'une passe sur deux.

`run_autotune` pré-initialise la recherche avec les **prédictions LSTM** du
profil quand elles améliorent l'objectif du proxy (`_lstm_seed`), enregistre
le résultat dans la base d'apprentissage (table `tuning` de la `FeedbackDB`,
par profil) et peut exporter la configuration optimisée (JSON versionné).
Le résultat est ensuite **appliqué automatiquement** aux exécutions suivantes
du même profil par `apply_learned` — c'est la « mémoire » de l'IA entre
exécutions. Points d'entrée : CLI `astroframe autotune` et onglet
**Auto-tune** de l'interface Gradio.

## 4.3 LSTM (`ai.lstm`, NumPy pur)

Une seule **cellule LSTM 1 couche** (portes i/f/o/g, forward + backward avec
backprop-through-time) implémentée à la main en NumPy — sans dépendance ;
PyTorch n'est qu'une accélération facultative (`torch_available()`). Deux
usages :

- **`LSTMTuner`** — s'entraîne sur l'historique de feedback (une exécution =
  un pas de temps : notes par étoiles, 5 métriques visuelles, deltas) et
  **prédit le vecteur de deltas** de la prochaine exécution. C'est le point
  de départ de l'auto-réglage (`_lstm_seed`) : convergence plus rapide quand
  le modèle a appris.
- **`TrajectoryPredictor`** — prédit la position du disque au frame suivant
  pour l'**anti-tremblement temporel** : extrapolation **linéaire** (moindres
  carrés) sur l'historique récent des centroïdes, avec **raffinement LSTM
  facultatif** (cellule 2→8) entraînée hors ligne sur des **trajectoires
  synthétiques** (`train_trajectory_model`, déterministe). Activé par
  `ai.lstm_trajectory`.

Les modèles sont des `.npz` **versionnés** (`Logs/weights/lstm.npz`) ; un
fichier corrompu ou de mauvaise version retombe **silencieusement** sur la
régression linéaire — le comportement d'origine.

## 4.4 CNN (`ai.cnn`, NumPy pur)

Petit réseau convolutif (conv 2D 3×3, ReLU, pooling moyen global, tête MLP),
entraîné **hors ligne et de façon déterministe** (graine fixe, early-stop) ;
les gradients sont vérifiés par **différences finies** dans les tests. Deux
têtes sur un corps partagé :

- **Résiduelle (`fit_residual` / `ResidualEnhancer`)** — apprend le résidu
  `r = y − x` entre l'entrée et la cible propre (paires bruitées → propres) ;
  il supprime bruit/smearing et s'applique en **étape post-unsharp** de
  `enhance_image` (`ai.cnn_enhance=true`) : canal **L du LAB**, tuiles
  **64×64 avec chevauchement**, couleurs préservées. Sans modèle → image
  intacte.
- **Classificatrice (`fit_classifier` / `DiskFilter`)** — apprend à
  distinguer un **disque réel du bruit** (positifs : ground truth de
  `calibration.json` ; négatifs : faux positifs) et note chaque candidat de
  `find_all_disks` (`confidence` = P(disque)). `ai.disk_filter > 0.0`
  filtre les candidats sous le seuil — et **ne vide jamais** la liste
  détectée.

Modèles : `Logs/weights/enhancer_cnn.npz` et `Logs/weights/disk_filter.npz`
(versionnés ; corrompus → repli silencieux).

### Entraînement des réseaux

- **CNN de détection** — l'entraînement automatique du `validator.py`
  collecte des patches de disques à chaque série (`cnn_positives` des
  cercles du guide, `cnn_negatives` des formes rejetées + recadrages
  aléatoires déterministes exclus par IoU) et ré-entraîne `DiskFilter` entre
  les séries (`--cnn`). Le résultat est comparé au **champion** de la table
  `models` par `score` ; s'il est strictement meilleur il est promu vers le
  chemin canonique et la série suivante fait un warm-start des poids du
  champion.
- **CNN résiduelle** — `enhancer_trainer.py` (GUI Valide/Rejeté ou `--auto`)
  accumule des paires (entrée, sortie CNN) / (entrée, entrée) et appelle
  `train_enhancer_round`, qui compare par `mean_delta` (1 − MSE) au champion
  et promeut si meilleure.
- Les deux entraîneurs journalisent chaque tour dans les tables `models` et
  `logs`.

## 4.5 Boucle de feedback et intégration dans la pipeline

La base SQLite (`Logs/logs/system/feedback.db`, surchargeable par
`ASTROFRAME_FEEDBACK_DB`) contient deux tables : `runs` (exécutions,
métriques, évaluations en étoiles) et `tuning` (auto-réglages par profil).
Depuis la v0.8.0 elle contient aussi `models` (artefacts NN : type, métrique,
drapeau promu, statut champion, série) et `logs` (historique d'événements par
composant).
Au démarrage de chaque exécution, `apply_learned` **additionne les nudges
par étoiles et les deltas d'auto-réglage** du profil — toujours clamppés par
le registre — ou retourne la configuration inchangée s'il n'y a rien
d'appris.

L'IA s'insère à trois points du pipeline existant :

1. **Détection** — `DiskFilter` filtre les candidats de `find_all_disks`
   (faux positifs) avant la stabilisation ;
2. **Stabilisation** — `TrajectoryPredictor` fournit la position prédite du
   disque aux frames sans détection (anti-tremblement) ;
3. **Amélioration** — `ResidualEnhancer` ajoute son étape après l'unsharp ;
   l'auto-réglage optimise l'ensemble des paramètres de ces étapes.

Le flux reste identique sans IA : `find_all_disks` → `AntiJitterStabilizer`
→ CLAHE → débruitage → unsharp → polissage → évaluation.

---

# Modules pour étendre la pipeline vidéo

Pour étendre cette pipeline aux vidéos du caméscope numérique Samsung avec
mouvements rapides :

1. Stacking / Lucky Imaging (bibliothèques recommandées) :

   - Utiliser scikit-image ou OpenCV pour lire la vidéo frame par frame.
   - Calculer `cv2.Laplacian(frame, cv2.CV_64F).var()`. Si la valeur est
     inférieure à un seuil prédéfini, rejeter la frame (elle a été capturée
     pendant un mouvement rapide de la caméra).

2. Interpolation de mouvement en cas de "sauts" :

   - Si la caméra se déplace trop vite et perd quelques bonnes frames, on peut
     utiliser le modèle RIFE (Real-Time Intermediate Flow Estimation) via
     PyTorch pour interpoler les frames en douceur entre les corrections
     manuelles d'élévation et de direction.
