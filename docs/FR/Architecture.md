# Architecture de la solution

Le système se divise en trois étapes principales :

- Stabilisation par géométrie,
- Amélioration automatique de l'image,
- Interface minimale

---

# 1 Suivi et stabilisation vidéo (annulation du tremblement)

Au lieu de tenter de stabiliser le fond (sombre ou uniforme), l'algorithme
détecte le centroïde du Soleil/de la Lune dans chaque frame et déplace l'image
pour garder l'éclipse toujours au centre exact de la frame.

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

Masquage de netteté (Unsharp Masking) : met en évidence les bords exacts de la
Lune sur le disque solaire.

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


def auto_enhance_eclipse_frame(img):
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
    result = auto_enhance_eclipse_frame(stabilized)
    return result


# Interface minimale avec Gradio
with gr.Blocks(title="Eclipse Auto-Enhancer") as demo:
    gr.Markdown("# 🌒 Eclipse Auto-Enhancer AI System")
    gr.Markdown("Amélioration automatique et stabilisation géométrique pour photos et frames d'éclipse.")

    with gr.Row():
        input_img = gr.Image(label="Photo/Frame originale")
        output_img = gr.Image(label="Résultat stabilisé et traité")

    btn = gr.Button("Traiter l'image", variant="primary")
    btn.click(fn=process_image_pipeline, inputs=input_img, outputs=output_img)

if __name__ == "__main__":
    demo.launch()
```

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
