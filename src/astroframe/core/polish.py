"""Polimento por astros: cada astro é realçado individualmente e remontado.

Depois da melhoria (CLAHE + denoise + nitidez), o polimento:

- **deteta todos os astros** (`find_all_disks`) — o astro maior e os
  outros corpos, cada um com a sua deteção própria;
- **processa cada astro individualmente** — realce local (esticamento de
  contraste) e brilho extra, calculados com as estatísticas do próprio astro
  (um astro escuro e uniforme é preservado intacto);
- **recorta um pouco além do astro maior** (`corona_scale` × raio) — o anel
  entre o bordo do astro e a linha de recorte é **diluído** até ao fundo;
- **remonta sem costuras** — as máscaras individuais (com feather) são
  combinadas por média ponderada entre sobreposições: onde dois astros se
  tocam ou sobrepõem, o resultado é a média suave dos dois realces;
- **fundo = média do fundo original** (`background_fill`, em vez de preto) ou
  preto puro (`black_background`);
- **remove reflexos da lente** — círculos-ghost pequenos (raio inferior a
  `GHOST_RADIUS_RATIO` × o do astro maior) com o centro fora do astro são
  eliminados (a área é preenchida com o fundo);
- círculos demasiado pequenos (`reflection_min_radius`) são ignorados (são
  estrelas/ruído, não astros).

Sem deteção válida, a imagem é devolvida inalterada (`polish_image` nunca
falha no fluxo principal).
"""

from __future__ import annotations

import logging
import math

import cv2
import numpy as np

from astroframe.config import AstroFrameConfig
from astroframe.core.stabilizer import DiskDetection, GHOST_RADIUS_RATIO, find_all_disks

logger = logging.getLogger(__name__)


def _feather_mask(shape: tuple[int, int], cx: int, cy: int, radius: float, feather: float) -> np.ndarray:
    """Máscara circular com borda suavizada (0..1), redonda e sem dentes."""
    height, width = shape
    ys, xs = np.ogrid[:height, :width]
    inside = np.hypot(xs - cx, ys - cy) <= radius
    if feather <= 0.0:
        return inside.astype(np.float32)
    blur = max(3, int(round(feather * radius)) | 1)
    return cv2.GaussianBlur(inside.astype(np.float32), (blur, blur), 0)


def _band_mask(
    shape: tuple[int, int], cx: int, cy: int, r_in: float, r_out: float, feather: float
) -> np.ndarray:
    """Máscara do astro: 1 dentro do disco, a descer linearmente até `r_out`
    (a linha de recorte). O anel `r_in → r_out` é a zona de diluição com o
    fundo — "do astro até à linha de recorte", sem cortes duros."""
    height, width = shape
    ys, xs = np.ogrid[:height, :width]
    dist = np.hypot(xs - cx, ys - cy)
    band = np.clip((r_out - dist) / max(r_out - r_in, 1.0), 0.0, 1.0).astype(np.float32)
    if feather > 0.0:
        blur = max(3, int(round(feather * r_out)) | 1)
        band = cv2.GaussianBlur(band, (blur, blur), 0)
    return band


def _applies_to(image: np.ndarray, detection: DiskDetection, config: AstroFrameConfig) -> bool:
    """Verifica se há algo para polir (deteção válida e dentro dos limites)."""
    cfg = config.polish
    if not cfg.enabled or detection.radius <= 0:
        return False
    height, width = image.shape[:2]
    return 0 <= detection.cx < width and 0 <= detection.cy < height


def _background_color(image: np.ndarray, cx: int, cy: int, radius: float, cfg) -> np.ndarray:
    """Média BGR do fundo original (fora da linha de recorte do astro maior)."""
    height, width = image.shape[:2]
    ys, xs = np.ogrid[:height, :width]
    outside = np.hypot(xs - cx, ys - cy) > radius * cfg.corona_scale
    if not outside.any():
        return np.zeros(3, dtype=np.float32)
    return image[outside].mean(axis=0).astype(np.float32)


def _astro_boost(image: np.ndarray, gray: np.ndarray, disk: DiskDetection, cfg) -> np.ndarray:
    """Realce individual de um astro (esticamento local de contraste + brilho).

    Astros escuros e uniformes são devolvidos intactos — esticar ou levantar
    o brilho destruiria o contraste do astro.
    """
    height, width = image.shape[:2]
    ys, xs = np.ogrid[:height, :width]
    inner = np.hypot(xs - disk.cx, ys - disk.cy) <= disk.radius
    if not inner.any():
        return image.astype(np.float32)
    values = gray[inner]
    median = float(np.median(values))
    if median < 48.0:
        return image.astype(np.float32)
    low, high = np.percentile(values, 2), np.percentile(values, 98)
    span = float(high - low)
    if span < 16.0:
        return image.astype(np.float32)
    scale = 240.0 / span
    boosted = (image.astype(np.float32) - low) * scale + 8.0
    if cfg.brightness > 0.0:
        smooth = _feather_mask((height, width), disk.cx, disk.cy, disk.radius, cfg.feather)
        boosted = boosted + cfg.brightness * 255.0 * smooth[..., None]
    return np.clip(boosted, 0.0, 255.0)


def _astro_region(boost: np.ndarray, inner: np.ndarray, band_px: int, feather: float) -> np.ndarray:
    """Propaga o conteúdo do astro para a banda de diluição.

    A banda (do bordo do astro até à linha de recorte) é preenchida com o
    valor do bordo do disco (dilatação em escala reduzida, rápida) — é esse
    conteúdo que depois se dilui linearmente com o fundo, sem cortes duros.
    """
    if boost.ndim == 3:
        content = boost * inner[..., None]
    else:
        content = boost * inner
    if band_px <= 2:
        return content
    factor = min(8, max(1, band_px // 6))
    small = cv2.resize(
        content,
        (content.shape[1] // factor, content.shape[0] // factor),
        interpolation=cv2.INTER_AREA,
    )
    kernel_size = max(3, 2 * (band_px // factor) + 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    diluted = cv2.dilate(small, kernel)
    return cv2.resize(diluted, (content.shape[1], content.shape[0]), interpolation=cv2.INTER_LINEAR)


def polish_image(
    image: np.ndarray,
    detection: DiskDetection | None,
    config: AstroFrameConfig | None = None,
) -> np.ndarray:
    """Aplica o polimento por astros (realce individual + remontagem sem costuras).

    `detection` deve usar as coordenadas da própria `image` (após
    estabilização, o disco está centrado). Sem disco detetado devolve a
    imagem inalterada.
    """
    config = config or AstroFrameConfig()
    cfg = config.polish
    if detection is None or not _applies_to(image, detection, config):
        return image

    height, width = image.shape[:2]
    cx, cy = int(detection.cx), int(detection.cy)
    radius = float(detection.radius)
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 1) separar astros de reflexos: ghosts (reflexos da lente) são círculos
    #    pequenos com o centro fora do astro maior; os restantes são corpos
    #    celestes reais e são polidos individualmente.
    disks = find_all_disks(image, config)
    astros: list[DiskDetection] = []
    for disk in disks:
        if disk.radius < cfg.reflection_min_radius:
            continue
        inside_primary = math.hypot(disk.cx - cx, disk.cy - cy) < radius
        if not inside_primary and disk.radius < GHOST_RADIUS_RATIO * radius:
            if cfg.remove_reflections:
                continue
        astros.append(disk)

    # 2) fundo: média do fundo original (ou preto puro).
    if cfg.black_background:
        background = np.zeros(3, dtype=np.float32)
    elif cfg.background_fill:
        background = _background_color(image, cx, cy, radius, cfg)
    else:
        return image

    # 3) processo individual + máscaras (bandas com diluição até ao recorte).
    #    O recorte máximo é a banda do astro maior (a "linha de recorte" do
    #    utilizador); nenhuma máscara passa dela.
    primary_band = _band_mask((height, width), cx, cy, radius, radius * cfg.corona_scale, cfg.feather)
    # Limite do recorte do astro maior, SEM feather: a banda de outro corpo
    # é limitada por este recorte, mas só na forma dura — usar a banda suave
    # (borrada pelo feather) atenuaria o corpo inteiro junto ao bordo do recorte.
    ys, xs = np.ogrid[:height, :width]
    primary_crop = (
        (np.hypot(xs - cx, ys - cy) <= radius * cfg.corona_scale).astype(np.float32)
    )
    masks: list[np.ndarray] = []
    boosted: list[np.ndarray] = []
    for astro in astros:
        r_out = astro.radius * cfg.corona_scale
        band = _band_mask((height, width), astro.cx, astro.cy, astro.radius, r_out, cfg.feather)
        # A banda de um corpo que se sobrepõe ao recorte do astro maior é
        # limitada por esse recorte; um corpo real fora do recorte (planeta,
        # estrela) mantém a própria banda — senão a máscara anulava-se ali.
        if math.hypot(astro.cx - cx, astro.cy - cy) < radius * cfg.corona_scale:
            band = np.minimum(band, primary_crop)
        masks.append(band)
        boosted.append(_astro_boost(image, gray, astro, cfg))

    # o astro maior não invade o interior dos outros astros (cada um manda
    # na sua área; as sobreposições são suavizadas pelo blend ponderado)
    if len(masks) > 1:
        primary_mask = masks[0].copy()
        for i in range(1, len(masks)):
            inner = _feather_mask((height, width), astros[i].cx, astros[i].cy, astros[i].radius, cfg.feather)
            primary_mask = np.clip(primary_mask - inner, 0.0, 1.0)
        masks[0] = primary_mask

    # 4) remontagem sem costuras: média ponderada dos realces onde as
    #    máscaras se sobrepõem, fundo onde nada cobre.
    acc = np.zeros((height, width), dtype=np.float32)
    acc_color = np.zeros((height, width, 3), dtype=np.float32)
    for astro, band, boost in zip(astros, masks, boosted, strict=True):
        inner = _feather_mask((height, width), astro.cx, astro.cy, astro.radius, cfg.feather)
        band_px = max(1, int(round(astro.radius * cfg.corona_scale - astro.radius)))
        region = _astro_region(boost, inner, band_px, cfg.feather)
        acc += band
        if region.ndim == 3:
            acc_color += region * band[..., None]
        else:
            acc_color += region[..., None] * band[..., None]
    coverage = np.minimum(acc, 1.0)
    normalized = acc_color / np.maximum(acc, 1e-6)[..., None]
    if background.ndim:
        bg = background[None, None, :]
    else:
        bg = float(background)
    result = normalized * coverage[..., None] + bg * (1.0 - coverage)[..., None]
    if image.ndim == 2:
        result = result[..., 0]
    return np.clip(result, 0, 255).astype(np.uint8)
