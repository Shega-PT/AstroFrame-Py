"""Estabilização geométrica: localiza o disco do Sol/Lua e centraliza o frame.

Em vez de estabilizar o fundo (escuro ou uniforme), o algoritmo deteta o
centroide do disco em cada frame (HoughCircles com fallback por contornos)
e translada a imagem para manter o eclipse no centro exato. Em vídeos com
trepidação, `AntiJitterStabilizer` suaviza o centroide ao longo do tempo e
reutiliza o último deslocamento válido quando um frame não tem deteção.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import cv2
import numpy as np

from astroframe.config import AstroFrameConfig

logger = logging.getLogger(__name__)

_MIN_DETECTABLE_DIM = 32
_HALF_RES_THRESHOLD = 1200
_CROP_MARGIN = 6


@dataclass(frozen=True)
class DiskDetection:
    """Disco solar/lunar detetado.

    `cx`/`cy` referem-se às coordenadas da imagem de origem; após a
    estabilização o disco fica no centro do frame e `radius` é ajustado
    ao eventual recorte/redimensionamento aplicado.
    """

    cx: int
    cy: int
    radius: int


def _intensity_centroid(gray: np.ndarray, cx: int, cy: int, radius: int) -> tuple[int, int]:
    """Refina o centro usando o centroide ponderado pela intensidade do disco.

    Mais robusto ao re-escala/interpolação do que a deteção de formas sozinha.
    """
    height, width = gray.shape[:2]
    half = max(8, int(radius * 1.25))
    x0, x1 = max(0, cx - half), min(width, cx + half)
    y0, y1 = max(0, cy - half), min(height, cy + half)
    crop = gray[y0:y1, x0:x1].astype(np.float32)
    lo, hi = float(crop.min()), float(crop.max())
    if hi - lo < 1:
        return cx, cy
    mask = crop > (lo + (hi - lo) * 0.5)
    if int(mask.sum()) < 16:
        return cx, cy
    weights = mask * (crop - (lo + (hi - lo) * 0.5))
    total = float(weights.sum())
    ys, xs = np.mgrid[0 : crop.shape[0], 0 : crop.shape[1]]
    return int(x0 + float((xs * weights).sum()) / total), int(y0 + float((ys * weights).sum()) / total)


def _effective_radius_limits(height: int, width: int, cfg) -> tuple[int, int]:
    """Raios Hough derivados da resolução do frame (evita limites fixos em px)."""
    half = min(height, width) // 2
    return min(cfg.min_radius, half), max(cfg.max_radius, half)


def find_disk_center(image: np.ndarray, config: AstroFrameConfig | None = None) -> DiskDetection | None:
    """Devolve o centro/raio do disco solar/lunar, ou None se não for detetado.

    É o melhor candidato de `find_all_disks` — em frames grandes a deteção
    corre em meia-resolução e o resultado é re-escalado.
    """
    disks = find_all_disks(image, config)
    return disks[0] if disks else None


def find_all_disks(image: np.ndarray, config: AstroFrameConfig | None = None) -> list[DiskDetection]:
    """Todos os discos candidatos detetados, ordenados por plausibilidade.

    O primeiro é o disco principal (Sol/Lua); os seguintes são normalmente
    reflexos internos da lente (ghosts) — a UI desenha-os a vermelho e o
    polimento pode removê-los como ruído. Ordenação: raio decrescente
    (contornos) ou confiança Hough decrescente, sempre raio decrescente.
    """
    config = config or AstroFrameConfig()
    cfg = config.stabilizer

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape[:2]
    if min(height, width) < _MIN_DETECTABLE_DIM:
        return []

    scale = 1.0
    if min(height, width) >= _HALF_RES_THRESHOLD:
        scale = 0.5
        gray = cv2.resize(gray, (width // 2, height // 2), interpolation=cv2.INTER_AREA)

    blurred = cv2.GaussianBlur(gray, (cfg.gaussian_kernel_size, cfg.gaussian_kernel_size), cfg.gaussian_sigma)
    min_radius, max_radius = _effective_radius_limits(*gray.shape[:2], cfg)

    disks: list[DiskDetection] = []

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=cfg.dp,
        minDist=cfg.min_dist,
        param1=cfg.param1,
        param2=cfg.param2,
        minRadius=min_radius,
        maxRadius=max_radius,
    )
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for x, y, r in sorted(circles[0], key=lambda c: int(c[2]), reverse=True):
            cx, cy = _intensity_centroid(gray, int(x), int(y), int(r))
            disks.append(DiskDetection(int(cx / scale), int(cy / scale), int(r / scale)))

    if cfg.contour_fallback:
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            moments = cv2.moments(largest)
            if moments["m00"] > 0:
                cx = int(moments["m10"] / moments["m00"] / scale)
                cy = int(moments["m01"] / moments["m00"] / scale)
                (_, _), radius = cv2.minEnclosingCircle(largest)
                found = DiskDetection(cx, cy, int(radius / scale))
                if not any(abs(d.cx - found.cx) + abs(d.cy - found.cy) < 4 for d in disks):
                    disks.append(found)

    disks.sort(key=lambda d: d.radius, reverse=True)
    unique: list[DiskDetection] = []
    for disk in disks:
        if any(_is_concentric_duplicate(disk, kept) for kept in unique):
            continue
        unique.append(disk)
    return unique


def _is_concentric_duplicate(candidate: DiskDetection, kept: DiskDetection) -> bool:
    """Círculo interior quase-concêntrico com um disco já aceite (é o mesmo
    bordo detetado duas vezes — ex.: silhueta da Lua dentro do Sol)."""
    tolerance = max(2, int(0.12 * kept.radius))
    return math.hypot(candidate.cx - kept.cx, candidate.cy - kept.cy) <= tolerance


def _auto_crop(stabilized: np.ndarray, dx: int, dy: int, radius: int, cfg) -> tuple[np.ndarray, float]:
    """Remove as bordas pretas introduzidas pela translação, sem cortar o disco.

    Devolve (imagem reenquadrada, fator de escala do raio).
    """
    if not cfg.auto_crop:
        return stabilized, 1.0
    height, width = stabilized.shape[:2]

    disk_w = min(width, 2 * (radius + _CROP_MARGIN))
    disk_h = min(height, 2 * (radius + _CROP_MARGIN))
    crop_w = min(width, max(disk_w, width - 2 * abs(dx)))
    crop_h = min(height, max(disk_h, height - 2 * abs(dy)))

    if crop_w >= width and crop_h >= height:
        return stabilized, 1.0

    x0 = (width - crop_w) // 2
    y0 = (height - crop_h) // 2
    cropped = stabilized[y0 : y0 + crop_h, x0 : x0 + crop_w]
    resized = cv2.resize(cropped, (width, height), interpolation=cv2.INTER_LINEAR)
    return resized, max(width / crop_w, height / crop_h)


def _translate(stabilized: np.ndarray, dx: int, dy: int, radius: int, cfg) -> tuple[np.ndarray, float]:
    matrix = np.float32([[1, 0, dx], [0, 1, dy]])
    height, width = stabilized.shape[:2]
    shifted = cv2.warpAffine(stabilized, matrix, (width, height))
    return _auto_crop(shifted, dx, dy, radius, cfg)


def center_and_stabilize(
    image: np.ndarray, config: AstroFrameConfig | None = None
) -> tuple[np.ndarray, DiskDetection | None]:
    """Calcula o vetor de deslocamento do disco até ao centro do frame e re-alinha a imagem."""
    config = config or AstroFrameConfig()
    detection = find_disk_center(image, config)
    if detection is None:
        logger.warning("Disco não detetado; frame devolvido sem alteração.")
        return image, None

    height, width = image.shape[:2]
    dx = width // 2 - detection.cx
    dy = height // 2 - detection.cy

    stabilized, scale = _translate(image, dx, dy, detection.radius, config.stabilizer)
    radius = int(detection.radius * scale)
    return stabilized, DiskDetection(detection.cx, detection.cy, radius)


class AntiJitterStabilizer:
    """Estabilização temporal com suavização do centroide (EMA).

    Evita saltos frame-a-frame quando a deteção varia ligeiramente e mantém
    o último deslocamento válido quando um frame não tem disco detetado
    (blur extremo, câmara em guinada rápida).
    """

    def __init__(self, config: AstroFrameConfig | None = None, alpha: float | None = None):
        self.config = config or AstroFrameConfig()
        self.alpha = alpha if alpha is not None else self.config.stabilizer.jitter_alpha
        self._smooth: tuple[float, float] | None = None
        self._radius: int | None = None
        self._all_disks: list[DiskDetection] = []
        self._last_detection: DiskDetection | None = None

    @property
    def last_all_disks(self) -> list[DiskDetection]:
        """Todos os discos detetados no frame mais recente (principal + reflexos)."""
        return list(self._all_disks)

    @property
    def last_detection(self) -> DiskDetection | None:
        """Última posição/raio conhecidos do disco principal (mesmo em frames sem deteção)."""
        if self._smooth is None:
            return None
        center = DiskDetection(int(round(self._smooth[0])), int(round(self._smooth[1])), self._radius or 0)
        if self._last_detection is None:
            return center
        return DiskDetection(center.cx, center.cy, self._last_detection.radius)

    def stabilize(self, frame: np.ndarray) -> tuple[np.ndarray, DiskDetection | None]:
        height, width = frame.shape[:2]
        all_disks = find_all_disks(frame, self.config)
        detection = all_disks[0] if all_disks else None

        if detection is not None:
            if self._smooth is None:
                self._smooth = (float(detection.cx), float(detection.cy))
            else:
                self._smooth = (
                    self.alpha * detection.cx + (1.0 - self.alpha) * self._smooth[0],
                    self.alpha * detection.cy + (1.0 - self.alpha) * self._smooth[1],
                )
            self._radius = detection.radius
            self._last_detection = detection
        elif self._smooth is None:
            self._all_disks = []
            return frame, None

        if all_disks:
            self._all_disks = all_disks
        dx = width // 2 - int(round(self._smooth[0]))
        dy = height // 2 - int(round(self._smooth[1]))
        radius = self._radius if self._radius is not None else 0

        stabilized, scale = _translate(frame, dx, dy, radius, self.config.stabilizer)
        if detection is not None:
            detection = DiskDetection(detection.cx, detection.cy, int(detection.radius * scale))
        return stabilized, detection
