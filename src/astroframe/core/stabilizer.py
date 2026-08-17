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
_HALF_RES_THRESHOLD = 1000
_CROP_MARGIN = 6

# Limites de raio/distância derivados da resolução do frame: sem valores fixos
# em px (dependem de imagem para imagem), escalam com o menor lado.
_MIN_RADIUS_RATIO = 0.02
_MIN_DIST_RATIO = 0.1

# Passe de sensibilidade: procura discos fracos/pequenos com um acumulador
# mais permissivo; os falsos positivos extra são removidos pelos filtros
# seguintes (_same_edge/_is_occluded_artifact/limite de discos).
_SENSITIVITY_PARAM2_FACTOR = 0.6

# Contraste mínimo do bordo interior (perfil radial) para aceitar um
# companheiro concêntrico (Lua em eclipse total) — evita falsos positivos
# com o limb darkening do Sol ou pontos brilhantes internos.
_MIN_RING_CONTRAST = 25.0

# Cache do filtro CNN (carregado uma única vez; `None` = sem modelo).
_disk_filter: object | None = None


@dataclass(frozen=True)
class DiskDetection:
    """Disco solar/lunar detetado.

    `cx`/`cy` referem-se às coordenadas da imagem de origem; após a
    estabilização o disco fica no centro do frame e `radius` é ajustado
    ao eventual recorte/redimensionamento aplicado.

    `ry` é o raio vertical para formas elípticas (ground truth manual da
    calibração); `None` significa círculo (`radius` é o raio).
    """

    cx: int
    cy: int
    radius: int
    ry: int | None = None


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
    """Limites Hough derivados da resolução do frame.

    O raio mínimo é ~2% do menor lado (nunca fixo em px — adapta-se à
    imagem/frame); o máximo é o teto `cfg.max_radius`, com o menor lado como
    limite absoluto. Calculados sobre as dimensões de trabalho (escala já
    aplicada), eliminando mínimos desalinhados em meia-resolução.
    """
    half = min(height, width) // 2
    min_radius = max(2, int(min(height, width) * _MIN_RADIUS_RATIO))
    return min_radius, max(cfg.max_radius, half)


def _derived_min_dist(height: int, width: int) -> int:
    """Distância mínima entre centros (~10% do menor lado, mínimo 60 px).

    Grande de propósito: suprime o enxame de círculos concêntricos que o
    Hough gera junto aos bordos do astro maior (centros quase coincidentes,
    incluindo o "envelope" que envolve o Sol e os planetas colados a ele) e
    impede que um deles se passe pelo disco principal com um raio inflado.

    Discos realmente separados (planetas, ghosts, companheiros excêntricos)
    têm centros mais afastados e não são afetados; a Lua em eclipse **total**
    (centros coincidentes) é encontrada pelo passe concêntrico
    (`_concentric_companion`), não pelo Hough.
    """
    return max(60, int(min(height, width) * _MIN_DIST_RATIO))


def find_disk_center(image: np.ndarray, config: AstroFrameConfig | None = None) -> DiskDetection | None:
    """Devolve o centro/raio do disco solar/lunar, ou None se não for detetado.

    É o melhor candidato de `find_all_disks` — em frames grandes a deteção
    corre em meia-resolução e o resultado é re-escalado.
    """
    disks = find_all_disks(image, config)
    return disks[0] if disks else None


def find_all_disks(image: np.ndarray, config: AstroFrameConfig | None = None) -> list[DiskDetection]:
    """Todos os discos candidatos detetados, ordenados por raio decrescente.

    O primeiro é o astro maior (Sol); os seguintes podem ser:

    - **companheiros de eclipse** — círculos interiores com raio próprio
      (ex.: a Lua a entrar); a distância mínima entre centros é derivada da
      resolução (pequena), por isso não suprime círculos concêntricos;
    - **planetas alinhados** — discos separados de tamanhos muito diferentes;
      um passe de sensibilidade (acumulador mais permissivo) apanha os
      pequenos/desbotados que o passe principal perde;
    - **reflexos da lente (ghosts)** — círculos afastados, normalmente mais
      pequenos (a UI desenha-os a vermelho e o polimento pode removê-los).

    Dedup: são fundidos apenas círculos do **mesmo bordo** (centros próximos
    E raios quase-iguais); círculos concêntricos de raios diferentes
    (Sol + Lua) convivem na lista. O número máximo de discos é
    `cfg.max_disks` (omissão: 8).
    """
    config = config or AstroFrameConfig()
    cfg = config.stabilizer

    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape[:2]
    if min(height, width) < _MIN_DETECTABLE_DIM:
        return []

    scale = 1.0
    if min(height, width) >= _HALF_RES_THRESHOLD:
        scale = 0.5
        gray = cv2.resize(gray, (width // 2, height // 2), interpolation=cv2.INTER_AREA)

    blurred = cv2.GaussianBlur(gray, (cfg.gaussian_kernel_size, cfg.gaussian_kernel_size), cfg.gaussian_sigma)
    min_radius, max_radius = _effective_radius_limits(*gray.shape[:2], cfg)
    min_dist = _derived_min_dist(*gray.shape[:2])
    detail_dist = max(25, min_dist // 4)

    main_candidates = _hough_pass(blurred, cfg, min_dist, min_radius, max_radius, scale)
    extras: list[DiskDetection] = []
    if detail_dist < min_dist:
        extras += _hough_pass(blurred, cfg, detail_dist, min_radius, max_radius, scale)
    if cfg.param2 > 1:
        sensitivity = max(1, int(cfg.param2 * _SENSITIVITY_PARAM2_FACTOR))
        if sensitivity < cfg.param2:
            extras += _hough_pass(
                blurred, cfg, min_dist, min_radius, max_radius, scale, param2_override=sensitivity
            )
    candidates = main_candidates + [
        c
        for c in extras
        if not any(_same_center(c, m) for m in main_candidates)
        and not any(_hugging_envelope(c, m) for m in main_candidates)
    ]

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
                if not any(abs(d.cx - found.cx) + abs(d.cy - found.cy) < 4 for d in candidates):
                    candidates.append(found)

    candidates.sort(key=lambda d: d.radius, reverse=True)
    unique: list[DiskDetection] = []
    for disk in candidates:
        if any(_same_edge(disk, kept) for kept in unique):
            continue
        if any(_concentric_envelope(disk, kept) for kept in unique):
            continue
        if len(unique) >= cfg.max_disks:
            break
        if any(_is_occluded_artifact(blurred, disk, kept, scale, cfg) for kept in unique):
            continue
        unique.append(disk)

    # Companheiro concêntrico (eclipse total): Hough suprime círculos com o
    # mesmo centro, por isso só quando o disco primário é o único encontrado
    # se procura a Lua no perfil radial do Sol.
    if len(unique) == 1:
        companion = _concentric_companion(blurred, unique[0], min_radius, cfg, scale)
        if companion is not None:
            if not any(_same_edge(companion, kept) for kept in unique) and not any(
                _is_occluded_artifact(blurred, companion, kept, scale, cfg) for kept in unique
            ):
                unique.append(companion)

    # Filtro CNN (opcional): remove falsos positivos do Hough com confiança
    # inferior ao limiar; **nunca** esvazia a lista quando havia deteções —
    # a deteção nunca regride. Os patches são extraídos da imagem original
    # (resolução total, não da cópia reduzida).
    if config.ai.disk_filter > 0.0 and unique:
        global _disk_filter
        if _disk_filter is None:
            from astroframe.ai.cnn import DiskFilter

            _disk_filter = DiskFilter()
        unique = _disk_filter.filter_disks(unique, image, config.ai.disk_filter)
    return unique


def _is_occluded_artifact(
    gray: np.ndarray, candidate: DiskDetection, kept: DiskDetection, scale: float, cfg
) -> bool:
    """Candidato quase totalmente dentro de um disco já aceite (interior ao
    astro maior) que **não é um astro real**: um companheiro de eclipse (a
    Lua) é muito mais escuro que o anel à sua volta; um círculo deitado pelos
    dois bordos (Sol+Lua na mesma deteção) tem contraste fraco e é descartado.

    `cfg.occluded_ring` é o raio do anel de comparação (× raio do candidato) e
    `cfg.occluded_ratio` o limiar de brilho interior que decide artefacto —
    ambos treináveis pela validação (nunca tamanhos/distâncias).

    A comparação usa a sobreposição de **área** (e não só o centro — o
    refinamento do centroide pode arrastar o centro de um objeto afastado
    para perto do astro maior).

    `candidate`/`kept` estão em coordenadas da imagem de origem; `scale` é a
    escala aplicada (0.5 em meia-resolução) e converte para a imagem de
    trabalho onde `gray` foi calculado.
    """
    height, width = gray.shape[:2]
    r_c, r_k = candidate.radius * scale, kept.radius * scale
    cx_c, cy_c = candidate.cx * scale, candidate.cy * scale
    cx_k, cy_k = kept.cx * scale, kept.cy * scale
    d = math.hypot(cx_c - cx_k, cy_c - cy_k)
    if d >= kept.radius:
        return False
    ring_r = cfg.occluded_ring * r_c
    x0 = max(0, int(math.floor(cx_c - ring_r)))
    x1 = min(width, int(math.ceil(cx_c + ring_r)))
    y0 = max(0, int(math.floor(cy_c - ring_r)))
    y1 = min(height, int(math.ceil(cy_c + ring_r)))
    if x1 <= x0 or y1 <= y0:
        return False
    crop = gray[y0:y1, x0:x1]
    ys, xs = np.ogrid[y0:y1, x0:x1]
    dx_c, dy_c = xs - cx_c, ys - cy_c
    r2_c = r_c * r_c
    in_candidate = dx_c * dx_c + dy_c * dy_c <= r2_c
    n_c = int(in_candidate.sum())
    if n_c == 0:
        return False
    if d + r_c > r_k:
        dx_k, dy_k = xs - cx_k, ys - cy_k
        inside_kept = dx_k * dx_k + dy_k * dy_k <= r_k * r_k
        if float(in_candidate[inside_kept].sum()) < 0.9 * n_c:
            return False
    ring = (~in_candidate) & (dx_c * dx_c + dy_c * dy_c <= ring_r * ring_r)
    inside_mean = float(crop[in_candidate].mean())
    ring_mean = float(crop[ring].mean()) if ring.any() else 0.0
    return ring_mean > 0 and inside_mean >= cfg.occluded_ratio * ring_mean


def _concentric_companion(
    blurred: np.ndarray, primary: DiskDetection, min_radius: int, cfg, scale: float
) -> DiskDetection | None:
    """Companheiro concêntrico do disco primário (ex.: Lua em eclipse total
    ou quase total, centro a poucos px do astro maior).

    O Hough suprime círculos com o MESMO centro (ou centro mais próximo que
    `min_dist`), por isso um disco interior centrado no Sol é invisível a
    ele. Aqui o perfil radial médio (anéis de 4 px em torno do centro
    primário) é calculado e procurada uma **depressão central escura**:

    - o mínimo do perfil fica no interior (raios ≤ 60% do raio máximo);
    - o centro é muito mais escuro que a região exterior (verdadeira
      depressão, não um centro brilhante com mancha);
    - o contraste é forte (`_MIN_RING_CONTRAST`).

    O raio do companheiro é o bordo de subida após a depressão (para um
    disco excentricamente centrado, a transição é suave e o bordo fica
    próximo do raio real + excentricidade).

    `primary` está em coordenadas da imagem de origem; `scale` converte para
    a imagem de trabalho onde `blurred` foi calculado.
    """
    height, width = blurred.shape[:2]
    cx, cy = primary.cx * scale, primary.cy * scale
    r_prim = primary.radius * scale
    r_max = int(min(r_prim - 3, cx, width - cx, cy, height - cy))
    if r_max <= min_radius + 3:
        return None
    ys, xs = np.ogrid[:height, :width]
    dist2 = (xs - cx) * (xs - cx) + (ys - cy) * (ys - cy)
    radii = np.arange(min_radius, r_max + 1)
    profile = np.array(
        [
            float(blurred[(dist2 >= (r - 2) * (r - 2)) & (dist2 < (r + 2) * (r + 2))].mean())
            for r in radii
        ]
    )
    i_min = int(np.argmin(profile))
    if radii[i_min] > 0.6 * r_max:
        return None
    dark = profile[i_min]
    bright = float(profile[-len(profile) // 5 :].mean())
    if bright - dark < _MIN_RING_CONTRAST:
        return None
    if dark > 0.7 * bright:
        return None
    grad = profile[1:] - profile[:-1]
    i = i_min + int(np.argmax(grad[i_min:]))
    return DiskDetection(primary.cx, primary.cy, int(int(radii[i]) / scale))


def _hough_pass(
    blurred: np.ndarray,
    cfg,
    min_dist: int,
    min_radius: int,
    max_radius: int,
    scale: float,
    param2_override: int | None = None,
) -> list[DiskDetection]:
    """Um passe Hough com parâmetros próprios.

    `param2_override` permite um passe de sensibilidade mais permissivo
    (procura discos fracos sem alterar a configuração principal).
    """
    param2 = cfg.param2 if param2_override is None else param2_override
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=cfg.dp,
        minDist=min_dist,
        param1=cfg.param1,
        param2=param2,
        minRadius=min_radius,
        maxRadius=max_radius,
    )
    found: list[DiskDetection] = []
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for x, y, r in sorted(circles[0], key=lambda c: int(c[2]), reverse=True):
            cx, cy = _intensity_centroid(blurred, int(x), int(y), int(r))
            found.append(DiskDetection(int(cx / scale), int(cy / scale), int(r / scale)))
    return found


def _same_edge(candidate: DiskDetection, kept: DiskDetection) -> bool:
    """O mesmo bordo detetado duas vezes (centros próximos E raios quase-iguais).

    Círculos concêntricos com raios diferentes (ex.: Lua dentro do Sol) não
    são fundidos — são astros distintos.
    """
    tolerance = max(2, int(0.12 * max(candidate.radius, kept.radius)))
    return (
        math.hypot(candidate.cx - kept.cx, candidate.cy - kept.cy) <= tolerance
        and abs(candidate.radius - kept.radius) <= tolerance
    )


def _same_center(candidate: DiskDetection, kept: DiskDetection) -> bool:
    """Centros praticamente coincidentes (mesmo objeto, bordo possivelmente
    diferente). Usado para não deixar os passos de detalhe/sensibilidade
    voltarem a detetar o astro maior com um raio inflado (envelope)."""
    return math.hypot(candidate.cx - kept.cx, candidate.cy - kept.cy) <= 8


def _hugging_envelope(candidate: DiskDetection, kept: DiskDetection) -> bool:
    """Candidato dos passos extra que "abraça" um disco do passe principal.

    Após o refinamento do centroide, os envelopes (círculos que envolvem um
    astro e os discos colados a ele) ficam com o centro a uma pequena fração
    do próprio raio do disco principal e são visivelmente maiores — um
    duplicado do mesmo objeto com raio inflado: descartado.
    """
    return math.hypot(candidate.cx - kept.cx, candidate.cy - kept.cy) <= 0.15 * candidate.radius and (
        candidate.radius >= 1.12 * kept.radius
    )


def _concentric_envelope(candidate: DiskDetection, kept: DiskDetection) -> bool:
    """Envelope concêntrico de um disco já aceite (mesmo centro, raio maior).

    Nos passos adicionais (detalhe/sensibilidade) o Hough volta a detetar o
    astro maior com um raio inflado — o círculo que envolve o disco e os
    astros colados a ele. Centro quase idêntico + raio não menor que o do
    aceite = duplicado do mesmo objeto com um raio pior: descartado.
    """
    if math.hypot(candidate.cx - kept.cx, candidate.cy - kept.cy) > 8:
        return False
    return candidate.radius >= 0.9 * kept.radius


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
    (blur extremo, câmara em guinada rápida). Com `ai.lstm_trajectory`
    ativo, o centroide é **previsto** (extrapolação linear + refinamento
    LSTM opcional) em vez de ficar congelado nos frames sem deteção.
    """

    def __init__(self, config: AstroFrameConfig | None = None, alpha: float | None = None):
        self.config = config or AstroFrameConfig()
        self.alpha = alpha if alpha is not None else self.config.stabilizer.jitter_alpha
        self._smooth: tuple[float, float] | None = None
        self._radius: int | None = None
        self._all_disks: list[DiskDetection] = []
        self._last_detection: DiskDetection | None = None
        self._trajectory: object | None = None
        if self.config.ai.lstm_trajectory:
            try:
                from astroframe.ai.lstm import TrajectoryPredictor

                self._trajectory = TrajectoryPredictor(use_lstm=True)
            except Exception:  # pragma: no cover - nunca bloqueia o runtime
                self._trajectory = None

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
            if self._trajectory is not None:
                self._trajectory.push(self._smooth[0], self._smooth[1])
        elif self._smooth is None:
            self._all_disks = []
            return frame, None
        elif self._trajectory is not None and self._smooth is not None:
            predicted = self._trajectory.predict()
            if predicted is not None:
                self._smooth = predicted

        if all_disks:
            self._all_disks = all_disks
        dx = width // 2 - int(round(self._smooth[0]))
        dy = height // 2 - int(round(self._smooth[1]))
        radius = self._radius if self._radius is not None else 0

        stabilized, scale = _translate(frame, dx, dy, radius, self.config.stabilizer)
        if detection is not None:
            detection = DiskDetection(detection.cx, detection.cy, int(detection.radius * scale))
        return stabilized, detection
