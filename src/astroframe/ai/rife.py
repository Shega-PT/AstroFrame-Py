"""Interpolação de movimento RIFE via PyTorch.

O PyTorch é uma dependência obrigatória desde a v0.9.0 (instalado por
omissão; no CI usa-se o wheel CPU). O modelo é carregado de forma preguiçosa
via `torch.hub` apenas quando a CLI `astroframe video --interp N` é usada;
se o carregamento falhar, o CLI avisa e continua sem interpolação.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

_RIFE_HINT = (
    "O modelo RIFE não pôde ser carregado (falha de rede/`torch.hub` ou "
    "repositório indisponível). Verifique a ligação e tente novamente; "
    "a CLI `astroframe video` continua a funcionar sem interpolação."
)


class RifeInterpolator:
    """Wrapper de interpolação RIFE com importação preguiçosa.

    Aceita frames BGR (numpy uint8) e devolve frames intermédios em BGR.
    A interface exata do modelo varia entre versões dos repositórios RIFE;
    ajuste `_infer` ao modelo concreto se necessário.
    """

    def __init__(
        self,
        repo: str,
        source: str = "github",
        model_name: str = "IFNet",
        device: str | None = None,
    ):
        import torch  # Importação preguiçosa

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.hub.load(repo, model_name, source=source, pretrained=True)
        self.model.to(self.device).eval()
        logger.info("RIFE carregado no dispositivo %s", self.device)

    @staticmethod
    def available() -> bool:
        try:
            import torch  # noqa: F401

            return True
        except ImportError:
            return False

    def _infer(self, frame_a: np.ndarray, frame_b: np.ndarray, timestep: float) -> np.ndarray:
        import torch

        def to_tensor(image: np.ndarray) -> torch.Tensor:
            rgb = image[..., ::-1]  # BGR -> RGB
            tensor = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1)))
            return tensor.float().to(self.device) / 255.0

        with torch.no_grad():
            img0 = to_tensor(frame_a)
            img1 = to_tensor(frame_b)
            imean = (img0 + img1) / 2.0
            inputs = torch.cat((img0, imean, img1), dim=0).unsqueeze(0)

            output = self.model(inputs, torch.tensor(timestep, device=self.device), True)[0]
            output = output.squeeze(0).permute(1, 2, 0).cpu().numpy()

        output = np.clip(output, 0.0, 1.0)
        return (output[..., ::-1] * 255.0).astype(np.uint8)

    def interpolate(self, frame_a: np.ndarray, frame_b: np.ndarray, n_interp: int = 1) -> list[np.ndarray]:
        """Devolve `n_interp` frames intermédios ordenados entre A e B."""
        if not self.available():
            raise RuntimeError(_RIFE_HINT)
        if n_interp < 0:
            raise ValueError("n_interp deve ser >= 0")
        if n_interp == 0:
            return []
        steps = np.linspace(1.0 / (n_interp + 1), n_interp / (n_interp + 1), n_interp)
        return [self._infer(frame_a, frame_b, float(timestep)) for timestep in steps]
