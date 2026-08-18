"""Persistência do ground truth manual da calibração.

O ficheiro é JSON (v1), em `Logs/train/calibration.json` por omissão (com
fallback para `samples/calibration.json` quando o ficheiro global ainda não
existe), com uma entrada por item de calibração:

```json
{"version": 1, "items": {
  "sol.jpg": {"path": "sol.jpg", "kind": "image", "frame": null,
                  "width": 1920, "height": 1080,
                  "circles": [{"cx": 960, "cy": 540, "radius": 400},
                              {"cx": 200, "cy": 300, "radius": 80, "ry": 50}]}
}}
```

A chave é a `item_key` do `scan` (path relativo + `#frame` para vídeos), o que
mantém o ground truth válido mesmo que a pasta de exemplos seja reorganizada
por pastas. `ry` é opcional e só aparece para elipses (ausente = círculo).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from astroframe.core.stabilizer import DiskDetection

logger = logging.getLogger(__name__)

STORE_VERSION = 1


@dataclass
class CalibrationItem:
    """Ground truth de um item: os círculos (astros) ajustados manualmente."""

    path: str
    kind: str
    frame: int | None
    width: int
    height: int
    circles: list[DiskDetection] = field(default_factory=list)


def _circle_to_dict(circle: DiskDetection) -> dict:
    raw = {"cx": circle.cx, "cy": circle.cy, "radius": circle.radius}
    if circle.ry is not None:
        raw["ry"] = circle.ry
    return raw


def _circle_from_dict(raw: dict) -> DiskDetection:
    ry = raw.get("ry")
    ry_int = int(ry) if ry is not None else None
    return DiskDetection(int(raw["cx"]), int(raw["cy"]), int(raw["radius"]), ry_int)


class CalibrationStore:
    """Base de ground truth em JSON, com carga/gravação idempotentes.

    Ficheiros inexistentes, JSON inválido ou versões desconhecidas resultam
    num store vazio (nunca levantam exceções).
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.items: dict[str, CalibrationItem] = {}
        self.load()

    def load(self) -> None:
        self.items = {}
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Calibração ilegível (%s), a começar vazia: %s", self.path, exc)
            return
        if not isinstance(data, dict) or data.get("version") != STORE_VERSION:
            logger.warning("Calibração com versão desconhecida (%s), a começar vazia", self.path)
            return
        for key, raw in (data.get("items") or {}).items():
            try:
                self.items[key] = CalibrationItem(
                    path=raw.get("path", key),
                    kind=raw.get("kind", "image"),
                    frame=raw.get("frame"),
                    width=int(raw.get("width", 0)),
                    height=int(raw.get("height", 0)),
                    circles=[_circle_from_dict(c) for c in raw.get("circles", [])],
                )
            except (KeyError, TypeError, ValueError):
                logger.warning("Item de calibração ignorado (inválido): %s", key)

    def save(self) -> None:
        data = {
            "version": STORE_VERSION,
            "items": {
                key: {
                    "path": item.path,
                    "kind": item.kind,
                    "frame": item.frame,
                    "width": item.width,
                    "height": item.height,
                    "circles": [_circle_to_dict(c) for c in item.circles],
                }
                for key, item in self.items.items()
            },
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    def upsert_item(self, key: str, item: CalibrationItem) -> None:
        self.items[key] = item
        self.save()

    def get_item(self, key: str) -> CalibrationItem | None:
        return self.items.get(key)
