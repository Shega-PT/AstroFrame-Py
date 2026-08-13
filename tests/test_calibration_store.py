"""Testes da persistência do ground truth da calibração (JSON)."""

from __future__ import annotations

import json

from astroframe.calibration.store import STORE_VERSION, CalibrationItem, CalibrationStore
from astroframe.core.stabilizer import DiskDetection


def test_store_vazio_sem_ficheiro(tmp_path):
    store = CalibrationStore(tmp_path / "cal.json")
    assert store.items == {}


def test_upsert_e_roundtrip(tmp_path):
    path = tmp_path / "sub" / "cal.json"
    store = CalibrationStore(path)
    item = CalibrationItem(
        path="eclipse.jpg",
        kind="image",
        frame=None,
        width=1920,
        height=1080,
        circles=[DiskDetection(960, 540, 400), DiskDetection(1000, 520, 60)],
    )
    store.upsert_item("eclipse.jpg", item)
    assert path.exists()

    reloaded = CalibrationStore(path)
    assert set(reloaded.items) == {"eclipse.jpg"}
    got = reloaded.get_item("eclipse.jpg")
    assert got.path == "eclipse.jpg"
    assert got.frame is None
    assert got.circles == [DiskDetection(960, 540, 400), DiskDetection(1000, 520, 60)]


def test_upsert_video_com_frame_e_substituicao(tmp_path):
    store = CalibrationStore(tmp_path / "cal.json")
    store.upsert_item(
        "v.mp4#3",
        CalibrationItem("v.mp4", "video", 3, 12, 12, [DiskDetection(5, 5, 3)]),
    )
    store.upsert_item(
        "v.mp4#3",
        CalibrationItem("v.mp4", "video", 3, 12, 12, [DiskDetection(6, 6, 4)]),
    )
    assert len(store.items) == 1
    assert store.get_item("v.mp4#3").circles == [DiskDetection(6, 6, 4)]


def test_get_item_inexistente(tmp_path):
    store = CalibrationStore(tmp_path / "cal.json")
    assert store.get_item("nada") is None


def test_json_invalido_comeca_vazio(tmp_path):
    path = tmp_path / "cal.json"
    path.write_text("{isto nao e json", encoding="utf-8")
    store = CalibrationStore(path)
    assert store.items == {}


def test_versao_desconhecida_comeca_vazio(tmp_path):
    path = tmp_path / "cal.json"
    path.write_text(json.dumps({"version": 99, "items": {}}), encoding="utf-8")
    store = CalibrationStore(path)
    assert store.items == {}


def test_item_invalido_ignorado(tmp_path):
    path = tmp_path / "cal.json"
    path.write_text(
        json.dumps(
            {
                "version": STORE_VERSION,
                "items": {
                    "ok.jpg": {
                        "path": "ok.jpg",
                        "kind": "image",
                        "frame": None,
                        "width": 2,
                        "height": 2,
                        "circles": [{"cx": 1, "cy": 1, "radius": 1}],
                    },
                    "mau.jpg": {
                        "path": "mau.jpg",
                        "kind": "image",
                        "frame": None,
                        "width": 2,
                        "height": 2,
                        "circles": [{"cx": "errado"}],
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    store = CalibrationStore(path)
    assert set(store.items) == {"ok.jpg"}


def test_tag_tipo_encoding(tmp_path):
    path = tmp_path / "cal.json"
    store = CalibrationStore(path)
    store.upsert_item("x.jpg", CalibrationItem("x.jpg", "image", None, 1, 1, [DiskDetection(0, 0, 1)]))
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw["version"] == STORE_VERSION
    assert raw["items"]["x.jpg"]["circles"][0] == {"cx": 0, "cy": 0, "radius": 1}
