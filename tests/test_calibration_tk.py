"""Testes da UI desktop de calibração (Tk real, janela oculta).

Exercita `CalibrationTkApp` com o mesmo padrão dos testes da validação:
`DISPLAY` + janela retirada do ecrã, loop de eventos bombeado com
`root.update()` e deteção substituída por `monkeypatch` para ser
determinística.
"""

from __future__ import annotations

import time
import types
from pathlib import Path

import cv2
import pytest

from astroframe.calibration.scan import scan_samples
from astroframe.calibration.store import CalibrationItem, CalibrationStore
from astroframe.config import AstroFrameConfig
from astroframe.core.stabilizer import DiskDetection
from astroframe.ui import calibration_tk
from tests.helpers import make_disk_image

DETECTED = [DiskDetection(240, 180, 80)]


@pytest.fixture()
def root():
    tk_root = calibration_tk.tk.Tk()
    tk_root.withdraw()
    yield tk_root
    try:
        tk_root.destroy()
    except calibration_tk.tk.TclError:
        pass


def make_samples(tmp_path: Path, n: int = 1, gt: bool = False) -> Path:
    """Pasta `samples` com `n` imagens; `gt` pré-guarda o ground truth."""
    samples = tmp_path / "samples"
    samples.mkdir()
    store = CalibrationStore(samples / "calibration.json")
    for i in range(n):
        name = f"sample_{i}.jpg"
        image, cx, cy = make_disk_image()
        cv2.imwrite(str(samples / name), image)
        if gt:
            store.items[name] = CalibrationItem(
                name, "image", None, 480, 360, [DiskDetection(cx, cy, 90)]
            )
    if gt:
        store.save()
    return samples


def build(root, samples: Path, monkeypatch=None) -> calibration_tk.CalibrationTkApp:
    app = calibration_tk.build_app(root, samples_dir=str(samples))
    if monkeypatch is not None:
        monkeypatch.setattr(app.canvas, "winfo_width", lambda: 900)
        monkeypatch.setattr(app.canvas, "winfo_height", lambda: 700)
    return app


def pump(app, root, timeout: float = 10.0) -> None:
    """Espera o trabalho em thread terminar e entrega a fila à UI."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline and app._busy:
        root.update()
        time.sleep(0.01)
    end = time.monotonic() + 0.2
    while time.monotonic() < end:
        root.update()
        time.sleep(0.01)
    assert not app._busy, "trabalho em thread não terminou a tempo"


def click(app, ix: float, iy: float) -> types.SimpleNamespace:
    cx, cy = calibration_tk.image_to_canvas(ix, iy, app.scale, app.ox, app.oy)
    return types.SimpleNamespace(x=cx, y=cy)


def press(app, ix: float, iy: float) -> None:
    app._on_press_left(click(app, ix, iy))


def drag(app, ix: float, iy: float) -> None:
    app._on_drag_left(click(app, ix, iy))


# ---------------------------------------------------------------- fluxo --


def test_carrega_primeira_passagem_sem_ground_truth(root, tmp_path):
    app = build(root, make_samples(tmp_path))
    try:
        assert app.current_index == 0
        assert app.shapes == []
        assert app.selected is None
        assert not app._busy
        assert "1.ª passagem" in app.status.get()
        assert app.listbox.get(0) == "IMG sample_0.jpg"
    finally:
        app.root.destroy()


def test_redraw_desenha_formas_selecionadas_e_pegas(root, tmp_path, monkeypatch):
    app = build(root, make_samples(tmp_path), monkeypatch)
    try:
        app.shapes = [
            DiskDetection(300, 140, 90),
            DiskDetection(100, 80, 50, 30),
        ]
        app.selected = 0
        app.redraw()
        assert len(app.canvas.find_all()) > 5
        assert app._photo is not None
        assert app._photo_scale == app.scale
    finally:
        app.root.destroy()


def test_criar_mover_redimensionar_com_o_rato(root, tmp_path, monkeypatch):
    app = build(root, make_samples(tmp_path), monkeypatch)
    try:
        press(app, 60, 40)
        assert app.shapes == [DiskDetection(60, 40, 40)]
        assert app.selected == 0
        assert app._drag_mode == "resize:right"
        assert "Selecionada: centro (60, 40)" in app.shape_info.cget("text")

        drag(app, 120, 40)
        assert app.shapes[0].radius == 60

        press(app, 60, 40)
        assert app._drag_mode == "move"
        drag(app, 70, 45)
        assert (app.shapes[0].cx, app.shapes[0].cy) == (70, 45)

        press(app, 130, 45)
        assert app._drag_mode == "resize:right"
        drag(app, 110, 45)
        assert app.shapes[0].radius == 40

        app.shape_kind.set("ellipse")
        press(app, 200, 200)
        assert app.shapes[1] == DiskDetection(200, 200, 60, 40)

        press(app, 200, 160)
        assert app._drag_mode == "resize:top"
        drag(app, 200, 170)
        assert app.shapes[1].ry == 30

        app._on_release()
        assert app._drag_mode is None

        drag(app, 300, 300)
        app._drag_mode = "move"
        app.selected = None
        drag(app, 300, 300)
        app.selected = 9
        drag(app, 300, 300)
        assert len(app.shapes) == 2

        app._busy = True
        press(app, 400, 300)
        assert len(app.shapes) == 2
        app._busy = False
    finally:
        app.root.destroy()


# ------------------------------------------------------------- deteção --


def test_deteccao_automatica_ao_carregar(root, tmp_path, monkeypatch):
    monkeypatch.setattr(calibration_tk, "find_all_disks", lambda frame, config: DETECTED)
    app = build(root, make_samples(tmp_path))
    try:
        app.auto_detect.set(True)
        app.load_sample(0)
        assert app._busy
        pump(app, root)
        assert app.shapes == DETECTED
        assert "1 disco(s) detetado(s)" in app.status.get()
        assert not app._busy
    finally:
        app.root.destroy()


def test_on_detect_done_job_antigo_e_erro(root, tmp_path, monkeypatch):
    monkeypatch.setattr(calibration_tk, "find_all_disks", lambda frame, config: DETECTED)
    app = build(root, make_samples(tmp_path))
    try:
        app._queue.put(("detect", app._job_id + 99, DETECTED, None))
        pump(app, root)
        assert app.shapes == []

        app._queue.put(("detect", app._job_id, None, RuntimeError("boom")))
        pump(app, root)
        assert "Erro na deteção" in app.status.get()
    finally:
        app.root.destroy()


def test_detect_now_alternador_e_parametros(root, tmp_path, monkeypatch):
    monkeypatch.setattr(calibration_tk, "find_all_disks", lambda frame, config: DETECTED)
    app = build(root, make_samples(tmp_path))
    try:
        app.frame = None
        app.detect_now()
        assert not app._busy
        app.frame = app.frame or None
        app.load_sample(0)

        app.auto_detect.set(True)
        app._on_auto_detect_toggle()
        pump(app, root)
        assert app.shapes == DETECTED

        app._on_auto_detect_toggle()
        assert "Deteção ligada" in app.status.get()

        app.auto_detect.set(False)
        app._on_auto_detect_toggle()
        assert "desligada (1.ª passagem manual)" in app.status.get()

        app.param2_var.set(30)
        app._on_param_slider("30")
        assert app.config.stabilizer.param2 == 30

        app.auto_detect.set(True)
        app.param2_var.set(31)
        app._on_param_slider("31")
        assert app._busy
        pump(app, root)
        assert app.config.stabilizer.param2 == 31

        app.detect_now()
        assert "A detetar…" in app.status.get()
        pump(app, root)
        assert "1 disco(s) detetado(s)" in app.status.get()
    finally:
        app.root.destroy()


# ------------------------------------------------------- amostras/store --


def test_carrega_ground_truth_guardado(root, tmp_path):
    app = build(root, make_samples(tmp_path, gt=True))
    try:
        assert app.shapes == [DiskDetection(300, 140, 90)]
        assert app.selected is None
        assert "Ground truth carregado" in app.status.get()
    finally:
        app.root.destroy()


def test_erro_ao_carregar_imagem_corrompida(root, tmp_path):
    samples = make_samples(tmp_path)
    (samples / "sample_0.jpg").write_bytes(b"nao e uma imagem")
    app = build(root, samples)
    try:
        assert not app._busy
        assert "Erro ao carregar" in app.status.get()
    finally:
        app.root.destroy()


def test_goto_listbox_selecao_e_sem_amostras(root, tmp_path):
    app = build(root, make_samples(tmp_path, n=2))
    try:
        app.goto(+1)
        assert app.current_index == 1
        app.goto(-1)
        assert app.current_index == 0

        app.listbox.selection_clear(0, calibration_tk.tk.END)
        app.listbox.selection_set(1)
        app._on_listbox_select()
        assert app.current_index == 1
    finally:
        app.root.destroy()

    root2 = calibration_tk.tk.Tk()
    root2.withdraw()
    app = calibration_tk.build_app(root2, samples_dir=str(tmp_path / "vazio"))
    try:
        assert app.samples == []
        assert app.current_index == 0
        app.load_sample(0)
        app.goto(+1)
        assert app.current_index == 0
    finally:
        app.root.destroy()


# ---------------------------------------------------------- zoom e pan --


def test_zoom_rodinha_e_pan(root, tmp_path, monkeypatch):
    app = build(root, make_samples(tmp_path), monkeypatch)
    try:
        before = app.scale
        app._zoom_at(450, 350, 2.0)
        assert app.scale == pytest.approx(before * 2.0)
        app._zoom_at(450, 350, 0.5)
        assert app.scale == pytest.approx(before)

        app._on_wheel(types.SimpleNamespace(delta=-1, x=100, y=100))
        assert app.scale > before
        app._on_wheel(types.SimpleNamespace(delta=1, x=100, y=100))
        assert app.scale == pytest.approx(before)

        app._on_drag_pan(types.SimpleNamespace(x=0, y=0))
        ox0, oy0 = app.ox, app.oy
        app._on_press_pan(types.SimpleNamespace(x=10, y=20))
        app._on_drag_pan(types.SimpleNamespace(x=30, y=40))
        assert app.ox == pytest.approx(ox0 + 20)
        assert app.oy == pytest.approx(oy0 + 20)

        app.frame = None
        app._zoom_at(450, 350, 2.0)
        app.fit_view()
        app.redraw()
    finally:
        app.root.destroy()


# --------------------------------------------------------- teclado etc. --


def test_eliminar_escape_e_nudge(root, tmp_path, monkeypatch):
    app = build(root, make_samples(tmp_path), monkeypatch)
    try:
        press(app, 100, 100)
        app._nudge(1, 0, types.SimpleNamespace(state=0))
        assert app.shapes[0].cx == 101
        app._nudge(1, 0, types.SimpleNamespace(state=1))
        assert app.shapes[0].cx == 111

        app.selected = None
        app._nudge(1, 0, types.SimpleNamespace(state=0))
        assert app.shapes[0].cx == 111

        app.selected = 0
        app._on_delete()
        assert app.shapes == []
        app._on_delete()
        assert app.shapes == []

        press(app, 200, 200)
        assert app.selected == 0
        app._on_escape()
        assert app.selected is None
    finally:
        app.root.destroy()


def test_sliders_de_raio_sincronizam_a_selecao(root, tmp_path, monkeypatch):
    app = build(root, make_samples(tmp_path), monkeypatch)
    try:
        press(app, 100, 100)
        app._on_rx_slider("60")
        assert app.shapes[0].radius == 60
        app._on_ry_slider("50")
        assert app.shapes[0].ry == 50
        app._on_ry_slider("30")
        assert app.shapes[0].ry == 30
        app._on_rx_slider("1")
        assert app.shapes[0].radius == 2
        app._on_rx_slider("6000")
        assert app.shapes[0].radius == 5000

        app.selected = None
        app._sync_sliders()
        assert app.shape_info.cget("text") == ""
        app._on_rx_slider("40")
        assert app.shapes[0].radius == 5000
        app._syncing = True
        app._on_rx_slider("40")
        assert app.shapes[0].radius == 5000
        app._syncing = False
        app.selected = 0
        app._on_ry_slider("40")
        assert app.shapes[0].ry == 40
        app.selected = None
        app._on_ry_slider("40")
        assert app.shapes[0].ry == 40
    finally:
        app.root.destroy()


# ------------------------------------------------------ guardar/validar --


def test_guardar_escreve_ground_truth(root, tmp_path, monkeypatch):
    app = build(root, make_samples(tmp_path), monkeypatch)
    try:
        press(app, 240, 180)
        app.save()
        assert "Guardado ✓ (1 forma(s))" in app.status.get()
        store = CalibrationStore(tmp_path / "samples" / "calibration.json")
        item = store.get_item("sample_0.jpg")
        assert item is not None
        assert item.circles == [DiskDetection(240, 180, 40)]
        assert (item.width, item.height) == (480, 360)
        assert item.kind == "image" and item.frame is None

        app.frame = None
        app.save()
        app.frame = None
        app.samples = []
        app.save()
    finally:
        app.root.destroy()

    root2 = calibration_tk.tk.Tk()
    root2.withdraw()
    samples2 = scan_samples(tmp_path / "samples")
    store2 = CalibrationStore(tmp_path / "samples" / "calibration.json")
    app2 = calibration_tk.CalibrationTkApp(root2, samples2, store2, AstroFrameConfig())
    try:
        app2.shape_kind.set("ellipse")
        press(app2, 100, 80)
        app2.save()
        assert "Guardado ✓" in app2.status.get()
        item2 = store2.get_item("sample_0.jpg")
        assert item2 is not None and "sample_0.jpg" in item2.path
        assert item2.circles == [DiskDetection(240, 180, 40), DiskDetection(100, 80, 60, 40)]
    finally:
        app2.root.destroy()


def test_validar_tudo_report_erros_e_sugestoes(root, tmp_path, monkeypatch):
    samples = make_samples(tmp_path, n=2)
    store = CalibrationStore(samples / "calibration.json")
    store.items["sample_0.jpg"] = CalibrationItem(
        "sample_0.jpg", "image", None, 480, 360, [DiskDetection(300, 140, 90)]
    )
    store.save()
    (samples / "sample_1.jpg").write_bytes(b"corrompida")
    monkeypatch.setattr(
        calibration_tk, "find_all_disks", lambda frame, config: [DiskDetection(301, 140, 89)]
    )
    app = build(root, samples)
    try:
        app._busy = True
        app.validate_all()
        assert app._busy
        app._busy = False

        app.validate_all()
        assert "A validar todas as amostras" in app.report.get("1.0", "end")
        pump(app, root)
        text = app.report.get("1.0", "end")
        assert "Score global" in text
        assert "Recall 100%" in text
        assert "sample_0.jpg: 1/1" in text
        assert "corrompida" in text or "erro" in text
        assert "Validação concluída." in app.status.get()
        assert not app._busy

        app._queue.put(("validate", app._job_id + 7, None, []))
        pump(app, root)

        app._queue.put(("validate", app._job_id + 1, None, []))
        pump(app, root)
        assert "Erro na validação." in app.status.get()
    finally:
        app.root.destroy()


# ---------------------------------------------------------- run/build --


def test_build_app_com_config_path(root, tmp_path):
    config_path = tmp_path / "config.yaml"
    AstroFrameConfig().to_yaml(config_path)
    samples = make_samples(tmp_path)
    app = calibration_tk.build_app(root, samples_dir=str(samples), config_path=str(config_path))
    try:
        assert isinstance(app.config, AstroFrameConfig)
        assert app.samples
    finally:
        app.root.destroy()


def test_run_mainloop_e_lancamento_da_janela(root, tmp_path, monkeypatch):
    recorded: dict = {}

    app = build(root, make_samples(tmp_path))
    monkeypatch.setattr(app.root, "mainloop", lambda: recorded.__setitem__("mainloop", True))
    app.run()
    assert recorded["mainloop"] is True
    app.root.destroy()

    fake_root = types.SimpleNamespace()
    monkeypatch.setattr(calibration_tk.tk, "Tk", lambda: fake_root)

    class FakeApp:
        def run(self):
            recorded["run_called"] = True

    def fake_build(root, samples_dir, config_path=None):
        recorded["root"] = root
        recorded["dir"] = samples_dir
        recorded["config"] = config_path
        return FakeApp()

    monkeypatch.setattr(calibration_tk, "build_app", fake_build)
    calibration_tk.run(samples_dir="samples", config_path=None)
    assert recorded["root"] is fake_root
    assert recorded["dir"] == "samples"
    assert recorded["config"] is None
    assert recorded["run_called"] is True
