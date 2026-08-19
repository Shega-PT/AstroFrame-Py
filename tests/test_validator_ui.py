"""Testes da UI desktop da validação (Tk real, janela oculta).

A suíte usa `DISPLAY` (Tk real com a janela retirada do ecrã) para exercitar
`ValidatorTkApp`, `AutoTrainWindow`, `FinalReportWindow` e `Tooltip` — o
loop de eventos é bombeado manualmente com `root.update()` e a deteção é
substituída por `monkeypatch` para ser determinística.
"""

from __future__ import annotations

import json
import time
import types
from pathlib import Path

import cv2
import pytest
import validator

from astroframe.calibration.store import CalibrationItem, CalibrationStore
from astroframe.calibration.validate import validate_all
from astroframe.core.stabilizer import DiskDetection
from astroframe.paths import train_dir
from tests.helpers import make_disk_image

CIRCLE = DiskDetection(300, 140, 90)
GHOST = DiskDetection(120, 60, 25)


@pytest.fixture()
def root():
    tk_root = validator.tk.Tk()
    tk_root.withdraw()
    yield tk_root
    try:
        tk_root.destroy()
    except validator.tk.TclError:
        pass


@pytest.fixture(autouse=True)
def _restore_weights():
    before = (dict(validator.REWARD_DELTAS), dict(validator.PUNISH_DELTAS))
    yield
    validator.REWARD_DELTAS.clear()
    validator.REWARD_DELTAS.update(before[0])
    validator.PUNISH_DELTAS.clear()
    validator.PUNISH_DELTAS.update(before[1])


def make_samples_dir(tmp_path: Path, n: int = 2) -> Path:
    """Pasta `samples` com `n` imagens + ground truth (calibration.json)."""
    root = tmp_path / "samples"
    root.mkdir()
    store = CalibrationStore(root / "calibration.json")
    for i in range(n):
        name = f"sample_{i}.jpg"
        image, cx, cy = make_disk_image()
        cv2.imwrite(str(root / name), image)
        store.items[name] = CalibrationItem(name, "image", None, 480, 360, [CIRCLE])
    store.save()
    return root


def pump_detect(app, root, timeout: float = 10.0) -> None:
    """Espera a deteção em thread terminar e entrega a fila à UI."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline and app._busy:
        root.update()
        time.sleep(0.01)
    app._poll_queue()
    root.update()
    assert not app._busy, "deteção não terminou a tempo"


def flush(root, seconds: float = 0.8) -> None:
    """Processa os eventos pendentes (ex.: `after(600, ...)`)."""
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        root.update()
        time.sleep(0.01)


@pytest.fixture()
def detect_one(monkeypatch):
    """Deteção determinística: devolve sempre [CIRCLE]."""
    monkeypatch.setattr(validator, "find_all_disks", lambda _f, _c: [CIRCLE])


@pytest.fixture()
def fake_final_window(monkeypatch):
    """Substitui a janela do relatório final por um gravador (sem Toplevel)."""
    recorded: list[tuple] = []

    class FakeFinal:
        def __init__(self, app, report, state, lines):
            recorded.append((report, state, lines))

    monkeypatch.setattr(validator, "FinalReportWindow", FakeFinal)
    return recorded


# --------------------------------------------------------------- fluxo manual --


def test_app_fluxo_completo_deteccao_julgamento_e_relatorio(root, tmp_path, detect_one, fake_final_window):
    samples = make_samples_dir(tmp_path)
    app = validator.build_app(root, samples_dir=str(samples), state_path=str(samples / "v.json"))
    try:
        assert app.state.round == 1
        pump_detect(app, root)
        assert app.session.current == CIRCLE
        assert "pendente" in app.status.get()
        assert str(app.valid_btn.cget("state")) == "normal"

        app.accept()
        pump_detect(app, root)
        assert app.state.rewards == 1
        assert "completa" in app.status.get()

        flush(root)
        pump_detect(app, root)
        app.accept()
        pump_detect(app, root)
        assert app.state.done_count(app.samples) == 2
        assert len(fake_final_window) == 1
        assert (samples / validator.DEFAULT_EXPORT_NAME).exists()
        assert app.state.rounds[-1]["ended"] is not None
    finally:
        app.root.destroy()


def test_app_rejeicao_reavalia_e_termina(root, tmp_path, fake_final_window, monkeypatch):
    samples = make_samples_dir(tmp_path, n=1)
    calls = {"n": 0}

    def fake_detect(_frame, _config):
        calls["n"] += 1
        return [CIRCLE, GHOST] if calls["n"] == 1 else [CIRCLE]

    monkeypatch.setattr(validator, "find_all_disks", fake_detect)
    app = validator.build_app(root, samples_dir=str(samples), state_path=str(samples / "v.json"))
    try:
        pump_detect(app, root)
        assert len(app.session.pending) == 2
        app.accept()
        pump_detect(app, root)
        assert app.session.current == GHOST
        app.reject()
        pump_detect(app, root)
        assert app.state.rewards == 1
        assert app.state.punishments == 1
        assert "Treino concluído" in app.status.get()
        assert app.state.is_done("sample_0.jpg")
    finally:
        app.root.destroy()


def test_app_recomecar_amostra(root, tmp_path, detect_one):
    samples = make_samples_dir(tmp_path, n=1)
    app = validator.build_app(root, samples_dir=str(samples), state_path=str(samples / "v.json"))
    try:
        pump_detect(app, root)
        app.accept()
        pump_detect(app, root)
        record = app.state.record("sample_0.jpg")
        assert record["accepted"] == [CIRCLE]
        app.restart_sample()
        pump_detect(app, root)
        assert app.state.record("sample_0.jpg")["accepted"] == []
        assert app.session.current == CIRCLE
    finally:
        app.root.destroy()


def test_app_novo_treino_com_e_sem_confirmacao(root, tmp_path, detect_one, monkeypatch):
    samples = make_samples_dir(tmp_path, n=1)
    monkeypatch.setattr(validator.messagebox, "askyesno", lambda *a, **k: False)
    app = validator.build_app(root, samples_dir=str(samples), state_path=str(samples / "v.json"))
    try:
        pump_detect(app, root)
        app.new_round()
        assert app.state.round == 1
        monkeypatch.setattr(validator.messagebox, "askyesno", lambda *a, **k: True)
        app.new_round()
        assert app.state.round == 2
        assert "Série 2" in app.progress_label.cget("text")
    finally:
        app.root.destroy()


def test_app_params_text_e_labels(root, tmp_path, detect_one):
    samples = make_samples_dir(tmp_path, n=1)
    app = validator.build_app(root, samples_dir=str(samples), state_path=str(samples / "v.json"))
    try:
        app.state.deltas = {"param2": 5.0, "dp": 0.3}
        text = app._params_text()
        assert "param2=35 (+5)" in text
        assert "dp=1.50 (+0.30)" in text
        app._update_labels()
        assert "Série 1" in app.progress_label.cget("text")
        assert "Recompensas: 0" in app.stats_label.cget("text")
    finally:
        app.root.destroy()


def test_app_sem_amostras(root, tmp_path):
    samples = tmp_path / "samples"
    samples.mkdir()
    app = validator.build_app(root, samples_dir=str(samples), state_path=str(samples / "v.json"))
    try:
        assert "Sem amostras" in app.status.get()
        assert app._first_undone() == 0
        app.goto_next_undone()
        app.goto(1, relative=True)
    finally:
        app.root.destroy()


def test_app_desenho_zoom_e_pan(root, tmp_path, detect_one, monkeypatch):
    samples = make_samples_dir(tmp_path, n=1)
    app = validator.build_app(root, samples_dir=str(samples), state_path=str(samples / "v.json"))
    try:
        monkeypatch.setattr(app.canvas, "winfo_width", lambda: 900)
        monkeypatch.setattr(app.canvas, "winfo_height", lambda: 700)
        pump_detect(app, root)
        app.session.accepted = [CIRCLE]
        app.session.rejected = [GHOST]
        app.redraw()
        assert len(app.canvas.find_all()) > 0
        app.fit_view()
        app._zoom_at(10, 10, 1.1)
        app._on_wheel(types.SimpleNamespace(delta=1, x=5, y=5))
        app._on_press_pan(types.SimpleNamespace(x=10, y=10))
        app._on_drag_pan(types.SimpleNamespace(x=30, y=40))
        app._report("resumo")
        assert "resumo" in app.report.get("1.0", "end")
    finally:
        app.root.destroy()


# ------------------------------------------------------------ treino automático --


def _fake_run_series_factory(window):
    def fake_run_series(self, progress=None, should_stop=None, on_detect=None):
        record = self.state.record(self.samples[0].key)
        record["accepted"] = [CIRCLE]
        record["done"] = True
        self.state.save()
        frame = make_disk_image()[0]
        if on_detect is not None:
            on_detect(frame, [CIRCLE], "sample_0.jpg", 0.90)
        if progress is not None:
            progress(1, 1, "sample_0.jpg", 0.90)
        return validator.AutoSeriesReport(
            series=1,
            samples_done=1,
            samples_total=1,
            rewards=1,
            punishments=0,
            threshold_end=0.90,
            report=None,
            errors=["leitura falhada"],
        )

    return fake_run_series


def test_auto_train_window_preview_e_sessao(root, tmp_path, fake_final_window, monkeypatch):
    samples = make_samples_dir(tmp_path, n=1)
    app = validator.build_app(root, samples_dir=str(samples), state_path=str(samples / "v.json"))
    win = validator.AutoTrainWindow(app)
    try:
        assert win._preview_enabled
        frame = make_disk_image()[0]
        win._preview_msg(frame, [CIRCLE], "a", 0.90)
        win._preview_msg(frame, [CIRCLE], "a", 0.90)
        assert win._queue.qsize() == 1  # throttle: a segunda é descartada
        win._poll()
        root.update()
        assert len(win.preview.find_all()) > 0

        win.preview_var.set(False)
        root.update()
        assert not win._preview_enabled
        win._preview_msg(frame, [CIRCLE], "a", 0.90)
        assert win._queue.qsize() == 0
        win.preview_var.set(True)

        monkeypatch.setattr(validator.AutoTrainer, "run_series", _fake_run_series_factory(win))
        win.series_var.set(2)
        win.start()
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline and win._running:
            win._poll()
            root.update()
            time.sleep(0.01)
        win._poll()
        root.update()
        assert not win._running
        assert "concluído" in win.status.get()
        assert len(fake_final_window) == 1
        assert win._preview_label == "sample_0.jpg"
    finally:
        win._on_close()
        app.root.destroy()


def test_auto_train_window_parar_e_erro(root, tmp_path, monkeypatch):
    samples = make_samples_dir(tmp_path, n=1)
    app = validator.build_app(root, samples_dir=str(samples), state_path=str(samples / "v.json"))

    def failing_run_series(self, progress=None, should_stop=None, on_detect=None):
        raise RuntimeError("falha sintética")

    monkeypatch.setattr(validator.AutoTrainer, "run_series", failing_run_series)
    win = validator.AutoTrainWindow(app)
    try:
        win.stop()
        assert win._stop
        win.start()
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline and win._running:
            win._poll()
            root.update()
            time.sleep(0.01)
        win._poll()
        root.update()
        assert not win._running
        assert "Erro no treino automático" in win.status.get()
    finally:
        win._on_close()
        app.root.destroy()


# ------------------------------------------------------- relatório final (pop-up) --


def test_final_report_window_aplica_e_salva_pesos(root, tmp_path, detect_one):
    samples = make_samples_dir(tmp_path, n=1)
    app = validator.build_app(root, samples_dir=str(samples), state_path=str(samples / "v.json"))
    report = validate_all([("sample_0.jpg", [CIRCLE], [CIRCLE])])
    win = validator.FinalReportWindow(app, report, app.state, ["linha 1", "linha 2"])
    try:
        assert win.WEIGHT_KEYS == validator.TRAINABLE_PARAMS
        assert len(win._vars) == 2 * len(validator.TRAINABLE_PARAMS)
        assert "linha 2" in win.text.get("1.0", "end")

        win._vars[("reward", "param2")].set(-1.5)
        win._vars[("punish", "occluded_ratio")].set(0.02)
        win.iou_var.set(0.97)
        win.apply()
        assert validator.REWARD_DELTAS["param2"] == -1.5
        assert "aplicados" in win.status_var.get()

        win.save()
        data = json.loads(app.state.path.read_text(encoding="utf-8"))
        assert data["weights"]["reward"]["param2"] == -1.5
        assert data["weights"]["iou"] == 0.97

        loaded = validator.ValidatorState(app.state.path)
        assert loaded.weights["reward"]["param2"] == -1.5
        loaded.reset()
        assert loaded.weights["reward"]["param2"] == -1.5
        validator.apply_state_weights(loaded)
        assert validator.REWARD_DELTAS["param2"] == -1.5
        assert validator.state_iou(loaded) == pytest.approx(0.97)
    finally:
        win.top.destroy()
        app.root.destroy()


def test_final_report_window_sem_relatorio(root, tmp_path):
    samples = make_samples_dir(tmp_path, n=1)
    app = validator.build_app(root, samples_dir=str(samples), state_path=str(samples / "v.json"))
    win = validator.FinalReportWindow(app, None, app.state, ["linha de aviso"])
    try:
        assert "linha de aviso" in win.text.get("1.0", "end")
        status_label = win.top.winfo_children()[0].winfo_children()[0]
        assert "Sem guia" in status_label.cget("text")
        win._load_weights()
        assert win._vars[("reward", "param2")].get() == validator.REWARD_DELTAS["param2"]
    finally:
        win.top.destroy()
        app.root.destroy()


def test_tooltip_mostra_e_esconde(root):
    label = validator.ttk.Label(root, text="ⓘ")
    tooltip = validator.Tooltip(label, "ajuda")
    try:
        tooltip._show(types.SimpleNamespace(x_root=50, y_root=60))
        assert tooltip._top is not None
        top = tooltip._top
        tooltip._show(types.SimpleNamespace(x_root=51, y_root=61))
        assert tooltip._top is top
        tooltip._hide()
        assert tooltip._top is None
    finally:
        label.destroy()


# ------------------------------------------------------- modos sem janela (CLI) --


def test_run_check_imprime_relatorio(root, tmp_path, monkeypatch, capsys):
    samples = make_samples_dir(tmp_path, n=1)
    monkeypatch.setattr(validator, "find_all_disks", lambda _f, _c: [CIRCLE])
    state = validator.ValidatorState(samples / "v.json")
    state.deltas = {"param2": 5.0}
    state.save()
    validator.run_check(str(samples), state_path=str(state.path))
    out = capsys.readouterr().out
    assert "Deltas aprendidos: param2 +5.00" in out
    assert "Score vs guia manual: 100.0/100" in out


def test_run_auto_headless_exporta(root, tmp_path, monkeypatch, capsys):
    samples = make_samples_dir(tmp_path, n=1)
    monkeypatch.setattr(validator, "find_all_disks", lambda _f, _c: [CIRCLE])
    state = validator.ValidatorState(samples / "v.json")
    state.deltas = {"param2": 5.0}
    state.weights["iou"] = 0.95
    state.save()
    validator.run_auto_headless(str(samples), state_path=str(state.path), series=2)
    out = capsys.readouterr().out
    assert "treino automático (1 amostras, 2 série(s)" in out
    assert "IoU mínimo 0.95" in out
    data = json.loads((train_dir() / validator.DEFAULT_EXPORT_NAME).read_text(encoding="utf-8"))
    # delta 5.0 menos 2 recompensas de 0.25 (uma por série) = 4.5 → 34
    assert data["stabilizer"]["param2"] == 34


def test_main_cli_check_auto_reset_e_erro(root, tmp_path, monkeypatch, capsys):
    samples = make_samples_dir(tmp_path, n=1)
    monkeypatch.setattr(validator, "find_all_disks", lambda _f, _c: [CIRCLE])
    args = ["--samples", str(samples), "--state", str(samples / "v.json")]

    assert validator.main(args + ["--check"]) == 0
    assert validator.main(args + ["--auto", "--series", "1"]) == 0
    assert (train_dir() / validator.DEFAULT_EXPORT_NAME).exists()

    assert validator.main(args + ["--reset-state", "--check"]) == 0
    out = capsys.readouterr().out
    assert "Estado de validação reposto" in out
    state = validator.ValidatorState(samples / "v.json")
    assert state.deltas == {}

    monkeypatch.setattr(validator, "run", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x")))
    assert validator.main(["--samples", str(samples), "--state", str(samples / "v.json")]) == 1


def test_build_parser_iou_default_none():
    parser = validator.build_parser()
    args = parser.parse_args(["--samples", "s"])
    assert args.iou is None
    assert args.series == 3


# ------------------------------------------------------- ramos de cobertura (UI) --


def test_app_selecao_na_listbox_carrega_amostra(root, tmp_path, detect_one):
    samples = make_samples_dir(tmp_path, n=1)
    app = validator.build_app(root, samples_dir=str(samples), state_path=str(samples / "v.json"))
    try:
        pump_detect(app, root)
        app.listbox.selection_set(0)
        app._on_listbox_select()
        assert "A analisar imagem" in app.status.get()
        pump_detect(app, root)
    finally:
        app.root.destroy()


def test_app_erro_de_carga_mostra_mensagem(root, tmp_path, detect_one, monkeypatch):
    samples = make_samples_dir(tmp_path, n=1)

    def broken_load(_sample):
        raise ValueError("ficheiro partido")

    monkeypatch.setattr(validator, "load_frame", broken_load)
    app = validator.build_app(root, samples_dir=str(samples), state_path=str(samples / "v.json"))
    try:
        app.goto(0)
        assert "Erro ao carregar" in app.status.get()
        assert app.frame is None
    finally:
        app.root.destroy()


def test_app_amostra_ja_completa_nao_reanalisa(root, tmp_path, detect_one):
    samples = make_samples_dir(tmp_path, n=1)
    app = validator.build_app(root, samples_dir=str(samples), state_path=str(samples / "v.json"))
    try:
        record = app.state.record("sample_0.jpg")
        record["accepted"] = [CIRCLE]
        record["done"] = True
        app.state.save()
        app.goto(0)
        assert "Amostra completa" in app.status.get()
        assert not app._busy
    finally:
        app.root.destroy()


def test_app_deteçao_obsoleta_ignorada(root, tmp_path, detect_one):
    samples = make_samples_dir(tmp_path, n=1)
    app = validator.build_app(root, samples_dir=str(samples), state_path=str(samples / "v.json"))
    try:
        app._job_id += 1
        app._on_detect_done(app._job_id - 1, [CIRCLE], None)
        assert app._busy
    finally:
        app.root.destroy()


def test_app_deteçao_com_erro_mostra_mensagem(root, tmp_path, detect_one):
    samples = make_samples_dir(tmp_path, n=1)
    app = validator.build_app(root, samples_dir=str(samples), state_path=str(samples / "v.json"))
    try:
        app._on_detect_done(app._job_id, None, RuntimeError("falhou"))
        assert "Erro na deteção" in app.status.get()
        assert not app._busy
    finally:
        app.root.destroy()


def test_app_deteçao_teimosa_reavalia_ate_completar(root, tmp_path, fake_final_window, monkeypatch):
    samples = make_samples_dir(tmp_path, n=1)
    monkeypatch.setattr(validator, "find_all_disks", lambda _f, _c: [GHOST])
    app = validator.build_app(root, samples_dir=str(samples), state_path=str(samples / "v.json"))
    try:
        pump_detect(app, root)
        assert app.session.current == GHOST
        app.reject()
        deadline = time.monotonic() + 30.0
        while not app.state.is_done("sample_0.jpg") and time.monotonic() < deadline:
            pump_detect(app, root)
        assert app.state.is_done("sample_0.jpg")
        assert app.state.punishments >= 2
        assert "Treino concluído" in app.status.get()
    finally:
        app.root.destroy()


def test_app_aceitar_e_rejeitar_em_busca_ignorados(root, tmp_path, detect_one):
    samples = make_samples_dir(tmp_path, n=1)
    app = validator.build_app(root, samples_dir=str(samples), state_path=str(samples / "v.json"))
    try:
        app.accept()
        app.reject()
        assert app.state.rewards == 0
        assert app.state.punishments == 0
    finally:
        app.root.destroy()


def test_app_acao_completa_finaliza_amostra(root, tmp_path, detect_one, fake_final_window):
    samples = make_samples_dir(tmp_path, n=2)
    app = validator.build_app(root, samples_dir=str(samples), state_path=str(samples / "v.json"))
    try:
        pump_detect(app, root)
        app._handle_action("complete")
        assert app.state.is_done("sample_0.jpg")
        assert "Amostra completa" in app.status.get()
        flush(root)
    finally:
        app.root.destroy()


def test_app_present_pending_sem_pendentes_finaliza(root, tmp_path, detect_one):
    samples = make_samples_dir(tmp_path, n=1)
    app = validator.build_app(root, samples_dir=str(samples), state_path=str(samples / "v.json"))
    try:
        pump_detect(app, root)
        app.session.pending = []
        app._present_pending()
        assert app.state.is_done("sample_0.jpg")
    finally:
        app.root.destroy()


def test_app_gt_hint_sem_guia(root, tmp_path, detect_one):
    samples = tmp_path / "samples"
    samples.mkdir()
    image, _, _ = make_disk_image()
    cv2.imwrite(str(samples / "a.jpg"), image)
    app = validator.build_app(root, samples_dir=str(samples), state_path=str(samples / "v.json"))
    try:
        pump_detect(app, root)
        assert "Sem guia" in app.gt_hint.cget("text")
    finally:
        app.root.destroy()


def test_app_relatorio_final_sem_amostras_concluidas(root, tmp_path, detect_one):
    samples = make_samples_dir(tmp_path, n=1)
    app = validator.build_app(root, samples_dir=str(samples), state_path=str(samples / "v.json"))
    try:
        pump_detect(app, root)
        app.accept()
        pump_detect(app, root)
        app.state.clear_progress()
        app._show_final_report()
        assert "Sem amostras concluídas" in app.report.get("1.0", "end")
    finally:
        app.root.destroy()


def test_app_novo_treino_em_busca_ignorado(root, tmp_path, detect_one):
    samples = make_samples_dir(tmp_path, n=1)
    app = validator.build_app(root, samples_dir=str(samples), state_path=str(samples / "v.json"))
    try:
        app.new_round()
        assert app.state.round == 1
    finally:
        app.root.destroy()


def test_app_open_auto_train_durante_treino_e_normal(root, tmp_path, detect_one, monkeypatch):
    samples = make_samples_dir(tmp_path, n=1)
    app = validator.build_app(root, samples_dir=str(samples), state_path=str(samples / "v.json"))
    created = []

    def fake_window(owner):
        created.append(owner)

    monkeypatch.setattr(validator, "AutoTrainWindow", fake_window)
    try:
        app._training = True
        app.open_auto_train()
        assert created == []
        app._training = False
        app.open_auto_train()
        assert created == [app]
    finally:
        app.root.destroy()


def test_app_restart_amostra_em_busca_ignorado(root, tmp_path, detect_one):
    samples = make_samples_dir(tmp_path, n=1)
    app = validator.build_app(root, samples_dir=str(samples), state_path=str(samples / "v.json"))
    try:
        app.restart_sample()
        assert app.state.record("sample_0.jpg")["accepted"] == []
    finally:
        app.root.destroy()


def test_app_sem_imagem_metodos_de_desenho_sao_noops(root, tmp_path):
    samples = tmp_path / "samples"
    samples.mkdir()
    app = validator.build_app(root, samples_dir=str(samples), state_path=str(samples / "v.json"))
    try:
        assert app.frame is None
        app.fit_view()
        app._zoom_at(10, 10, 1.1)
        app._on_wheel(types.SimpleNamespace(delta=-1, x=5, y=5))
        app._on_drag_pan(types.SimpleNamespace(x=3, y=4))
        app._on_press_pan(types.SimpleNamespace(x=1, y=1))
        app._on_drag_pan(types.SimpleNamespace(x=3, y=4))
        app.redraw()
    finally:
        app.root.destroy()


def test_app_run_encerra_com_quit(root, tmp_path, detect_one):
    samples = make_samples_dir(tmp_path, n=1)
    app = validator.build_app(root, samples_dir=str(samples), state_path=str(samples / "v.json"))
    try:
        root.after(100, root.quit)
        app.run()
    finally:
        app.root.destroy()


def test_run_funcao_abre_e_fecha_janela(tmp_path, monkeypatch):
    samples = tmp_path / "samples"
    samples.mkdir()

    class FakeApp:
        def __init__(self, root):
            self.root = root

        def run(self):
            self.root.after(20, self.root.destroy)
            self.root.mainloop()

    monkeypatch.setattr(validator, "build_app", lambda root, **kw: FakeApp(root))
    validator.run(str(samples), state_path=str(samples / "v.json"))


def test_run_check_com_erro_de_leitura(tmp_path, monkeypatch, capsys):
    samples = make_samples_dir(tmp_path, n=1)

    def broken_load(_sample):
        raise OSError("ficheiro partido")

    monkeypatch.setattr(validator, "load_frame", broken_load)
    assert validator.run_check(str(samples), state_path=str(samples / "v.json")) == 0
    out = capsys.readouterr().out
    assert "sample_0.jpg" in out and "erro" in out.lower()


def test_run_auto_headless_com_erro_de_leitura(tmp_path, monkeypatch, capsys):
    samples = make_samples_dir(tmp_path, n=1)

    def broken_load(_sample):
        raise OSError("ficheiro partido")

    monkeypatch.setattr(validator, "load_frame", broken_load)
    assert validator.run_auto_headless(str(samples), state_path=str(samples / "v.json"), series=1) == 0
    out = capsys.readouterr().out
    assert "! IMG" in out and "erro ao ler" in out


def test_main_run_sem_erro_devolve_0(tmp_path, monkeypatch):
    samples = tmp_path / "samples"
    samples.mkdir()
    monkeypatch.setattr(validator, "run", lambda *a, **k: None)
    assert validator.main(["--samples", str(samples)]) == 0


def test_main_module_guard(tmp_path, monkeypatch):
    import runpy

    samples = tmp_path / "samples"
    samples.mkdir()
    monkeypatch.setattr(validator, "run", lambda *a, **k: None)
    monkeypatch.setattr(validator.sys, "argv", ["validator", "--samples", str(samples)])
    with pytest.raises(SystemExit):
        runpy.run_path(str(validator.__file__), run_name="__main__")


# --------------------------------------------------- ramos (janela de treino auto) --


def test_auto_train_window_iou_slider_atualiza_limiar(root, tmp_path, detect_one):
    samples = make_samples_dir(tmp_path, n=1)
    app = validator.build_app(root, samples_dir=str(samples), state_path=str(samples / "v.json"))
    win = validator.AutoTrainWindow(app)
    try:
        win.iou_var.set(0.93)
        win._on_iou_slider()
        assert win.app.auto_iou_min == pytest.approx(0.93)
        assert win.iou_label.cget("text") == "0.93"
    finally:
        win.top.destroy()
        app.root.destroy()


def test_auto_train_window_start_duplo_ignorado(root, tmp_path, monkeypatch):
    samples = make_samples_dir(tmp_path, n=1)
    app = validator.build_app(root, samples_dir=str(samples), state_path=str(samples / "v.json"))
    win = validator.AutoTrainWindow(app)
    monkeypatch.setattr(validator.AutoTrainer, "run_series", _fake_run_series_factory(win))
    try:
        win.start()
        win.start()
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline and win._running:
            win._poll()
            root.update()
            time.sleep(0.01)
        win._poll()
        root.update()
        assert not win._running
        assert "concluído" in win.status.get()
    finally:
        win._on_close()
        app.root.destroy()


def test_auto_train_window_parar_antes_da_primeira_serie(root, tmp_path, monkeypatch):
    samples = make_samples_dir(tmp_path, n=1)
    app = validator.build_app(root, samples_dir=str(samples), state_path=str(samples / "v.json"))
    win = validator.AutoTrainWindow(app)
    monkeypatch.setattr(validator.AutoTrainer, "run_series", _fake_run_series_factory(win))

    real_thread = validator.threading.Thread

    class StopThread(real_thread):
        def start(self):
            win._stop = True
            return super().start()

    monkeypatch.setattr(validator.threading, "Thread", StopThread)
    try:
        win.start()
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline and win._running:
            win._poll()
            root.update()
            time.sleep(0.01)
        win._poll()
        root.update()
        assert not win._running
        assert "interrompido" in win.status.get()
    finally:
        win._on_close()
        app.root.destroy()


def test_auto_train_window_parar_entre_series(root, tmp_path, monkeypatch):
    samples = make_samples_dir(tmp_path, n=1)
    app = validator.build_app(root, samples_dir=str(samples), state_path=str(samples / "v.json"))
    win = validator.AutoTrainWindow(app)

    def stopping_run_series(self, progress=None, should_stop=None, on_detect=None):
        win._stop = True
        return validator.AutoSeriesReport(
            series=1,
            samples_done=1,
            samples_total=1,
            rewards=0,
            punishments=0,
            threshold_end=0.90,
        )

    monkeypatch.setattr(validator.AutoTrainer, "run_series", stopping_run_series)
    try:
        win.series_var.set(2)
        win.start()
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline and win._running:
            win._poll()
            root.update()
            time.sleep(0.01)
        win._poll()
        root.update()
        assert not win._running
        assert "interrompido" in win.status.get()
    finally:
        win._on_close()
        app.root.destroy()


def test_auto_train_window_preview_desligado_nao_desenha(root, tmp_path, detect_one):
    samples = make_samples_dir(tmp_path, n=1)
    app = validator.build_app(root, samples_dir=str(samples), state_path=str(samples / "v.json"))
    win = validator.AutoTrainWindow(app)
    try:
        win.preview_var.set(False)
        root.update()
        win._draw_preview(make_disk_image()[0], [CIRCLE], "a")
        assert getattr(win, "_preview_frame", None) is None
    finally:
        win.top.destroy()
        app.root.destroy()


def test_auto_train_window_mensagens_diretas(root, tmp_path, detect_one):
    samples = make_samples_dir(tmp_path, n=1)
    app = validator.build_app(root, samples_dir=str(samples), state_path=str(samples / "v.json"))
    win = validator.AutoTrainWindow(app)
    try:
        win._queue.put(
            (
                "series_done",
                1,
                validator.AutoSeriesReport(
                    series=1,
                    samples_done=1,
                    samples_total=1,
                    rewards=1,
                    punishments=0,
                    threshold_end=0.90,
                    errors=["leitura falhada"],
                ),
            )
        )
        win._poll()
        assert "Série 1 concluída" in win.series_label.cget("text")
        assert "leitura falhada" in win.report.get("1.0", "end")
        win._queue.put(("stopped", 1, None))
        win._poll()
        assert "interrompido" in win.status.get()
    finally:
        win.top.destroy()
        app.root.destroy()


def test_auto_train_window_cnn_real_treina_e_promove(root, tmp_path, fake_final_window, monkeypatch):
    """Séries reais com patches suficientes: a CNN treina, é promovida e o
    relatório final mostra o balanço da CNN."""
    samples = make_samples_dir(tmp_path, n=2)
    app = validator.build_app(root, samples_dir=str(samples), state_path=str(samples / "v.json"))
    win = validator.AutoTrainWindow(app)
    try:
        win.series_var.set(1)
        win.start()
        deadline = time.monotonic() + 60.0
        while time.monotonic() < deadline and win._running:
            win._poll()
            root.update()
            time.sleep(0.01)
        win._poll()
        root.update()
        assert not win._running
        assert "concluído" in win.status.get()
        state = app.state
        assert len(state.cnn_series) == 1
        assert state.cnn_series[0]["promoted"] is True
        report_text = win.report.get("1.0", "end")
        assert "CNN de deteção:" in report_text
        assert "PROMOVIDA" in report_text
        assert len(fake_final_window) == 1
    finally:
        win._on_close()
        app.root.destroy()
