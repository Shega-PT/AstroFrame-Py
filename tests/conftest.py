"""Fixtures partilhadas.

Nota: o GC cíclico é desligado durante a sessão — no Python 3.12, o GC a
correr numa thread de trabalho durante o bootstrap (com cobertura ativa)
aborta o processo (`Fatal Python error: Aborted`); a recolha é feita de forma
segura na main thread, após cada teste.

Modo headless (sem janelas):
- `Xvfb` é arrancado automaticamente quando não há `DISPLAY` (CI, servidores);
- todas as janelas Tk (`Tk`, `Toplevel`) são retiradas do ecrã e **destruídas
  no fim de cada teste** — nenhum teste pode deixar uma janela aberta;
- `mainloop` fecha a janela automaticamente ao fim de ~60 ms — nenhum teste
  pode bloquear à espera de interação;
- `cv2.imshow`/`cv2.waitKey` (janelas OpenCV) estão bloqueados e levantam
  erro claro se um teste os usar;
- matplotlib corre sempre com o backend `Agg` (sem janela).
"""

from __future__ import annotations

import gc
import os
import shutil
import subprocess
import time

import cv2
import pytest

from tests.helpers import make_disk_image, make_noisy_image

# matplotlib nunca abre janelas durante os testes (mesmo que um dia seja usado)
os.environ.setdefault("MPLBACKEND", "Agg")

_LIVE_ROOTS: set = set()


def _destroy_root(root) -> None:
    try:
        root.destroy()
    except Exception:  # TclError se já destruída
        pass


@pytest.fixture(scope="session", autouse=True)
def _gc_desligado_durante_a_sessao():
    gc.disable()
    yield
    gc.enable()


@pytest.fixture(autouse=True)
def _recolha_de_lixo_segura():
    yield
    gc.collect()


@pytest.fixture(scope="session", autouse=True)
def _ecra_virtual_automatico():
    """Arranca `Xvfb` quando não existe `DISPLAY` (sem bloqueios nem falhas).

    Com `DISPLAY` já definido (desktop local, ou `xvfb-run` no CI) não faz
    nada; sem `DISPLAY`, sobe um servidor X virtual para a sessão e derruba-o
    no fim. Sem `Xvfb` instalado, a suíte é ignorada com mensagem clara.
    """
    if os.environ.get("DISPLAY"):
        yield
        return
    xvfb = shutil.which("Xvfb")
    if xvfb is None:
        pytest.skip(
            "Sem DISPLAY e sem Xvfb instalado: os testes com janelas (tkinter) "
            "não podem correr headless. Instala o xvfb (apt: `sudo apt install xvfb`)."
        )
        return
    proc = subprocess.Popen(
        [xvfb, ":99", "-screen", "0", "1280x1024x24", "-nolisten", "tcp"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    os.environ["DISPLAY"] = ":99"
    time.sleep(0.5)
    try:
        yield
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        os.environ.pop("DISPLAY", None)


@pytest.fixture(autouse=True)
def _janelas_controladas(monkeypatch):
    """Garante que nenhum teste abre uma janela interativa (headless):

    - `Tk`/`Toplevel` ficam retirados do ecrã, são registados e destruídos
      no fim do teste (mesmo que o teste se esqueça de os fechar);
    - `mainloop` agenda o fecho automático da janela — nunca bloqueia;
    - `cv2.imshow`/`cv2.waitKey` levantam `RuntimeError` com mensagem clara.
    """
    import tkinter as tk

    _real_tk = tk.Tk
    _real_toplevel = tk.Toplevel
    _real_mainloop = tk.Misc.mainloop

    def _tracked_tk(*args, **kwargs):
        root = _real_tk(*args, **kwargs)
        _LIVE_ROOTS.add(root)
        root.withdraw()
        return root

    def _tracked_toplevel(master=None, *args, **kwargs):
        if master is None and tk._default_root is None:
            _tracked_tk()  # root implícito também fica controlado/destruído
        top = _real_toplevel(master, *args, **kwargs)
        try:
            top.withdraw()
        except tk.TclError:
            pass
        return top

    def _quit_seguro(widget):
        try:
            widget.quit()
        except Exception:
            pass

    def _mainloop_que_fecha(self, n=0):
        try:
            self.after(60, _quit_seguro, self)
        except Exception:
            pass
        return _real_mainloop(self, n)

    def _bloqueado(nome: str):
        def _f(*args, **kwargs):
            raise RuntimeError(
                f"{nome} está bloqueado nos testes (modo headless): "
                "os testes não podem abrir janelas nem esperar por teclas."
            )

        return _f

    monkeypatch.setattr(cv2, "imshow", _bloqueado("cv2.imshow"))
    monkeypatch.setattr(cv2, "waitKey", _bloqueado("cv2.waitKey"))
    monkeypatch.setattr(tk, "Tk", _tracked_tk)
    monkeypatch.setattr(tk, "Toplevel", _tracked_toplevel)
    monkeypatch.setattr(tk.Misc, "mainloop", _mainloop_que_fecha)

    yield

    for root in list(_LIVE_ROOTS):
        _destroy_root(root)
    _LIVE_ROOTS.clear()
    tk._default_root = None


@pytest.fixture
def disk_image():
    return make_disk_image()


@pytest.fixture
def noisy_image():
    return make_noisy_image()


@pytest.fixture(autouse=True)
def _ai_isolado(tmp_path, monkeypatch):
    """Isola todos os artefactos de IA (banco, modelos canónicos e staging).

    Sem isto, `FeedbackDB()`, `DiskFilter()` e o treino entre séries
    escreveriam nos `Logs/` reais do projeto durante os testes. O
    `ASTROFRAME_DATA_DIR` redireciona toda a estrutura `Logs/` (logs, train,
    weights) para a pasta temporária; o banco e os caminhos canónicos são
    apontados explicitamente para não dependerem de módulos já importados.
    """
    import enhancer_trainer
    import validator

    import astroframe.ai.cnn as cnn

    monkeypatch.setenv("ASTROFRAME_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("ASTROFRAME_FEEDBACK_DB", str(tmp_path / "feedback.db"))
    monkeypatch.setattr(cnn, "_FILTER_MODEL", tmp_path / "disk_filter.npz")
    monkeypatch.setattr(cnn, "_ENHANCER_MODEL", tmp_path / "enhancer_cnn.npz")
    monkeypatch.setattr(validator, "CNN_MODEL_DIR", tmp_path / "models")
    monkeypatch.setattr(validator, "CNN_CANONICAL_PATH", tmp_path / "disk_filter.npz")
    monkeypatch.setattr(enhancer_trainer, "MODEL_DIR", tmp_path / "models")
    monkeypatch.setattr(enhancer_trainer, "ENHANCER_CANONICAL_PATH", tmp_path / "enhancer_cnn.npz")
    yield
