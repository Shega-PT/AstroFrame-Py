"""Interface desktop (tkinter) da calibração: edição direta de círculos/elipses.

Substitui o editor do navegador por uma janela Python nativa: vês a imagem,
**clicas** para criar círculos/elipses, **arrastas o interior** para mover o
centro e **arrastas as pegas** (direita = raio horizontal, topo = raio
vertical) para ajustar as dimensões em tempo real.

Workflow em duas passagens:

1. **1.ª passagem (deteção DESLIGADA, omissão)** — desenhas à mão os astros de
   todas as amostras e carregas em **Guardar**: fica o ground truth em
   `calibration.json`.
2. **2.ª passagem (deteção LIGADA)** — ao carregar uma amostra sem ground
   truth, a deteção automática preenche os círculos; amostras já guardadas
   abrem como as deixaste. Ajustas o que for preciso (ou aceitas) e voltas a
   guardar. **Validar tudo** compara a deteção com o ground truth em todas as
   amostras e mostra o relatório (recall, precisão, IoU) + sugestões.

Interações no canvas:

- Clique em espaço vazio → cria uma forma nova (círculo ou elipse, conforme o
  seletor) nesse ponto.
- Clique numa forma → seleciona-a.
- Arrastar no interior da forma selecionada → move o centro em tempo real.
- Arrastar a pega direita → ajusta o raio horizontal; pega de topo → raio
  vertical (elipses).
- Rodinha do rato → zoom centrado no cursor; arrastar com o botão direito ou
  do meio → deslocar a imagem.
- Delete → elimina a forma selecionada; setas → nudge de 1 px (Shift = 10 px).

As funções puras de geometria (`canvas_to_image`, `point_in_shape`,
`hit_handle`, …) são testáveis sem ecrã; a janela só é criada por
`run`/`CalibrationTkApp`.
"""

from __future__ import annotations

import logging
import math
import queue
import threading
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from astroframe.calibration.scan import SampleRef, load_frame, scan_samples
from astroframe.calibration.store import CalibrationItem, CalibrationStore
from astroframe.calibration.validate import suggest_parameters, validate_all
from astroframe.config import AstroFrameConfig
from astroframe.core.stabilizer import DiskDetection, find_all_disks
from astroframe.paths import calibration_json

logger = logging.getLogger(__name__)

try:  # tkinter é stdlib mas o módulo só é necessário em runtime
    import tkinter as tk
    from tkinter import ttk

    from PIL import Image, ImageTk
except ImportError as exc:  # pragma: no cover
    tk = None  # type: ignore[assignment]
    ttk = None  # type: ignore[assignment]
    Image = None  # type: ignore[assignment]
    ImageTk = None  # type: ignore[assignment]
    _TK_IMPORT_ERROR = exc
else:
    _TK_IMPORT_ERROR = None

MIN_SHAPE_SIZE = 2
MAX_SHAPE_SIZE = 5000
DEFAULT_SHAPE_SIZE = 40
HANDLE_HIT_CANVAS = 8
ZOOM_FACTOR = 1.15


@dataclass(frozen=True)
class ShapeHandle:
    """Pegas de redimensionamento de uma forma (em coordenadas de imagem)."""

    kind: str  # "right" | "top"
    x: float
    y: float


def canvas_to_image(cx: float, cy: float, scale: float, ox: float, oy: float) -> tuple[float, float]:
    """Coordenadas de canvas → coordenadas de imagem."""
    return (cx - ox) / scale, (cy - oy) / scale


def image_to_canvas(ix: float, iy: float, scale: float, ox: float, oy: float) -> tuple[float, float]:
    """Coordenadas de imagem → coordenadas de canvas."""
    return ix * scale + ox, iy * scale + oy


def point_in_shape(ix: float, iy: float, shape: DiskDetection) -> bool:
    """Teste ponto-dentro-da-forma (elipse ou círculo)."""
    rx, ry = float(shape.radius), float(shape.ry if shape.ry is not None else shape.radius)
    dx, dy = (ix - shape.cx) / rx, (iy - shape.cy) / ry
    return dx * dx + dy * dy <= 1.0


def shape_handles(shape: DiskDetection) -> tuple[ShapeHandle, ShapeHandle]:
    """Pegas da forma: `right` (extremo horizontal) e `top` (extremo vertical)."""
    ry = shape.ry if shape.ry is not None else shape.radius
    return (
        ShapeHandle("right", shape.cx + shape.radius, shape.cy),
        ShapeHandle("top", shape.cx, shape.cy - ry),
    )


def hit_handle(ix: float, iy: float, shape: DiskDetection, scale: float) -> ShapeHandle | None:
    """Devolve a pega sob o ponto (tolerância em pixels de canvas), se houver."""
    tol = HANDLE_HIT_CANVAS / scale
    for handle in shape_handles(shape):
        if math.hypot(ix - handle.x, iy - handle.y) <= tol:
            return handle
    return None


def hit_shape(ix: float, iy: float, shapes: list[DiskDetection]) -> int | None:
    """Índice da forma sob o ponto (a mais recente primeiro), ou `None`."""
    for i in range(len(shapes) - 1, -1, -1):
        if point_in_shape(ix, iy, shapes[i]):
            return i
    return None


def resize_shape(shape: DiskDetection, handle: ShapeHandle, ix: float, iy: float) -> DiskDetection:
    """Aplica o arrasto de uma pega: novo `radius`/`ry`, com limites seguros.

    A pega `right` altera o raio horizontal (`radius`); a pega `top` altera
    apenas o raio vertical (`ry`) e, se a forma for um círculo, converte-a em
    elipse mantendo o raio horizontal.
    """
    if handle.kind == "right":
        rx = clamp_radius(abs(ix - shape.cx))
        return DiskDetection(shape.cx, shape.cy, rx, shape.ry)
    ry = clamp_radius(abs(iy - shape.cy))
    return DiskDetection(shape.cx, shape.cy, shape.radius, ry)


def move_shape(shape: DiskDetection, dx: float, dy: float, width: int, height: int) -> DiskDetection:
    """Move o centro, mantendo dentro dos limites da imagem."""
    cx = min(width - 1, max(0, round(shape.cx + dx)))
    cy = min(height - 1, max(0, round(shape.cy + dy)))
    return DiskDetection(cx, cy, shape.radius, shape.ry)


def clamp_radius(value: float) -> int:
    return min(MAX_SHAPE_SIZE, max(MIN_SHAPE_SIZE, round(value)))


class CalibrationTkApp:
    """Janela de calibração: canvas + painel de controlo."""

    def __init__(
        self,
        root: tk.Tk,
        samples: list[SampleRef],
        store: CalibrationStore,
        config: AstroFrameConfig,
        samples_root: str | Path | None = None,
    ):
        if tk is None:  # pragma: no cover
            raise RuntimeError(f"tkinter indisponível: {_TK_IMPORT_ERROR}")
        self.root = root
        self.samples = samples
        self.store = store
        self.config = config
        self.samples_root = Path(samples_root) if samples_root else None

        self.shapes: list[DiskDetection] = []
        self.selected: int | None = None
        self.frame: np.ndarray | None = None
        self.img_w = 0
        self.img_h = 0

        self.scale = 1.0
        self.ox = 0.0
        self.oy = 0.0
        self._photo: ImageTk.PhotoImage | None = None
        self._photo_scale: float | None = None

        self.auto_detect = tk.BooleanVar(value=False)
        self.shape_kind = tk.StringVar(value="circle")
        self.rx_var = tk.DoubleVar(value=DEFAULT_SHAPE_SIZE)
        self.ry_var = tk.DoubleVar(value=DEFAULT_SHAPE_SIZE)
        self.param2_var = tk.IntVar(value=config.stabilizer.param2)
        self.max_radius_var = tk.IntVar(value=config.stabilizer.max_radius)

        self._job_id = 0
        self._busy = False
        self._syncing = False
        self._queue: queue.Queue[tuple] = queue.Queue()
        self.root.after(50, self._poll_queue)
        self._drag_mode: str | None = None
        self._drag_ix = 0.0
        self._drag_iy = 0.0
        self._grab_dx = 0.0
        self._grab_dy = 0.0
        self._pan_start: tuple[float, float] | None = None
        self._pan_ox = 0.0
        self._pan_oy = 0.0

        self.current_index = 0

        self._build_ui()
        if self.samples:
            self.load_sample(0)
        else:  # pragma: no cover
            self.status.set("Sem amostras na pasta de exemplos.")

    # ---------------------------------------------------------------- UI --

    def _build_ui(self) -> None:
        self.root.title("AstroFrame — Calibração")
        self.root.geometry("1280x800")

        paned = ttk.Panedwindow(self.root, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(paned, bg="#1c1c1e", highlightthickness=0)
        paned.add(self.canvas, weight=3)

        panel = ttk.Frame(paned, padding=8, width=300)
        paned.add(panel, weight=1)

        row = 0

        ttk.Label(panel, text=f"Amostras ({len(self.samples)})", font=("", 10, "bold")).grid(
            row=row, column=0, columnspan=2, sticky="w"
        )
        row += 1

        self.listbox = tk.Listbox(panel, height=10, activestyle="none", exportselection=False)
        self.listbox.grid(row=row, column=0, columnspan=2, sticky="ew")
        sb = ttk.Scrollbar(panel, orient=tk.VERTICAL, command=self.listbox.yview)
        sb.grid(row=row, column=2, sticky="ns")
        self.listbox.configure(yscrollcommand=sb.set)
        for sample in self.samples:
            self.listbox.insert(tk.END, sample.label)
        self.listbox.bind("<<ListboxSelect>>", self._on_listbox_select)
        row += 1

        nav = ttk.Frame(panel)
        nav.grid(row=row, column=0, columnspan=2, sticky="ew", pady=(4, 0))
        ttk.Button(nav, text="◀ Anterior", command=lambda: self.goto(-1)).pack(side=tk.LEFT)
        ttk.Button(nav, text="Seguinte ▶", command=lambda: self.goto(+1)).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Button(nav, text="Ajustar", command=self.fit_view).pack(side=tk.RIGHT)
        row += 1

        ttk.Separator(panel).grid(row=row, column=0, columnspan=2, sticky="ew", pady=8)
        row += 1

        ttk.Label(panel, text="Nova forma (clica na imagem):", font=("", 10, "bold")).grid(
            row=row, column=0, columnspan=2, sticky="w"
        )
        row += 1
        shapes_row = ttk.Frame(panel)
        shapes_row.grid(row=row, column=0, columnspan=2, sticky="w")
        ttk.Radiobutton(shapes_row, text="Círculo", value="circle", variable=self.shape_kind).pack(
            side=tk.LEFT
        )
        ttk.Radiobutton(shapes_row, text="Elipse", value="ellipse", variable=self.shape_kind).pack(
            side=tk.LEFT, padx=(8, 0)
        )
        row += 1

        ttk.Label(panel, text="Forma selecionada:").grid(row=row, column=0, sticky="w")
        row += 1
        ttk.Label(panel, text="Raio X (px)").grid(row=row, column=0, sticky="w")
        ttk.Scale(
            panel, from_=MIN_SHAPE_SIZE, to=2000, variable=self.rx_var, command=self._on_rx_slider
        ).grid(row=row, column=1, sticky="ew")
        row += 1
        ttk.Label(panel, text="Raio Y (px)").grid(row=row, column=0, sticky="w")
        ttk.Scale(
            panel, from_=MIN_SHAPE_SIZE, to=2000, variable=self.ry_var, command=self._on_ry_slider
        ).grid(row=row, column=1, sticky="ew")
        row += 1
        self.shape_info = ttk.Label(panel, text="")
        self.shape_info.grid(row=row, column=0, columnspan=2, sticky="w")
        row += 1

        ttk.Separator(panel).grid(row=row, column=0, columnspan=2, sticky="ew", pady=8)
        row += 1

        ttk.Checkbutton(
            panel,
            text="Deteção automática ao carregar",
            variable=self.auto_detect,
            command=self._on_auto_detect_toggle,
        ).grid(row=row, column=0, columnspan=2, sticky="w")
        row += 1
        ttk.Label(
            panel,
            text="1.ª passagem: desligada, desenhas tudo à mão e guardas;\n"
            "2.ª passagem: ligada, a deteção preenche e validas.",
            foreground="#666",
        ).grid(row=row, column=0, columnspan=2, sticky="w")
        row += 1
        ttk.Label(panel, text="Parâmetros (re-detetam ao largar, com deteção ligada):").grid(
            row=row, column=0, columnspan=2, sticky="w", pady=(6, 0)
        )
        row += 1

        self._param_rows: list[tuple[str, tk.IntVar, int]] = [
            ("param2", self.param2_var, 2),
            ("Raio máx.", self.max_radius_var, 2000),
        ]
        for label, var, to in self._param_rows:
            ttk.Label(panel, text=label).grid(row=row, column=0, sticky="w")
            ttk.Scale(panel, from_=1, to=to, variable=var, command=self._on_param_slider).grid(
                row=row, column=1, sticky="ew"
            )
            row += 1

        ttk.Button(panel, text="Detetar agora (substitui as formas)", command=self.detect_now).grid(
            row=row, column=0, columnspan=2, sticky="ew", pady=(4, 0)
        )
        row += 1

        ttk.Separator(panel).grid(row=row, column=0, columnspan=2, sticky="ew", pady=8)
        row += 1

        buttons = ttk.Frame(panel)
        buttons.grid(row=row, column=0, columnspan=2, sticky="ew")
        ttk.Button(buttons, text="Guardar (Ctrl+S)", command=self.save).pack(side=tk.LEFT)
        ttk.Button(buttons, text="Validar tudo", command=self.validate_all).pack(side=tk.RIGHT)
        row += 1

        self.status = tk.StringVar(value="")
        ttk.Label(panel, textvariable=self.status, foreground="#0a7", wraplength=290).grid(
            row=row, column=0, columnspan=2, sticky="w", pady=(6, 0)
        )
        row += 1

        self.report = tk.Text(panel, height=10, width=38, state=tk.DISABLED)
        self.report.grid(row=row, column=0, columnspan=2, sticky="ew", pady=(4, 0))
        row += 1

        panel.columnconfigure(1, weight=1)

        # Eventos do canvas
        self.canvas.bind("<ButtonPress-1>", self._on_press_left)
        self.canvas.bind("<B1-Motion>", self._on_drag_left)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)
        self.canvas.bind("<ButtonPress-3>", self._on_press_pan)
        self.canvas.bind("<ButtonPress-2>", self._on_press_pan)
        self.canvas.bind("<B3-Motion>", self._on_drag_pan)
        self.canvas.bind("<B2-Motion>", self._on_drag_pan)
        self.canvas.bind("<MouseWheel>", self._on_wheel)
        self.canvas.bind("<Button-4>", lambda e: self._zoom_at(e.x, e.y, 1.0 / ZOOM_FACTOR))  # Linux
        self.canvas.bind("<Button-5>", lambda e: self._zoom_at(e.x, e.y, ZOOM_FACTOR))
        self.canvas.bind("<Delete>", self._on_delete)
        self.canvas.bind("<Escape>", self._on_escape)
        self.canvas.bind("<Left>", lambda e: self._nudge(-1, 0, e))
        self.canvas.bind("<Right>", lambda e: self._nudge(1, 0, e))
        self.canvas.bind("<Up>", lambda e: self._nudge(0, -1, e))
        self.canvas.bind("<Down>", lambda e: self._nudge(0, 1, e))

        self.root.bind("<Control-s>", lambda e: self.save())
        self.root.bind("<Prior>", lambda e: self.goto(-1))
        self.root.bind("<Next>", lambda e: self.goto(+1))

    # ------------------------------------------------------------ amostras --

    def goto(self, delta: int) -> None:
        if not self.samples:
            return
        self.load_sample((self.current_index + delta) % len(self.samples))

    def _on_listbox_select(self, _event=None) -> None:
        selection = self.listbox.curselection()
        if selection:
            self.load_sample(selection[0])

    def load_sample(self, index: int) -> None:
        if not self.samples:
            return
        sample = self.samples[index]
        self.current_index = index
        self._job_id += 1
        self._busy = True
        self._report("")
        try:
            frame = load_frame(sample)
        except Exception as exc:
            self._busy = False
            self.status.set(f"Erro ao carregar: {exc}")
            return
        self.frame = frame
        self.img_h, self.img_w = frame.shape[:2]

        item = self.store.get_item(sample.key)
        if item is not None:
            self.shapes = list(item.circles)
            self.selected = None
            self._busy = False
            self.status.set(f"Ground truth carregado: {len(self.shapes)} forma(s) guardadas.")
        elif self.auto_detect.get():
            self.shapes = []
            self.selected = None
            self.status.set("Sem ground truth — a detetar automaticamente…")
            self._start_detect()
        else:
            self.shapes = []
            self.selected = None
            self._busy = False
            self.status.set("1.ª passagem: desenha os astros à mão (deteção desligada).")

        self.listbox.selection_clear(0, tk.END)
        self.listbox.selection_set(index)
        self.listbox.see(index)
        self.fit_view()
        self._sync_sliders()
        self.redraw()

    # ------------------------------------------------------------- deteção --

    def _start_detect(self) -> None:
        job_id = self._job_id
        frame = self.frame.copy() if self.frame is not None else None
        config = self.config
        messages = self._queue

        def work() -> None:
            try:
                assert frame is not None
                detected = find_all_disks(frame, config)
                messages.put(("detect", job_id, detected, None))
            except Exception as exc:  # pragma: no cover
                messages.put(("detect", job_id, None, exc))

        threading.Thread(target=work, daemon=True).start()

    def _poll_queue(self) -> None:
        """Entrega as mensagens das threads de trabalho no main loop (Tk é
        single-threaded: `after()` só pode ser chamado na main thread)."""
        try:
            while True:
                message = self._queue.get_nowait()
                kind = message[0]
                if kind == "detect":
                    self._on_detect_done(*message[1:])
                elif kind == "validate":
                    self._on_validate_done(*message[1:])
        except queue.Empty:
            pass
        self.root.after(50, self._poll_queue)

    def _on_detect_done(
        self, job_id: int, detected: list[DiskDetection] | None, exc: Exception | None
    ) -> None:
        if job_id != self._job_id:
            return
        self._busy = False
        if exc is not None:
            self.status.set(f"Erro na deteção: {exc}")
            return
        self.shapes = list(detected or [])
        self.selected = None
        self.status.set(f"Deteção automática: {len(self.shapes)} disco(s) detetado(s).")
        self._sync_sliders()
        self.redraw()

    def detect_now(self) -> None:
        if self.frame is None or self._busy:
            return
        self._job_id += 1
        self._busy = True
        self.status.set("A detetar…")
        self._start_detect()

    def _on_auto_detect_toggle(self) -> None:
        if self.auto_detect.get() and self.frame is not None and not self.shapes:
            self._start_detect()
        elif self.auto_detect.get():
            self.status.set("Deteção ligada: carrega em 'Detetar agora' ou muda de amostra.")
        else:
            self.status.set("Deteção desligada (1.ª passagem manual).")

    def _on_param_slider(self, _value=None) -> None:
        self.config.stabilizer.param2 = int(self.param2_var.get())
        self.config.stabilizer.max_radius = int(self.max_radius_var.get())
        if self.auto_detect.get() and self.frame is not None and not self._busy:
            self.detect_now()

    # ------------------------------------------------------------- desenho --

    def fit_view(self) -> None:
        if self.frame is None:
            return
        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        if cw < 10 or ch < 10:
            cw, ch = 900, 700
        self.scale = max(0.05, min(cw / self.img_w, ch / self.img_h))
        self.ox = (cw - self.img_w * self.scale) / 2.0
        self.oy = (ch - self.img_h * self.scale) / 2.0
        self._photo_scale = None

    def _zoom_at(self, cx: float, cy: float, factor: float) -> None:
        if self.frame is None:
            return
        ix, iy = canvas_to_image(cx, cy, self.scale, self.ox, self.oy)
        new_scale = max(0.02, min(40.0, self.scale * factor))
        self.scale = new_scale
        self.ox = cx - ix * self.scale
        self.oy = cy - iy * self.scale
        self._photo_scale = None
        self.redraw()

    def _on_wheel(self, event) -> None:
        factor = 1.0 / ZOOM_FACTOR if event.delta > 0 else ZOOM_FACTOR
        self._zoom_at(event.x, event.y, factor)

    def _image_from_pipeline(self, frame_bgr: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    def redraw(self) -> None:
        if self.frame is None:
            return
        canvas = self.canvas
        canvas.delete("all")
        cw, ch = canvas.winfo_width(), canvas.winfo_height()
        if cw < 10 or ch < 10:
            return
        canvas.create_rectangle(0, 0, cw, ch, fill="#1c1c1e")

        if self._photo_scale != self.scale:
            rgb = self._image_from_pipeline(self.frame)
            pil = Image.fromarray(rgb)
            size = (max(1, round(self.img_w * self.scale)), max(1, round(self.img_h * self.scale)))
            scaled = pil.resize(size, Image.LANCZOS)
            self._photo = ImageTk.PhotoImage(scaled, master=self.canvas)
            self._photo_scale = self.scale
        canvas.create_image(self.ox, self.oy, anchor=tk.NW, image=self._photo)

        for i, shape in enumerate(self.shapes):
            selected = i == self.selected
            self._draw_shape(shape, selected=selected)

    def _draw_shape(self, shape: DiskDetection, selected: bool) -> None:
        canvas = self.canvas
        rx = shape.radius * self.scale
        ry = (shape.ry if shape.ry is not None else shape.radius) * self.scale
        cx, cy = image_to_canvas(shape.cx, shape.cy, self.scale, self.ox, self.oy)
        outline = "#ffd23f" if selected else "#3ee66f"
        width = 3 if selected else 2
        canvas.create_oval(cx - rx, cy - ry, cx + rx, cy + ry, outline=outline, width=width)
        canvas.create_line(cx - 6, cy, cx + 6, cy, fill=outline, width=1)
        canvas.create_line(cx, cy - 6, cx, cy + 6, fill=outline, width=1)
        if selected:
            s = 6
            for handle in shape_handles(shape):
                hx, hy = image_to_canvas(handle.x, handle.y, self.scale, self.ox, self.oy)
                canvas.create_rectangle(hx - s, hy - s, hx + s, hy + s, fill=outline, outline="")

    # --------------------------------------------------------------- rato --

    def _on_press_left(self, event) -> None:
        if self.frame is None or self._busy:
            return
        self.canvas.focus_set()
        ix, iy = canvas_to_image(event.x, event.y, self.scale, self.ox, self.oy)
        self._drag_ix, self._drag_iy = ix, iy
        self._drag_mode = None

        if self.selected is not None and self.selected < len(self.shapes):
            handle = hit_handle(ix, iy, self.shapes[self.selected], self.scale)
            if handle is not None:
                self._drag_mode = f"resize:{handle.kind}"
                return
        index = hit_shape(ix, iy, self.shapes)
        if index is not None:
            self.selected = index
            shape = self.shapes[index]
            self._grab_dx = ix - shape.cx
            self._grab_dy = iy - shape.cy
            self._drag_mode = "move"
            self._sync_sliders()
            self.redraw()
            return
        self.selected = None
        rx = DEFAULT_SHAPE_SIZE
        ry = DEFAULT_SHAPE_SIZE
        if self.shape_kind.get() == "ellipse":
            rx, ry = 3 * DEFAULT_SHAPE_SIZE // 2, DEFAULT_SHAPE_SIZE
        cx = min(self.img_w - 1, max(0, round(ix)))
        cy = min(self.img_h - 1, max(0, round(iy)))
        shape = (
            DiskDetection(cx, cy, rx) if self.shape_kind.get() == "circle" else DiskDetection(cx, cy, rx, ry)
        )
        self.shapes.append(shape)
        self.selected = len(self.shapes) - 1
        self._drag_mode = f"resize:{'right'}"
        self._sync_sliders()
        self.redraw()

    def _on_drag_left(self, event) -> None:
        if self._drag_mode is None or self.selected is None or self._busy:
            return
        if not (0 <= self.selected < len(self.shapes)):
            return
        ix, iy = canvas_to_image(event.x, event.y, self.scale, self.ox, self.oy)
        shape = self.shapes[self.selected]
        if self._drag_mode == "move":
            dx = ix - self._grab_dx - shape.cx
            dy = iy - self._grab_dy - shape.cy
            self.shapes[self.selected] = move_shape(shape, dx, dy, self.img_w, self.img_h)
        elif self._drag_mode.startswith("resize:"):
            handle = shape_handles(shape)[0] if self._drag_mode.endswith("right") else shape_handles(shape)[1]
            self.shapes[self.selected] = resize_shape(shape, handle, ix, iy)
        self._drag_ix, self._drag_iy = ix, iy
        self._sync_sliders()
        self.redraw()

    def _on_release(self, _event=None) -> None:
        self._drag_mode = None

    def _on_press_pan(self, event) -> None:
        self._pan_start = (event.x, event.y)
        self._pan_ox, self._pan_oy = self.ox, self.oy

    def _on_drag_pan(self, event) -> None:
        if self._pan_start is None:
            return
        self.ox = self._pan_ox + (event.x - self._pan_start[0])
        self.oy = self._pan_oy + (event.y - self._pan_start[1])
        self.redraw()

    # ------------------------------------------------------------ teclado --

    def _on_delete(self, _event=None) -> None:
        if self.selected is not None and 0 <= self.selected < len(self.shapes):
            self.shapes.pop(self.selected)
            self.selected = None
            self._sync_sliders()
            self.redraw()

    def _on_escape(self, _event=None) -> None:
        self.selected = None
        self.redraw()

    def _nudge(self, dx: int, dy: int, event) -> None:
        if self.selected is None or not (0 <= self.selected < len(self.shapes)):
            return
        step = 10 if event.state & 0x0001 else 1
        shape = self.shapes[self.selected]
        self.shapes[self.selected] = move_shape(shape, dx * step, dy * step, self.img_w, self.img_h)
        self._sync_sliders()
        self.redraw()

    # ------------------------------------------------------------- sliders --

    def _sync_sliders(self) -> None:
        self._syncing = True
        try:
            if self.selected is not None and 0 <= self.selected < len(self.shapes):
                shape = self.shapes[self.selected]
                self.rx_var.set(shape.radius)
                ry = shape.ry if shape.ry is not None else shape.radius
                self.ry_var.set(ry)
                self.shape_info.config(
                    text=f"Selecionada: centro ({shape.cx}, {shape.cy}) · raio {shape.radius}×{ry} px"
                )
            else:
                self.shape_info.config(text="")
        finally:
            self._syncing = False

    def _on_rx_slider(self, value) -> None:
        if self._syncing or self.selected is None or not (0 <= self.selected < len(self.shapes)):
            return
        shape = self.shapes[self.selected]
        rx = clamp_radius(float(value))
        self.shapes[self.selected] = DiskDetection(shape.cx, shape.cy, rx, shape.ry)
        self.redraw()

    def _on_ry_slider(self, value) -> None:
        if self._syncing or self.selected is None or not (0 <= self.selected < len(self.shapes)):
            return
        shape = self.shapes[self.selected]
        ry = clamp_radius(float(value))
        if shape.ry is None:
            self.shapes[self.selected] = DiskDetection(shape.cx, shape.cy, shape.radius, ry)
        else:
            self.shapes[self.selected] = DiskDetection(shape.cx, shape.cy, shape.radius, ry)
        self.redraw()

    # ----------------------------------------------------------- guardar --

    def save(self) -> None:
        if self.frame is None or not self.samples:
            return
        sample = self.samples[self.current_index]
        if self.samples_root:
            rel = sample.path.relative_to(self.samples_root).as_posix()
        else:
            rel = sample.path.as_posix()
        item = CalibrationItem(
            path=rel,
            kind=sample.kind,
            frame=sample.frame,
            width=self.img_w,
            height=self.img_h,
            circles=list(self.shapes),
        )
        self.store.upsert_item(sample.key, item)
        self.status.set(f"Guardado ✓ ({len(self.shapes)} forma(s)) em {self.store.path.name}.")

    # ---------------------------------------------------------- validar --

    def validate_all(self) -> None:
        if not self.samples or self._busy:
            return
        self._busy = True
        self._report("A validar todas as amostras…\n")
        job_id = self._job_id + 1
        samples = list(self.samples)
        config = self.config
        messages = self._queue

        def work() -> None:
            rows: list[tuple[str, list[DiskDetection], list[DiskDetection]]] = []
            errors: list[str] = []
            try:
                for sample in samples:
                    try:
                        frame = load_frame(sample)
                        detected = find_all_disks(frame, config)
                    except Exception as exc:
                        errors.append(f"{sample.label}: erro ({exc})")
                        continue
                    item = self.store.get_item(sample.key)
                    manual = list(item.circles) if item is not None else []
                    rows.append((sample.label, manual, detected))
                report = validate_all(rows)
                messages.put(("validate", job_id, report, errors))
            except Exception as exc:  # pragma: no cover
                messages.put(("validate", job_id, None, exc))

        threading.Thread(target=work, daemon=True).start()

    def _on_validate_done(self, job_id: int, report, errors: list[str]) -> None:
        if job_id != self._job_id + 1:
            return
        self._busy = False
        if report is None:
            self.status.set("Erro na validação.")
            return
        lines = [
            (
                f"Score global: {report.score:.1f}/100"
                if report.score is not None
                else "Sem ground truth para validar."
            ),
            (
                f"Recall {report.recall * 100:.0f}% · Precisão {report.precision * 100:.0f}%"
                f" · IoU médio {report.mean_iou:.2f}"
            ),
            f"Total: {report.total_matched} emparelhado(s), "
            f"{report.total_false_negatives} falso(s) negativo(s), "
            f"{report.total_false_positives} falso(s) positivo(s).",
        ]
        if report.mean_center_error is not None:
            lines.append(f"Erro médio do centro: {report.mean_center_error:.1f} px")
        if report.mean_radius_error_pct is not None:
            lines.append(f"Erro médio do raio: {report.mean_radius_error_pct:+.0f}%")
        lines.append("")
        for item in report.items:
            iou = f"{item.mean_iou:.2f}" if item.mean_iou is not None else "—"
            lines.append(
                f"{item.label}: {item.n_matched}/{item.n_manual} · IoU {iou} · "
                f"FN {item.n_false_negatives} FP {item.n_false_positives}"
            )
        lines.append("")
        lines.extend(suggest_parameters(report, self.config))
        if errors:
            lines.append("")
            lines.extend(errors)
        self._report("\n".join(lines))
        self.status.set("Validação concluída.")

    def _report(self, text: str) -> None:
        self.report.config(state=tk.NORMAL)
        self.report.delete("1.0", tk.END)
        self.report.insert("1.0", text)
        self.report.config(state=tk.DISABLED)

    # ------------------------------------------------------------ ciclo --

    def run(self) -> None:
        self.root.mainloop()


def build_app(root: tk.Tk, samples_dir: str = "samples", config_path: str | None = None) -> CalibrationTkApp:
    """Constrói a janela (sem `mainloop`), para testes e para `run`."""
    config = AstroFrameConfig.from_yaml(config_path) if config_path else AstroFrameConfig()
    samples = scan_samples(samples_dir)
    store = CalibrationStore(calibration_json(samples_dir))
    return CalibrationTkApp(root, samples, store, config, samples_root=samples_dir)


def run(samples_dir: str = "samples", config_path: str | None = None) -> None:
    """Lança a interface desktop de calibração."""
    if tk is None:  # pragma: no cover
        raise SystemExit(
            "tkinter não está disponível neste ambiente.\nNo Debian/Ubuntu: sudo apt install python3-tk"
        )
    root = tk.Tk()
    build_app(root, samples_dir=samples_dir, config_path=config_path).run()
