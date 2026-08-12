# Referência da API — AstroFrame

Referência dos módulos e contratos do pacote `astroframe`. Para utilização
prática, ver [USO.md](USO.md).

**Convenções globais**

- Imagens: numpy `np.ndarray` **BGR** (convenção OpenCV), `uint8`, `(H, W, 3)`.
  Entradas em escala de cinza `(H, W)` e RGBA `(H, W, 4)` (interpretado RGBA)
  são normalizadas para BGR automaticamente.
- Configuração: `AstroFrameConfig`; qualquer função de processamento aceita-o
  opcionalmente (usa os valores por omissão se omitido).

## `astroframe.config`

Dataclasses com os parâmetros (ver [USO.md](USO.md#configuração-configyaml)
para a tabela campo a campo):

- `CLAHEConfig`, `DenoiseConfig`, `UnsharpConfig`
- `StabilizerConfig`, `LuckyConfig`, `StackingConfig`
- `AstroFrameConfig` — raiz com um campo por subconfiguração.

Métodos de `AstroFrameConfig`:

```python
cfg.to_dict() -> dict
cfg.to_yaml(path) -> None
cfg.from_yaml(path) -> AstroFrameConfig   # classmethod
```

`from_yaml` valida os tipos e avisa (logging) sobre chaves desconhecidas e
tipos inesperados, sem falhar.

## `astroframe.core`

### `core.stabilizer`

```python
@dataclass(frozen=True)
class DiskDetection:
    cx: int
    cy: int          # centro detetado nas coordenadas da imagem de origem
    radius: int      # raio ajustado após recorte/re-escala

find_disk_center(image, config=None) -> DiskDetection | None
center_and_stabilize(image, config=None) -> tuple[np.ndarray, DiskDetection | None]
class AntiJitterStabilizer(config=None, alpha=None): ...
```

- `find_disk_center` — HoughCircles + fallback por contornos + refinamento por
  centroide de intensidade. Em frames ≥1200 px a deteção corre em meia-resolução.
- `center_and_stabilize` — translada o frame para centrar o disco e recorta as
  bordas pretas (`stabilizer.auto_crop`), devolvendo o raio ajustado.
  Sem disco detetado devolve a imagem inalterada e `None`.
- `AntiJitterStabilizer.stabilize(frame) -> (frame, DiskDetection | None)` —
  estado interno: EMA do centroide (`jitter_alpha`) e reutilização do último
  deslocamento válido em frames sem deteção.

### `core.enhancer`

```python
clahe_enhance(image, config) -> np.ndarray
denoise(image, config) -> np.ndarray
unsharp_mask(image, config) -> np.ndarray
enhance_image(image, config=None, use_denoise=True) -> np.ndarray
```

- Ordem: CLAHE no canal L do LAB → `fastNlMeansDenoisingColored` → unsharp.
- `use_denoise=False` omite o passo mais lento (usado pelo `--fast`).

### `core.pipeline`

```python
@dataclass
class ProcessResult:
    original: np.ndarray
    stabilized: np.ndarray
    enhanced: np.ndarray
    detection: DiskDetection | None

process_image(image, config=None) -> ProcessResult
process_path(path, config=None) -> ProcessResult   # ValueError se ilegível
```

## `astroframe.video`

### `video.reader`

```python
class FrameReader(path):
    .fps -> float
    .frame_count -> int          # 0 quando desconhecido
    .size -> tuple[int, int]     # (largura, altura)
    .close() / context manager
    iterável: frames BGR
    # ValueError se o vídeo não abrir
```

### `video.select` (lucky imaging)

```python
sharpness(frame, config=None) -> float        # variância do Laplaciano
estimate_sharpness_threshold(scores, percentile=25.0) -> float
select_sharp_frames(frames, config=None, minimum=None) -> list[(idx, frame, score)]
```

- Ordem de limiar: `minimum` → `config.lucky.min_sharpness` → percentil estimado
  da própria sequência.

### `video.stacking`

```python
stack_frames(frames, stacking=None) -> np.ndarray   # ValueError se vazio ou shapes diferentes
select_best(frames, n_best, config=None) -> list[np.ndarray]
```

- `stack_frames`: mediana (`use_median=True`) ou média; avisa sobre memória
  acima de 1080p com muitos frames.

## `astroframe.ui`

### `ui.gradio_app`

```python
build_app(config=None) -> gr.Blocks
run(config_path=None, host="127.0.0.1", port=7860, share=False, inbrowser=True) -> None
```

- A UI converte RGB→BGR na entrada e BGR→RGB nas saídas (funções `_to_pipeline`
  / `_from_pipeline`); os valores e o pipeline são partilhados com a CLI.
- `run()` aceita `inbrowser` para abrir o navegador automaticamente; o ponto de
  entrada único equivalente é `python main.py` na raiz do repositório.

### `ui.cli`

```python
main(argv=None) -> int                 # ponto de entrada do script `astroframe`
build_parser() -> argparse.ArgumentParser
process_images(paths, output_dir, config) -> tuple[int, int]   # (sucessos, falhas)
process_video(path, output, config, mode, stack_n, fast) -> str  # caminho de saída
```

- Subcomandos: `serve`, `process`, `video` (`--mode stabilize|enhance|stack`,
  `--fast`), `config-template`.
- `process_images` continua após falhas individuais e levanta `RuntimeError`
  se nada for processado. `mode="stack"` centraliza os frames antes de
  empilhar. Exportação de vídeo não copia áudio (limitado pelo OpenCV).

## `astroframe.ai` (opcional)

```python
class RifeInterpolator(repo, source="github", model_name="IFNet", device=None):
    .available() -> bool            # stateless: False se PyTorch não instalado
    .interpolate(frame_a, frame_b, n_interp=1) -> list[np.ndarray]
```

- Preciso de `pip install -e ".[rife]"`. Aceita BGR; devolve `n_interp` frames
  intermédios em BGR. A interface do modelo depende do repositório RIFE usado
  (o `_infer` internal é o ponto a ajustar entre versões); sem PyTorch levanta
  `RuntimeError` com instruções.