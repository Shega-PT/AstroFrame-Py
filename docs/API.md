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
- `StabilizerConfig`, `PolishConfig`, `FeedbackConfig`, `LuckyConfig`, `StackingConfig`
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

find_all_disks(image, config=None) -> list[DiskDetection]
find_disk_center(image, config=None) -> DiskDetection | None
center_and_stabilize(image, config=None) -> tuple[np.ndarray, DiskDetection | None]
class AntiJitterStabilizer(config=None, alpha=None): ...
```

- `find_all_disks` — **dois passes** de `HoughCircles` (o segundo com
  `minDist` 1/4 do normal, para círculos interiores ao astro maior) +
  fallback por contornos; até **5 discos**, ordenados por raio decrescente.
  Dedup apenas de círculos do **mesmo bordo** (centros próximos E raios
  quase-iguais, tolerância 12% do raio), e rejeição de **círculos-ghost**:
  um candidato quase totalmente dentro de um disco já aceite (≥90% da área)
  com contraste fraco face ao anel à sua volta é descartado (`_is_occluded_artifact`).
  Aceita BGR ou escala de cinza.
- `find_disk_center` — primeiro elemento de `find_all_disks`; HoughCircles +
  fallback por contornos + refinamento por centroide de intensidade. Em frames
  ≥1200 px a deteção corre em meia-resolução.
- `center_and_stabilize` — translada o frame para centrar o disco e recorta as
  bordas pretas (`stabilizer.auto_crop`), devolvendo o raio ajustado.
  Sem disco detetado devolve a imagem inalterada e `None`.
- `AntiJitterStabilizer.stabilize(frame) -> (frame, DiskDetection | None)` —
  estado interno: EMA do centroide (`jitter_alpha`) e reutilização do último
  deslocamento válido em frames sem deteção (`last_detection` — propriedade com
  o último disco detetado, usada pelo vídeo para polimento/preview).

### `core.polish`

```python
polish_image(image, detection, config=None) -> np.ndarray
```

- Polimento **por astros**: deteta todos os discos (`find_all_disks`),
  separa companheiros de eclipse (centro dentro do astro maior) de reflexos
  da lente (centro fora), realça **cada astro individualmente** (`_astro_boost`:
  esticamento local de contraste + `polish.brightness`; silhuetas escuras e
  uniformes — como a Lua em eclipse — são preservadas intactas) e **remonta
  sem costuras** por blend de máscaras com feather (`_band_mask` +
  `_astro_region`): a linha de recorte `corona_scale × raio` dilui o anel até
  ao fundo e as sobreposições entre astros são a média suave dos realces.
  O fundo é a **média do fundo original** (`background_fill`) ou preto puro
  (`black_background`); reflexos (raio ≥ `reflection_min_radius` px) são
  preenchidos com o fundo se `remove_reflections`. Sem deteção devolve a
  imagem inalterada.

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
    enhanced: np.ndarray       # estabilizado + CLAHE + denoise + unsharp + polimento
    enhanced_raw: np.ndarray   # o mesmo, sem polimento (base da avaliação)
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

## `astroframe.meta`

Leitura de metadados e sugestões de parâmetros (implementação própria, MIT —
inspirada na mesma ideia do MetadataExplorer, sem código copiado).

### `meta.extractor`

```python
@dataclass(frozen=True)
class MediaMetadata:
    path: str | None
    kind: str                       # "image" | "video"
    width: int | None
    height: int | None
    aspect_ratio: float | None      # largura / altura
    fps: float | None
    frame_count: int | None
    duration: float | None          # segundos
    codec: str | None
    bitrate: int | None             # bits/segundo
    format_name: str | None
    iso: int | None                 # sensibilidade ISO (EXIF)
    exposure_time: float | None     # segundos
    focal_length: float | None      # mm
    aperture: float | None          # f/ número
    camera_make: str | None
    camera_model: str | None
    captured_at: str | None         # data/hora EXIF
    raw: dict                       # tudo o que foi lido (source→chave→valor)

extract_metadata(path) -> MediaMetadata
```

- Vídeo: cascata **ffprobe** (se instalado; codec/bitrate/duração/formato) →
  **OpenCV** (resolução/fps/frames — sempre disponível).
- Imagem: EXIF via PIL (ISO, exposição, abertura, distância focal, câmara, data).
- `ValueError` se o caminho não existir; `kind="unknown"` com o que der para
  ler se nem ffprobe nem OpenCV abrirem o ficheiro.
- `aspect_ratio` arredondado a 3 casas (0.0 → `None`); o texto de
  apresentação (ex. `5616×3744 · 3:2`) é `aspect_text` em `extractor` (16:9,
  3:2, 4:3, 1:1, quadrado ou mudança decimal).

### `meta.suggest`

```python
suggest_config(meta: MediaMetadata) -> AstroFrameConfig
summary_fields(meta: MediaMetadata) -> dict[str, str]
```

- Heurísticas: raios de deteção proporcionais à resolução (`min = 8%` do
  semieixo menor, `max = 45%`); `denoise.h` escalado pelo ISO (`2 + ISO/1600*4`,
  limitado a `[2, 15]`, usado por defeito se o config não o defina) com
  `unsharp` 0.4/0.6; em vídeos muito comprimidos (< 0.1 bit/pixel) o denoise é
  reduzido ~30% (menos risco de "plastificar").
- `summary_fields` devolve o dicionário exibido no painel "Proporção /
  qualidade / sugestões" da interface.

## `astroframe.ui`

### `ui.gradio_app`

```python
build_app(config=None) -> gr.Blocks
run(config_path=None, host="127.0.0.1", port=7860, share=False, inbrowser=True) -> None

def inspect_video_upload(video_path, db=None, config=None) -> tuple[str, dict, dict, dict, dict, dict]
def process_video(video_path, export=False, denoise_h=None, ...) -> Generator[tuple]
def process_image_input(image, clip_limit=None, denoise_h=None, ..., db=None, config=None) -> tuple[np.ndarray, np.ndarray, np.ndarray, str, dict, str]
def manual_feedback(state, stars, db=None, config=None) -> tuple[str, str]
```

- A UI converte RGB→BGR na entrada e BGR→RGB nas saídas (funções `_to_pipeline`
  / `_from_pipeline`); os valores e o pipeline são partilhados com a CLI.
- Dois separadores: **Imagem** (entrada, estabilizado, processado, zoom,
  avaliação, sliders, avaliação manual + log de aprendizagem) e **Vídeo**
  (upload, painel de metadados, sliders pré-preenchidos, processamento ao vivo
  com discos desenhados, avaliação automática e manual, exportação opcional).
- `inspect_video_upload` chama `meta.extractor` + `meta.suggest` +
  `apply_learned` (avaliações anteriores do mesmo perfil de câmara) e devolve,
  respetivamente: HTML do resumo (proporção/qualidade/sugestões), os metadados
  crus e os `update()` dos sliders.
- `process_video` é um **gerador** (consumido pelo `gr.Progress.track` do
  Gradio); a cada frame devolve:
  `(live_rgb, preview_rgb, out_video_path_ou_None, status, progress, rating_html,
  run_state, log_html)` — `live` é o frame original em tempo real com os discos
  detetados (`_draw_disks`: **verde** = astro maior, **amarelo** = companheiros
  de eclipse, **vermelho** = reflexos — separados por `_split_disks`, que usa o
  centro do disco vs. raio do astro maior), `preview` é o resultado final
  mostrado apenas em frames espaçados (`_preview_every`), os outros campos com
  `None`/fração no meio do passe. Sem disco detetado em **nenhum** frame, o
  resultado final sai sem polimento e a avaliação é calculada sem deteção (aviso
  no estado). Se `export=True`, escreve o vídeo completo com polimento (.mp4,
  codec `mp4v`, sem áudio) e devolve o caminho no último frame.
- `process_image_input` devolve `(estabilizado, processado, zoom, html da
  avaliação, estado, log de aprendizagem)`; o estado (perfil, avaliação, parâmetros)
  alimenta o `manual_feedback` que grava a avaliação em estrelas e reporta o ajuste
  aprendido no log.
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

## `astroframe.ai` (opcional até 0.3: RIFE)

```python
class RifeInterpolator(repo, source="github", model_name="IFNet", device=None):
    .available() -> bool            # stateless: False se PyTorch não instalado
    .interpolate(frame_a, frame_b, n_interp=1) -> list[np.ndarray]
```

- Preciso de `pip install -e ".[rife]"`. Aceita BGR; devolve `n_interp` frames
  intermédios em BGR. A interface do modelo depende do repositório RIFE usado
  (o `_infer` internal é o ponto a ajustar entre versões); sem PyTorch levanta
  `RuntimeError` com instruções.

### `ai.score` (avaliação automática)

```python
@dataclass
class StarRating:
    stars: float            # 0.0–5.0 (peso das métricas = 1)
    score: float            # 0.0–1.0 não ponderado
    metrics: dict[str, float]  # noise | contrast | size | corona; 0 (mau) a 1 (bom)
    explanation: str        # texto humano com o porquê

score_image(image, detection=None, config=None) -> StarRating
package_rating(original, stabilized, detection, config=None) -> StarRating
score_from_stars(stars, metrics=None) -> StarRating   # para testes/externalização
```

- `noise` = variância do Laplaciano (sem ruído → 1), `contrast` = relação
  percentil 99/50 da luminância, `size` = raio do disco vs. frame,
  `corona` = brilho médio do anel coroa (1–2× raio) vs. disco.
- `score_image` funciona **sem deteção** (métricas de ruído/contraste apenas).

### `ai.feedback` (aprendizagem por avaliação)

```python
@dataclass(frozen=True)
class ConfigNudge:
    clip_limit: dict       # {multiplicador, offset}   ex.: {'m': 1.0, 'b': 0.0}
    denoise: dict          #                           ex.: {'h': {'m': 1.0, 'b': 1.5}}
    unsharp: dict          # {'amount': {...}, 'sigma': {'m': 0.7, 'b': 1.2}}
    polish: dict           # {'brightness': {...}}
    explanation: dict[str, str]  # texto por parâmetro (porquê do ajuste)
    judicial_override: bool      # True: a avaliação má impõe logo a correção
    factor: float          # escala global da correção (feedback.learning_rate)

@dataclass(frozen=True)
class RunRecord:
    id: int; profile: str; kind: str; params: dict; metrics: dict
    stars: float; source: str; at: str; modified: dict | None

def profile_for(kind, width, height) -> str     # ex.: "video@5616x3744"
def format_profile(profile) -> str              # "5616×3744 · vídeo" (interface)
def record_run(db, kind, profile, config, params, rating, source="cli") -> RunRecord
def recent_nudges(db, profile, limit=5) -> list[RunRecord]
def apply_learned(config, profile, db=None) -> AstroFrameConfig
def _learning_db(config, db=None) -> FeedbackDB | None   # feedback.enabled?
def _learning_log_html(profile, db) -> str               # histórico (interface)
class FeedbackDB(path=None):               # SQLite com locking retry (WAL)
    .history(profile, limit=50, base=None) -> list[RunRecord]
    .latest_ids(profile, limit=5) -> list[int]
    .nudges(profile_runs, nudge_params, factor) -> ConfigNudge  # regras
    .store_run(kind, profile, config, params, stars, source, metrics) -> RunRecord
    .apply_nudge(config, nudge) -> AstroFrameConfig
```

- Banco SQLite em `~/.astroframe/feedback.db` (ou `$ASTROFRAME_FEEDBACK_DB`);
  criado ao primeiro uso, com retry em base bloqueada e `history_limit` por perfil.
- `apply_learned` devolve a config original se não houver histórico (ou a
  `judicial_override`/`factor` for nula); regras: avaliações boas e consistentes
  suavizam o ajuste (`user_weight`), avaliações más aplicam denoise extra
  com ruído (métricas >`1.0`), coroa fraca aumenta o brilho do polimento, disco
  pequeno reduz os raios do detector; os valores são limitados aos intervalos
  válidos.
- `FeedbackDB.default_path() -> Path`, `.path -> Path`, `.close()`.