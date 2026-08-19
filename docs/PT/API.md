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
- `TuningConfig`, `AIConfig` — auto-tuning e redes neuronais pequenas (v0.7.0)
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
  Aceita BGR ou escala de cinza. Com `ai.disk_filter > 0.0` os candidatos são
  pontuados pelo classificador CNN (`ai.cnn.DiskFilter`) e os que ficam abaixo
  do limiar de confiança são descartados — a lista **nunca** é esvaziada.
- `find_disk_center` — primeiro elemento de `find_all_disks`; HoughCircles +
  fallback por contornos + refinamento por centroide de intensidade. Em frames
  ≥1200 px a deteção corre em meia-resolução.
- `center_and_stabilize` — translada o frame para centrar o disco e recorta as
  bordas pretas (`stabilizer.auto_crop`), devolvendo o raio ajustado.
  Sem disco detetado devolve a imagem inalterada e `None`.
- `AntiJitterStabilizer.stabilize(frame) -> (frame, DiskDetection | None)` —
  estado interno: EMA do centroide (`jitter_alpha`) e reutilização do último
  deslocamento válido em frames sem deteção (`last_detection` — propriedade com
  o último disco detetado, usada pelo vídeo para polimento/preview). Com
  `ai.lstm_trajectory` o próximo centroide é **previsto** (extrapolação linear
  + refinamento LSTM opcional, `ai.lstm.TrajectoryPredictor`) em vez de ficar
  congelado nos frames sem deteção.

### `core.polish`

```python
polish_image(image, detection, config=None) -> np.ndarray
```

- Polimento **por astros**: deteta todos os discos (`find_all_disks`),
  separa discos secundários (centro dentro do astro maior; ex.: a Lua
  diante do Sol) de reflexos da lente (centro fora), realça **cada astro
  individualmente** (`_astro_boost`:
  esticamento local de contraste + `polish.brightness`; silhuetas escuras e
  uniformes — como um disco secundário concêntrico (ex.: a Lua diante do Sol)
  — são preservadas intactas) e **remonta sem costuras** por blend de máscaras
  com feather (`_band_mask` +
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
  Com `ai.cnn_enhance` um **passo residual CNN** (remoção aprendida de
  ruído/smearing, `ai.cnn.ResidualEnhancer`) corre a seguir ao unsharp (canal L
  do LAB, tiles 64×64 com overlap; v0.7.0).
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

## `astroframe.calibration`

Calibração da deteção contra exemplos (fotos/vídeos), com ground truth manual.

### `calibration.scan`

```python
@dataclass(frozen=True)
class SampleRef:
    kind: str            # "image" | "video"
    path: Path           # caminho absoluto do ficheiro
    frame: int | None    # índice de frame (None para imagens)
    key: str             # "path_relativo#frame" (chave estável no store)
    label: str           # "IMG path" / "VID path #frame" (interface)

scan_samples(root, frames_per_video=8) -> list[SampleRef]
sample_video_frames(frame_count, n=8) -> list[int]
load_frame(sample) -> np.ndarray          # BGR (imagem ou frame amostrado)
item_key(relpath, frame=None) -> str
item_label(kind, relpath, frame=None) -> str
```

- Varrimento **recursivo** da pasta; imagens (jpg/jpeg/png/bmp/tif/tiff/webp)
  entram tal como estão e vídeos (mp4/avi/mov/mkv/m4v) contribuem com N frames
  **equidistantes e determinísticos** (meios-intervalos — reproduzível na
  validação). Vídeos ilegíveis são ignorados com aviso.
- `load_frame` lê a imagem via `cv2.imread` ou o frame do vídeo via
  `FrameReader.frame_at(index)` (novo — procura `CAP_PROP_POS_FRAMES`; erros
  levantam `ValueError`).

### `calibration.store`

```python
@dataclass
class CalibrationItem:
    path: str            # caminho relativo à pasta de amostras
    kind: str
    frame: int | None
    width: int
    height: int
    circles: list[DiskDetection] = []   # ground truth manual

class CalibrationStore(path):           # JSON v1 (samples/calibration.json)
    .load() -> None                     # idempotente; ilegível/versão -> vazio
    .save() -> None
    .upsert_item(key, item) -> None     # grava logo
    .get_item(key) -> CalibrationItem | None
```

### `calibration.circles`

```python
circles_to_layers(image_rgb, circles) -> {"background": ..., "layers": [...]}
layers_to_circles(layers) -> list[DiskDetection]
```

- `circles_to_layers` constrói o valor do `gr.ImageEditor`: o fundo + **uma
  camada RGBA por círculo** (disco translúcido + bordo opaco). As camadas são
  **arrastáveis** na interface → mover um círculo = arrastar a camada;
  pintar por cima adiciona; a borracha remove.
- `layers_to_circles` converte o que o utilizador desenhou em círculos — um
  por **componente conexa** de cada camada (duas pinturas separadas na mesma
  camada = dois círculos); aceita camadas com alpha ou RGB.

### `calibration.validate`

```python
circle_iou(a, b) -> float                                  # interseção/união 0–1
match_circles(manual, detected, iou_threshold=0.5)
    -> (pairs: list[(i, j)], unmatched_manual: set, unmatched_detected: set)

@dataclass
class ItemReport:        # por amostra
    label, n_manual, n_detected, n_matched,
    n_false_negatives, n_false_positives,
    mean_iou, mean_center_error, mean_radius_error_pct   # None sem pares

@dataclass
class CalibrationReport: # agregado
    items, total_* , recall, precision,
    mean_iou, mean_center_error, mean_radius_error_pct,
    score: float | None  # 0–100 = 0.4·recall + 0.3·precisão + 0.3·IoU

validate_item(label, manual, detected) -> ItemReport      # erro de raio com sinal (%)
validate_all([(label, manual, detected), ...]) -> CalibrationReport
suggest_parameters(report, config=None) -> list[str]      # sugestões PT
```

- Correspondência **greedy por IoU decrescente** (limiar 0.5): manual↔deteção.

## `astroframe.ui`

### `ui.gradio_app`

```python
build_app(config=None) -> gr.Blocks
run(config_path=None, host="127.0.0.1", port=7860, share=False, inbrowser=True) -> None

def inspect_video_upload(video_path, db=None, config=None) -> tuple[str, dict, dict, dict, dict, dict]
def process_video(video_path, export=False, denoise_h=None, ...) -> Generator[tuple]
def process_image_input(image, clip_limit=None, denoise_h=None, ..., db=None, config=None) -> tuple[np.ndarray, np.ndarray, np.ndarray, str, dict, str]
def manual_feedback(state, stars, db=None, config=None) -> tuple[str, str]
def run_autotune_tab(samples_dir, budget, params, anneal, register, config=None) -> Generator[tuple[str, dict | None]]
```

- A UI converte RGB→BGR na entrada e BGR→RGB nas saídas (funções `_to_pipeline`
  / `_from_pipeline`); os valores e o pipeline são partilhados com a CLI.
- Três separadores: **Imagem** (entrada, estabilizado, processado, zoom,
  avaliação, sliders, avaliação manual + log de aprendizagem), **Vídeo**
  (upload, painel de metadados, sliders pré-preenchidos, processamento ao vivo
  com discos desenhados, avaliação automática e manual, exportação opcional) e
  **Auto-tune** (pasta de samples, orçamento em segundos, parâmetros e
  registo no banco → progresso, relatório e configuração resultante).
- `inspect_video_upload` chama `meta.extractor` + `meta.suggest` +
  `apply_learned` (avaliações anteriores do mesmo perfil de câmara **e deltas
  do auto-tuning**) e devolve, respetivamente: HTML do resumo
  (proporção/qualidade/sugestões), os metadados crus e os `update()` dos
  sliders.
- `run_autotune_tab` é um **gerador** (consumido pelo `gr.Progress.track`):
  devolve pares `(mensagem, config_dict_ou_None)` — primeiro o aviso de pasta
  em falta (se aplicável), depois o progresso e por fim o relatório com a
  configuração otimizada; chama `ai.tuner.run_autotune` e regista no banco
  quando `register=True`.
- `process_video` é um **gerador** (consumido pelo `gr.Progress.track` do
  Gradio); a cada frame devolve:
  `(live_rgb, preview_rgb, out_video_path_ou_None, status, progress, rating_html,
  run_state, log_html)` — `live` é o frame original em tempo real com os discos
  detetados (`_draw_disks`: **verde** = astro maior, **amarelo** = discos
  secundários, **vermelho** = reflexos — separados por `_split_disks`, que usa o
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

### `ui.calibration_app`

```python
build_calibration_app(samples_dir="samples", config=None, store=None) -> gr.Blocks
run(samples_dir="samples", config_path=None, host="127.0.0.1", port=7860, share=False, inbrowser=True) -> None

def load_item_payload(key, samples_dir, config=None, store=None) -> (dict, str)
def auto_detect_payload(key, samples_dir, config=None) -> (dict, str)
def save_item_circles(editor_value, key, samples_dir, store=None) -> str
def validate_all_report(samples_dir, config=None, store=None) -> (rows, summary_html, suggestions_html)
```

- Layout: dropdown de amostras + `gr.ImageEditor` (camadas RGBA por círculo,
  pincel/borracha) + botões "Deteção automática" / "Guardar ajustes" / 
  "Validar todas as amostras" → tabela por amostra, resumo global (score
  0–100, recall, precisão, IoU, erros) e sugestões de parâmetros.
- `load_item_payload` dá prioridade ao ground truth guardado; sem ele usa a
  deteção automática como ponto de partida. `save_item_circles` converte as
  camadas do editor em círculos e grava no store. `validate_all_report`
  percorre **todas** as amostras (imagens + frames amostrados dos vídeos).
- Ponto de entrada equivalente: `python calibrate.py` na raiz do repositório
  ou `astroframe calibrate` (CLI).

### `ui.cli`

```python
main(argv=None) -> int                 # ponto de entrada do script `astroframe`
build_parser() -> argparse.ArgumentParser
process_images(paths, output_dir, config) -> tuple[int, int]   # (sucessos, falhas)
process_video(path, output, config, mode, stack_n, fast) -> str  # caminho de saída
```

- Subcomandos: `serve`, `process`, `video` (`--mode stabilize|enhance|stack`,
  `--fast`), `config-template`, `calibrate` (interface de calibração) e
  `autotune` (auto-tuning: `--samples DIR`, `--budget N`, `--seed N`,
  `--no-anneal`, `--params nome1,nome2`, `--profile NOME`, `--export FILE`,
  `--reset`).
- `process_images` continua após falhas individuais e levanta `RuntimeError`
  se nada for processado. `mode="stack"` centraliza os frames antes de
  empilhar. Exportação de vídeo não copia áudio (limitado pelo OpenCV).

## `astroframe.ai` (aprendizagem e auto-tuning)

Camada de IA: auto-tuning e redes neuronais pequenas com **núcleo NumPy
puro** (o PyTorch, obrigatório desde a v0.9.0, alimenta apenas o RIFE).
Módulos: `params`
(registry de parâmetros ajustáveis), `tuner` (auto-tuning), `lstm`
(aprendizagem temporal), `cnn` (melhoria residual + filtro de deteção),
`feedback` (aprendizagem por avaliação), `score` (avaliação automática) e
`rife` (interpolação).

**Segurança**: tudo está **desligado por omissão** (`tuning.enabled=false`,
`ai.*`); um modelo em falta ou corrompido degrada silenciosamente e nunca
bloqueia a pipeline.

```python
class RifeInterpolator(repo, source="github", model_name="IFNet", device=None):
    .available() -> bool            # stateless: False se PyTorch não instalado
    .interpolate(frame_a, frame_b, n_interp=1) -> list[np.ndarray]
```

- Usado pela CLI `astroframe video --interp N`. Aceita BGR; devolve
  `n_interp` frames intermédios em BGR. A interface do modelo depende do
  repositório RIFE usado (o `_infer` internal é o ponto a ajustar entre
  versões); o CLI avisa e continua sem interpolação se o modelo não carregar.

### `ai.params` (registry de parâmetros ajustáveis)

Fonte única de verdade dos **intervalos seguros**, passos de otimização e
deltas de treino de todos os parâmetros ajustáveis da pipeline (antes
espalhados por `validator.py` e `feedback.py`). Todo o valor aprendido passa
por `clamp_value` — os clamps são sempre aplicados via registry.

```python
@dataclass(frozen=True)
class ParamSpec:
    path: str                    # "stabilizer.param2"
    low: float; high: float      # intervalo seguro (clamp)
    step: float                  # passo inicial da subida de colina
    dtype: type                  # int | float (ints são arredondados)
    odd: bool                    # True: tem de ficar ímpar (kernels Gaussianos)
    group: str                   # detect | geometry | enhance | stack | polish | score | meta
    costly: bool                 # True: avaliação lenta (denoising)
    punish: float; reward: float # deltas de treino do validator (grupo "detect")

PARAM_SPECS: dict[str, ParamSpec]     # caminhos de parâmetros registados
FEEDBACK_PARAMS: tuple[str, ...]      # os 5 parâmetros visuais do feedback por estrelas

specs(group=None) -> list[ParamSpec]  # ordem de declaração; opcionalmente por grupo
spec(path) -> ParamSpec
spec_by_name(name) -> ParamSpec
bounds(path) -> tuple[float, float]   # intervalo seguro
step(path) -> float                   # passo inicial de otimização
clamp_value(path, value) -> int | float   # clamps + arredonda ints + força ímpar
get_param(config, path) -> float
set_param(config, path, value) -> None    # sem clamp
apply_deltas(config, deltas) -> AstroFrameConfig   # cópia com clamps aplicados
deltas_dict(config, paths) -> dict[str, float]     # deltas vs. valores por omissão
default_punish_deltas() / default_reward_deltas() -> dict  # grupo "detect"
```

- Grupos: `detect` (os 7 pesos do validator: `param2`, `param1`, `dp`,
  `gaussian_kernel_size`, `gaussian_sigma`, `occluded_ratio`,
  `occluded_ring`), `geometry` (`max_radius`, `max_disks`, `jitter_alpha`),
  `enhance` (CLAHE, denoise, unsharp, lucky imaging), `stack`, `polish`,
  `score` e `meta`.
- `clamp_value` é o único ponto por onde passa qualquer valor aprendido — a
  estabilidade do treino depende de nunca falhar.

### `ai.tuner` (auto-tuning)

Otimiza os parâmetros de deteção/melhoria contra as amostras de calibração
através de uma pesquisa determinística com orçamento de tempo.

```python
@dataclass
class TuneReport:
    objective: float            # 0–1 (deteção 0.6 · estrelas 0.4 por omissão)
    stars: float                # 0–5 sobre as amostras melhoradas
    detection: float | None     # 0.4·recall + 0.3·precisão + 0.3·IoU; None sem ground truth
    recall: float; precision: float; mean_iou: float
    elapsed_s: float; n_items: int; n_scored: int
    to_dict() -> dict

@dataclass
class TuneResult:
    config: AstroFrameConfig    # melhor configuração encontrada
    deltas: dict[str, float]    # ajustes vs. a base
    base: AstroFrameConfig
    report: TuneReport
    evaluations: int
    lines: list[str]            # relatório humano (parâmetro · base → ajustado)
    to_dict() -> dict

class ProxyEval(samples_dir, work_scale=0.5, frames_per_sample=3,
                detection_weight=0.6, seed=42):
    .evaluate(config) -> TuneReport   # em cache pelos parâmetros efetivos
    .clear_cache() -> None

class BoundedHillClimb(specs, budget_s=60.0, seed=42, anneal=True,
                       patience=3, improve_eps=1e-4):
    .optimize(evaluate, base, start_deltas=None) -> TuneResult

run_autotune(samples_dir, config=None, budget_s=60.0, seed=42, anneal=True,
             params_filter=None, export_path=None, profile=DEFAULT_PROFILE,
             db=None, work_scale=0.5, frames_per_sample=3,
             detection_weight=0.6) -> TuneResult
export_trained_config(config, deltas, report, path) -> Path
tuning_table_lines(result) -> list[str]
```

- **Avaliação por proxy** (`ProxyEval`): corre a pipeline nas imagens de
  calibração (pasta `samples/` + ground truth em `Logs/train/calibration.json`), mede o
  **IoU médio** entre os discos detetados (Hough) e os esperados, com
  **penalizações por discos extra/em falta**, e pontua alguns frames
  melhorados com estrelas. Os frames são reduzidos para ~480p (escala de
  trabalho máxima 0.5, nunca ampliados); os relatórios ficam em cache pelos
  valores efetivos dos parâmetros — a otimização só reavalia o que mudou.
- **Pesquisa** (`BoundedHillClimb`): passes de +passo/−passo por parâmetro
  sobre o registry (aceita o melhor se melhorar ≥ `improve_eps`), momentum
  (duplica o passo após 2 aceites consecutivos na mesma direção), falhas
  reduzem o passo a metade (mínimo passo/8); com `anneal=True` candidatos
  piores são aceites com probabilidade `exp(−Δ/T)`, T a decair por passe —
  escapa a mínimos locais sem sair dos intervalos seguros do registry.
  Parâmetros caros (denoising) só são tentados nos passes pares.
  Determinístico (`seed` fixa) com orçamento de tempo (`budget_s`).
- `run_autotune` pode ser **pré-semeado com previsões LSTM** (`_lstm_seed`:
  deltas do `ai.lstm.LSTMTuner` sobre o histórico do perfil, mantidos apenas
  quando melhoram o objetivo do proxy) e regista o resultado no banco de
  feedback (tabela `tuning`) quando recebe um `db` ou `feedback.enabled`
  está ativo. A CLI (`astroframe autotune`) e o separador Auto-tune chamam-na.
- `export_trained_config` escreve um JSON com os `params` efetivos, os
  `deltas`, o `report` do proxy e uma secção `stabilizer` (compatível com o
  export antigo); a CLI escreve `Logs/train/trained_config.json` por omissão.

### `ai.lstm` (aprendizagem temporal)

Célula LSTM de uma camada implementada à mão em NumPy puro (forward + backward
com backprop-through-time, portas i/f/o/g vetorizadas — sem novas dependências),
usada por dois preditores.

```python
torch_available() -> bool          # True se PyTorch estiver instalado (usado pelo RIFE)

class LSTMCell(n_in, n_hidden, rng=None):
    .forward(x_seq, h0=None) -> (h, cache)    # (T, n_in) → h, cache
    .forward_full(x_seq) -> np.ndarray        # saídas de todos os timesteps
    .backward(x_seq, cache, dh_next=None) -> dict
    .save(path) -> Path
    .load(path) -> LSTMCell | None

@dataclass
class FitHistory:
    epochs: int; final_loss: float; best_loss: float; best_epoch: int

class LSTMTuner(n_hidden=24, seed=42):
    .fit(history, epochs=200, lr=0.05, seq_len=8, val_fraction=0.2,
         patience=6) -> FitHistory
    .predict_next_delta(history, seq_len=8) -> dict[str, float]  # {} sem dados
    .save(path=None) -> Path            # Logs/weights/lstm.npz
    .load(path=None) -> LSTMTuner | None

class TrajectoryPredictor(maxlen=8, use_lstm=False, model_path=None):
    .push(cx, cy) -> None; .clear() -> None; len() -> int
    .predict() -> tuple[float, float] | None   # None sem histórico suficiente

train_trajectory_model(trajectories, path=None, seed=42, epochs=60) -> Path
trajectory_model_path(path=None) -> Path
```

- **`LSTMTuner`** — treinado **offline** (gradiente descendente em batch
  completo, 20% de validação, early stop) sobre o histórico de feedback
  (estrelas + métricas, 9 características por execução, janelas deslizantes)
  e prevê os **deltas de parâmetros** para a próxima execução (os 5
  parâmetros visuais do feedback). O auto-tuning (`ai.tuner.run_autotune`)
  usa esta previsão como pré-semente; sem histórico ou convergência suficientes
  devolve `{}` — a pesquisa parte da base.
- **`TrajectoryPredictor`** — prevê o próximo centroide do disco a partir das
  últimas deteções: **regressão linear** (mínimos quadrados sobre o histórico)
  como base, com um **refinamento LSTM** opcional (célula 2→8, treinada em
  trajetórias sintéticas por `train_trajectory_model`) quando `use_lstm=True`
  e existe um modelo compatível. Usado pelo `AntiJitterStabilizer` com
  `ai.lstm_trajectory`; sem histórico devolve `None` — nada muda.
- Os modelos são guardados como `.npz` **versionados**
  (`Logs/weights/lstm.npz` por omissão); um ficheiro corrompido ou com a
  versão errada carrega como `None` (fallback silencioso).

### `ai.cnn` (rede convolucional pequena)

Rede convolucional pequena em NumPy puro (2× conv 3×3 + ReLU + pooling +
cabeça) com duas cabeças permutáveis, treino offline determinístico (seed
fixa) e gradientes verificados por diferenças finitas.

```python
@dataclass
class FitReport:
    epochs: int; final_loss: float; best_loss: float; best_epoch: int

class SmallCNN(mode="residual", k=8, seed=42, n_in=1):
    .forward(x) -> (out, cache)         # x: (N, 1, H, W)
    .predict_class(x) -> np.ndarray     # P(disco) por patch (modo classify)
    .backward_residual / .backward_classify(grad, cache) -> dict
    .save(path) -> Path
    .load(path) -> SmallCNN | None

fit_residual(pairs, model=None, epochs=40, lr=0.05, batch_size=8,
             val_fraction=0.2, seed=42) -> tuple[SmallCNN, FitReport]
fit_classifier(positives, negatives, model=None, epochs=60, lr=0.05,
               batch_size=8, val_fraction=0.2, seed=42) -> tuple[SmallCNN, FitReport]

class ResidualEnhancer(model=None, model_path=None):
    .available -> bool
    .apply(image_bgr) -> np.ndarray     # inalterada sem modelo

class DiskFilter(model=None, model_path=None):
    .available -> bool
    .patch(image, cx, cy, radius) -> np.ndarray   # patch cinza, 2× raio → 48×48
    .confidence(image, cx, cy, radius) -> float   # P(disco); 0.5 sem modelo
    .filter_disks(disks, image, threshold) -> list  # nunca esvazia a lista
```

- **`fit_residual`** treina a cabeça residual em pares `(entrada, alvo)`:
  aprende `r = y − x` (MSE, 20% de validação, early stop patience 5). O
  `ResidualEnhancer` treinado é aplicado por `enhance_image` **a seguir ao
  passo unsharp** quando `ai.cnn_enhance`: o residual é somado ao canal L do
  LAB em tiles 64×64 com overlap (as cores preservam-se); sem modelo a imagem
  sai inalterada.
- **`fit_classifier`** treina a cabeça disco/ruído em patches positivos
  (disco) e negativos (cross-entropy). O `DiskFilter` pontua cada deteção
  (`confidence`) e o `find_all_disks` descarta os candidatos abaixo de
  `ai.disk_filter` (0–1) — **nunca** esvazia a lista detetada.
- Modelos: `Logs/weights/enhancer_cnn.npz` e
  `Logs/weights/disk_filter.npz` (`.npz` versionados; corrompidos ou com a
  versão errada → fallback silencioso).

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
    .add_tuning(profile, base_params, deltas, report, source="autotune") -> None
    .history_all(limit=32) -> list[dict]   # histórico de tuning (tabela tuning)
    .recent_tuning(profile=None, limit=1) -> list[dict[str, float]]  # deltas
    .reset_tuning(profile=None) -> int     # limpa o histórico de tuning
```

- Banco SQLite em `Logs/logs/system/feedback.db` (ou `$ASTROFRAME_FEEDBACK_DB`);
  criado ao primeiro uso, com retry em base bloqueada e `history_limit` por perfil.
  Além da tabela `runs` (avaliações), existe a tabela `tuning` (resultados do
  auto-tuning: base_params, deltas, relatório e fonte).
- `apply_learned` soma os **nudges** por estrelas e os **deltas do
  auto-tuning** (tabela `tuning`), sempre com clamp do registry unificado; sem
  nada aprendido devolve a configuração original inalterada (as regras do
  feedback por estrelas continuam como antes: avaliações boas e consistentes
  suavizam o ajuste (`user_weight`), avaliações más aplicam denoise extra com
  ruído (métricas >`1.0`), coroa fraca aumenta o brilho do polimento, disco
  pequeno reduz os raios do detetor; os valores são limitados aos intervalos
  válidos).
- `FeedbackDB.default_path() -> Path`, `.path -> Path`, `.close()`.