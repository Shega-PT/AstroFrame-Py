# Changelog

Todas as mudanças notáveis do AstroFrame serão documentadas neste ficheiro.

O formato segue [Keep a Changelog](https://keepachangelog.com/pt-PT/1.1.0/),
e o versionamento [SemVer](https://semver.org/lang/pt-BR/).

## [0.7.0] - 2026-08-17

### Adicionado

- **Auto-tuning de todos os parâmetros** (`src/astroframe/ai/tuner.py`) —
  otimização automática dos parâmetros de deteção e melhoria contra o ground
  truth de calibração:
  - **avaliação por proxy** (`ProxyEval`) — corre a pipeline sobre as amostras
    reduzidas a ~480p e pontua com IoU médio disco-a-disco (penalizações por
    discos extra/em falta) + estrelas dos frames melhorados, com cache pelos
    parâmetros efetivos;
  - **pesquisa determinística** (`BoundedHillClimb`) — subida de colina em
    etapas sobre o registry (momentum, redução do passo nas falhas, recozimento
    opcional `exp(−Δ/T)`, orçamento de tempo e semente fixa);
  - **pré-semente LSTM** (`_lstm_seed`) — a pesquisa pode partir da previsão
    do `LSTMTuner` sobre o histórico do perfil quando esta melhora o objetivo;
  - **CLI** `astroframe autotune` (`--samples/--budget/--seed/--no-anneal/
    --params/--profile/--export/--reset`) e **separador Auto-tune** no Gradio
    (progresso, relatório e configuração otimizada);
  - o resultado é **registado no banco de aprendizagem** (tabela `tuning`) e
    **aplicado automaticamente** às próximas execuções do mesmo perfil
    (`apply_learned` soma os deltas com clamp do registry).
- **Registry de parâmetros** (`src/astroframe/ai/params.py`) — fonte única de
  verdade dos intervalos seguros, passos e deltas de treino (17 parâmetros em
  `detect`/`enhance`/`stabilizer`); todo o valor aprendido passa por
  `clamp_value` (clamp + arredondamento de ints + paridade ímpar).
- **LSTM em NumPy puro** (`src/astroframe/ai/lstm.py`) — célula LSTM de uma
  camada implementada à mão (forward/backward vetorizados, sem dependências
  novas; torch é opcional):
  - `LSTMTuner` treina offline sobre o histórico de feedback e prevê os deltas
    dos 5 parâmetros visuais;
  - `TrajectoryPredictor` prevê o centroide do disco (regressão linear +
    refinamento LSTM opcional); integrado na anti-trepidação com
    `ai.lstm_trajectory` — frames sem deteção usam a previsão em vez de
    congelar o último deslocamento;
  - modelos `.npz` versionados em `~/.astroframe/lstm.npz` (corrompidos →
    fallback silencioso).
- **CNN em NumPy puro** (`src/astroframe/ai/cnn.py`) — rede convolucional
  pequena (2× conv 3×3 + ReLU + pooling + cabeça) com gradientes verificados
  por diferenças finitas e treino offline determinístico:
  - `fit_residual` + `ResidualEnhancer` — passo residual de remoção de
    ruído/smearing após o unsharp (canal L do LAB, tiles 64×64 com overlap)
    com `ai.cnn_enhance`;
  - `fit_classifier` + `DiskFilter` — classificador disco/ruído que pontua
    cada deteção; com `ai.disk_filter` (0–1) os candidatos abaixo do limiar
    são descartados (a lista detetada nunca é esvaziada);
  - modelos `~/.astroframe/enhancer_cnn.npz` e `~/.astroframe/disk_filter.npz`.
- **Configuração nova** — `[tuning]` (`enabled=false`, `budget_s`, `seed`,
  `anneal`, `proxy_scale`, `frames_per_sample`, `detection_weight`, `params`)
  e `[ai]` (`backend="numpy"`, `lstm_trajectory=false`, `cnn_enhance=false`,
  `disk_filter=0.0`).

### Segurança

- Toda a IA está **desligada por omissão** (`tuning.enabled=false`, `ai.*`);
  modelos em falta/corrompidos degradam silenciosamente e nunca bloqueiam a
  pipeline; todos os valores aprendidos passam pelo clamp do registry.

### Testes

- **558 testes, cobertura a 100%** de `src/astroframe/` — novos suítes para
  `ai/params.py`, `ai/tuner.py`, `ai/lstm.py`, `ai/cnn.py` e testes de
  cobertura dos ramos em falta (early-stop, fallbacks, integração Gradio/CLI);
  `ruff check` limpo.

### Documentação

- Docs PT/EN/FR atualizadas: auto-tuning (CLI + separador), LSTM, CNN,
  registry, configuração `[tuning]`/`[ai]` e segurança.

## [0.6.0] - 2026-08-14

### Adicionado

- **Validação e treino da deteção** (`validator.py` na raiz) — interface
  desktop nativa (tkinter) que percorre as amostras de `samples/` uma a uma,
  mostra a deteção (principal + companheiros) sobre a imagem, e deixa
  **aceitar/rejeitar** cada forma comparando-a com o guia manual
  (`calibration.json`):
  - pesos treináveis por forma: **7 parâmetros** (`param2`, `param1`, `dp`,
    `gaussian_kernel_size`, `gaussian_sigma`, `occluded_ratio`,
    `occluded_ring`) com deltas de recompensa/punição, limites e histórico;
  - **treino automático** (`--auto`): séries de re-deteção com auto-avaliação
    contra o guia (IoU mínimo configurável), recompensa/punição por forma
    detetada ou falhada, punição dobrada para rejeições teimosas, até 100% do
    material processado;
  - **relatório final** com score, pesos treinados e tooltips ⓘ por parâmetro
    + botão **Salvar** que exporta a configuração treinada para
    `trained_config.json` (aplicável ao sistema real);
  - **preview ao detetar** (`on_detect`) — a deteção é desenhada em cima da
    imagem em tempo real antes de seres convidado a julgar;
  - **estado persistente** em `validator_state.json` (rondas, séries,
    histórico de pesos e deltas) com `--reset-state` para começar de novo;
  - modos `--check` (relatório sem interface) e interface com sliders de IoU
    mínimo, zoom/pan e desenho dos discos.
- **Editor de calibração desktop** (`src/astroframe/ui/calibration_tk.py`) —
  janela nativa (tkinter) em `calibrate.py` (por omissão) substituindo o
  editor do navegador:
  - clique cria círculo/elipse, arrastar move o centro, **pegas** ajustam os
    raios horizontal/vertical, sliders Raio X/Raio Y em tempo real, rodinha =
    zoom, botão direito = pan, Delete/setas para eliminar e nudge (Shift = 10);
  - **duas passagens**: 1.ª manual (deteção desligada) → ground truth;
    2.ª com **deteção automática ao carregar** para preencher/validar;
  - sliders de `param2`/raio máximo re-caem a deteção ao largar;
  - **elipses** suportadas no ground truth (`ry` no JSON) e na validação
    (IoU por máscara + raio geométrico para os erros);
  - `calibrate.py --ui gradio` mantém o editor antigo no navegador.
- **Deteção sem `min_radius`/`min_dist` explícitos** — os raios passam a
  derivar automaticamente da imagem (resolução, diâmetro principal) e a
  distância mínima é inferida da deteção; a calibração sugere apenas os
  parâmetros que ainda existem (`param2`/`param1`).
- **Cobertura de testes a 100%** em todo o código (`validator.py` +
  `src/astroframe/`, ~435 testes): testes de interface com Tk real (janela
  oculta), threads determinísticas por `monkeypatch`, e infraestrutura que
  contorna o aborto do GC do Python 3.12 durante o bootstrap de threads com
  cobertura ativa (`gc.disable()` + recolha segura na main thread).

### Corrigido

- `RuntimeError: main thread is not in main loop` no treino automático — o
  valor do slider de IoU era lido dentro da thread de trabalho; agora é
  capturado na thread principal antes de arrancar a série.
- Aborto intermitente da suíte (`Fatal Python error: Aborted`) ao correr
  cobertura em testes com threads + Tk (GC a recolher durante o bootstrap de
  uma thread de trabalho) — GC cíclico desligado na sessão de testes.
- `_pan_start` não inicializado no editor de validação (arrastar sem clique
  prévio levantava `AttributeError`).

## [0.5.0] - 2026-08-13

### Adicionado

- **Calibração com exemplos** (novo pacote `astroframe/calibration/`):
  - `scan_samples` varre a pasta de exemplos (`samples/` por omissão)
    recursivamente — imagens (jpg/png/bmp/tif/webp) entram tal como estão e
    vídeos (mp4/avi/mov/mkv/m4v) contribuem com **8 frames equidistantes e
    determinísticos** (reproduzíveis na validação).
  - `CalibrationStore` guarda o **ground truth manual** em
    `samples/calibration.json` (JSON v1, chave = path relativo + `#frame`).
  - `circles_to_layers` / `layers_to_circles` convertem círculos em **camadas
    RGBA** do `gr.ImageEditor` (arrastar = mover, pincel = adicionar,
    borracha = remover; um círculo por componente conexa).
  - `validate_all` compara a deteção automática (`find_all_disks`) com o
    ground truth em todas as amostras: correspondência greedy por IoU (≥0.5),
    recall/precisão, IoU médio, erros de centro (px) e raio (%) com sinal,
    score de calibração 0–100 (recall 0.4 · precisão 0.3 · IoU 0.3) e
    **sugestões de parâmetros** em PT (ex.: baixar `min_radius` se discos
    pequenos falham, subir `param2` com deteções falsas).
- **Interface de calibração** (`astroframe/ui/calibration_app.py`): dropdown
  de amostras + editor de círculos + botões "Deteção automática", "Guardar
  ajustes" e "Validar todas as amostras" (tabela por amostra + resumo global +
  sugestões).
- **Entradas**: `calibrate.py` na raiz (espelho do `main.py`, com
  `--samples/--config/--host/--port/--share/--no-browser`) e subcomando
  `astroframe calibrate` na CLI.
- `FrameReader.frame_at(index)` — leitura direta de um frame por índice
  (`CAP_PROP_POS_FRAMES`).
- `samples/README.md` reescrito com a estrutura recomendada (imagens/vídeos,
  subpastas por assunto: eclipse, lua, sol, planetas).

### Documentação (multilingue)

- A documentação passou a ser **trilingue**:
  - `docs/PT/` — `API.md`, `Arquitetura.md`, `USO.md` (movidos, com a secção
    de calibração); o `CHANGELOG.md` da raiz mantém-se como canónico (PT).
  - `docs/EN/` — `API.md`, `Architecture.md`, `Usage.md`, `CHANGELOG.md`
    traduzidos para inglês.
  - `docs/FR/` — `API.md`, `Architecture.md`, `Usage.md`, `CHANGELOG.md`
    traduzidos para francês.
  - `README-EN.md` e `README-FR.md` na raiz (traduções do README); o
    `README.md` passou a apontar para `docs/PT/` e as versões EN/FR.

### Testes

- Suíte expandida para **~260 testes com cobertura de 100%** do pacote
  (`tests/test_calibration_{scan,store,circles,validate,app}.py` +
  `astroframe calibrate` na CLI + `frame_at`).

## [0.4.0] - 2026-08-13

### Adicionado

- **Companheiros de eclipse (ex.: a Lua a entrar no Sol)**: `find_all_disks`
  agora faz um **segundo passe Hough com `minDist` reduzido** (1/4 do normal)
  para encontrar círculos interiores ao astro maior, que o passe normal
  descartaria. A interface desenha-os a **amarelo** (astro maior a verde,
  reflexos da lente a vermelho), tanto no separador Imagem como no Vídeo.
- **Filtro de círculos-ghost por área** (`_is_occluded_artifact`): um círculo
  quase totalmente contido no astro maior é descartado quando o contraste com
  o anel à sua volta é fraco (o bordos Sol+Lua detetados como um só círculo);
  compara a sobreposição de **área** (e não só o centro), resistente ao
  refinamento por centroide.
- **Polimento por astros** (`core/polish.py` reescrito): cada astro recebe o
  seu próprio realce (esticamento local de contraste + brilho, com silhuetas
  escuras e uniformes — ex.: Lua em eclipse — preservadas intactas) e a
  imagem é **remontada sem costuras** por blend de máscaras com feather
  (sobreposições = média suave dos realces). A linha de recorte
  (`corona_scale`) dilui o anel até ao fundo.
- **Fundo = média do fundo original** (`polish.background_fill`, agora por
  omissão) em vez de preto puro; `polish.black_background` volta a optar pelo
  preto e `polish.brightness` controla o brilho extra dos astros.
- **Capa de discos**: `find_all_disks` devolve no máximo 5 discos (`_MAX_DISKS`).
- `find_all_disks` aceita imagens em escala de cinza `(H, W)`.

### Corrigido

- Reflexos a desenhar-se a vermelho dentro do astro maior (o separador
  principal/companheiro/reflexo agora é pelo centro em relação ao raio do
  astro maior).
- Polimento a apagar companheiros de eclipse: só círculos com o centro
  **fora** do astro maior são removidos como reflexos.

### Documentação

- `docs/PT/USO.md` e `docs/PT/API.md` atualizados para o polimento por astros, o
  novo `PolishConfig` e a deteção em dois passes; suíte com **221 testes e
  cobertura de 100%**.

## [0.3.0] - 2026-08-13

### Adicionado

- **Deteção de múltiplos discos** (`find_all_disks` em `core/stabilizer.py`):
  em vez de apenas o principal, são detetados o disco principal e os seus
  **reflexos** (Hough + contornos, com fusão de duplicados e preservação do
  mais luminoso a cada centro). O estabilizador continua a usar o principal e
  mantém a última deteção em frames sem disco (`last_detection`).
- **Polimento** (`core/polish.py`): `polish_image()` aplica brilho ao disco
  principal (mantendo a coroa desfocada), remoção dos reflexos e é usado no
  preview/frame final e no vídeo exportado.
- **Avaliação automática** (`ai/score.py`): `score_image()` calcula estrelas
  (0–5) a partir de ruído, contraste, tamanho do disco e cor da coroa; a
  interface mostra o resultado em "Avaliação automática" (imagem **e** vídeo).
- **Base de aprendizagem por feedback** (`ai/feedback.py`): cada execução fica
  registada (perfil de câmara + parâmetros + métricas + avaliação); o
  utilizador pode avaliar manualmente (0–5 estrelas) e o sistema **ajusta os
  sliders automaticamente** nas próximas execuções (mais suave com boas
  avaliações, mais forte com más; denoise extra para ruído, brilho para
  coroa fraca, etc.). Log de aprendizagem com o histórico e as razões em SQLite
  (variável `ASTROFRAME_FEEDBACK_DB` para localização).
- **Vídeos sem disco**: o pipeline estabiliza/preview pula o polimento e a
  avaliação funciona sem deteção (antes ocorria falha).
- Suíte expandida para **205 testes com cobertura de 100%** do pacote.

### Corrigido

- O polimento **apagava um círculo no centro da imagem**: círculos interiores
  quase-concêntricos com o disco principal (ex.: silhueta da Lua dentro do
  Sol) eram detetados como "reflexos" e removidos — `polish_image` agora
  só remove reflexos cujo **centro esteja fora do disco principal**, e
  `find_all_disks` funde círculos concêntricos (tolerância de 12% do raio),
  evitando duplicados do mesmo bordo nos dois sentidos (polimento e desenho
  ao vivo).

### Documentação

- `docs/PT/USO.md`: avaliação automática/manual, log de aprendizagem e secção de
  vídeo reescrita; `docs/PT/API.md` com `find_all_disks`, `polish_image`,
  `score_image` e o novo pacote `ai/`.

## [0.2.0] - 2026-08-12

### Adicionado

- **Nova interface de vídeo ao vivo** (separador "Vídeo"): o painel esquerdo
  mostra o vídeo em tempo real conforme é processado, com o círculo (bounding
  box) do disco detetado; o direito atualiza em frames espaçados o resultado
  final (estabilizado + CLAHE + denoise + nitidez). Exportação opcional do
  vídeo processado (.mp4, sem áudio). `_best_frame_from_video` foi substituído
  por este fluxo completo.
- **Leitura de metadados** (novo pacote `meta/`, implementação própria MIT):
  vídeo via cascata ffprobe → OpenCV (codec, bitrate, duração, fps, resolução)
  e imagem via PIL/EXIF (ISO, exposição, abertura, distância focal, câmara,
  data); sem dependências novas no pip.
- **Sugestões automáticas de parâmetros** (`meta/suggest.py`): raios do
  estabilizador proporcionais à resolução, `denoise.h` escalado pelo ISO,
  redução de denoise em vídeos com bitrate muito comprimido; aplicadas aos
  sliders no carregamento do vídeo (mantêm-se editáveis).
- Painel de "proporção/qualidade" na interface (resolução, aspect ratio, fps,
  codec, bitrate, ISO, exposição, câmara) + `gr.JSON` com os metadados crus.
- Interface reorganizada em separadores ("Imagem" / "Vídeo"); o processamento
  de imagem passou para `process_image_input()` (função de módulo testável).
- Deteção em meia-resolução coberta e suíte expandida para **131 testes com
  cobertura de 100%** do pacote (incluindo o RIFE sem PyTorch, via módulo
  `torch` falso nos testes).

## [0.1.2] - 2026-08-12

### Adicionado

- Interface Gradio aceita vídeos (`.mp4/.avi/.mov`): o frame mais nítido do
  vídeo é selecionado automaticamente (lucky imaging) e processado como imagem
  (`_best_frame_from_video` em `ui.gradio_app`).
- Testes para a seleção do frame mais nítido a partir de vídeo sintético.

## [0.1.1] - 2026-08-12

### Adicionado

- `main.py` na raiz: ponto de entrada único que arranca o frontend (Gradio) e
  o backend (pipeline) juntos, abrindo o navegador automaticamente
  (`python main.py [--config|--host|--port|--share|--no-browser]`).
- Parâmetro `inbrowser` em `ui.gradio_app.run()` (abre o navegador por omissão).

### Documentação

- README: secção de instalação com aviso sobre o PEP 668 (Debian/Ubuntu) e
  `python main.py` como primeiro comando de uso rápido.
- `docs/PT/USO.md`: interface web documentada com `python main.py`.
- `docs/PT/API.md`: nova assinatura de `run()` com `inbrowser`.
- `.gitignore`: padrões genéricos para vídeos (`*.mp4`, `*.MP4`, `*.MOV`, `*.mkv`).

## [0.1.0] - 2026-08-12

### Adicionado

- Pipeline completa: estabilização geométrica (HoughCircles + contornos), melhoria automática (CLAHE/denoise/unsharp) e orquestração (`core/`).
- Vídeo: leitura frame-a-frame, lucky imaging com limiar estatístico e stacking (`video/`).
- Interfaces: Gradio (Antes/Depois, sliders, zoom) e CLI (`astroframe serve|process|video|config-template`).
- Configuração externa via YAML (`astroframe config-template`), com validação e avisos.
- Estabilização temporal (EMA do centroide) com reutilização do último deslocamento em frames sem deteção.
- Recorte automático pós-translação (sem bordas pretas, sem cortar o disco) e raios de deteção relativos à resolução do frame.
- Deteção em meia-resolução em frames grandes (≥1200 px).
- Modo `--fast` (omite o denoising) para vídeos.
- Interpolação RIFE opcional (`astroframe[rife]`), com importação preguiçosa.
- Licença MIT, CI GitHub Actions (pytest 3.10/3.12 + ruff) e 43 testes.

### Corrigido

- Canais RGB/BGR trocados na interface Gradio (cores agora corretas).
- Crash do CLAHE em imagens mais pequenas que o tile grid.
- Stacking sem alinhamento dos frames (agora centraliza antes de empilhar).
- Lote de fotos a abortar na primeira falha (agora continua e resume o resultado).
- Chaves/tipos inválidos em `config.yaml` a serem aceites em silêncio (agora avisam).

### Conhecido

- O vídeo exportado não inclui áudio (usar ffmpeg para juntar a faixa).