# Changelog

Todas as mudanças notáveis do AstroFrame serão documentadas neste ficheiro.

O formato segue [Keep a Changelog](https://keepachangelog.com/pt-PT/1.1.0/),
e o versionamento [SemVer](https://semver.org/lang/pt-BR/).

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

- **Detecção de múltiplos discos** (`find_all_disks` em `core/stabilizer.py`):
  em vez de apenas o principal, são detetados o disco principal e os seus
  **reflexos** (Hough + contornos, com fusão de duplicados e preservação do
  mais luminoso a cada centro). O estabilizador continua a usar o principal e
  mantém a última detecção em frames sem disco (`last_detection`).
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
  avaliação funciona sem detecção (antes ocorria falha).
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