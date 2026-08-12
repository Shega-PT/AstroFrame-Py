# Changelog

Todas as mudanças notáveis do AstroFrame serão documentadas neste ficheiro.

O formato segue [Keep a Changelog](https://keepachangelog.com/pt-PT/1.1.0/),
e o versionamento [SemVer](https://semver.org/lang/pt-BR/).

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