# AstroFrame

Estabilização geométrica e melhoria automática de fotos e vídeos de eclipses solares e lunares.

![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue) ![License MIT](https://img.shields.io/badge/license-MIT-green)

## Funcionalidades

- **Estabilização por geometria** — deteta o disco do Sol/Lua (`cv2.HoughCircles` + fallback por contornos + refinamento por centroide de intensidade) e re-alinha cada frame para manter o eclipse sempre no centro exato, sem bordas pretas.
- **Melhoria automática** — CLAHE no espaço LAB (sem estourar o brilho), denoising Non-Local Means (útil para ISO alto) e máscara de nitidez (unsharp) para destacar a borda da Lua.
- **Lucky imaging** — descarte de frames borrados por variância do Laplaciano, com limiar estimado estatisticamente a partir do próprio vídeo.
- **Stacking** — combinação (mediana ou média) dos N melhores frames, alinhados por centralização, para reduzir ruído.
- **Anti-trepidação temporal** — suavização do centroide (EMA) e reutilização do último deslocamento válido quando um frame não tem deteção.
- **Deteção de múltiplos discos** — além do disco principal, são detetados os **reflexos** (Hough + contornos); o polimento elimina os reflexos e o vídeo ao vivo mostra-os a vermelho.
- **Polimento e avaliação automática** — `polish_image()` dá brilho ao disco mantendo a coroa; `score_image()` atribui **estrelas (0–5)** ao resultado (ruído, contraste, tamanho e cor da coroa).
- **Calibração com exemplos** — interface desktop nativa (`python calibrate.py`) que carrega as fotos e vídeos de `samples/`, permite **desenhar círculos/elipses à mão** (clique cria, arrastar move, pegas redimensionam) numa 1.ª passagem, ligar a **deteção automática** na 2.ª para preencher/validar as restantes amostras, e comparar tudo contra o ground truth em todas as amostras (recall, precisão, IoU, erros + sugestões de parâmetros).
- **Validação e treino da deteção** — `validator.py` (janela desktop nativa) percorre as amostras, mostra a deteção com zoom/pan, e aprende **recompensando e punindo 7 parâmetros do detetor** forma a forma contra o guia manual; o **treino automático** (`--auto`) re-deteça em séries até 100% e exporta os **pesos treinados** para o sistema real.
- **Aprendizagem por feedback** — cada execução fica em SQLite; para além da avaliação automática, pode avaliar manualmente (0–5 estrelas) e o AstroFrame **ajusta os sliders automaticamente** na próxima execução com o mesmo perfil de câmara, mostrando o histórico/log no próprio interface.
- **Interface Gradio** — dois separadores: **Imagem** (Antes/Depois, sliders, zoom na coroa/borda) e **Vídeo** (processamento ao vivo com os discos detetados, preview final em frames espaçados e exportação opcional). Ao carregar um vídeo, os **metadados** são lidos (ffprobe/OpenCV/EXIF) e os **parâmetros são sugeridos automaticamente** (ISO → denoising, resolução → raios do detetor, bitrate → compressão), mantendo-se editáveis.
- **CLI** — lote de fotos, vídeos (estabilizar/melhorar/stack), logs e barra de progresso.

## Instalação

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

Alternativa simples: `pip install -r requirements.txt`. Requer Python 3.10+.

> [!IMPORTANT]
> Em Debian/Ubuntu recentes, `pip install` no Python do sistema falha com
> `error: externally-managed-environment` (PEP 668). Use sempre a virtualenv acima
> (`source .venv/bin/activate` antes de instalar), ou force com `--break-system-packages`
> por sua conta e risco.

> [!TIP]
> Para metadados de vídeo ricos (codec, bitrate, duração) instale o `ffmpeg`
> do sistema — sem ele, o AstroFrame usa apenas o OpenCV (resolução/fps/frames).

## Uso rápido

```bash
python main.py                             # interface web (Gradio) — frontend + backend juntos
python calibrate.py                        # interface de calibração (samples/)
python validator.py                        # validação/treino da deteção (samples/)
astroframe serve                          # equivalente via CLI instalada
astroframe process --input foto1.jpg foto2.jpg --output-dir outputs/
astroframe video --input eclipse.mp4 --mode enhance
astroframe video --input eclipse.mp4 --mode stack --stack-n 20
astroframe video --input eclipse.mp4 --mode enhance --fast   # sem denoise (mais rápido)
astroframe config-template                # gera config.yaml editável
```

`main.py` é o ponto de entrada único: arranca o servidor Gradio que serve o
frontend no navegador e processa as imagens no backend (o motor em `core/` corre
no mesmo processo, a cada clique em **Processar**). Opções: `--config`,
`--host`, `--port`, `--share` e `--no-browser`.

`validator.py` é a **validação/treino da deteção** (janela desktop nativa;
`--check` para relatório sem interface, `--auto` para treino automático):
compara a deteção com o guia manual de `calibration.json`, **recompensa/pune
os parâmetros** por forma e termina com um relatório + pesos treinados
exportáveis para o sistema real.

`calibrate.py` é a **interface de calibração** (janela desktop nativa; usa
`--ui gradio` para o navegador). Workflow em duas passagens:

1. **1.ª passagem (deteção desligada)** — desenhas os astros à mão em todas as
   amostras (clique cria círculo/elipse, arrastar move, pegas redimensionam) e
   guardas: fica o ground truth em `calibration.json`.
2. **2.ª passagem (deteção ligada)** — as amostras sem ground truth são
   preenchidas automaticamente; as guardadas abrem como as deixaste; ajustas o
   que for preciso e voltas a guardar. **Validar tudo** compara a deteção com o
   ground truth em todas as amostras (recall, precisão, IoU) + sugestões.

## Documentação

- [docs/PT/USO.md](docs/PT/USO.md) — guia prático: CLI, configuração YAML campo a campo, interface, calibração e workflow de vídeo (PT).
- [docs/PT/API.md](docs/PT/API.md) — referência dos módulos `core/`, `video/`, `meta/`, `ai/`, `calibration/` e `config.py` (PT).
- [docs/PT/Arquitetura.md](docs/PT/Arquitetura.md) — especificação original da solução (referência, PT).
- [docs/EN/](docs/EN/) — mesma documentação em inglês (API, Architecture, Usage, CHANGELOG).
- [docs/FR/](docs/FR/) — même documentation en français (API, Architecture, Usage, CHANGELOG).
- [README-EN.md](README-EN.md) / [README-FR.md](README-FR.md) — este README em inglês e francês.

## Limitações conhecidas

- O vídeo exportado **não contém áudio** (`cv2.VideoWriter`); para preservar o som, junte a faixa original com ffmpeg:
  `ffmpeg -i original.mp4 -i processado.mp4 -c copy -map 0:a -map 1:v saida.mp4`
- A interpolação RIFE é opcional e exige PyTorch (`pip install -e ".[rife]"`); a interface do modelo varia entre versões dos repositórios RIFE.
- O denoising é o passo mais lento (~1 s/frame a 480p); use `--fast` em vídeos grandes.

## Desenvolvimento

```bash
pytest                      # ~260 testes (pixel-tests com imagens sintéticas)
pytest --cov=astroframe     # cobertura (100% do pacote)
ruff check .                # lint
ruff format .               # formatação
```

CI (GitHub Actions): pytest em Python 3.10/3.12 + ruff, em `.github/workflows/ci.yml`.

## Estrutura

```
src/astroframe/
├── core/         estabilizador geométrico, melhoria automática e pipeline
├── video/        leitura de frames, lucky imaging e stacking
├── meta/         leitura de metadados (ffprobe/OpenCV/EXIF) e sugestões de parâmetros
├── calibration/  varrimento de exemplos, ground truth e validação da deteção
├── ui/           interface Gradio (Imagem/Vídeo + Calibração) e CLI
└── ai/           interpolação RIFE opcional (requer PyTorch)
```

## Licença

MIT — ver [LICENSE](LICENSE).