# AstroFrame

Estabilização geométrica e melhoria automática de fotos e vídeos de eclipses solares e lunares.

![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue) ![License MIT](https://img.shields.io/badge/license-MIT-green)

## Funcionalidades

- **Estabilização por geometria** — deteta o disco do Sol/Lua (`cv2.HoughCircles` + fallback por contornos + refinamento por centroide de intensidade) e re-alinha cada frame para manter o eclipse sempre no centro exato, sem bordas pretas.
- **Melhoria automática** — CLAHE no espaço LAB (sem estourar o brilho), denoising Non-Local Means (útil para ISO alto) e máscara de nitidez (unsharp) para destacar a borda da Lua.
- **Lucky imaging** — descarte de frames borrados por variância do Laplaciano, com limiar estimado estatisticamente a partir do próprio vídeo.
- **Stacking** — combinação (mediana ou média) dos N melhores frames, alinhados por centralização, para reduzir ruído.
- **Anti-trepidação temporal** — suavização do centroide (EMA) e reutilização do último deslocamento válido quando um frame não tem deteção.
- **Interface Gradio** — comparação Antes/Depois, sliders e zoom na coroa/borda.
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

## Uso rápido

```bash
python main.py                             # interface web (Gradio) — frontend + backend juntos
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

## Documentação

- [docs/USO.md](docs/USO.md) — guia prático: CLI, configuração YAML campo a campo, interface e workflow de vídeo.
- [docs/API.md](docs/API.md) — referência dos módulos `core/`, `video/`, `ai/` e `config.py`.
- [docs/Arquitetura.md](docs/Arquitetura.md) — especificação original da solução (referência).

## Limitações conhecidas

- O vídeo exportado **não contém áudio** (`cv2.VideoWriter`); para preservar o som, junte a faixa original com ffmpeg:
  `ffmpeg -i original.mp4 -i processado.mp4 -c copy -map 0:a -map 1:v saida.mp4`
- A interpolação RIFE é opcional e exige PyTorch (`pip install -e ".[rife]"`); a interface do modelo varia entre versões dos repositórios RIFE.
- O denoising é o passo mais lento (~1 s/frame a 480p); use `--fast` em vídeos grandes.

## Desenvolvimento

```bash
pytest          # 43 testes (pixel-tests com imagens sintéticas)
ruff check .    # lint
ruff format .   # formatação
```

CI (GitHub Actions): pytest em Python 3.10/3.12 + ruff, em `.github/workflows/ci.yml`.

## Estrutura

```
src/astroframe/
├── core/    estabilizador geométrico, melhoria automática e pipeline
├── video/   leitura de frames, lucky imaging e stacking
├── ui/      interface Gradio e CLI
└── ai/      interpolação RIFE opcional (requer PyTorch)
```

## Licença

MIT — ver [LICENSE](LICENSE).