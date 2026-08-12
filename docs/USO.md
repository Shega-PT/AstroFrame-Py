# Guia de Utilização — AstroFrame

Guia prático para instalar, configurar e executar o AstroFrame. A especificação
da solução está em [Arquitetura.md](Arquitetura.md) e a referência de código
em [API.md](API.md).

## Índice

1. [Instalação](#instalação)
2. [Interface web (Gradio)](#interface-web-gradio)
3. [Linha de comando](#linha-de-comando)
4. [Configuração (config.yaml)](#configuração-configyaml)
5. [Workflow de vídeo](#workflow-de-vídeo)
6. [Limitações e notas](#limitações-e-notas)

---

## Instalação

Requer Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

Sem ambiente virtual (`pip install -r requirements.txt` funciona, mas em
Debian/Ubuntu 24+ é obrigatório o venv — PEP 668).

## Interface web (Gradio)

O ponto de entrada mais simples é o `main.py` na raiz do repositório: arranca o
frontend (Gradio) e o backend (motor de processamento em `core/`) no mesmo
processo e abre o navegador automaticamente.

```bash
python main.py
```

Abre em `http://127.0.0.1:7860`. Para configurar, porta ou link público:

```bash
python main.py --config config.yaml --port 7861 --share
python main.py --no-browser        # não abre o navegador (útil em servidores)
```

O mesmo servidor está disponível via CLI instalada, equivalente a:

```bash
astroframe serve                   # equivalente a python main.py
astroframe serve --config config.yaml --port 7861 --share
```

> `--share` cria um URL público temporário (via túnel do Gradio) — não o use
> com material sensível.

Na interface:

- **Entrada** — carregar a foto/frame (formato abitrátrio de imagem).
- **Estabilizado** — disco centralizado (com círculo verde do disco detetado).
- **Processado** — CLAHE + denoising + nitidez.
- **Zoom** — ampliação centrada na coroa/borda.
- **Parâmetros** — CLAHE clip limit, força do denoising, nitidez e zoom,
  com valores iniciais vindos do `config.yaml`.

## Linha de comando

Os subcomandos completos (`astroframe --help`):

| Comando | Descrição |
|---|---|
| `serve` | Inicia a interface Gradio |
| `process` | Processa fotos em lote (`--input a.jpg b.jpg --output-dir pasta/`) |
| `video` | Processa um vídeo (`--mode stabilize\|enhance\|stack`) |
| `config-template` | Gera `config.yaml` com os valores por omissão |

### Fotografias em lote

```bash
astroframe process --input foto1.jpg foto2.jpg --output-dir outputs/ --config config.yaml
```

- Cada ficheiro é processado de forma independente: se um estiver corrompido,
  o lote **continua** e o resumo sai no fim (contagem de falhas).
- As saídas são PNG com o sufixo `_processed.png`.

### Vídeo

```bash
astroframe video --input eclipse.mp4                                  # modo enhance (padrão)
astroframe video --input eclipse.mp4 --mode stabilize                 # só centra o disco
astroframe video --input eclipse.mp4 --mode stack --stack-n 20        # stack dos 20 melhores frames
astroframe video --input eclipse.mp4 --mode enhance --fast            # sem denoising (rápido)
astroframe video --input eclipse.mp4 --output saida.mp4               # nome do ficheiro de saída
```

- **enhance / stabilize** — o vídeo é **estabilizado frame a frame** e
  re-exportado como MP4 (`<nome>_stabilized.mp4` por omissão) com barra de
  progresso. A anti-trepidação temporal suaviza o centroide (EMA) e mantém o
  último deslocamento quando um frame não tem deteção.
- **stack** — seleciona os N frames mais nítidos (lucky imaging), **centraliza
  cada um** e combina-os (mediana por omissão) num único PNG.
- `--fast` omite o passo mais lento (denoising) e reduz muito o tempo de
  processamento em vídeos grandes.

## Configuração (config.yaml)

Gere o modelo e edite apenas o necessário (o resto mantém os valores por
omissão):

```bash
astroframe config-template --output config.yaml
```

Todos os campos e tipos:

### `clahe`
| Campo | Tipo | Padrão | Descrição |
|---|---|---|---|
| `clip_limit` | float | `3.0` | Limite de recorte do CLAHE (maior = mais contraste) |
| `tile_grid_size` | int | `8` | Tamanho da grelha (reduzido automaticamente se a imagem for mais pequena) |

### `denoise`
| Campo | Tipo | Padrão | Descrição |
|---|---|---|---|
| `h` | float | `5.0` | Força do denoising (subir com ISO alto ~ σ do ruído) |
| `template_window_size` | int | `7` | Janela de template do Non-Local Means |
| `search_window_size` | int | `21` | Janela de pesquisa (menor = mais rápido) |

### `unsharp`
| Campo | Tipo | Padrão | Descrição |
|---|---|---|---|
| `sigma` | float | `2.0` | Desvio do desfoque gaussiano |
| `amount` | float | `0.5` | Intensidade da nitidez |

### `stabilizer`
| Campo | Tipo | Padrão | Descrição |
|---|---|---|---|
| `min_radius` / `max_radius` | int | `30` / `400` | Limites de raio do disco (ajustados à resolução do frame) |
| `dp`, `min_dist`, `param1`, `param2` | — | `1.2` / `100` / `50` / `30` | Parâmetros do `HoughCircles` |
| `gaussian_kernel_size`, `gaussian_sigma` | — | `9` / `2.0` | Desfoque de pré-deteção |
| `contour_fallback` | bool | `true` | Fallback por contornos quando o Hough falha |
| `auto_crop` | bool | `true` | Remove as bordas pretas da translação (sem cortar o disco) |
| `jitter_alpha` | float | `0.5` | Suavização EMA do centroide (1 = sem suavização) |

### `lucky`
| Campo | Tipo | Padrão | Descrição |
|---|---|---|---|
| `min_sharpness` | float\|null | `null` | Limiar fixo de nitidez; `null` = estimar do vídeo |
| `sharpness_percentile` | float | `25.0` | Percentil usado na estimativa automática |
| `gaussian_kernel_size`, `gaussian_sigma` | — | `5` / `1.5` | Desfoque antes do Laplaciano |

### `stacking`
| Campo | Tipo | Padrão | Descrição |
|---|---|---|---|
| `n_best` | int | `10` | Nº de frames a empilhar (usado se `--stack-n` não for dado) |
| `use_median` | bool | `true` | `true` = mediana (robusta), `false` = média |

**Validação:** chaves desconhecidas e tipos inesperados geram avisos no log
(por exemplo `clip_limit: "abc"`), mas nunca derrubam o arranque.

## Workflow de vídeo

1. **Captura** — gravar o eclipse com a câmara estática; trepidação lenta é
   aceitável (a estabilização absoluta pelo disco compensa-a).
2. **Pré-seleção** (opcional): `astroframe video --input clip.mp4 --mode stack --stack-n 30`
   devolve um único PNG com o melhor "instantâneo" possível.
3. **Estabilização total**: `astroframe video --input clip.mp4 --mode enhance`
   — centro constante e imagem melhorada. Para vídeos 1080p/4K use `--fast`.
4. **Pós-execução**: juntar o áudio com ffmpeg (ver limitações).

## Limitações e notas

- **Áudio**: o exportador usa OpenCV e **não copia áudio**:
  ```bash
  ffmpeg -i original.mp4 -i processado.mp4 -c copy -map 0:a -map 1:v saida.mp4
  ```
- **Denoising lento**: ~1 s/frame a 480p; a 1080p pode chegar a vários
  segundos por frame. `--fast` ou reduzir `search_window_size`.
- **Stacking de alta resolução**: frames acima de 1080p empilhados usam
  float32 em memória (aviso no log) — reduza `n_best` se exceder o necessário.
- **Frames sem disco**: `center_and_stabilize` devolve o frame inalterado
  (com aviso); em vídeo, o `AntiJitterStabilizer` reaproveita o último
  deslocamento válido.
- **RIFE** (interpolação em saltos) é opcional e exige PyTorch; ver [API.md](API.md).