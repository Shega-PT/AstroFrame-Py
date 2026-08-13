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

A interface tem dois separadores:

### Separador Imagem

- **Entrada** — carregar a foto/frame (formato arbitrátrio de imagem).
- **Estabilizado** — disco centralizado, com os discos detetados desenhados:
  **verde** = astro maior, **amarelo** = companheiros de eclipse (ex.: a Lua
  a entrar no Sol), **vermelho** = reflexos da lente.
- **Processado** — CLAHE + denoising + nitidez + **polimento por astros**
  (cada astro realçado individualmente e remontado sem costuras; fundo = média
  do fundo original; reflexos removidos).
- **Zoom** — ampliação centrada na coroa/borda.
- **Parâmetros** — CLAHE clip limit, força do denoising, nitidez, zoom e escala
  da coroa mantida no polimento, com valores iniciais vindos do `config.yaml`.
- **Avaliação automática** — estrelas (0–5) calculadas a partir de ruído,
  contraste, tamanho do disco e cor da coroa.
- **Avaliação manual + aprendizagem** — deslize o número de estrelas que o
  resultado merece e carregue em *Guardar avaliação manual*: a execução fica
  gravada e, na próxima vez com o mesmo perfil de câmara, os sliders **ajustam-se
  automaticamente** (com leve correção para boas avaliações, forte para más).
  O separador *Log de aprendizagem* mostra o histórico e o porquê de cada ajuste.

### Separador Vídeo

1. **Carregar o vídeo** (`.mp4/.avi/.mov`). Nesse momento os **metadados** são
   lidos — ffprobe (se instalado; senão apenas OpenCV: resolução/fps/frames)
   para vídeo, EXIF (PIL) para imagens — e são mostrados no painel
   **Proporção / qualidade / sugestões** (resolution, aspect ratio, fps, codec,
   bitrate, ISO, exposição, câmara). Os **sliders são pré-preenchidos** com as
   sugestões de otimização **e com o que a IA já aprendeu** (avaliações
   anteriores do mesmo perfil), mas continuam editáveis.
2. **Processar vídeo** — enquanto a pipeline corre:
   - **Esquerda (ao vivo)** — o frame original em tempo real com os discos
     detetados: **verde** = astro maior, **amarelo** = companheiros de eclipse,
     **vermelho** = reflexos da lente.
   - **Direita (resultado final)** — em frames bem espaçados, o resultado com
     todas as correções (estabilizado + CLAHE + denoising + nitidez +
     polimento por astros).
   - Barra de estado com o frame atual e o progresso, e **avaliação automática**
     do resultado final.
3. **Exportação opcional** — marcar *"Exportar vídeo processado (.mp4, sem
   áudio)"* para gravar o vídeo completo no fim (mesmo passe, sem saltar frames).
4. **Avaliação manual + aprendizagem** — tal como no separador Imagem, dê
   estrelas ao resultado do vídeo; o ajuste é aplicado nos próximos carregamentos
   do mesmo tipo de vídeo e aparece no log de aprendizagem.

### Base de aprendizagem (onde fica guardado)

As execuções (parâmetros usados, métricas e avaliações) ficam num ficheiro
SQLite em `~/.astroframe/feedback.db`. Pode mudar o local com a variável de
ambiente `ASTROFRAME_FEEDBACK_DB` (por exemplo, para partilhar a aprendizagem
entre várias máquinas).

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

### `polish`
| Campo | Tipo | Padrão | Descrição |
|---|---|---|---|
| `enabled` | bool | `true` | Liga/desliga o polimento |
| `corona_scale` | float | `1.6` | Linha de recorte (× raio do astro): entre o bordo e a linha a imagem é diluída até ao fundo |
| `feather` | float | `0.02` | Suavização (fração do raio) do contorno e das sobreposições entre astros |
| `background_fill` | bool | `true` | Fundo = média do fundo original (fora da linha de recorte) |
| `black_background` | bool | `false` | `true` = fundo preto puro em vez da média |
| `brightness` | float | `0.15` | Brilho extra adicionado aos astros (0 = só esticamento de contraste) |
| `remove_reflections` | bool | `true` | Remove círculos-ghost (centro fora do astro maior) |
| `reflection_min_radius` | int | `8` | Raio mínimo (px) de um reflexo a remover (mais pequeno = estrela/ruído) |

### `feedback` (aprendizagem)
| Campo | Tipo | Padrão | Descrição |
|---|---|---|---|
| `enabled` | bool | `true` | Guarda avaliações e aplica o ajuste aprendido aos sliders |
| `db_path` | str | `~/.astroframe/feedback.db` | Base SQLite com o histórico de execuções e ajustes |
| `learning_rate` | float | `0.3` | Fração do delta aplicado por execução |
| `user_weight` | float | `2.0` | Multiplicador quando o utilizador avalia manualmente |
| `history_limit` | int | `12` | Execuções recentes consideradas por perfil |

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