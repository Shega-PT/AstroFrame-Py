# Guia de Utilização — AstroFrame

Guia prático para instalar, configurar e executar o AstroFrame. A especificação
da solução está em [Arquitetura.md](Arquitetura.md) e a referência de código
em [API.md](API.md).

## Índice

1. [Instalação](#instalação)
2. [Interface web (Gradio)](#interface-web-gradio)
3. [Calibração](#calibração)
4. [Validação e treino da deteção](#validação-e-treino-da-deteção)
5. [Auto-tuning (IA)](#auto-tuning-ia)
6. [Linha de comando](#linha-de-comando)
7. [Configuração (config.yaml)](#configuração-configyaml)
8. [Workflow de vídeo](#workflow-de-vídeo)
9. [Limitações e notas](#limitações-e-notas)

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

A interface tem três separadores:

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

### Separador Auto-tune

Otimiza **todos os parâmetros da pipeline** contra a pasta de samples
(`samples/` + ground truth `calibration.json`) — deteção (IoU contra o guia)
+ melhoria (estrelas) — com uma pesquisa determinística e com orçamento de
tempo:

- **Pasta de samples** — onde vive o material de calibração (por omissão
  `samples`).
- **Orçamento (segundos)** — tempo permitido para a otimização (por omissão
  60).
- **Parâmetros** — subconjunto de parâmetros ajustáveis (vazio = todos).
- **Recozimento** — permite aceitar candidatos piores para escapar de mínimos
  locais.
- **Registar no banco de aprendizagem** — guarda o resultado na tabela
  `tuning`, para ser aplicado automaticamente nas próximas execuções do mesmo
  perfil (ver [Base de aprendizagem](#base-de-aprendizagem-onde-fica-guardado)).
- **Otimizar** — mostra o progresso e depois o relatório (parâmetro · base →
  ajustado · delta · passo, objetivo, estrelas, deteção, nº de avaliações e
  tempo decorrido) mais a **configuração otimizada** em JSON.
- **Limpar histórico de auto-tuning** — apaga o histórico `tuning` da base.

### Base de aprendizagem (onde fica guardado)

As execuções (parâmetros usados, métricas e avaliações) ficam num ficheiro
SQLite em `~/.astroframe/feedback.db`. Pode mudar o local com a variável de
ambiente `ASTROFRAME_FEEDBACK_DB` (por exemplo, para partilhar a aprendizagem
entre várias máquinas). Os **resultados do auto-tuning** ficam na mesma base
(tabela `tuning`) e são aplicados automaticamente às próximas execuções do
mesmo perfil, juntamente com os ajustes por estrelas (`apply_learned`).

## Calibração

O AstroFrame inclui uma **interface dedicada à calibração**: carrega as fotos e
vídeos da pasta [samples/](../../samples/README.md), mostra a deteção com
círculos (bounding box circulares) que podem ser **adicionados, removidos e
movidos manualmente**, e valida a deteção automática contra o ground truth em
**todas** as amostras.

```bash
python calibrate.py                          # janela desktop (tkinter)
python calibrate.py --ui gradio              # interface no navegador
python calibrate.py --samples samples
astroframe calibrate --samples samples       # equivalente (CLI instalada)
```

### O que entra na calibração

- **Imagens** (jpg/png/bmp/tif/webp) — cada uma é um item.
- **Vídeos** (mp4/avi/mov/mkv/m4v) — cada um contribui com 8 frames amostrados
  de forma equidistante e determinística (reproduzível na validação).
- A pasta é varrida recursivamente: organiza por assunto como quiseres
  (eclipse, lua, sol, planetas — subpastas em `samples/images/` e
  `samples/videos/`).

### Fluxo de trabalho

O fluxo divide-se em **duas passagens**:

1. **1.ª passagem — manual (deteção desligada por omissão):**
   1. **Escolher a amostra** — a lista do painel mostra todos os itens
      (`IMG …` para imagens, `VID … #frame` para frames de vídeo).
   2. **Desenhar os astros** — no canvas:
      - **clique em espaço vazio** → cria um círculo (ou elipse, conforme o
        seletor) nesse ponto;
      - **arrastar o interior** da forma selecionada → move o centro;
      - **arrastar a pega direita** → ajusta o raio horizontal; **pega de
        topo** → raio vertical (elipse); os sliders Raio X/Raio Y fazem o
        mesmo ajuste fino em tempo real;
      - **roda do rato** → zoom no cursor; arrastar com o botão direito/médio
        → deslocar; **Delete** elimina a forma selecionada, setas movem 1 px
        (Shift = 10 px).
   3. **Guardar (Ctrl+S)** — grava o ground truth do item em
      `samples/calibration.json` (ficheiro local, ignorado pelo git).
2. **2.ª passagem — validação (liga "Deteção automática ao carregar):**
   4. **Amostras sem ground truth** são preenchidas automaticamente pela
      deteção; as guardadas abrem exatamente como as deixaste. **Ajusta** o
      que for preciso (mesmos gestos) e volta a guardar.
   5. **Validar tudo** — corre a deteção automática em todas as amostras e
      compara com o ground truth manual: por amostra e global devolve
      recall, precisão, IoU médio, erro do centro (px) e do raio (%),
      falsos negativos/positivos, um **score de calibração (0–100)** e
      **sugestões de parâmetros** (ex.: baixar `min_radius` se discos
      pequenos falham, subir `param2` se houver deteções falsas).
   6. Os sliders de parâmetros re-correm a deteção ao largar (com a deteção
      ligada), para afinar `param2`/raios/distância sem sair da amostra.

> Elipses são guardadas como objetos (com `ry` no JSON); a validação usa
> IoU por máscara quando há elipses e o raio geométrico para os erros.

### Para que serve a calibração

Os círculos manuais são o "resposta certa" que o sistema compara com a
deteção automática. Com uma pasta variada (eclipses, Lua, Sol, planetas —
discos grandes e pequenos, alto e baixo contraste), a validação mostra onde a
deteção falha e o que ajustar no `config.yaml` antes de processar o material
real.

## Validação e treino da deteção

O `validator.py` usa esse mesmo ground truth para **afinar a deteção por
forma**: percorre as amostras uma a uma, mostra o que o `find_all_disks`
encontrou (disco principal + companheiros de eclipse) sobre a imagem, e
aprende a distinguir deteções corretas de falsas.

```bash
python validator.py                          # janela desktop (tkinter)
python validator.py --check                  # relatório sem interface
python validator.py --auto --series 3        # treino automático (3 séries)
python validator.py --auto --iou 0.7         # exigência mínima com o guia
python validator.py --reset-state --check    # recomeça do zero e verifica
```

### Como funciona

1. **Ronda manual** — em cada amostra vês a deteção e o guia manual
   (`calibration.json`); **Aceitar/Rejeitar** diz se a forma está certa.
   - Com um **preview ao detetar**: a deteção desenha-se em cima da imagem em
     tempo real antes de pedir o veredito.
2. **Treino automático (`--auto`)** — sem janela: cada série re-deteça as
   amostras e **auto-avalia** cada forma contra o guia (IoU mínimo configurável
   com `--iou`). Cada forma correta dá **recompensa** aos parâmetros que a
   encontraram; cada forma falsa ou falhada dá **punição** (dobrada para
   rejeições teimosas). O processo termina com 100% do material processado.
3. **Pesos treináveis (7)** — `param2`, `param1`, `dp`,
   `gaussian_kernel_size`, `gaussian_sigma`, `occluded_ratio`,
   `occluded_ring`: cada um tem deltas de recompensa/punição, limites mínimos
   e máximos e histórico de aplicação.
4. **Relatório final** — score da deteção, pesos treinados com **tooltips ⓘ**
   a explicar cada parâmetro, e o botão **Salvar** exporta a configuração
   treinada para `trained_config.json` (na pasta de samples), pronta a usar no
   sistema real.

### Estado

- O progresso fica em `validator_state.json` (na pasta de samples, por
  omissão): rondas, séries, histórico de pesos/deltas e o IoU mínimo atual.
- `--reset-state` apaga tudo (incluindo o histórico) e recomeça; sozinho abre
  depois a interface, combinado com `--check`/`--auto` corre sem janela.
- `--state ficheiro.json` muda o local do estado; `--export saida.json` muda o
  destino do relatório salvo.

## Auto-tuning (IA)

[v0.7.0] Otimiza automaticamente os **parâmetros de deteção/melhoria** contra
o material de calibração. Precisa de uma pasta de samples com ground truth
(`samples/` + `calibration.json`, ver [Calibração](#calibração)) — sem ele o
proxy não consegue pontuar a deteção e o ajuste não tem com que comparar.

```bash
astroframe autotune --samples samples --budget 60
astroframe autotune --samples samples --seed 42 --no-anneal \
    --params param2,clip_limit,denoise.h
astroframe autotune --samples samples --profile "video@5616x3744" --export tuned.json
astroframe autotune --samples samples --reset       # limpa o histórico de tuning primeiro
```

| Opção | Descrição |
|---|---|
| `--samples DIR` | Pasta com as amostras e `calibration.json` (por omissão `samples`) |
| `--budget N` | Orçamento de tempo do otimizador em segundos (por omissão 60) |
| `--seed N` | Semente determinística (por omissão 42) |
| `--no-anneal` | Desliga o recozimento (não aceita candidatos piores) |
| `--params p1,p2` | Subconjunto de parâmetros ajustáveis (por omissão: todos os registados) |
| `--profile NOME` | Perfil de câmara usado no banco de aprendizagem |
| `--export FILE` | Exporta a configuração otimizada (por omissão `samples/trained_config.json`) |
| `--reset` | Limpa o histórico de tuning do perfil antes de correr |
| `--config FILE` | `config.yaml` base (a pesquisa parte dele) |

### Como funciona

1. **Avaliação por proxy** — cada configuração candidata corre a pipeline
   sobre as amostras reduzidas a ~480p (escala de trabalho 0.5, nunca
   ampliadas); a deteção é comparada com o ground truth (IoU médio entre
   discos detetados e esperados, penalizações por discos extra/em falta) e
   alguns frames melhorados são pontuados com estrelas. Os resultados ficam
   em cache pelos parâmetros efetivos.
2. **Pesquisa** — subida de colina em etapas: passes +passo/−passo por
   parâmetro com momentum, redução do passo nas falhas e recozimento opcional
   para escapar de mínimos locais; determinística (semente) e limitada ao
   orçamento de tempo e aos intervalos seguros do registry de parâmetros.
   Parâmetros caros (denoising) são tentados com menos frequência. A pesquisa
   pode partir da previsão LSTM do histórico do perfil quando esta melhora o
   objetivo.
3. **Resultado** — os deltas ajustados são **registados no banco de
   aprendizagem** (tabela `tuning`) e **aplicados automaticamente às próximas
   execuções** do mesmo perfil (`apply_learned`, juntamente com os ajustes por
   estrelas); a configuração otimizada é também exportada para o ficheiro
   `--export` (JSON: parâmetros efetivos, deltas, relatório do proxy e secção
   `stabilizer`).

> Toda a IA está desligada por omissão — o auto-tuning só corre quando o
> invocas (CLI ou separador *Auto-tune*); o resultado aprendido, no entanto,
> continua a ser aplicado pelo `apply_learned` tal como as avaliações manuais.

## Linha de comando

Os subcomandos completos (`astroframe --help`):

| Comando | Descrição |
|---|---|
| `serve` | Inicia a interface Gradio |
| `process` | Processa fotos em lote (`--input a.jpg b.jpg --output-dir pasta/`) |
| `video` | Processa um vídeo (`--mode stabilize\|enhance\|stack`) |
| `config-template` | Gera `config.yaml` com os valores por omissão |
| `calibrate` | Abre a interface de calibração (`--samples pasta/`) |
| `autotune` | Auto-afina os parâmetros contra as amostras (`--samples pasta/`, `--budget N`, `--seed N`, `--no-anneal`, `--params p1,p2`, `--profile NOME`, `--export ficheiro`, `--reset`); ver [Auto-tuning (IA)](#auto-tuning-ia) |

A validação/treino da deteção é um script à parte (ver
[Validação e treino da deteção](#validação-e-treino-da-deteção)):
`python validator.py [--check|--auto|--reset-state|--iou N]`.

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

### `tuning` (auto-tuning)
| Campo | Tipo | Padrão | Descrição |
|---|---|---|---|
| `enabled` | bool | `false` | Liga/desliga o auto-tuning (toda a IA está desligada por omissão) |
| `budget_s` | float | `60.0` | Orçamento de tempo do otimizador (segundos) |
| `seed` | int | `42` | Semente determinística da pesquisa |
| `anneal` | bool | `true` | Aceita candidatos piores (recozimento) para escapar de mínimos locais |
| `proxy_scale` | float | `0.5` | Escala de trabalho da avaliação por proxy (cap 480p, nunca amplia) |
| `frames_per_sample` | int | `3` | Frames de vídeo por amostra pontuados com estrelas |
| `detection_weight` | float | `0.6` | Peso da deteção vs. estrelas no objetivo |
| `params` | list\|null | `null` | Subconjunto de parâmetros ajustáveis (`null` = todos os registados) |

### `ai` (redes neuronais)
| Campo | Tipo | Padrão | Descrição |
|---|---|---|---|
| `backend` | str | `numpy` | Backend de cálculo (`numpy` no núcleo; `torch` aceleração opcional) |
| `lstm_trajectory` | bool | `false` | Prevê a trajetória do disco (anti-trepidação) com a LSTM |
| `cnn_enhance` | bool | `false` | Passo residual CNN a seguir à máscara de nitidez (remoção de ruído/smearing) |
| `disk_filter` | float | `0.0` | Limiar de confiança da CNN para filtrar deteções (0 = desligado; nunca esvazia a lista) |

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
  deslocamento válido (ou usa a previsão de trajetória com `ai.lstm_trajectory`).
- **RIFE** (interpolação em saltos) é opcional e exige PyTorch; ver [API.md](API.md).
- **IA desligada por omissão** — `tuning.enabled`, `ai.lstm_trajectory`,
  `ai.cnn_enhance` e `ai.disk_filter` só ativam o que está explícito; sem
  modelos treinados em `~/.astroframe/` (`.npz` versionados) a pipeline
  degrada silenciosamente e nunca bloqueia.