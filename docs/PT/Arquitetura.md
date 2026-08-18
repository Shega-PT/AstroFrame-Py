# Arquitetura da Solução

O sistema divide-se em três etapas principais: 

- Estabilização por Geometria,
- Melhoria Automática de Imagem,
- Interface Mínima

---

# 1 Rastreamento e Estabilização de Vídeo (Cancelamento de Trepidação)

Em vez de tentar estabilizar o fundo (que é escuro ou uniforme), o algoritmo deteta o centroide do Sol/Lua em cada frame e move a imagem para manter o astro sempre no centro exato do frame.

Deteção de Forma: Utiliza a Transformada de Hough para Círculos (cv2.HoughCircles) ou deteção de contornos maiores (cv2.findContours).

Reenquadramento Automático: Se a câmara der um salto rápido, o algoritmo calcula o vetor de deslocamento do centro $(x, y)$ do Sol em relação ao centro do frame e re-alinha a imagem.

Rejeição de Frames Borrados (Lucky Imaging): Frames capturados durante movimentos muito rápidos da camcorder ficam desfocados por motion blur. O Python pode calcular a variância do Laplaciano da imagem (nível de nitidez) e ignorar os frames muito borrados.

---

# 2 Processamento e Melhoria Automática (Fotos e Vídeos)

Gama e Contraste Adaptativo (CLAHE): O algoritmo Contrast Limited Adaptive Histogram Equalization melhora os detalhes da coroa solar ou da transição de iluminação sem estourar o brilho.

Redução de Ruído: Aplicação de filtro Non-Local Means Denoising (especialmente útil nas fotos do A52 com ISO alterado).

Máscara de Nitidez (Unsharp Masking): Destaca as bordas exatas da Lua sobre o disco solar

---

# 3 Frontend Mínimo (Gradio ou Streamlit)

A forma mais rápida de criar a interface gráfica em Python é utilizar o Gradio. Ele roda localmente no navegador com seletores de arquivos, sliders e visualização lado a lado (Antes vs. Depois).

Exemplo do Código da Pipeline (Python)Aqui está uma implementação funcional de base utilizando OpenCV para a lógica de processamento e Gradio para a interface.

```Python
import cv2
import numpy as np
import gradio as gr


def auto_enhance_frame(img):
    """
    Aplica equalização adaptativa e nitidez focada em astrofotografia.
    """
    # Converter para escala de cinza para análise de bordas
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1. Filtro CLAHE para realçar contraste da coroa/borda sem estourar
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

    # Aplicar no canal L do espaço de cor LAB para preservar as cores originais
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l_enhanced = clahe.apply(l)
    updated_lab = cv2.merge((l_enhanced, a, b))
    enhanced_bgr = cv2.cvtColor(updated_lab, cv2.COLOR_LAB2BGR)

    # 2. Redução de ruído suave
    denoised = cv2.fastNlMeansDenoisingColored(enhanced_bgr, None, 5, 5, 7, 21)

    # 3. Unsharp Masking para destacar o contorno da Lua
    gaussian_3 = cv2.GaussianBlur(denoised, (0, 0), 2.0)
    unsharp = cv2.addWeighted(denoised, 1.5, gaussian_3, -0.5, 0)

    return unsharp


def center_and_stabilize(img):
    """
    Localiza o Sol/Lua por geometria e centraliza a imagem.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    # Busca por formas circulares
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=100, param1=50, param2=30, minRadius=30, maxRadius=400
    )

    if circles is not None:
        circles = np.uint16(np.around(circles))
        # Pega o maior círculo encontrado (Sol)
        x, y, r = circles[0][0]

        # Calcular deslocamento até o centro da imagem
        h, w = img.shape[:2]
        center_x, center_y = w // 2, h // 2
        dx = center_x - x
        dy = center_y - y

        # Matriz de translação
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        stabilized = cv2.warpAffine(img, M, (w, h))
        return stabilized

    return img


def process_image_pipeline(input_image):
    if input_image is None:
        return None
    # 1. Centralizar pelo disco solar
    stabilized = center_and_stabilize(input_image)
    # 2. Aplicar melhorias automáticas
    result = auto_enhance_frame(stabilized)
    return result


# Interface Mínima com Gradio
with gr.Blocks(title="Astro Auto-Enhancer") as demo:
    gr.Markdown("# 🌒 Astro Auto-Enhancer")
    gr.Markdown("Melhoria automática e estabilização geométrica para fotos e vídeos de astros.")

    with gr.Row():
        input_img = gr.Image(label="Foto/Frame Original")
        output_img = gr.Image(label="Resultado Estabilizado e Processado")

    btn = gr.Button("Processar Imagem", variant="primary")
    btn.click(fn=process_image_pipeline, inputs=input_img, outputs=output_img)

if __name__ == "__main__":
    demo.launch()
```

---

# Módulos para Expansão da Pipeline de Vídeo

Para estender este pipeline aos vídeos da Samsung Digital Camcorder com movimentos rápidos:

1. Stacking / Lucky Imaging (Bibliotecas recomendadas):

   - Use o scikit-image ou OpenCV para ler o vídeo frame a frame.
   - Calcule cv2.Laplacian(frame, cv2.CV_64F).var(). Se o valor for menor que um limite pré-definido, descarte o frame (foi capturado durante uma guinada rápida da câmara).

2. Interpolação de Movimento Se Acontecerem "Saltos":

   - Se a câmara se mover muito rápido e perder alguns frames bons, você pode utilizar o modelo RIFE (Real-Time Intermediate Flow Estimation) via PyTorch para interpolar frames suavemente entre as correções manuais de elevação e direção.

---

# 4 Arquitetura de IA (v0.7.0)

A camada de IA acrescenta auto-tuning e pequenas redes neuronais com um
**núcleo NumPy puro** (PyTorch é opcional e nunca obrigatório). Tudo está
**desligado por omissão** (`tuning.enabled=false`, `ai.*`); um modelo em
falta ou corrompido degrada silenciosamente e nunca bloqueia a pipeline.

## 4.1 Registry de parâmetros (`ai.params`)

Fonte única de verdade dos intervalos seguros, passos e deltas de treino de
todos os parâmetros ajustáveis (17 parâmetros em `detect`/`enhance`/
`stabilizer`). Cada valor aprendido — do validator, do feedback por estrelas
ou do auto-tuning — passa por `clamp_value`, que aplica o clamp, arredonda
ints e força a paridade ímpar (kernels Gaussianos). A estabilidade do treino
depende deste ponto nunca falhar.

## 4.2 Auto-tuning (`ai.tuner`)

- **Avaliação por proxy** (`ProxyEval`): corre a pipeline nas amostras de
  calibração (`samples/` + `Logs/train/calibration.json`, ground truth desenhado à mão)
  e mede o IoU médio entre discos detetados (Hough) e esperados, com
  penalizações por discos extra/em falta; alguns frames melhorados são ainda
  pontuados com estrelas (`ai.score`). Os frames são reduzidos a ~480p e os
  relatórios ficam em cache pelos parâmetros efetivos.
- **Pesquisa** (`BoundedHillClimb`): subida de colina determinística com
  passes +passo/−passo sobre o registry, momentum (o passo duplica após 2
  aceites na mesma direção), redução do passo nas falhas, recozimento
  opcional (aceita piores com probabilidade `exp(−Δ/T)`) e orçamento de
  tempo. Parâmetros caros (denoising) são tentados apenas em passes pares.
- **Pré-semente LSTM** (`_lstm_seed`): antes da pesquisa, o histórico de
  feedback do perfil é usado pelo `ai.lstm.LSTMTuner` para prever deltas;
  só são mantidos se melhorarem o objetivo do proxy.
- O resultado é registado na tabela `tuning` do banco de aprendizagem e
  aplicado automaticamente nas execuções seguintes por `apply_learned` —
  a "memória" da IA entre utilizações.

## 4.3 LSTM (`ai.lstm`)

Célula LSTM de uma camada implementada à mão em NumPy (backprop-through-time,
portas i/f/o/g vetorizadas). Duas utilizações:

- **`LSTMTuner`** — treinado offline sobre o histórico de feedback (9
  características por execução, janelas deslizantes) para prever os deltas
  dos 5 parâmetros visuais; usado como pré-semente do auto-tuning.
- **`TrajectoryPredictor`** — prevê o centroide do disco no frame seguinte:
  regressão linear (mínimos quadrados) como base e refinamento LSTM opcional
  (célula 2→8 treinada em trajetórias sintéticas). Integrado no
  `AntiJitterStabilizer`: com `ai.lstm_trajectory`, os frames sem deteção
  usam a previsão em vez de congelar o último deslocamento.

Os modelos são `.npz` versionados (`Logs/weights/lstm.npz`); ficheiros
corrompidos ou com versão errada carregam como `None` (fallback silencioso).

## 4.4 CNN (`ai.cnn`)

Rede convolucional pequena em NumPy puro (2× conv 3×3 + ReLU + pooling +
cabeça), treino offline determinístico e gradientes verificados por
diferenças finitas. Duas cabeças permutáveis:

- **Residual** (`fit_residual` + `ResidualEnhancer`): aprende `r = y − x`
  para remover ruído/smearing; aplicada por `enhance_image` a seguir ao
  unsharp (canal L do LAB, tiles 64×64 com overlap) com `ai.cnn_enhance`.
- **Classificador** (`fit_classifier` + `DiskFilter`): pontua cada deteção
  com P(disco); `find_all_disks` descarta os candidatos abaixo de
  `ai.disk_filter` (0–1) — a lista detetada **nunca** é esvaziada.

Modelos: `Logs/weights/enhancer_cnn.npz` e `Logs/weights/disk_filter.npz`.

### Treino dos modelos

- **CNN de deteção** — o treino automático do `validator.py` recolhe patches
  de discos em cada série (`cnn_positives` dos círculos do guia,
  `cnn_negatives` de formas rejeitadas + recortes aleatórios determinísticos
  excluídos por IoU) e retreina o `DiskFilter` entre séries (`--cnn`). O
  resultado é comparado com o **campeão** da tabela `models` por `score`; se
  estritamente melhor é promovido para o caminho canónico e a série seguinte
  faz warm-start dos pesos do campeão.
- **CNN residual** — `enhancer_trainer.py` (GUI Válido/Rejeitado ou `--auto`)
  acumula pares (entrada, saída CNN) / (entrada, entrada) e chama
  `train_enhancer_round`, que compara por `mean_delta` (1 − MSE) com o
  campeão e promove se melhor.
- Ambos os treinadores registam cada ronda nas tabelas `models` e `logs`.

## 4.5 Segurança e ciclo de vida

1. Tudo desligado por omissão; cada componente decide por si se está
   disponível (`available`/`torch_available`/`load()`).
2. Modelos versionados; carga falhada → `None` → comportamento idêntico ao
   modo sem IA.
3. Todos os valores aprendidos passam pelo clamp do registry.
4. A pipeline principal não depende de nenhum destes módulos: são camadas
   opcionais por cima de `core/`.