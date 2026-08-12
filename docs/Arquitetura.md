# Arquitetura da Solução

O sistema divide-se em três etapas principais: 

- Estabilização por Geometria,
- Melhoria Automática de Imagem,
- Interface Mínima

---

# 1 Rastreamento e Estabilização de Vídeo (Cancelamento de Trepidação)

Em vez de tentar estabilizar o fundo (que é escuro ou uniforme), o algoritmo detecta o centroide do Sol/Lua em cada frame e move a imagem para manter o eclipse sempre no centro exato do frame.

Detecção de Forma: Utiliza a Transformada de Hough para Círculos (cv2.HoughCircles) ou detecção de contornos maiores (cv2.findContours).

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


def auto_enhance_eclipse_frame(img):
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
    result = auto_enhance_eclipse_frame(stabilized)
    return result


# Interface Mínima com Gradio
with gr.Blocks(title="Eclipse Auto-Enhancer") as demo:
    gr.Markdown("# 🌒 Eclipse Auto-Enhancer AI System")
    gr.Markdown("Melhoria automática e estabilização geométrica para fotos e frames do eclipse.")

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