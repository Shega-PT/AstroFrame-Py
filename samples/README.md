# Samples (exemplos para calibração)

Esta pasta guarda **fotos e vídeos de exemplo** usados para **calibrar** o
AstroFrame: a interface de calibração carrega-os, permite ajustar os círculos
(astros) à mão e valida a deteção automática contra o ground truth.

```bash
python calibrate.py                 # abre a interface de calibração
python calibrate.py --samples samples --port 7861
astroframe calibrate                # equivalente (CLI instalada)
```

## Estrutura recomendada

A pasta é varrida **recursivamente** — qualquer organização serve — mas a
estrutura sugerida é separar por tipo e por assunto:

```
samples/
├── images/          # fotografias (jpg/png/bmp/tif/webp)
│   ├── eclipse/     # eclipse solar/lunar
│   ├── lua/         # Lua (cheia, crescente, ...)
│   ├── sol/         # Sol (manchas, proeminências, ...)
│   └── planetas/    # planetas (Mercúrio, Vénus, Júpiter, ...)
├── videos/          # gravações (mp4/avi/mov/mkv/m4v)
│   ├── eclipse/
│   └── lua/
└── calibration.json # ground truth manual (gerado pela calibração)
```

## Como funciona a calibração

1. **Imagens** entram tal como estão; **vídeos** contribuem com N frames
   amostrados de forma equidistante e determinística (8 por omissão).
2. Na interface: o círculo pré-preenchido é o ground truth guardado (ou a
   deteção automática como ponto de partida). **Arrastar** uma camada move o
   círculo, **pintar** por cima adiciona, **borracha** remove.
3. **Guardar ajustes** grava os círculos em `samples/calibration.json`.
4. **Validar todas as amostras** corre a deteção automática em tudo e compara
   com o manual: recall, precisão, IoU, erros de centro/raio + sugestões de
   parâmetros.

> O repositório não inclui mídia real por defeito (`.gitignore` ignora
> `*.mp4`/`*.MOV`/`*.mkv`). O `calibration.json` também é local (referencia
> os ficheiros desta pasta).
