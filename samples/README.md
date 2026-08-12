# Samples

Coloque aqui fotos e vídeos de exemplo (frames do eclipse, gravações da camcorder, etc.) para testar a pipeline:

```bash
astroframe process --input samples/eclipse.jpg --output-dir outputs/
astroframe video --input samples/eclipse.mp4 --mode enhance
astroframe video --input samples/eclipse.mp4 --mode stack --stack-n 20
```

O repositório não inclui mídia real por defeito.