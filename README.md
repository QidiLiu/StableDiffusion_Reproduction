# StableDiffusion_Reproduction - vqvae-mnist

Reproduce VQVAE for Latent Diffusion Model[^1][^2]. Most of the code is referenced from explainingai-code/StableDiffusion-PyTorch[^3].

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip3 install PyYAML tqdm
```

```bash
python -m script.train
python -m script.test
```

[^1]: Van Den Oord, A., & Vinyals, O. (2017). Neural discrete representation learning. Advances in neural information processing systems, 30.
[^2]: Rombach, R., Blattmann, A., Lorenz, D., Esser, P., & Ommer, B. (2022). High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition (pp. 10684-10695).
[^3]: [GitHub - explainingai-code/StableDiffusion-PyTorch](https://github.com/explainingai-code/StableDiffusion-PyTorch/tree/main)
