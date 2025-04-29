# StableDiffusion_Reproduction

Reproduce Stable Diffusion (Latent Diffusion Model)[^1][^2]. Most of the code is referenced from explainingai-code/StableDiffusion-PyTorch[^3].

To improve readability and modularity, this project uses separate Git branches for each major component:

- **main**: Overview of this repository.
- [**vqvae-mnist**](https://github.com/QidiLiu/StableDiffusion_Reproduction/tree/vqvae-mnist): The most basic reproduction of VQVAE.
- [**ldm-mnist-uncond**](https://github.com/QidiLiu/StableDiffusion_Reproduction/tree/ldm-mnist-uncond): The most basic reproduction of LDM without condition input.
- [**ldm-mnist-cond**](https://github.com/QidiLiu/StableDiffusion_Reproduction/tree/ldm-mnist-cond): The most basic reproduction of LDM with condition input.

[^1]: Van Den Oord, A., & Vinyals, O. (2017). Neural discrete representation learning. Advances in neural information processing systems, 30.
[^2]: Rombach, R., Blattmann, A., Lorenz, D., Esser, P., & Ommer, B. (2022). High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition (pp. 10684-10695).
[^3]: [GitHub - explainingai-code/StableDiffusion-PyTorch](https://github.com/explainingai-code/StableDiffusion-PyTorch/tree/main)
