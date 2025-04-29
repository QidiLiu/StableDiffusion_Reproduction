# StableDiffusion_Reproduction - ldm-mnist-cond

The most basic reproduction of LDM with condition input.

Prerequisite:
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip3 install PyYAML tqdm einops
```

Run pretrained VQVAE model to get latent files for MNIST dataset (for more details please see the test script of [StableDiffusion_Reproduction - vqvae-mnist](https://github.com/QidiLiu/StableDiffusion_Reproduction/tree/vqvae-mnist)), then edit yaml files under folder `config` and run the script:
```bash
python -m script.train
python -m script.test
```
