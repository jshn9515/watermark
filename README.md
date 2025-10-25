# Watermark Embedding and Detection in Diffusion Models

This repository serves as an implementation and reproduction framework for watermark embedding and detection methods introduced in multiple research studies. It is designed to reproduce and validate the findings of these papers, offering a consistent environment for experimental comparison and further development. For more details about the original research and implementation, please refer to the link below.

- [A recipe for watermarking diffusion models](arXiv-2303.10137)

## For Autodl Users

If you are using the Autodl platform to run your experiments, please note that there are some restrictions when cloning GitHub repositories or downloading datasets from official sources. Here are a few tips to help you get started.

### Enabling network acceleration

Before cloning this repository, enable the acceleration service with:

```bash
source /etc/network_turbo
```

After finishing your tasks, remember to turn it off:

```bash
unset http_proxy && unset https_proxy
```

> [!NOTE]
> The Autodl platform does not guarantee the stability of the acceleration service. Use at your own risk.

### Accessing datasets

If you are using datasets from `torchvision`, check the Autodl public dataset registry first. If the dataset you need is available there, you can directly copy it without downloading from the official website. Otherwise, you’ll need to manually download the dataset and upload it to your workspace.

To copy a public dataset to your workspace:

```bash
cp -r /autodl-pub/data/<dataset_name> ./datasets
```
