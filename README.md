
# Diffusion Posterior Sampling MNIST

This repository is a slightly modified version of [Diffusion Posterior Sampling Repository](https://github.com/DPS2022/diffusion-posterior-sampling). For a given inverse problem (inpainting), the method generates samples from the image's posterior distribution given the measurements, using a pre-trained diffusion model as an image prior.


## Installation

To install the required dependencies, run the following commands:

```bash
git clone https://github.com/cekmekci/diffusion-posterior-sampling-mnist.git
cd diffusion-posterior-sampling-mnist/
git clone https://github.com/VinAIResearch/blur-kernel-space-exploring bkse
git clone https://github.com/LeviBorodenko/motionblur motionblur
```

## Configuration Files

Pretrained models are available in the `models` directory. The configuration of these models can be found in the `configs/model_config.yaml` file, while further details about the diffusion process are located in the `configs/diffusion_config.yaml` file. The imaging setup configuration is specified in the `configs/inpainting_config.yaml` file.

## Training

The training code utilized in this repository is based on the code provided [here](https://github.com/cekmekci/diffusion-model-mnist). Please ensure that the training configurations in that repository match the configurations specified in this repo through the `.yaml` files.

## Running the Code

To run the code, follow these steps:

1. Ensure that all the configuration files are properly set up according to your requirements.
2. Execute the following command to run the code:

```
python sample_dps.py --model_config=configs/model_config.yaml --diffusion_config=configs/diffusion_config.yaml --task_config=configs/inpainting_config.yaml
```



