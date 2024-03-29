Implementations and results for [An Empirical Comparison of GANs and Normalizing Flows for Density Estimation.](https://arxiv.org/abs/2006.10175)

## Abstract

Generative adversarial networks (GANs) and normalizing flows are both approaches to density estimation that use deep neural networks to transform samples from an uninformative prior distribution to an approximation of the data distribution. There is great interest in both for general-purpose statistical modeling, but the two approaches have seldom been compared to each other for modeling non-image data. The difficulty of computing likelihoods with GANs, which are implicit models, makes conducting such a comparison challenging. We work around this difficulty by considering several low-dimensional synthetic datasets. An extensive grid search over GAN architectures, hyperparameters, and training procedures suggests that no GAN is capable of modeling our simple low-dimensional data well, a task we view as a prerequisite for an approach to be considered suitable for general-purpose statistical modeling. Several normalizing flows, on the other hand, excelled at these tasks, even substantially outperforming WGAN in terms of Wasserstein distance -- the metric that WGAN alone targets. Scientists and other practitioners should be wary of relying on WGAN for applications that require accurate density estimation.

## Dependency

- Python 3

Some important dependencies:

- Numpy
- Pandas
- Pytorch
- torchdiffeq
- Seaborn
- Matplotlib
- Raytune
- statsmodels (This package is necessary for Seaborn to use KDE bandwidth correctly.)

`requirements.txt` contains the full list.

Our experiments of [FFJORD](https://github.com/rtqichen/ffjord) and [Gaussianization Flows](https://github.com/chenlin9/Gaussianization_Flows) are based on existed repos, more detailed requirements can be found there.

## Usage

Different scripts are provided for different models. The dataset: unimodal/multimodal can be specified by args `--gu_num` and the number of GPU to use can be specified by args `--cuda`. Further details of `args` are provided within the script. Only some important args are explained here. We use 50,000 iterations and batch size 2,048 by default.

#### Unimodal Dataset:

WGAN: 

`python3 -m hyperopt_wgan --gu_num 1`

- use `--auto` to start a *baseline* random search with `AHSA`, use further `--residual_block, --clr, --prior uniform, --dropout, --auto_full` to use  *resnet, cyclic learning rate, uniform prior, dropout* tweaks or *includes all tweaks* as described in paper. 
- use `--niters` to specify the total iterations and use `--log_interval` to specify how many iterations to save a checkpoint (trained model and plot). 

FFJORD:

`python3 -m ffjord.gu_maf --gu_num 1`

Gaussianization flows:

`python3 -m Gaussianization_Flows.gu_gaus_flow --gu_num 1`



#### Multimodal

WGAN: 

`python3 -m hyperopt_wgan --gu_num 8`

FFJORD:

`python3 -m ffjord.gu_maf --gu_num 8`

Gaussianization flows:

`python3 -m Gaussianization_Flows.gu_gaus_flow --gu_num 8`



#### Trained models

Trained models are included in **model_results** and **wgan_results** folds with `.pth` format which can be loaded by `torch.load(model_path, map_location=device)` directly without network instance. However, in order to use WGAN to generate new samples, corresponding configs are still needed to specify the prior distribution and dimension. All these *best* configs of WGAN can be found in `wgan_best_configs.py`. 

`model_plot.py` and `wgan_plot.py` can be directly used to plot the trained distribution by

`python3 -m model_plot` 

`python3 -m wgan_plot`

