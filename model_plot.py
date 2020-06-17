from Gaussianization_Flows.gu_gaus_flow import parser as gaus_flow_parser
import ffjord.gu_ffjord as gu_ffjord
from normalizing_flows.gu_maf import *
from hyperopt_wgan import Generator, Critic
from util import *
from gu import *
from wgan_best_configs import *
from torch.autograd import Variable
from torch import autograd
import torch.nn.utils.spectral_norm as spectral_norm
import torch.nn as nn
import seaborn.apionly as sns
import matplotlib.pyplot as plt
import os
import sys
from functools import partial

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(curPath)


plt.style.use('seaborn-paper')


device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')


def plot_separately(
    data, wgan_config=None, kde_num=500, labels=[
        'a', 'b', 'c', 'd'], xranges=(
            0, 100), yranges=(
                0, 1), rounds_num=2):
    config = wgan_config

    if data == 'unimodal':
        dataloader = GausUniffMixture(
            n_mixture=1,
            mean_dist=5,
            sigma=0.1,
            unif_intsect=5,
            unif_ratio=3,
            device=device,
            seed=2020,
            extend_dim=False)
    else:
        dataloader = GausUniffMixture(
            n_mixture=8,
            mean_dist=10,
            sigma=2,
            unif_intsect=1.5,
            unif_ratio=1.,
            device=device,
            seed=2020,
            extend_dim=False)

    real = dataloader.get_sample(eval_size)
    real_sample = real.cpu().data.numpy().squeeze()
    min_value, max_value = xranges
    step = (max_value - min_value) / eval_size
    real_range = np.arange(min_value, max_value, step)
    real_density = dataloader.density(real_range)

    model_path = os.path.join(curPath, 'model_results', data)

    # Load gaussianization_flow
    gaussianization_flow = torch.load(
        model_path + '/gaussianzation_flow.pth',
        map_location=device)

    # Load ffjord
    ffjord = torch.load(model_path + '/ffjord.pth', map_location=device)

    # Load wgan
    generator = torch.load(model_path + '/generator.pth', map_location=device)
    critic = torch.load(model_path + '/critic.pth', map_location=device)

    # gaussianization_flow
    args = gaus_flow_parser.parse_args()
    DATA = dataloader.get_sample(args.total_datapoints)
    DATA = DATA.view(DATA.shape[0], -1)

    prior = gaussianization_flow.base_dist.sample((eval_size,)).to(device)
    fake = gaussianization_flow.sampling(
        DATA, prior, process_size=eval_size, sample_num=eval_size)
    w_distance_gaussianization = w_distance(real, fake)
    fake_gaussianization = fake.cpu().data.numpy().squeeze()

    # ffjord
    sample_fn, density_fn = gu_ffjord.get_transforms(ffjord)
    z = torch.randn(eval_size, 1).type(torch.float32).to(device)
    zk, inds = [], torch.arange(0, z.shape[0]).to(torch.int64)
    for ii in torch.split(inds, int(10000)):
        zk.append(sample_fn(z[ii]))
    fake = torch.cat(zk, 0)
    w_distance_ffjord = w_distance(real, fake)
    fake_ffjord = fake.cpu().data.numpy().squeeze()

    # wgan
    prior = torch.randn if config['prior'] == 'uniform' else partial(
        torch.normal, mean=0., std=1.)
    z = prior(size=(eval_size, config['prior_size']), device=device)
    fake = generator(z)
    w_distance_est = critic(real).mean() - critic(fake).mean()
    w_distance_est = abs(round(w_distance_est.item(), 4))
    w_distance_wgan = w_distance(real, fake)
    fake_wgan = fake.cpu().data.numpy().squeeze()

    plt.cla()
    fig = plt.figure(
        figsize=(
            FIG_W,
            FIG_H *
            4) if vertical else (
            FIG_W *
            4,
            FIG_H))

    ax = fig.add_subplot(411) if vertical else fig.add_subplot(141)
    ax.set_title(f'({labels[0]})', fontsize=FONTSIZE * 1.2)

    ax.set_facecolor('whitesmoke')
    ax.grid(
        True,
        color='white',
        linewidth=3,
        linestyle='--',
        alpha=shade_alpha)

    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    # real data

    ax.fill_between(
        real_range,
        0,
        real_density,
        color='green',
        alpha=shade_alpha)
    ax.plot(
        real_range,
        real_density,
        label='Data',
        color='darkgreen',
        linewidth=20)

    ax.tick_params(axis='x', labelsize=FONTSIZE * 0.7)
    ax.tick_params(axis='y', labelsize=FONTSIZE * 0.7, direction='in')
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height *
                     0.1, box.width, box.height * 0.9])
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
              fontsize=FONTSIZE * 0.7, fancybox=True, shadow=True, ncol=5)
    ax.set_xlim(xranges)
    ax.set_ylim(yranges)
    ax.set_ylabel('Density', fontsize=FONTSIZE * 0.7, labelpad=labelpad)
    start, end = yranges
    jump = (end - start) / (y_num + 1)
    ax.yaxis.set_ticks(
        np.round(
            np.arange(
                start +
                jump,
                end,
                jump),
            rounds_num))

    # wgan
    ax = fig.add_subplot(412) if vertical else fig.add_subplot(142)
    ax.set_title(f'({labels[1]})', fontsize=FONTSIZE * 1.2)

    ax.set_facecolor('whitesmoke')
    ax.grid(
        True,
        color='white',
        linewidth=3,
        linestyle='--',
        alpha=shade_alpha)

    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    # wgan
    min_value, max_value = min(fake_wgan), max(fake_wgan)
    kde_width = kde_num * (max_value - min_value) / eval_size
    sns.kdeplot(
        fake_wgan,
        bw=kde_width,
        label=f'WGAN',
        color='coral',
        shade=True,
        linewidth=12,
        alpha=shade_alpha,
        ax=ax)
    sns.kdeplot(
        fake_wgan,
        bw=kde_width,
        color='coral',
        shade=False,
        linewidth=12,
        ax=ax)

    # real data
    if data_line:
        ax.plot(
            real_range,
            real_density,
            label='Data',
            color='darkgreen',
            linewidth=20)

    ax.tick_params(axis='x', labelsize=FONTSIZE * 0.7)
    ax.tick_params(axis='y', labelsize=FONTSIZE * 0.7, direction='in')
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height *
                     0.1, box.width, box.height * 0.9])
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
              fontsize=FONTSIZE * 0.7, fancybox=True, shadow=True, ncol=5)
    ax.set_xlim(xranges)
    ax.set_ylim(yranges)
    ax.set_ylabel('Density', fontsize=FONTSIZE * 0.7, labelpad=labelpad)
    start, end = yranges
    jump = (end - start) / (y_num + 1)
    ax.yaxis.set_ticks(
        np.round(
            np.arange(
                start +
                jump,
                end,
                jump),
            rounds_num))

    # ffjord
    ax = fig.add_subplot(413) if vertical else fig.add_subplot(143)
    ax.set_title(f'({labels[2]})', fontsize=FONTSIZE * 1.2)

    ax.set_facecolor('whitesmoke')
    ax.grid(
        True,
        color='white',
        linewidth=3,
        linestyle='--',
        alpha=shade_alpha)
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    min_value, max_value = min(fake_ffjord), max(fake_ffjord)
    kde_width = kde_num * (max_value - min_value) / eval_size
    sns.kdeplot(
        fake_ffjord,
        bw=kde_width,
        label=f'FFJORD',
        color='coral',
        shade=True,
        linewidth=12,
        alpha=shade_alpha,
        ax=ax)
    sns.kdeplot(
        fake_ffjord,
        bw=kde_width,
        color='coral',
        shade=False,
        linewidth=12,
        ax=ax)

    # real data
    if data_line:
        ax.plot(
            real_range,
            real_density,
            label='Data',
            color='darkgreen',
            linewidth=20)

    ax.tick_params(axis='x', labelsize=FONTSIZE * 0.7)
    ax.tick_params(axis='y', labelsize=FONTSIZE * 0.7, direction='in')
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height *
                     0.1, box.width, box.height * 0.9])
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
              fontsize=FONTSIZE * 0.7, fancybox=True, shadow=True, ncol=5)
    ax.set_xlim(xranges)
    ax.set_ylim(yranges)
    ax.set_ylabel('Density', fontsize=FONTSIZE * 0.7, labelpad=labelpad)
    start, end = yranges
    jump = (end - start) / (y_num + 1)
    ax.yaxis.set_ticks(
        np.round(
            np.arange(
                start +
                jump,
                end,
                jump),
            rounds_num))

    # gaussianization_flow
    ax = fig.add_subplot(414) if vertical else fig.add_subplot(144)
    ax.set_title(f'({labels[3]})', fontsize=FONTSIZE * 1.2)

    ax.set_facecolor('whitesmoke')
    ax.grid(
        True,
        color='white',
        linewidth=3,
        linestyle='--',
        alpha=shade_alpha)

    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    min_value, max_value = min(fake_gaussianization), max(fake_gaussianization)
    kde_width = kde_num * (max_value - min_value) / eval_size
    sns.kdeplot(
        fake_gaussianization,
        bw=kde_width,
        label=f'Gaussianization Flows',
        color='coral',
        shade=True,
        linewidth=12,
        alpha=shade_alpha,
        ax=ax)
    sns.kdeplot(
        fake_gaussianization,
        bw=kde_width,
        color='coral',
        shade=False,
        linewidth=12,
        ax=ax)

    # real data
    if data_line:
        ax.plot(
            real_range,
            real_density,
            label='Data',
            color='darkgreen',
            linewidth=20)

    ax.tick_params(axis='x', labelsize=FONTSIZE * 0.7)
    ax.tick_params(axis='y', labelsize=FONTSIZE * 0.7, direction='in')
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height *
                     0.1, box.width, box.height * 0.9])
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
              fontsize=FONTSIZE * 0.7, fancybox=True, shadow=True, ncol=5)
    ax.set_xlim(xranges)
    ax.set_ylim(yranges)
    ax.set_ylabel('Density', fontsize=FONTSIZE * 0.7, labelpad=labelpad)
    start, end = yranges
    jump = (end - start) / (y_num + 1)
    ax.yaxis.set_ticks(
        np.round(
            np.arange(
                start +
                jump,
                end,
                jump),
            rounds_num))

    cur_img_path = os.path.join(model_path, data + '.jpg')

    print('Saving to: ' + cur_img_path)
    plt.savefig(cur_img_path)
    plt.close()


if __name__ == '__main__':
    eval_size = 100000
    shade_alpha = 0.3
    kde_num = 500
    y_num = 4
    labelpad = 25

    vertical = False
    data_line = False

    xranges = (4, 6)
    yranges = (0., 2.2)
    plot_separately('unimodal', unimodal['base'],
                    labels=['a', 'b', 'c', 'd'],
                    rounds_num=1, xranges=xranges, yranges=yranges)

    xranges = (0, 90)
    yranges = (0., 0.04)
    plot_separately('multimodal', multimodal['base'],
                    labels=['e', 'f', 'g', 'h'],
                    rounds_num=3, xranges=xranges, yranges=yranges)

    print('Finish All...')
