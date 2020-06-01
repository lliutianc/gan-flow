import os
import sys
from functools import partial

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(curPath)

import matplotlib.pyplot as plt
import seaborn.apionly as sns

plt.style.use('seaborn-paper')


import torch.nn as nn
import torch.nn.utils.spectral_norm as spectral_norm
from torch import autograd
from torch.autograd import Variable


from gu import *
from util import *

from normalizing_flows.gu_maf import *
import ffjord.gu_ffjord as gu_ffjord

device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')

# gum_config = { 'activation_fn': 'tanh', 'activation_slope': 0.01, 'auto': True, 'batch_norm': 1, 'batch_size': 2048,
#                 'beta1': 0.8, 'beta2': 0.999, 'clr': False, 'clr_scale': 2, 'clr_size_up': 2000, 'cuda': 2,
#                 'device': 'cuda', 'eval_real': True, 'eval_size': 100000, 'exp_num': 100, 'gu_num': 1,
#                 'hidden_size': 64, 'init_method': 'xav_u', 'k': 100, 'l': 0.01, 'log_interval': 1000, 'lr': 5e-05,
#                 'n_hidden': 4, 'niters': 50000, 'no_batch_norm': False, 'no_spectral_norm': False, 'prior': 'gaussian',
#                 'prior_size': 5, 'residual_block': False, 'seed': 1, 'smoke_test': False, 'spect_norm': 0,
#                 'weight_decay': 0.0001
#               }

gum_config = { 'activation_fn': 'relu', 'activation_slope': 0.01, 'auto': True, 'batch_norm': 0, 'batch_size': 2048,
               'beta1': 0.5, 'beta2': 0.7, 'clr': False, 'clr_scale': 2, 'clr_size_up': 2000, 'cuda': 2,
               'device': 'cuda', 'eval_real': True, 'eval_size': 100000, 'exp_num': 100, 'gu_num': 1, 'hidden_size': 64,
               'init_method': 'default', 'k': 100, 'l': 0.0, 'log_interval': 1000, 'lr': 1e-05, 'n_hidden': 1,
               'niters': 50000, 'prior': 'gaussian', 'prior_size': 1, 'residual_block': False, 'seed': 1,
               'spect_norm': 1, 'weight_decay': 1e-05
               }

mgum_config = { 'activation_fn': 'relu', 'activation_slope': 0.01, 'auto': True, 'batch_norm': 1, 'batch_size': 2048,
               'beta1': 0.8, 'beta2': 0.7, 'clr': False, 'clr_scale': 2, 'clr_size_up': 2000, 'cuda': 2,
               'device': 'cuda', 'eval_real': True, 'eval_size': 100000, 'exp_num': 100, 'gu_num': 8,
               'hidden_size': 128,
               'init_method': 'default', 'k': 100, 'l': 0.1, 'log_interval': 1000, 'lr': 0.0005, 'n_hidden': 3,
               'niters': 50000, 'no_batch_norm': False, 'no_spectral_norm': False, 'prior': 'gaussian', 'prior_size': 3,
               'residual_block': False, 'seed': 1, 'smoke_test': False, 'spect_norm': 0, 'weight_decay': 5e-05
                }


class Generator(nn.Module):
    def __init__(self, input_size, n_hidden, hidden_size, activation_fn, activation_slope, init_method,
                 batch_norm=True, res_block=False):
        super().__init__()
        # Define activation function.
        if activation_fn == 'relu':
            activation = nn.ReLU(inplace=True)
        elif activation_fn == 'leakyrelu':
            activation = nn.LeakyReLU(inplace=True, negative_slope=activation_slope)
        elif activation_fn == 'tanh':
            activation = nn.Tanh()
        else:
            raise NotImplementedError('Check activation_fn.')

        # Define network architecture.
        modules = [nn.Linear(input_size, hidden_size)] + batch_norm * [nn.BatchNorm1d(hidden_size)]
        for _ in range(n_hidden):
            if res_block:
                modules += [activation, ResidualBlock(hidden_size, hidden_size, activation, False, batch_norm)]
            else:
                modules += [activation, nn.Linear(hidden_size, hidden_size)]
            modules += batch_norm * [nn.BatchNorm1d(hidden_size)]
        modules += [activation, nn.Linear(hidden_size, 1)]
        self.model = nn.Sequential(*modules)
        self.init_method = init_method
        self.model.apply(self.__init)

    def forward(self, x):
        return self.model(x)

    def __init(self, m):
        classname = m.__class__.__name__

        if self.init_method == 'default':
            return
        elif self.init_method == 'xav_u':
            if classname.find('Linear') != -1:
                nn.init.xavier_uniform_(m.weight, gain=1)
        else:
            raise NotImplementedError('Check init_method')


class Critic(nn.Module):
    def __init__(self, n_hidden, hidden_size, activation_fn, activation_slope, init_method,
                 spect_norm=True, batch_norm=False, res_block=False):
        super().__init__()
        # Define activation function.
        if activation_fn == 'relu':
            activation = nn.ReLU(inplace=True)
        elif activation_fn == 'leakyrelu':
            activation = nn.LeakyReLU(inplace=True, negative_slope=activation_slope)
        elif activation_fn == 'tanh':
            activation = nn.Tanh()
        else:
            raise NotImplementedError('Check activation_fn.')

        # Define network architecture.
        modules = [spectral_norm(nn.Linear(1, hidden_size)) if spect_norm else nn.Linear(1, hidden_size)]
        modules += batch_norm * [nn.BatchNorm1d(hidden_size)]
        for _ in range(n_hidden):
            modules += [activation]
            if res_block:
                modules += [ResidualBlock(hidden_size, hidden_size, activation, spect_norm, batch_norm)]
            else:
                modules += [spectral_norm(nn.Linear(hidden_size, hidden_size)) if spect_norm else nn.Linear(hidden_size, hidden_size)]
            modules += batch_norm * [nn.BatchNorm1d(hidden_size)]
        modules += [activation]
        modules += [spectral_norm(nn.Linear(hidden_size, 1)) if spect_norm else nn.Linear(hidden_size, 1)]
        self.model = nn.Sequential(*modules)
        self.init_method = init_method
        self.model.apply(self.__init)

    def forward(self, x):
        return self.model(x)

    def __init(self, m):
        classname = m.__class__.__name__

        if self.init_method == 'default':
            return
        elif self.init_method == 'xav_u':
            if classname.find('Linear') != -1:
                nn.init.xavier_uniform_(m.weight, gain=1)
        else:
            raise NotImplementedError('Check init_method')


def plot_together(data, wgan_config=None):
    config = wgan_config

    if data == 'gum':
        dataloader = GausUniffMixture(n_mixture=1, mean_dist=5, sigma=0.1, unif_intsect=5, unif_ratio=3, device=device,
                                seed=2020, extend_dim=False)
    else:
        dataloader = GausUniffMixture(n_mixture=8, mean_dist=10, sigma=2, unif_intsect=1.5, unif_ratio=1., device=device,
                                seed=2020, extend_dim=False)

    model_path = os.path.join(curPath, 'models_to_plot', data)

    # Load maf
    maf = torch.load(model_path + '/maf.pth', map_location=device)

    # Load ffjord
    ffjord = torch.load(model_path + '/ffjord.pth', map_location=device)

    # Load wgan
    # generator = torch.load(model_path + '/generator.pth', map_location=device)
    # critic = torch.load(model_path + '/critic.pth', map_location=device)

    generator = Generator(input_size=config['prior_size'], n_hidden=config['n_hidden'],
                     hidden_size=config['hidden_size'],
                     activation_slope=config['activation_slope'], init_method=config['init_method'],
                     activation_fn=config['activation_fn'], batch_norm=config['batch_norm'],
                     res_block=config['residual_block']).to(device)

    critic = Critic(n_hidden=config['n_hidden'], hidden_size=config['hidden_size'],
                    activation_slope=config['activation_slope'], init_method=config['init_method'],
                    activation_fn=config['activation_fn'], batch_norm=False,
                    res_block=config['residual_block'],
                    spect_norm=config['spect_norm']).to(device)
    generator.load_state_dict(torch.load(model_path + '/generator.pth', map_location=device))
    critic.load_state_dict(torch.load(model_path + '/critic.pth', map_location=device))

    torch.save(generator, model_path + '/generator.pth')
    torch.save(critic, model_path + '/critic.pth')


    real = dataloader.get_sample(eval_size)
    real_sample = real.cpu().data.numpy().squeeze()

    # maf
    prior = maf.base_dist.sample((eval_size,)).to(device)
    fake, _ = maf.inverse(prior)
    fake = fake[:, 0]
    w_distance_maf = w_distance(real, fake)
    fake_maf = fake.cpu().data.numpy().squeeze()

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
    prior = torch.randn() if config['prior'] == 'uniform' else partial(torch.normal, mean=0., std=1.)
    z = prior(size=(eval_size, config['prior_size']), device=device)
    fake = generator(z)
    w_distance_est = critic(real).mean() - critic(fake).mean()
    w_distance_est = abs(round(w_distance_est.item(), 4))
    w_distance_wgan = w_distance(real, fake)
    fake_wgan = fake.cpu().data.numpy().squeeze()

    plt.cla()
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax = fig.add_subplot(111)
    # ax.set_facecolor('whitesmoke')
    # ax.grid(True, color='white', linewidth=2)

    ax.set_facecolor('white')
    # ax.grid(True, color='whitesmoke', linewidth=2)

    # ax.spines["top"].set_visible(False)
    # ax.spines["bottom"].set_visible(False)
    # ax.spines["right"].set_visible(False)
    # ax.spines["left"].set_visible(False)
    ax.spines["top"].set_linewidth(6)
    ax.spines["bottom"].set_linewidth(6)
    ax.spines["right"].set_linewidth(6)
    ax.spines["left"].set_linewidth(6)
    # kde_num = 1000


    # maf
    min_value, max_value = min(fake_maf), max(fake_maf)
    kde_width = kde_num * (max_value - min_value) / eval_size
    sns.kdeplot(fake_maf, bw=kde_width, color='steelblue', shade=True, linewidth=1, alpha=shade_alpha)
    sns.kdeplot(fake_maf, bw=kde_width, label=f'MAF ({w_distance_maf})', color='steelblue', shade=False,
                linewidth=12, alpha=1)

    # ffjord
    min_value, max_value = min(fake_ffjord), max(fake_ffjord)
    kde_width = kde_num * (max_value - min_value) / eval_size
    sns.kdeplot(fake_ffjord, bw=kde_width, color='green', shade=True, linewidth=1, alpha=shade_alpha)
    sns.kdeplot(fake_ffjord, bw=kde_width, label=f'FFJORD ({w_distance_ffjord})', color='green', shade=False,
                linewidth=12, alpha=1)

    # wgan
    min_value, max_value = min(fake_ffjord), max(fake_ffjord)
    kde_width = kde_num * (max_value - min_value) / eval_size
    sns.kdeplot(fake_wgan, bw=kde_width, color='red', shade=True, linewidth=1, alpha=shade_alpha)
    sns.kdeplot(fake_wgan, bw=kde_width, label=f'WGAN ({w_distance_wgan}, ({w_distance_est}))', color='red', shade=False,
                linewidth=12, alpha=1)

    # real data
    min_value, max_value = min(real_sample), max(real_sample)
    kde_width = kde_num * (max_value - min_value) / eval_size
    # sns.kdeplot(real_sample, bw=kde_width, color='grey', shade=True, linewidth=1, alpha=shade_alpha)
    sns.kdeplot(real_sample, bw=kde_width, label='Data', color='black', shade=False, linewidth=12, alpha=1)


    ax.set_title(f'{data.upper()}: Model Name (True EM Distance, (Est. EM Distance))', fontsize=FONTSIZE * 0.6)
    ax.legend(loc=2, fontsize=FONTSIZE * 0.5)
    ax.set_ylabel('Estimated Density by KDE', fontsize=FONTSIZE * 0.6)
    ax.tick_params(axis='x', labelsize=FONTSIZE * 0.5)
    ax.tick_params(axis='y', labelsize=FONTSIZE * 0.5, direction='in')

    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                     box.width, box.height * 0.9])

    # Put a legend below current axis
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fontsize=FONTSIZE * 0.5, fancybox=True, shadow=True, ncol=5)

    cur_img_path = os.path.join(model_path, data + '.jpg')
    # plt.tight_layout()

    print('Saving to: ' + cur_img_path)
    plt.savefig(cur_img_path)
    plt.close()


def plot_separately(data, wgan_config=None, kde_num=500):
    config = wgan_config

    if data == 'gum':
        dataloader = GausUniffMixture(n_mixture=1, mean_dist=5, sigma=0.1, unif_intsect=5, unif_ratio=3, device=device,
                                seed=2020, extend_dim=False)
    else:
        dataloader = GausUniffMixture(n_mixture=8, mean_dist=10, sigma=2, unif_intsect=1.5, unif_ratio=1., device=device,
                                seed=2020, extend_dim=False)

    model_path = os.path.join(curPath, 'models_to_plot', data)

    # Load maf
    maf = torch.load(model_path + '/maf.pth', map_location=device)

    # Load ffjord
    ffjord = torch.load(model_path + '/ffjord.pth', map_location=device)

    # Load wgan
    generator = torch.load(model_path + '/generator.pth', map_location=device)
    critic = torch.load(model_path + '/critic.pth', map_location=device)


    real = dataloader.get_sample(eval_size)
    real_sample = real.cpu().data.numpy().squeeze()

    # maf
    prior = maf.base_dist.sample((eval_size,)).to(device)
    fake, _ = maf.inverse(prior)
    fake = fake[:, 0]
    w_distance_maf = w_distance(real, fake)
    fake_maf = fake.cpu().data.numpy().squeeze()

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
    prior = torch.randn() if config['prior'] == 'uniform' else partial(torch.normal, mean=0., std=1.)
    z = prior(size=(eval_size, config['prior_size']), device=device)
    fake = generator(z)
    w_distance_est = critic(real).mean() - critic(fake).mean()
    w_distance_est = abs(round(w_distance_est.item(), 4))
    w_distance_wgan = w_distance(real, fake)
    fake_wgan = fake.cpu().data.numpy().squeeze()


    plt.cla()
    fig = plt.figure(figsize=(FIG_W, FIG_H * 2.7))

    ax = fig.add_subplot(311)
    ax.set_facecolor('whitesmoke')
    ax.grid(True, color='white', linewidth=3,linestyle='--', alpha=shade_alpha)

    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    # ax.spines["top"].set_linewidth(6)
    # ax.spines["bottom"].set_linewidth(6)
    # ax.spines["right"].set_linewidth(6)
    # ax.spines["left"].set_linewidth(6)

    # wgan
    min_value, max_value = min(fake_ffjord), max(fake_ffjord)
    kde_width = kde_num * (max_value - min_value) / eval_size
    sns.kdeplot(fake_wgan, bw=kde_width, label=f'WGAN ({w_distance_wgan}, ({w_distance_est}))', color='coral', shade=True, linewidth=12, alpha=shade_alpha, ax=ax)
    sns.kdeplot(fake_wgan, bw=kde_width, color='coral', shade=False, linewidth=12, ax=ax)

    # real data
    min_value, max_value = min(real_sample), max(real_sample)
    kde_width = kde_num * (max_value - min_value) / eval_size
    sns.kdeplot(real_sample, bw=kde_width, label='Data', color='green', shade=True, linewidth=12, alpha=shade_alpha, ax=ax)
    sns.kdeplot(real_sample, bw=kde_width, color='green', shade=False, linewidth=12, ax=ax)

    ax.tick_params(axis='x', labelsize=FONTSIZE * 0.5)
    ax.tick_params(axis='y', labelsize=FONTSIZE * 0.5, direction='in')
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fontsize=FONTSIZE * 0.5, fancybox=True, shadow=True,
              ncol=5)
    ax.set_ylabel('WGAN', fontsize=FONTSIZE * 0.6)
    # ax.set_title('WGAN')

    # maf
    ax = fig.add_subplot(312)
    ax.set_facecolor('whitesmoke')
    ax.grid(True, color='white', linewidth=3,linestyle='--', alpha=shade_alpha)

    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    # ax.set_facecolor('white')
    # ax.grid(True, color='whitesmoke', linewidth=2)
    # ax.spines["top"].set_linewidth(6)
    # ax.spines["bottom"].set_linewidth(6)
    # ax.spines["right"].set_linewidth(6)
    # ax.spines["left"].set_linewidth(6)

    min_value, max_value = min(fake_maf), max(fake_maf)
    kde_width = kde_num * (max_value - min_value) / eval_size
    sns.kdeplot(fake_maf, bw=kde_width, label=f'MAF ({w_distance_maf})', color='coral', shade=True, linewidth=12, alpha=shade_alpha, ax=ax)
    sns.kdeplot(fake_maf, bw=kde_width, color='coral', shade=False, linewidth=12, ax=ax)

    # real data
    min_value, max_value = min(real_sample), max(real_sample)
    kde_width = kde_num * (max_value - min_value) / eval_size
    sns.kdeplot(real_sample, bw=kde_width, label='Data', color='green', shade=True, linewidth=12, alpha=shade_alpha, ax=ax)
    sns.kdeplot(real_sample, bw=kde_width, color='green', shade=False, linewidth=12, ax=ax)

    ax.tick_params(axis='x', labelsize=FONTSIZE * 0.5)
    ax.tick_params(axis='y', labelsize=FONTSIZE * 0.5, direction='in')
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fontsize=FONTSIZE * 0.5, fancybox=True, shadow=True,
              ncol=5)
    ax.set_ylabel('MAF', fontsize=FONTSIZE * 0.6)
    # ax.set_title('MAF')


    # ffjord
    ax = fig.add_subplot(313)

    ax.set_facecolor('whitesmoke')
    ax.grid(True, color='white', linewidth=3,linestyle='--', alpha=shade_alpha)
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    # ax.set_facecolor('white')
    # ax.grid(True, color='whitesmoke', linewidth=2)
    # ax.spines["top"].set_linewidth(6)
    # ax.spines["bottom"].set_linewidth(6)
    # ax.spines["right"].set_linewidth(6)
    # ax.spines["left"].set_linewidth(6)

    min_value, max_value = min(fake_ffjord), max(fake_ffjord)
    kde_width = kde_num * (max_value - min_value) / eval_size
    sns.kdeplot(fake_ffjord, bw=kde_width, label=f'FFJORD ({w_distance_ffjord})', color='coral', shade=True, linewidth=12, alpha=shade_alpha, ax=ax)
    sns.kdeplot(fake_ffjord, bw=kde_width, color='coral', shade=False, linewidth=12, ax=ax)

    # real data
    min_value, max_value = min(real_sample), max(real_sample)
    kde_width = kde_num * (max_value - min_value) / eval_size
    sns.kdeplot(real_sample, bw=kde_width, label='Data', color='green', shade=True, linewidth=12, alpha=shade_alpha, ax=ax)
    sns.kdeplot(real_sample, bw=kde_width, color='green', shade=False, linewidth=12, ax=ax)

    ax.tick_params(axis='x', labelsize=FONTSIZE * 0.5)
    ax.tick_params(axis='y', labelsize=FONTSIZE * 0.5, direction='in')
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,box.width, box.height * 0.9])
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fontsize=FONTSIZE * 0.5, fancybox=True, shadow=True, ncol=5)
    ax.set_ylabel('FFJORD', fontsize=FONTSIZE * 0.6)

    cur_img_path = os.path.join(model_path, data + '.jpg')
    # fig.text(0.04, 0.5, 'Estimated Density by KDE', va='center', rotation='vertical')
    # plt.subplots_adjust(top=0.85)

    # plt.tight_layout()
    # fig = ax.get_figure()
    # fig.suptitle(f'{data.upper()}. Model Name (True EM Distance, (Est. EM Distance))', fontsize=FONTSIZE * 0.7)
    # fig.tight_layout(h_pad=1., w_pad=1)
    # fig.subplots_adjust(top=0.95)

    print('Saving to: ' + cur_img_path)
    plt.savefig(cur_img_path)
    plt.close()



if __name__ == '__main__':
    eval_size = 100000
    shade_alpha = 0.3
    kde_num = 500

    # plot_together('gum', gum_config)
    # plot_together('mgum', mgum_config)

    plot_separately('gum', gum_config)
    plot_separately('mgum', mgum_config)

    print('Finish All...')