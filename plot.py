import os
import sys
from functools import partial

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(curPath)

import matplotlib.pyplot as plt
import seaborn.apionly as sns

plt.style.use('seaborn-paper')

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
#                 'weight_decay': 0.0001 }
#
# mgum_config = { 'activation_fn': 'relu', 'activation_slope': 0.01, 'auto': True, 'batch_norm': 1, 'batch_size': 2048,
#                'beta1': 0.8, 'beta2': 0.7, 'clr': False, 'clr_scale': 2, 'clr_size_up': 2000, 'cuda': 2,
#                'device': 'cuda', 'eval_real': True, 'eval_size': 100000, 'exp_num': 100, 'gu_num': 8,
#                'hidden_size': 128,
#                'init_method': 'default', 'k': 100, 'l': 0.1, 'log_interval': 1000, 'lr': 0.0005, 'n_hidden': 3,
#                'niters': 50000, 'no_batch_norm': False, 'no_spectral_norm': False, 'prior': 'gaussian', 'prior_size': 3,
#                'residual_block': False, 'seed': 1, 'smoke_test': False, 'spect_norm': 0, 'weight_decay': 5e-05 }
#


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
    generator = torch.load(model_path + '/generator.pth', map_location=device)
    critic = torch.load(model_path + '/critic.pth', map_location=device)

    # generator = Generator(input_size=config['prior_size'], n_hidden=config['n_hidden'],
    #                  hidden_size=config['hidden_size'],
    #                  activation_slope=config['activation_slope'], init_method=config['init_method'],
    #                  activation_fn=config['activation_fn'], batch_norm=config['batch_norm'],
    #                  res_block=config['residual_block']).to(device)
    #
    # critic = Critic(n_hidden=config['n_hidden'], hidden_size=config['hidden_size'],
    #                 activation_slope=config['activation_slope'], init_method=config['init_method'],
    #                 activation_fn=config['activation_fn'], batch_norm=False,
    #                 res_block=config['residual_block'],
    #                 spect_norm=config['spect_norm']).to(device)

    # all args use the same eval_size.
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
    ax.set_facecolor('whitesmoke')
    ax.grid(True, color='white', linewidth=2)

    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    kde_num = 200
    # real data
    min_value, max_value = min(real_sample), max(real_sample)
    kde_width = kde_num * (max_value - min_value) / eval_size
    sns.kdeplot(real_sample, bw=kde_width, label='Data', color='black', shade=False, linewidth=12)

    # maf
    min_value, max_value = min(fake_maf), max(fake_maf)
    kde_width = kde_num * (max_value - min_value) / eval_size
    sns.kdeplot(fake_maf, bw=kde_width, label=f'MAF ({w_distance_maf})', color='orange', shade=False, linewidth=12)

    # ffjord
    min_value, max_value = min(fake_ffjord), max(fake_ffjord)
    kde_width = kde_num * (max_value - min_value) / eval_size
    sns.kdeplot(fake_ffjord, bw=kde_width, label=f'FFJORD ({w_distance_ffjord})', color='purple', shade=False, linewidth=12)

    # wgan
    min_value, max_value = min(fake_ffjord), max(fake_ffjord)
    kde_width = kde_num * (max_value - min_value) / eval_size
    sns.kdeplot(fake_wgan, bw=kde_width, label=f'WGAN ({w_distance_wgan}, ({w_distance_est}))', color='green',
                shade=False, linewidth=12)

    ax.set_title(f'{data.upper()}: Model Name (True EM Distance, (Est. EM Distance))', fontsize=FONTSIZE * 0.7)
    ax.legend(loc=2, fontsize=FONTSIZE * 0.5)
    ax.set_ylabel('Estimated Density by KDE', fontsize=FONTSIZE * 0.7)
    ax.tick_params(axis='x', labelsize=FONTSIZE * 0.7)
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


if __name__ == '__main__':
    eval_size = 100000
    plot_together('gum')
    plot_together('mgum')

    print('Finish All...')