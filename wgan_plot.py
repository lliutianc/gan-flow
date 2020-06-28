from hyperopt_wgan import Generator, Critic
from wgan_best_configs import *
from util import *
from gu import *
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

# multimodal = { }
# multimodal['base'] = { 'activation_fn': 'relu', 'activation_slope': 0.01, 'auto': True, 'batch_norm': 1, 'batch_size': 2048,
#                  'beta1': 0.8, 'beta2': 0.7, 'clr': False, 'clr_scale': 2, 'clr_size_up': 2000, 'cuda': 2,
#                  'device': 'cuda', 'eval_real': True, 'eval_size': 100000, 'exp_num': 100, 'gu_num': 8,
#                  'hidden_size': 128, 'init_method': 'default', 'k': 100, 'l': 0.1, 'log_interval': 1000, 'lr': 0.0005,
#                  'n_hidden': 3, 'niters': 50000, 'no_batch_norm': False, 'no_spectral_norm': False, 'prior': 'gaussian',
#                  'prior_size': 3, 'residual_block': False, 'seed': 1, 'smoke_test': False, 'spect_norm': 0,
#                  'weight_decay': 5e-05
#                  }
# multimodal['resnet'] = { 'prior_size': 3, 'hidden_size': 128, 'n_hidden': 3, 'activation_slope': 0.01,
#                    'activation_fn': 'leakyrelu', 'init_method': 'default', 'lr': 1e-05, 'weight_decay': 1e-06,
#                    'beta1': 0.6, 'beta2': 0.999, 'clr_scale': 2, 'clr_size_up': 2000, 'k': 100, 'l': 1.0,
#                    'spect_norm': 1, 'batch_norm': 0, 'dropout': 0, 'cuda': 2, 'device': 'cuda', 'seed': 1, 'gu_num': 8,
#                    'prior': 'gaussian', 'residual_block': True, 'batch_size': 2048, 'niters': 50000, 'clr': False,
#                    'auto': True, 'eval_size': 100000, 'exp_num': 100, 'eval_real': True, 'log_interval': 1000,
#                    }
# multimodal['clr'] = { 'activation_fn': 'leakyrelu', 'activation_slope': 0.01, 'auto': True, 'batch_norm': 1,
#                 'batch_size': 2048, 'beta1': 0.6, 'beta2': 0.9, 'clr': True, 'clr_scale': 3, 'clr_size_up': 8000,
#                 'cuda': 2, 'device': 'cuda', 'eval_real': True, 'eval_size': 100000, 'exp_num': 100, 'gu_num': 8,
#                 'hidden_size': 256, 'init_method': 'default', 'k': 50, 'l': 0.01, 'log_interval': 1000, 'lr': 0.0001,
#                 'n_hidden': 2, 'niters': 50000, 'prior': 'gaussian', 'prior_size': 3, 'residual_block': False,
#                 'seed': 1, 'spect_norm': 1, 'weight_decay': 1e-05
#                 }
# multimodal['dropout'] = { 'activation_fn': 'leakyrelu', 'activation_slope': 0.01, 'auto': True, 'batch_norm': 1,
#                     'batch_size': 2048, 'beta1': 0.6, 'beta2': 0.8, 'clr': False, 'clr_scale': 2, 'clr_size_up': 2000,
#                     'cuda': 2, 'device': 'cuda', 'dropout': True, 'eval_real': True, 'eval_size': 100000,
#                     'exp_num': 100, 'gu_num': 8, 'hidden_size': 128, 'init_method': 'xav_u', 'k': 10, 'l': 0.1,
#                     'log_interval': 1000, 'lr': 0.0005, 'n_hidden': 2, 'niters': 50000, 'prior': 'gaussian',
#                     'prior_size': 3, 'residual_block': False, 'seed': 1, 'spect_norm': 1, 'weight_decay': 1e-05
#                     }
# multimodal['uniform'] = { 'activation_fn': 'relu', 'activation_slope': 0.01, 'auto': True, 'batch_norm': 1,
#                     'batch_size': 2048, 'beta1': 0.5, 'beta2': 0.999, 'clr': False, 'clr_scale': 2, 'clr_size_up': 2000,
#                     'cuda': 2, 'device': 'cuda', 'dropout': False, 'eval_real': True, 'eval_size': 100000,
#                     'exp_num': 100, 'gu_num': 8, 'hidden_size': 256, 'init_method': 'xav_u', 'k': 100, 'l': 0.01,
#                     'log_interval': 1000, 'lr': 0.0001, 'n_hidden': 2, 'niters': 50000, 'prior': 'uniform',
#                     'prior_size': 5, 'residual_block': False, 'seed': 1, 'spect_norm': 1, 'weight_decay': 1e-06
#                     }
# multimodal['all'] = { 'activation_fn': 'relu', 'activation_slope': 0.01, 'auto': True, 'auto_full': True, 'batch_size': 2048,
#                 'beta1': 0.6, 'beta2': 0.7, 'clr': 0, 'clr_scale': 2, 'clr_size_up': 2000, 'cuda': 2, 'device': 'cuda',
#                 'dropout': 0, 'eval_real': True, 'eval_size': 100000, 'exp_num': 100, 'gu_num': 8, 'hidden_size': 256,
#                 'init_method': 'xav_u', 'k': 5, 'l': 0.1, 'log_interval': 1000, 'lr': 5e-05, 'n_hidden': 4,
#                 'niters': 50000, 'norm': None, 'prior': 'uniform', 'prior_size': 3, 'residual_block': False, 'seed': 1,
#                 'spect_norm': 0, 'weight_decay': 5e-06
#                 }
#
# unimodal = { }
# unimodal['base'] = { 'activation_fn': 'relu', 'activation_slope': 0.01, 'auto': True, 'batch_norm': 0, 'batch_size': 2048,
#                 'beta1': 0.5, 'beta2': 0.7, 'clr': False, 'clr_scale': 2, 'clr_size_up': 2000, 'cuda': 2,
#                 'device': 'cuda', 'dropout': False, 'eval_real': True, 'eval_size': 100000, 'exp_num': 100, 'gu_num': 1,
#                 'hidden_size': 64, 'init_method': 'default', 'k': 100, 'l': 0.0, 'log_interval': 1000, 'lr': 1e-05,
#                 'n_hidden': 1, 'niters': 50000, 'prior': 'gaussian', 'prior_size': 1, 'residual_block': False,
#                 'seed': 1, 'spect_norm': 1, 'weight_decay': 1e-05
#                 }
# unimodal['resnet'] = { 'activation_fn': 'leakyrelu', 'activation_slope': 0.01, 'auto': True, 'batch_norm': 1,
#                   'batch_size': 2048, 'beta1': 0.9, 'beta2': 0.9, 'clr': False, 'clr_scale': 2, 'clr_size_up': 2000,
#                   'cuda': 2, 'device': 'cuda', 'dropout': False, 'eval_real': True, 'eval_size': 100000, 'exp_num': 100,
#                   'gu_num': 1, 'hidden_size': 128, 'init_method': 'default', 'k': 100, 'l': 0.01, 'log_interval': 1000,
#                   'lr': 0.0001, 'n_hidden': 3, 'niters': 50000, 'prior': 'gaussian', 'prior_size': 1,
#                   'residual_block': True, 'seed': 1, 'spect_norm': 0, 'weight_decay': 0.0001
#                   }
# unimodal['clr'] = { 'activation_fn': 'leakyrelu', 'activation_slope': 0.01, 'auto': True, 'batch_norm': 1,
#                'batch_size': 2048, 'beta1': 0.7, 'beta2': 0.8, 'clr': True, 'clr_scale': 4, 'clr_size_up': 8000,
#                'cuda': 2, 'device': 'cuda', 'eval_real': True, 'eval_size': 100000, 'exp_num': 100, 'gu_num': 1,
#                'hidden_size': 128, 'init_method': 'xav_u', 'k': 5, 'l': 0.0, 'log_interval': 1000, 'lr': 5e-05,
#                'n_hidden': 2, 'niters': 50000, 'prior': 'gaussian', 'prior_size': 3, 'residual_block': False, 'seed': 1,
#                'spect_norm': 1, 'weight_decay': 0.0
#                }
# unimodal['dropout'] = { 'activation_fn': 'leakyrelu', 'activation_slope': 0.01, 'auto': True, 'batch_norm': 0,
#                    'batch_size': 2048, 'beta1': 0.6, 'beta2': 0.7, 'clr': False, 'clr_scale': 2, 'clr_size_up': 2000,
#                    'cuda': 2, 'device': 'cuda', 'dropout': True, 'eval_real': True, 'eval_size': 100000, 'exp_num': 100,
#                    'gu_num': 1, 'hidden_size': 64, 'init_method': 'xav_u', 'k': 100, 'l': 0.01, 'log_interval': 1000,
#                    'lr': 0.001, 'n_hidden': 2, 'niters': 50000, 'prior': 'gaussian', 'prior_size': 5,
#                    'residual_block': False, 'seed': 1, 'spect_norm': 1, 'weight_decay': 0.0
#                    }
# unimodal['uniform'] = { 'activation_fn': 'leakyrelu', 'activation_slope': 0.01, 'auto': True, 'batch_norm': 1,
#                    'batch_size': 2048, 'beta1': 0.9, 'beta2': 0.9, 'clr': False, 'clr_scale': 2, 'clr_size_up': 2000,
#                    'cuda': 2, 'device': 'cuda', 'dropout': False, 'eval_real': True, 'eval_size': 100000,
#                    'exp_num': 100, 'gu_num': 1, 'hidden_size': 128, 'init_method': 'default', 'k': 50, 'l': 0.01,
#                    'log_interval': 1000, 'lr': 0.001, 'n_hidden': 4, 'niters': 50000, 'prior': 'uniform',
#                    'prior_size': 5, 'residual_block': False, 'seed': 1, 'spect_norm': 1, 'weight_decay': 1e-05
#                    }
# unimodal['all'] = { 'activation_fn': 'tanh', 'activation_slope': 0.01, 'auto': True, 'auto_full': True, 'batch_size': 2048,
#                'beta1': 0.8, 'beta2': 0.9, 'clr': 1, 'clr_scale': 2, 'clr_size_up': 2000, 'cuda': 2, 'device': 'cuda',
#                'dropout': 0, 'eval_real': True, 'eval_size': 100000, 'exp_num': 100, 'gu_num': 1, 'hidden_size': 256,
#                'init_method': 'xav_u', 'k': 100, 'l': 0.1, 'log_interval': 1000, 'lr': 0.0005, 'n_hidden': 2,
#                'niters': 50000, 'norm': None, 'prior': 'gaussian', 'prior_size': 1, 'residual_block': False, 'seed': 1,
#                'spect_norm': 0, 'weight_decay': 1e-06
#                }


def plot_separately(
    data,
    best_configs,
    kde_num=500,
    labels=[
        'a',
        'b',
        'c',
        'd',
        'e',
        'f',
        'g'],
        rounds_num=2,
        xranges=(
            4,
        6)):
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

    model_path = os.path.join(curPath, 'wgan_results', data)

    # real = dataloader.get_sample(eval_size)
    # real_sample = real.cpu().data.numpy().squeeze()

    plt.cla()
    fig, axes = plt.subplots(
        figsize=(
            FIG_W, FIG_H * len(labels)), nrows=len(labels), ncols=1)
    ax = axes[0]
    ax.set_title(f'({labels[0]})', fontsize=FONTSIZE * 1.5)

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
    for i, test in enumerate(
            ['base', 'uniform', 'resnet', 'dropout', 'clr', 'all']):
        # Load wgan
        config = best_configs[test]

        path = model_path + '/' + test
        print('Start: ' + path)
        generator = torch.load(path + '/generator.pth', map_location=device)
        critic = torch.load(path + '/critic.pth', map_location=device)

        try:
            print(
                f'Parameters number: generator: {count_parameters(generator)}, critic: {count_parameters(critic)}')
        except AttributeError:
            print(
                'Old version results: network state dict, try to overwrite files with model.')
            config['norm'] = 'batch' if config['batch_norm'] else None
            generator = Generator(
                input_size=config['prior_size'],
                n_hidden=config['n_hidden'],
                hidden_size=config['hidden_size'],
                activation_slope=config['activation_slope'],
                init_method=config['init_method'],
                activation_fn=config['activation_fn'],
                norm=config['norm'],
                res_block=config['residual_block'],
                dropout=config['dropout']).to(
                config['device'])

            critic = Critic(
                n_hidden=config['n_hidden'],
                hidden_size=config['hidden_size'],
                activation_slope=config['activation_slope'],
                init_method=config['init_method'],
                activation_fn=config['activation_fn'],
                norm=config['norm'],
                res_block=config['residual_block'],
                dropout=config['dropout'],
                spect_norm=config['spect_norm']).to(
                config['device'])

            generator.load_state_dict(torch.load(path + '/generator.pth'))
            critic.load_state_dict(torch.load(path + '/critic.pth'))

            torch.save(generator, path + '/generator.pth')
            torch.save(critic, path + '/critic.pth')

            print(
                f'Parameters number: generator: {count_parameters(generator)}, critic: {count_parameters(critic)}')

        generator.eval()
        critic.eval()

        prior = torch.randn if config['prior'] == 'uniform' else partial(
            torch.normal, mean=0., std=1.)
        z = prior(size=(eval_size, config['prior_size']), device=device)
        fake = generator(z)
        w_distance_est = critic(real).mean() - critic(fake).mean()
        w_distance_est = abs(round(w_distance_est.item(), 4))
        w_distance_wgan = w_distance(real, fake)
        fake_wgan = fake.cpu().data.numpy().squeeze()

        ax = axes[i + 1]
        ax.set_title(f'({labels[i + 1]})', fontsize=FONTSIZE * 1.5)

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
            label=f'{test.upper()}:({w_distance_wgan},({w_distance_est}))',
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
    plot_separately(
        'unimodal',
        unimodal,
        labels=[
            'a',
            'b',
            'c',
            'd',
            'e',
            'f',
            'g'],
        rounds_num=1,
        xranges=(
            4,
            6))

    xranges = (0, 90)
    yranges = (0., 0.04)
    plot_separately(
        'multimodal',
        multimodal,
        labels=[
            'h',
            'i',
            'j',
            'k',
            'l',
            'm',
            'n'],
        rounds_num=3,
        xranges=(
            0,
            90))

    print('Finish All...')
