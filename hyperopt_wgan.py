import os
import sys
import argparse
from functools import partial
import time

import matplotlib.pyplot as plt
import seaborn.apionly as sns

import torch.nn as nn
import torch.nn.utils.spectral_norm as spectral_norm
from torch import autograd
from torch.autograd import Variable

import ray.tune as tune
from ray.tune.schedulers import ASHAScheduler

from residualblock import ResidualBlock
from gu import *
from util import *


curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(curPath)

parser = argparse.ArgumentParser()
# action
parser.add_argument(
    '--cuda',
    type=int,
    default=2,
    help='Number of CUDA to use if available.')
# data
parser.add_argument('--seed', type=int, default=1, help='Random seed to use.')
parser.add_argument('--gu_num', type=int, default=8,
                    help='Components of GU clusters.')
# model parameters
parser.add_argument(
    '--prior',
    type=str,
    choices=[
        'uniform',
        'gaussian'],
    default='gaussian',
    help='Distribution of prior.')
parser.add_argument(
    '--prior_size',
    type=int,
    default=3,
    help='Dimension of prior.')
parser.add_argument('--hidden_size', type=int, default=256,
                    help='Hidden layer size for GAN/WGAN.')
parser.add_argument(
    '--n_hidden',
    type=int,
    default=3,
    help='Number of hidden layers(Residual blocks) in GAN/WGAN.')
parser.add_argument(
    '--activation_fn',
    type=str,
    choices=[
        'relu',
        'leakyrelu',
        'tanh'],
    default='leakyrelu',
    help='What activation function to use in GAN/WGAN.')
parser.add_argument('--activation_slope', type=float, default=1e-2,
                    help='Negative slope of LeakyReLU activation function.')

parser.add_argument(
    '--no_spectral_norm',
    action='store_true',
    help='Do not use spectral normalization in critic.')
# parser.add_argument('--no_batch_norm', action='store_true', help='Do not use batch norm')
parser.add_argument(
    '--residual_block',
    action='store_true',
    help='Use residual block')
parser.add_argument('--dropout', action='store_true', help='Use dropout')
parser.add_argument(
    '--norm',
    type=str,
    choices=[
        'layer',
        'batch',
        None],
    default='batch',
    help='Which normaliztion to be used.')
parser.add_argument(
    '--init_method',
    type=str,
    choices=[
        'default',
        'xav_u'],
    default='default',
    help='Use residual block')

# training params
parser.add_argument(
    '--batch_size',
    type=int,
    default=2048,
    help='Batch size in training.')
parser.add_argument('--niters', type=int, default=50000,
                    help='Total iteration numbers in training.')
parser.add_argument(
    '--lr',
    type=float,
    default=1e-4,
    help='Learning rate in Adam.')
parser.add_argument(
    '--weight_decay',
    type=float,
    default=1e-6,
    help='Weight decay in Adam.')
parser.add_argument('--beta1', type=float, default=0.9, help='Beta 1 in Adam.')
parser.add_argument(
    '--beta2',
    type=float,
    default=0.999,
    help='Beta 2 in Adam.')

parser.add_argument(
    '--clr',
    action='store_true',
    help='Use cyclic LR in training.')
parser.add_argument(
    '--clr_size_up',
    type=int,
    default=2000,
    help='Size of up step in cyclic LR.')
parser.add_argument('--clr_scale', type=int, default=3,
                    help='Scale of base lr in cyclic LR.')
parser.add_argument(
    '--k',
    type=int,
    default=5,
    help='Update times of critic in each iterations.')
parser.add_argument(
    '--l',
    type=float,
    default=0.1,
    help='Coefficient for Gradient penalty.')

parser.add_argument(
    '--auto',
    action='store_true',
    help='Using parameter searching to find the best result.')
parser.add_argument(
    '--auto_full',
    action='store_true',
    help='Using parameter searching to find the best result.')
parser.add_argument(
    '--eval_size',
    type=int,
    default=100000,
    help='Sample size in evaluation.')
parser.add_argument(
    '--exp_num',
    type=int,
    default=100,
    help='Number of experiments.')
parser.add_argument(
    '--eval_est',
    action='store_true',
    help='use w_distance_estimated to choose best model.')
parser.add_argument(
    '--log_interval',
    type=int,
    default=1000,
    help='How often to show loss statistics and save models/samples.')

config = {  # 'prior': tune.choice(['uniform', 'gaussian']),
    'prior_size': tune.choice([1, 3, 5]), 'hidden_size': tune.choice([64, 128, 256]),
    'n_hidden': tune.choice([1, 2, 3, 4]), 'activation_slope': 1e-2,
    'activation_fn': tune.choice(['relu', 'leakyrelu', 'tanh']), 'init_method': tune.choice(['default', 'xav_u']),

    'lr': tune.choice([1e-5, 5e-5, 1e-4, 5e-4, 1e-3]),
    'weight_decay': tune.choice([0., 1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 1e-3]),
    'beta1': tune.choice([0.5, 0.6, 0.7, 0.8, 0.9]), 'beta2': tune.choice([0.7, 0.8, 0.9, 0.999]),

    # In auto_full, these are not used
    'clr_scale': tune.choice([2, 3, 4, 5]), 'clr_size_up': tune.choice([2000, 4000, 6000, 8000]),

    'k': tune.choice([1, 5, 10, 50, 100]), 'l': tune.choice([0, 1e-2, 1e-1, 1, 10]),

    'norm': tune.choice(['batch', None]), 'spect_norm': tune.choice([1, 0]),
    # 'spect_norm': 1,  # try enforcing spect_norm in critic.
    # 'dropout': None,
    # 'clr': None,
}


class Generator (nn.Module):
    def __init__(
            self,
            input_size,
            n_hidden,
            hidden_size,
            activation_fn,
            activation_slope,
            init_method,
            norm='batch',
            res_block=False,
            dropout=False,
            dropout_p=0.5):
        super().__init__()
        # Define activation function.
        if activation_fn == 'relu':
            activation = nn.ReLU(inplace=True)
        elif activation_fn == 'leakyrelu':
            activation = nn.LeakyReLU(
                inplace=True, negative_slope=activation_slope)
        elif activation_fn == 'tanh':
            activation = nn.Tanh()
        else:
            raise NotImplementedError('Check activation_fn.')

        if norm == 'batch':
            norm = nn.BatchNorm1d
        elif norm == 'layer':
            norm = nn.LayerNorm
        else:
            norm = None

        modules = [
            nn.Linear(
                input_size,
                hidden_size),
            norm(hidden_size)] if norm else [
            nn.Linear(
                input_size,
                hidden_size)]
        for _ in range(n_hidden):
            # Add dropout.
            if dropout:
                modules += [nn.Dropout(dropout_p)]
            # Add act and layer.
            if res_block:
                modules += [activation,
                            ResidualBlock(hidden_size,
                                          hidden_size,
                                          activation,
                                          False,
                                          norm)]
            else:
                modules += [activation, nn.Linear(hidden_size, hidden_size)]
            if norm:
                modules += [norm(hidden_size)]
        if dropout:
            modules += [nn.Dropout(dropout_p)]
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


class Critic (nn.Module):
    def __init__(
            self,
            n_hidden,
            hidden_size,
            activation_fn,
            activation_slope,
            init_method,
            spect_norm=True,
            norm='layer',
            res_block=False,
            dropout=False,
            dropout_p=0.5):
        super().__init__()
        # Define activation function.
        if activation_fn == 'relu':
            activation = nn.ReLU(inplace=True)
        elif activation_fn == 'leakyrelu':
            activation = nn.LeakyReLU(
                inplace=True, negative_slope=activation_slope)
        elif activation_fn == 'tanh':
            activation = nn.Tanh()
        else:
            raise NotImplementedError('Check activation_fn.')

        if norm == 'layer':
            norm = nn.LayerNorm
        else:
            norm = None

        modules = [
            spectral_norm(
                nn.Linear(
                    1,
                    hidden_size)) if spect_norm else nn.Linear(
                1,
                hidden_size)]
        if norm:
            modules += [norm(hidden_size)]
        for _ in range(n_hidden):
            # Add dropout.
            if dropout:
                modules += [nn.Dropout(dropout_p)]
            # Add act and layer.
            if res_block:
                modules += [activation,
                            ResidualBlock(hidden_size,
                                          hidden_size,
                                          activation,
                                          spect_norm,
                                          norm)]
            else:
                modules += [
                    spectral_norm(
                        nn.Linear(
                            hidden_size,
                            hidden_size)) if spect_norm else nn.Linear(
                        hidden_size,
                        hidden_size)]
            if norm:
                modules += [norm(hidden_size)]
        if dropout:
            modules += [nn.Dropout(dropout_p)]
        modules += [activation]
        modules += [spectral_norm(nn.Linear(hidden_size, 1))
                    if spect_norm else nn.Linear(hidden_size, 1)]

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


class WGANTrainer (tune.Trainable):
    def _setup(self, config):
        self.config = config
        self.prior = torch.randn if self.config['prior'] == 'uniform' else partial(
            torch.normal, mean=0., std=1.)
        self.i = 0
        # model
        self.generator = Generator(
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

        self.critic = Critic(
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

        # data
        if self.config['gu_num'] == 8:
            self.dataloader = GausUniffMixture(
                n_mixture=self.config['gu_num'],
                mean_dist=10,
                sigma=2,
                unif_intsect=1.5,
                unif_ratio=1.,
                device=self.config['device'])
        else:
            self.dataloader = GausUniffMixture(
                n_mixture=self.config['gu_num'],
                mean_dist=5,
                sigma=0.1,
                unif_intsect=5,
                unif_ratio=3,
                device=self.config['device'])

        # optimizer
        self.optim_g = torch.optim.Adam(
            [
                p for p in self.generator.parameters() if p.requires_grad],
            lr=config['lr'],
            betas=(
                config['beta1'],
                config['beta2']),
            weight_decay=config['weight_decay'])
        self.optim_c = torch.optim.Adam(
            [
                p for p in self.critic.parameters() if p.requires_grad],
            lr=config['lr'],
            betas=(
                config['beta1'],
                config['beta2']),
            weight_decay=config['weight_decay'])

        if self.config['clr']:
            self.sche_g = torch.optim.lr_scheduler.CyclicLR(
                self.optim_g,
                base_lr=config['lr'] /
                config['clr_scale'],
                max_lr=config['lr'],
                step_size_up=config['clr_size_up'],
                cycle_momentum=False)
            self.sche_c = torch.optim.lr_scheduler.CyclicLR(
                self.optim_c,
                base_lr=config['lr'] /
                config['clr_scale'],
                max_lr=config['lr'],
                step_size_up=config['clr_size_up'],
                cycle_momentum=False)
        else:
            self.sche_g, self.sche_c = None, None

    def _train(self):
        if self.i == 0:
            self.start = time.time()

        self.i += 1
        self.generator.train()
        self.critic.train()

        for k in range(self.config['k']):
            real = self.dataloader.get_sample(self.config['batch_size'])
            prior = self.prior(
                size=(
                    self.config['batch_size'],
                    self.config['prior_size']),
                device=self.config['device'])
            fake = self.generator(prior)

            loss_c = self.critic(fake.detach()).mean() - \
                self.critic(real).mean()
            loss_c += self.config["l"] * self._gradient_penalty(real, fake)
            self.optim_c.zero_grad()
            loss_c.backward()
            self.optim_c.step()
            if self.sche_c:
                self.sche_c.step()

        prior = self.prior(
            size=(
                self.config['batch_size'],
                self.config['prior_size']),
            device=self.config['device'])
        fake = self.generator(prior)
        loss_g = - self.critic(fake).mean()
        self.optim_g.zero_grad()
        loss_g.backward()
        self.optim_g.step()
        if self.sche_g:
            self.sche_g.step()

        if self.i % self.config['log_interval'] == 0 and not self.config['auto']:
            cur_state_path = os.path.join(model_path, str(self.i))
            torch.save(self.generator, cur_state_path + '_' + 'generator.pth')
            torch.save(self.critic, cur_state_path + '_' + 'critic.pth')

            w_distance_real, w_distance_est = self._evaluate(
                display=True, niter=self.i)

            logger.info(
                f'Iter: {self.i} / {self.config["niters"]}, Time: {round (time.time () - self.start, 4)},  '
                f'w_distance_real: {w_distance_real}, w_distance_estimated: {w_distance_est}')

            self.start = time.time()

        w_distance_real, w_distance_est = self._evaluate(
            display=False, niter=self.config['niters'])

        return {
            'w_distance_estimated': w_distance_est,
            'w_distance_real': w_distance_real,
            'iteration': self.i}

    def _save(self, tmp_checkpoint_dir):
        generator_path = os.path.join(tmp_checkpoint_dir, 'generator.pth')
        critic_path = os.path.join(tmp_checkpoint_dir, 'critic.pth')

        torch.save(self.generator.state_dict(), generator_path)
        torch.save(self.critic.state_dict(), critic_path)

        return tmp_checkpoint_dir

    def _save_whole(self, tmp_checkpoint_dir):
        generator_path = os.path.join(tmp_checkpoint_dir, 'generator.pth')
        critic_path = os.path.join(tmp_checkpoint_dir, 'critic.pth')
        torch.save(self.generator.to('cpu'), generator_path)
        torch.save(self.critic.to('cpu'), critic_path)

        return tmp_checkpoint_dir

    def _restore(self, checkpoint_dir):
        generator_path = os.path.join(checkpoint_dir, 'generator.pth')
        critic_path = os.path.join(checkpoint_dir, 'critic.pth')

        self.generator.load_state_dict(torch.load(generator_path))
        self.critic.load_state_dict(torch.load(critic_path))

    def _evaluate(self, display, niter):
        with torch.no_grad():
            real = self.dataloader.get_sample(self.config['eval_size'])
            prior = self.prior(
                size=(
                    self.config['eval_size'],
                    self.config['prior_size']),
                device=self.config['device'])
            fake = self.generator(prior)

            w_distance_est = self.critic(
                real).mean() - self.critic(fake).mean()
            w_distance_est = abs(round(w_distance_est.item(), 5))
            w_distance_real = w_distance(real, fake)

            if display:
                # save images
                real_sample = real.cpu().data.numpy().squeeze()
                fake_sample = fake.cpu().data.numpy().squeeze()

                plt.cla()
                fig = plt.figure(figsize=(FIG_W, FIG_H))
                ax = fig.add_subplot(111)
                ax.set_facecolor('whitesmoke')
                ax.grid(True, color='white', linewidth=2)

                ax.spines["top"].set_visible(False)
                ax.spines["bottom"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.spines["left"].set_visible(False)
                ax.get_xaxis().tick_bottom()

                kde_num = 200
                min_real, max_real = min(real_sample), max(real_sample)
                kde_width_real = kde_num * \
                    (max_real - min_real) / args.eval_size
                min_fake, max_fake = min(fake_sample), max(fake_sample)
                kde_width_fake = kde_num * \
                    (max_fake - min_fake) / args.eval_size
                sns.kdeplot(
                    real_sample,
                    bw=kde_width_real,
                    label='Data',
                    color='green',
                    shade=True,
                    linewidth=6)
                sns.kdeplot(
                    fake_sample,
                    bw=kde_width_fake,
                    label='Model',
                    color='orange',
                    shade=True,
                    linewidth=6)

                ax.set_title(
                    f'True EM Distance: {w_distance_real}, '
                    f'Est. EM Distance: {w_distance_est}.',
                    fontsize=FONTSIZE)
                ax.legend(loc=2, fontsize=FONTSIZE)
                ax.set_ylabel('Estimated Density by KDE', fontsize=FONTSIZE)
                ax.tick_params(axis='x', labelsize=FONTSIZE * 0.7)
                ax.tick_params(
                    axis='y',
                    labelsize=FONTSIZE * 0.5,
                    direction='in')

                cur_img_path = os.path.join(image_path, str(niter) + '.jpg')
                plt.tight_layout()

                plt.savefig(cur_img_path)
                plt.close()

        return w_distance_real, w_distance_est

    def _gradient_penalty(self, real, fake):
        batch_size = fake.size(0)
        alpha = torch.rand(size=(batch_size, 1), device=self.config['device'])
        alpha = alpha.expand_as(real)
        interpolated = alpha * real + (1 - alpha) * fake
        interpolated = Variable(
            interpolated,
            requires_grad=True).to(
            self.config['device'])
        interpolation_loss = self.critic(interpolated)
        gradients = autograd.grad(
            outputs=interpolation_loss,
            inputs=interpolated,
            grad_outputs=torch.ones(
                interpolation_loss.size(),
                device=self.config['device']),
            create_graph=True,
            retain_graph=True)[0]

        gradients = gradients.view(gradients.size(0), -1)
        return ((gradients.norm(2, dim=1) - 1.) ** 2).mean()


if __name__ == '__main__':
    args = parser.parse_args()
    args.spect_norm = not args.no_spectral_norm
    if args.auto:
        args.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
    else:
        args.device = torch.device(
            f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')

    if args.auto_full:
        # Search over all tweaks, but don't search over clr parameters.
        # Further, since ResNet doesn't improve, don't search over it as well.
        config = {'prior': tune.choice(['uniform', 'gaussian']), 'prior_size': tune.choice([1, 3, 5]),
                  'hidden_size': tune.choice([64, 128, 256]), 'n_hidden': tune.choice([1, 2, 3, 4]),
                  'activation_slope': 1e-2, 'activation_fn': tune.choice(['relu', 'leakyrelu', 'tanh']),
                  'init_method': tune.choice(['default', 'xav_u']),

                  'lr': tune.choice([1e-5, 5e-5, 1e-4, 5e-4, 1e-3]),
                  'weight_decay': tune.choice([0., 1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 1e-3]),
                  'beta1': tune.choice([0.5, 0.6, 0.7, 0.8, 0.9]), 'beta2': tune.choice([0.7, 0.8, 0.9, 0.999]),

                  'k': tune.choice([1, 5, 10, 50, 100]), 'l': tune.choice([0, 1e-2, 1e-1, 1, 10]),

                  'spect_norm': tune.choice([1, 0]),
                  'norm': tune.choice(['batch', None]), 'dropout': tune.choice([1, 0]), 'clr': tune.choice([1, 0]),
                  }

    # Add constant params in args to config.
    dict_args = vars(args)
    for key in dict_args:
        if key in ['no_batch_norm', 'no_spectral_norm', 'batch_norm']:
            # redundant args.
            continue
        if key in config:
            if not args.auto:
                # In Manual experiment: overwrite existed config settings.
                config[key] = dict_args[key]
        else:
            config[key] = dict_args[key]

    if args.auto:
        if not args.clr:
            # Reset hyperparameter choices in clr if it is not a tuning field.
            config["clr_scale"] = 2
            config["clr_size_up"] = 2000
        if args.residual_block:
            # Set deeper depth of Resnet.
            config["n_hidden"] = tune.choice([1, 3, 5, 7])

    # save path
    search_type = 'automatic' if args.auto else 'manual'
    experiment = f'gu{args.gu_num}/wgan/{args.niters}|' + args.eval_real * 'w_distance_real|' + (
        args.eval_est) * 'w_distance_estimated|' + 'resnt|' * args.residual_block + 'fcnet|' * (
        not args.residual_block) + f'{args.prior}|' + f'clr|' * args.clr + f'dropout|' * args.dropout + f'{args.activation_fn}|' * (
        not args.auto) + f'{args.norm}_norm|' * (not args.auto) + (
                         'no_' * args.no_spectral_norm + 'spect_norm|') * (not args.auto) + (
        'no_' * (args.l != 0) + 'gradient_penalty|') * (
        not args.auto) + f'{args.init_method}_init|{args.k}_updates' * (not args.auto)

    model_path = os.path.join(curPath, search_type, 'models', experiment)
    image_path = os.path.join(curPath, search_type, 'images', experiment)

    if args.auto_full:
        model_path = os.path.join(
            curPath,
            search_type,
            f'models/gu{args.gu_num}/wgan/{args.niters}|full_new')
        image_path = os.path.join(
            curPath,
            search_type,
            f'images/gu{args.gu_num}/wgan/{args.niters}|full_new')

    makedirs(model_path, image_path)

    log_path = model_path + '/logs'
    logger = get_logger(log_path)

    logger.info('Trained model will save to: ' + model_path)
    logger.info('Result plot will save to : ' + image_path)
    logger.info('Search space: ')
    logger.info(config)
    logger.info(SEP)
    logger.info('Start training...')

    if args.auto:
        if args.eval_est:
            sched = ASHAScheduler(
                metric='w_distance_estimated',
                mode='min',
                grace_period=args.niters // 10,
                max_t=args.niters,
                time_attr="iteration")
        else:
            sched = ASHAScheduler(
                metric='w_distance_real',
                mode='min',
                grace_period=args.niters // 10,
                max_t=args.niters,
                time_attr="iteration")

        analysis = tune.run(WGANTrainer, name=experiment, scheduler=sched,  # search_alg=algo,
                            stop={"iteration": args.niters}, resources_per_trial={"cpu": 3, "gpu": 1},
                            num_samples=args.exp_num, checkpoint_at_end=True, config=config)

        if args.eval_real:
            best_config = analysis.get_best_config(
                metric='w_distance_real', mode='min')
            best_path = analysis.get_best_logdir(
                metric='w_distance_real', mode='min')
        else:
            best_config = analysis.get_best_config(
                metric='w_distance_estimated', mode='min')
            best_path = analysis.get_best_logdir(
                metric='w_distance_estimated', mode='min')

        results = analysis.dataframe()

        if args.eval_real:
            results.to_csv(model_path + f'results.csv')
        else:
            results.to_csv(model_path + f'results.csv')

        logger.info(f'Best config is: {best_config}')
        best_model_dir = retrieve_best_result_from_tune(best_path)

    else:

        trainer = WGANTrainer(config)
        for _ in range(1, config['niters'] + 1):
            _ = trainer._train()

        best_config = config
        best_model_dir = model_path
        logger.info(f'Saving to {model_path}')
        trainer._save(model_path)

    logger.info('Start evaluation...')

    eval_trainer = WGANTrainer(best_config)
    eval_trainer._restore(best_model_dir)
    eval_trainer._evaluate(display=True, niter=args.niters)

    logger.info('Saving to: ' + model_path)
    eval_trainer._save_whole(model_path)

    logger.info('Finish All...')
    logger.info(SEP)
