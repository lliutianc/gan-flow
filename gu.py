import numpy as np
import math

import torch
from torch.utils.data import Dataset


class GausUniffMixture:
    def __init__(
            self,
            n_mixture,
            mean_dist,
            sigma,
            unif_intsect,
            centralized=False,
            unif_ratio=1.,
            device="cpu",
            seed=1,
            extend_dim=False):
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.device = torch.device(device)

        self.n_mixture = n_mixture
        self.mean = np.cumsum(tuple(self.__cast(mean_dist)))
        if centralized:
            self.mean = self.mean - np.mean(self.mean)
        self.sigma = tuple(self.__cast(sigma))
        self.unif_intsect = tuple(self.__cast(unif_intsect))
        unif_ratio = self.__cast(unif_ratio)
        self.unif_p = tuple(unif_ratio / (unif_ratio + 1))

        self.extend_dim = extend_dim

    def get_sample(self, size):

        # prepare center(mean) for each cluster
        p = [1. / self.n_mixture for _ in range(self.n_mixture)]
        ith_center = np.random.choice(self.n_mixture, size, p=p)

        sample_points = np.zeros(1)
        for c in range(self.n_mixture):
            center = self.mean[c]
            sigma = self.sigma[c]
            unif_intsect = self.unif_intsect[c]
            unif_p = self.unif_p[c]

            ith_size = len(ith_center[ith_center == c])
            unif_size = np.random.binomial(n=ith_size, p=unif_p)
            unif_a, unif_b = center - unif_intsect * sigma, center + unif_intsect * sigma
            unif_sample = np.random.uniform(unif_a, unif_b, unif_size)
            gaus_sample = np.random.normal(
                loc=center, scale=sigma, size=(
                    ith_size - unif_size))
            sample_points = np.concatenate(
                [sample_points, unif_sample, gaus_sample])

        sample_points = sample_points[1:]
        np.random.shuffle(sample_points)
        dat = torch.from_numpy(sample_points.astype("float32")).to(self.device)
        dat = dat.view(size, -1)

        # In MAF/ RealNVP, univariate data may not be useful in trainning.
        if self.extend_dim:
            ext_dat = torch.rand(size=dat.size(), device=self.device)
            dat = torch.cat((dat, ext_dat), 1)

        return dat

    def density(self, x):
        p = [1. / self.n_mixture for _ in range(self.n_mixture)]

        x_density = 0.
        for i in range(self.n_mixture):
            center = self.mean[i]
            sigma = self.sigma[i]
            unif_intsect = self.unif_intsect[i]
            unif_p = self.unif_p[i]
            unif_a, unif_b = center - unif_intsect * sigma, center + unif_intsect * sigma

            # Calculate ith uniform density at x
            unif_density = 1 / (unif_b - unif_a) * \
                (x <= unif_b) * (x >= unif_a)

            # Calculate ith gaussian density at x
            gaus_density = np.exp(- ((x - center) / sigma)
                                  ** 2 / 2) / (np.sqrt(2 * math.pi) * sigma)
            i_density = gaus_density * (1 - unif_p) + unif_density * unif_p

            x_density += i_density * p[i]

        return x_density

    def __cast(self, val):
        if isinstance(val, (int, float)):
            return np.array([val for _ in range(self.n_mixture)])
        else:
            return np.array(val)


class GU (Dataset):
    def __init__(self, dataset_size=100000, gu_num=8, device="cpu"):
        """ Create a dataset based on Gaussian Uniform Mixture，
            This is far more slower so is not recommended to use. """
        if gu_num == 8:
            gu_std = 2
            n_mix = 8
            mean_dist = 10
            uc = 1.5
            ur = 1
        else:
            gu_std = 1 / 10
            n_mix = gu_num
            mean_dist = 5
            uc = 5
            ur = 3

        self.device = torch.device(device)
        self.gu = GausUniffMixture(
            n_mixture=n_mix,
            mean_dist=mean_dist,
            sigma=gu_std,
            unif_intsect=uc,
            unif_ratio=ur,
            centralized=False,
            device=self.device,
            seed=1)
        self.input_size = 2
        self.label_size = 1
        self.dataset_size = int(dataset_size)

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, i):
        real = self.gu.get_sample(1)
        extra_real = torch.rand(size=real.size(), device=self.device)
        real = torch.cat((real, extra_real), 1)
        return real, torch.zeros(self.label_size)
