import os
import logging
import shutil

from scipy.stats import wasserstein_distance
import numpy as np


def w_distance(real_tensor, fake_tensor):
    real = real_tensor.cpu().data.numpy().squeeze()
    fake = fake_tensor.cpu().data.numpy().squeeze()

    return np.round(wasserstein_distance(real, fake), 5)


def count_parameters(net):
    """from github: ffjord"""
    return sum(p.numel() for p in net.parameters() if p.requires_grad)


def get_logger(log_path):
    """from github: ffjord"""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    info_file_handler = logging.FileHandler(log_path, mode="a")
    info_file_handler.setLevel(logging.INFO)
    logger.addHandler(info_file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    logger.addHandler(console_handler)

    return logger


def makedirs(*dirnames):
    for dirname in dirnames:
        if not os.path.exists(dirname):
            os.makedirs(dirname)


def save_best_result_from_tune(best_model_dir, save_path):
    if os.path.exists(save_path):
        for net in os.listdir(best_model_dir):
            shutil.copy(os.path.join(best_model_dir, net), os.path.join(save_path, net))


def retrieve_best_result_from_tune(path):
    for file in os.listdir(path):
        if 'checkpoint' in file and os.path.isdir(os.path.join(path, file)):
            checkpoint_i = file
            break

    best_model_dir = os.path.join(path, checkpoint_i)
    return best_model_dir