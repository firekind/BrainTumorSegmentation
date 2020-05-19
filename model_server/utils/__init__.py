import os
from datetime import datetime

import tensorflow as tf

from .dataset import get_dataset


def sampling(args):
    """
    Re-parametrization trick by sampling from an isotropic unit Gaussian.
    From keras-team/keras/blob/master/examples/variational_autoencoder.py

    Args:
        args (tensor): mean and log of variance of Q(z|X)

    Returns:
        tensor: sampled latent vector
    """
    z_mean, z_var = args
    batch = tf.keras.backend.shape(z_mean)[0]
    dim = tf.keras.backend.int_shape(z_mean)[1]

    # by default, random_normal has mean = 0 and std = 1.0
    epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
    return z_mean + tf.keras.backend.exp(0.5 * z_var) * epsilon


class Checkpoint:
    def __init__(self, optimizer: tf.keras.optimizers.Optimizer, model: tf.keras.Model, directory: str = None,
                 max_to_keep: int = 5):
        if directory is None:
            directory = os.path.join("runs", datetime.now().strftime("%d_%m_%Y_%H_%M_%S"))

        self.checkpoint = tf.train.Checkpoint(epoch=tf.Variable(1), optimizer=optimizer, model=model)
        self.manager = tf.train.CheckpointManager(self.checkpoint, directory, max_to_keep)

    def restore(self, path: str = None):
        self.checkpoint.restore(self.manager.latest_checkpoint if path is None else path)

    def save(self) -> str:
        return self.manager.save()

    def get_save_counter(self) -> int:
        return int(self.checkpoint.save_counter)


__all__ = [sampling, get_dataset, Checkpoint]
