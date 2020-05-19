from typing import Callable, Tuple

import tensorflow as tf
from tsalib import get_dim_vars

K = tf.keras.backend
B, C, H, W, D = get_dim_vars("B C H W D")


def dice_loss(e: float = 1e-8) -> 'Callable[[(B, C, H, W, D), (B, C, H, W, D)], tf.float32]':
    """
    Calculates the dice loss between the input and the predicted values.
    
    Args:
        e (float, optional): The coefficient to use to avoid zero division
        error. Defaults to 1e-8.
    
    Returns:
        Callable[[(B, C, H, W, D), (B, C, H, W, D), tf.float32]: Returns a function that
        takes in two tensors, the correct result and the predicted result. 
        This function is returned as keras does not allow a custom loss function
        to have arguments other than the correct result and the predicted result
    """

    def _dice_loss(y_true: (B, C, H, W, D), y_pred: (B, C, H, W, D)) -> tf.float32:
        # calculating the numerator
        intersection: (B, C) = 2 * K.sum(K.abs(y_true * y_pred), axis=[-3, -2, -1])

        # calculating the denominator
        denominator: (B, C) = K.sum(K.square(y_true) + K.square(y_pred), axis=[-3, -2, -1]) + e

        # dividing the two and taking mean across all the channels
        return -K.mean(intersection / denominator, axis=[0, 1])

    return _dice_loss


def l2_loss(y_true: (B, C, H, W, D), y_pred: (B, C, H, W, D)) -> (B,):
    """
    Returns the L2 loss between the arguments.
    
    Args:
        y_true ((B, C, H, W, D)): The actual output
        y_pred ((B, C, H, W, D)): The predicted output
    
    Returns:
        (B,): The L2 loss.
    """
    return K.mean(K.square(y_true - y_pred), axis=[1, 2, 3, 4])


def kl_div_loss(z_mean: (B, C, H, W, D), z_var: (B, C, H, W, D), n: int) -> tf.float32:
    """
    Returns the KL divergence loss between the two layers that represent the mean 
    and variance given as inputs and zero mean and unit variance.
    
    Args:
        z_mean (tf.keras.layers.Layer): The layer representing the means
        z_var (tf.keras.layers.Layer): The layer representing the variances
        n (int): The total number of image voxels
    
    Returns:
        tf.float32: The loss
    """
    # not using log according to the actual KL divergence formula as 
    # z_var has already undergone the log operation after the sampling
    return (1 / n) * K.sum(
        K.square(z_mean) + K.exp(z_var) - z_var - 1.
    )


def loss_VAE(
        input_shape: Tuple[int, int, int, int],
        weight_l2: float = 0.1,
        weight_kl: float = 0.1
) -> 'Callable[[(B, C, H, W, D), (B, C, H, W, D), (B, C, H, W, D), (B, C, H, W, D)], tf.float32]':
    """
    Returns the total loss for the variational autoencoder, that is, 
    weight_L2 * L2 loss + weight_KL * KL divergence loss.
    
    Args:
        input_shape (Tuple[int, int, int, int]): The shape of the input to the model.
        weight_l2 (float, optional): The weight to be applied for the L2 loss. Defaults to 0.1.
        weight_kl (float, optional): The weight to be applied for the KL divergence loss. Defaults to 0.1.
    
    Returns:
        Callable[[tf.Tensor, tf.Tensor], tf.Tensor]: A function that takes in the true labels and predicted
        values, and outputs the loss.
    """

    def _loss_VAE(y_true: (B, C, H, W, D), y_pred: (B, C, H, W, D), z_mean: (B, C, H, W, D),
                  z_var: (B, C, H, W, D)) -> tf.float32:
        _C, _H, _W, _D = input_shape
        return K.mean(weight_l2 * l2_loss(y_true, y_pred) + weight_kl * kl_div_loss(z_mean, z_var, _C * _H * _W * _D))

    return _loss_VAE
