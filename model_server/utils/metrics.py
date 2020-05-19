import tensorflow as tf
from tsalib import get_dim_vars

K = tf.keras.backend
B, C, H, W, D = get_dim_vars("B C H W D")


def dice_coefficient(y_true: (B, C, H, W, D), y_pred: (B, C, H, W, D)) -> tf.float32:
    """
    Calculates the dice coefficient used for metrics.
    
    Args:
        y_true ((B, C, H, W, D)): The true values of the output
        y_pred ((B, C, H, W, D)): The predicted values of the model
    
    Returns:
        tf.float32: The dice coefficient
    """
    # calculating the numerator
    # noinspection PyTypeChecker
    intersection: (B, C) = K.sum(K.abs(y_true * y_pred), axis=[-3, -2, -1])

    # calculating the denominator
    denominator: (B, C) = K.sum(K.square(y_true) + K.square(y_pred), axis=[-3, -2, -1]) + 1e-8

    # dividing the two and taking mean across all the channels
    return K.mean(2 * intersection / denominator, axis=[0, 1])
