from typing import List, Tuple

import numpy as np
import skimage.color
import skimage.transform
from flask import current_app
from tsalib import get_dim_vars

import model_server.utils.losses as L
import model_server.utils.metrics as M
from model_server.utils.dataset import load_nii, preprocess_data, preprocess_label

B, C, H, W, D = get_dim_vars("B C H W D")


def prepare_data_for_overlay(path: bytes, out_shape: Tuple[int, int, int]) -> np.ndarray:
    """
    Prepares an MRI scan input image so that the mask can be
    overlaid on it.
    Args:
        path (bytes): The path to the scan
        out_shape (Tuple[int, int, int): The output shape of the image

    Returns:
        np.ndarray: The process input image.
    """

    # loading the image
    data = load_nii([path])[0]

    # resizing the image
    data = skimage.transform.resize(data, output_shape=out_shape)

    # converting to rgb image
    data = skimage.color.gray2rgb(data)

    # scaling the image to have values between 0 and 1
    data = data / data.max()

    # returning
    return data


def load_and_preprocess(paths: List[bytes], out_shape: Tuple[int, int, int]) -> np.ndarray:
    """
    Loads the data files and prepossesses them.
    Args:
        paths (List[str]): List containing the paths to the input MRI scans.
        (t1, t1c, t2 and flair)
        out_shape: The shape of each image / scan.

    Returns:
        np.ndarray: The preprocessed input.
    """

    # loading the nii files
    inp: np.ndarray = load_nii(paths)

    # prepossessing data
    processed_data: np.ndarray = preprocess_data(inp, out_shape=out_shape)

    # adding the batch dimension so tensorflow is happy and returning the
    # result.
    return np.expand_dims(processed_data, axis=0)


def load_and_preprocess_label(path: bytes) -> np.ndarray:
    """
    Loads and pre-processes the label.
    Args:
        path (bytes): The path to the label.

    Returns:
        np.ndarray: The pre-processed label.
    """

    label: (1, H, W, D) = load_nii([path])
    return preprocess_label(label[0], out_shape=(80, 96, 64))


def overlay(image: np.ndarray, mask: np.ndarray, multiplier: Tuple[float, float, float],
            threshold: float = 1.0) -> np.ndarray:
    """
    Overlays the segmentation mask over the image.

    Args:
        image (np.ndarray): The image on which the segmentation mask should be applied.
        mask (np.ndarray): The segmentation mask.
        multiplier (Tuple[int, int, int, int]: The color of the mask that needs to be applied.
        threshold (int): The value above which the values of the mask will be considered.

    Returns:
        np.ndarray: The overlaid image.
    """

    # getting the indices where the color has to be applied
    indices = np.argwhere(mask >= threshold)

    # creating a copy of the image
    out = np.copy(image)

    # applying the color to the image at the locations specified by [indices]
    for i in range(len(indices)):
        out[indices[i][0], indices[i][1], indices[i][2]] = multiplier

    # returning the result
    return out


def get_metrics(data, label, decoder_out, vae_out, z_mean, z_var):
    vae_loss_fn = L.loss_VAE(input_shape=current_app.config["input_shape"])
    decoder_loss_fn = L.dice_loss()
    acc_fn = M.dice_coefficient

    decoder_loss = float(decoder_loss_fn(label, decoder_out))
    vae_loss = float(vae_loss_fn(data, vae_out, z_mean, z_var))
    decoder_acc = float(acc_fn(label, decoder_out))
    vae_acc = float(acc_fn(data, vae_out))

    return {
        "decoder_loss": decoder_loss,
        "vae_loss": vae_loss,
        "decoder_acc": decoder_acc,
        "vae_acc": vae_acc
    }
