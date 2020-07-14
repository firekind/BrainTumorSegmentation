import logging

from tsalib import dim_vars

# initializing dimension variables
dim_vars("Batch(B) Width(W) Height(H) Depth(D) Channel(C)")

from pathlib import Path
from typing import Tuple, List, Union

import SimpleITK as sitk
import numpy as np
import tensorflow as tf
from flask import current_app

from model_server.models import NvNet
from model_server.utils.dataset import get_files
from model_server.utils.processing import prepare_data_for_overlay, load_and_preprocess, overlay, \
    load_and_preprocess_label, get_metrics

logger: logging.Logger = logging.getLogger()


def process_request(dir_path: Path, **kwargs) -> Union[np.ndarray, Tuple[Path, str]]:
    """
    Processes the request. Loads the data to the model and saves the result.
    Args:
        dir_path (Path): The path to the directory containing the data files.
        has to be sent for this to work.

    Returns:
        Union[np.ndarray, Tuple[Path, str]]: The path to the directory containing the the result
        and the name of the result file if save_output is True (which is the default), returns the result otherwise.
    """

    logger.info("Processing request.")

    if kwargs.get("check_integrity", False):
        _check_test_files_integrity(dir_path)

    # defining various variables
    input_shape: Tuple[int, int, int, int] = tuple(current_app.config["input_shape"])
    out_shape: Tuple[int, int, int, int] = tuple(current_app.config["output_shape"])
    model_path: str = current_app.config["model_path"]
    red_multiplier: Tuple[float, float, float] = tuple(current_app.config["red_multiplier"])
    yellow_multiplier: Tuple[float, float, float] = tuple(current_app.config["yellow_multiplier"])
    green_multiplier: Tuple[float, float, float] = tuple(current_app.config["green_multiplier"])
    overlay_threshold: float = current_app.config["overlay_threshold"]
    res_file: str = current_app.config["res_file_name"]

    # creating model and loading weights
    nvnet: tf.keras.Model = NvNet(input_shape=input_shape)
    nvnet.load_weights(model_path).expect_partial()
    logger.info("Network initialized. Pre-processing data...")

    # getting paths to data files (t1, t1c, t2, flair) and segmentation mask, if debug is true
    data_file_paths: List[bytes]
    seg_file_path: List[bytes]
    data_file_paths, seg_file_path = get_files(dir_path)

    # sorting the paths so that it will be in the following order: flair, t1, t1ce, t2
    data_file_paths.sort()

    # preparing data to be used as the base for the final overlay
    overlay_base = prepare_data_for_overlay(data_file_paths[0], out_shape=out_shape[1:])

    # pre-processing data
    inp = load_and_preprocess(data_file_paths, out_shape=out_shape[1:])
    logger.info("Data preprocessed. Beginning forward pass...")

    # forward prop
    decoder_out: tf.Tensor
    vae_out: tf.Tensor
    z_mean: tf.Tensor
    z_var: tf.Tensor
    decoder_out, vae_out, z_mean, z_var = nvnet(inp, training=False)
    logger.info("Forward pass complete. Generating results...")

    # applying overlay
    overlaid = overlay(overlay_base, decoder_out[0][0], red_multiplier, overlay_threshold)
    overlaid = overlay(overlaid, decoder_out[0][1], yellow_multiplier, overlay_threshold)
    overlaid = overlay(overlaid, decoder_out[0][2], green_multiplier, overlay_threshold)

    # if segmentation label was sent, calculating loss and accuracy and logging them
    if len(seg_file_path) != 0:
        logger.debug("Segmentation label file detected. Processing...")

        # loading and pre-processing label
        label = load_and_preprocess_label(seg_file_path[0])
        logger.debug("Preprocessed segmentation label. Calculating metrics...")

        # calculating metrics
        metrics = get_metrics(inp, label, decoder_out, vae_out, z_mean, z_var)
        logger.debug("decoder acc: %.2f, vae acc: %.2f, decoder loss: %.2f vae loss: %.2f", metrics["decoder_acc"],
                     metrics["vae_acc"], metrics["decoder_loss"], metrics["vae_loss"])

    # saving result
    if kwargs.get("save_output", True):
        if kwargs.get("as_numpy", True):
            logger.info("saving as numpy array")
            np.save(dir_path / res_file, overlaid)
        else:
            sitk.WriteImage(sitk.GetImageFromArray(overlaid), str(dir_path / res_file))
        logger.info("Results saved.")
        return dir_path, res_file

    return overlaid


def _check_test_files_integrity(transferred_file_path: Path):
    """
    Tests whether the sent files are same as the files in the directory specified by
    [compare_file] property of the config. Used to check whether the sent test files
    were corrupted somehow.

    Args:
        transferred_file_path: The path to the directory of the files that were transferred.

    Raises:
        AssertionError: If any of the files fail the test.
    """
    logger.debug("Running integrity test...")

    # defining function to load nii.gz file
    def load(path: str) -> np.ndarray:
        return sitk.GetArrayFromImage(sitk.ReadImage(path))

    # defining function to compare two numpy arrays
    def cmp(a: np.ndarray, b: np.ndarray) -> bool:
        return np.all(a == b)

    # defining function to get the path to a specific file given the type (t1, t2, t1ce, flair or seg)
    def get_file_path(paths: List[bytes], file_tp: bytes, is_transferred: bool = False) -> str:
        if is_transferred:
            return ([i for i in paths if file_tp + b".nii.gz" in i][0]).decode("utf-8")
        return ([i for i in paths if file_tp in i][0]).decode("utf-8")

    # defining the function that performs the test
    def test(t_paths: List[bytes], o_paths: List[bytes], file_tp: bytes):
        t_file = get_file_path(t_paths, file_tp, is_transferred=True)
        o_file = get_file_path(o_paths, file_tp)
        assert cmp(load(t_file), load(o_file)), f"{t_file} and {o_file} not similar"
        logger.debug("Comparing %s and %s...Passed", t_file, o_file)

    # getting files in path
    t_data_paths, t_seg_path = get_files(transferred_file_path)
    o_data_paths, o_seg_path = get_files(current_app.config["integrity_test_folder"])

    # testing data files
    for tp in [b"t1", b"t1ce", b"t2", b"flair"]:
        test(t_data_paths, o_data_paths, tp)
    logger.debug("Data files passed integrity test.")

    # testing seg files
    if len(t_seg_path) != 0:
        test(t_seg_path, o_seg_path, b"seg")
        logger.debug("Segmentation label file passed integrity test.")

    logger.debug("Integrity test complete.")
