import os
from pathlib import Path
from typing import Tuple, List, Union

import SimpleITK as sitk
import numpy as np
import tensorflow as tf
from scipy.ndimage import zoom


def get_dataset(dataset_path: Path, out_shape: Tuple[int, int, int] = None,
                validation: bool = False) -> tf.data.Dataset:
    """
    Gets a dataset object to use while training.

    Args:
        dataset_path (Path): The path to the dataset.
        out_shape (Tuple[int, int, int], optional): The shape to resize to. Defaults to None.
        validation (bool, optional): To specify whether the validation dataset should be used.
        Defaults to False.

    Returns:
        tf.data.Dataset: The dataset.
    """

    # creating a dataset of the list of files to use
    dataset = tf.data.Dataset.list_files(
        str(dataset_path / '*GG/*/') if not validation else str(dataset_path / '*/')
    )

    # processing the file in the dataset
    dataset = dataset.map(process_paths_wrapper(out_shape))

    return dataset


def resize(img: np.ndarray, shape: Tuple[int, int, int], mode: str = 'constant') -> np.ndarray:
    """
    Wrapper for scipy.ndimage.zoom suited for MRI images.

    Args:
        img (np.ndarray): The image to resize.
        shape (Tuple[int, int, int]): The shape to resize to.
        mode (str, optional): The mode to use while resizing. Defaults to 'constant'.

    Returns:
        np.ndarray: The resized image.
    """
    assert len(shape) >= 3, "Output shape cannot have more than 3 dimensions"

    orig_shape = img.shape
    factors = (
        shape[0] / orig_shape[0],
        shape[1] / orig_shape[1],
        shape[2] / orig_shape[2]
    )

    # Resize to the given shape
    return zoom(img, factors, mode=mode)


def load_nii(paths: List[bytes]) -> np.ndarray:
    """
    Reads the .nii files specified by their paths and returns the data.

    Args:
        paths (List[bytes]): The list of paths to the files to read.

    Returns:
        np.ndarray: A numpy array containing the data of each individual file.
    """

    # creating the array to store the data
    data: Union[np.ndarray, None] = None

    # for every path in the list of given paths
    for i in range(len(paths)):

        # read the file
        d = sitk.GetArrayFromImage(sitk.ReadImage(paths[i].decode('utf-8')))

        # allocating the variable to store the data if it has not been allocated
        # already
        if data is None:
            data = np.zeros((len(paths), *d.shape), dtype=np.float32)

        # assigning the file contents to the data array
        data[i] = d

    # returning the data
    return data


def get_files(parent_dir: Union[str, bytes, Path]) -> Tuple[List[bytes], List[bytes]]:
    """
    Gets the contents of the directory.

    Args:
        parent_dir (bytes): The directory to get the contents of.

    Returns:
        Tuple[List[bytes], List[bytes]]: An array of the paths to the data files,
        and an array to the path of the segmentation mask (label).
    """
    directory: bytes
    if isinstance(parent_dir, str):
        directory = parent_dir.encode("ascii")
    elif isinstance(parent_dir, Path):
        directory = str(parent_dir).encode("ascii")
    else:
        directory = parent_dir

    # getting the list of files in the directory
    files = os.listdir(directory)

    # filtering out the data files from the list of files in the directory
    data_files = [os.path.join(directory, i)
                  for i in files if b"seg" not in i]

    # filtering out the segmentation mask file (label) from the list of files in the directory
    seg_file = [os.path.join(directory, i) for i in files if b"seg" in i]

    # returning the two
    return data_files, seg_file


def preprocess_data(images: np.ndarray, out_shape: Tuple[int, int, int] = None) -> np.ndarray:
    """
    Preprocess the input data.

    Args:
        images (np.ndarray): The list of images to preprocess
        out_shape (Tuple[int, int, int, int], optional): The output shape of the image, if resizing
        is required. Defaults to None.

    Returns:
        np.ndarray: The array containing the resultant images.
    """

    # creating the array to store the resultant images
    out_imgs = np.zeros((len(images), *out_shape), dtype=np.float32)

    # for every images
    for i, img in enumerate(images):

        # if there is a need to resize the image, resize the image.
        if out_shape is not None:
            _img = resize(img, out_shape)
        else:
            _img = img

        # normalizing the image
        mean = _img.mean()
        std = _img.std()
        out_imgs[i] = (_img - mean) / std

    return out_imgs


def preprocess_label(seg_mask: np.ndarray, out_shape: Tuple[int, int, int] = None, mode: str = 'nearest') -> np.ndarray:
    """
    Separates out the 3 labels from the segmentation provided, namely:
    GD-enhancing tumor (ET — label 4), the peritumoral edema (ED — label 2))
    and the necrotic and non-enhancing tumor core (NCR/NET — label 1)

    Args:
        seg_mask (np.ndarray): The numpy array containing the data of the label.
        out_shape (Tuple[int, int, int], optional): The shape to resize to. Defaults to None.
        mode (str, optional): The mode to use while resizing. Defaults to 'nearest'.

    Returns:
        np.ndarray: The resultant image.
    """

    # extracting the labels from the segmentation mask
    ncr = seg_mask == 1  # Necrotic and Non-Enhancing Tumor (NCR/NET)
    ed = seg_mask == 2  # Peritumoral Edema (ED)
    et = seg_mask == 4  # GD-enhancing Tumor (ET)

    # resizing if required
    if out_shape is not None:
        ncr = resize(ncr, out_shape, mode=mode)
        ed = resize(ed, out_shape, mode=mode)
        et = resize(et, out_shape, mode=mode)

    return np.array([ncr, ed, et], dtype=np.float32)


def process_paths(parent_dir: str, out_shape: Tuple[int, int, int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads the images from the parent directory, processes them and returns the processed images.

    Args:
        parent_dir (str): The path to the parent directory.
        out_shape (Tuple[int, int, int], optional): The shape to resize to. Defaults to None.

    Returns:
        Tuple[np.ndarray, np.ndarray]: First, the data, second, the labels.
    """

    data_files, seg_file = get_files(parent_dir)

    data: np.ndarray = load_nii(data_files)
    seg: np.ndarray = load_nii(seg_file)[0]

    data = preprocess_data(data, out_shape)
    label = preprocess_label(seg, out_shape)

    return data, label


def process_paths_wrapper(out_shape: Tuple[int, int, int]):
    def _process_paths_wrapper(parent_dir):
        data, label = tf.numpy_function(
            process_paths, [parent_dir, out_shape], [tf.float32, tf.float32])

        data.set_shape([None for _ in range(len(out_shape) + 1)])
        label.set_shape([None for _ in range(len(out_shape) + 1)])

        return data, (label, tf.identity(data))

    return _process_paths_wrapper
