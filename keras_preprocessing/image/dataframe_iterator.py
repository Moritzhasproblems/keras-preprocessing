"""Utilities for real-time data augmentation on image data.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import warnings

import numpy as np

from .data_transformations import transform_output
from .iterator import BatchFromFilesMixin, Iterator
from .utils import validate_filename


class DataFrameIterator(BatchFromFilesMixin, Iterator):
    """Iterator capable of reading images from a directory on disk
        through a dataframe.

    # Arguments
        dataframe: Pandas dataframe containing the filepaths relative to
            `directory` (or absolute paths if `directory` is None) of the
            images in a string column. It should include other column/s
            depending on the `class_mode`:
            - if `class_mode` is `"categorical"` (default value) it must
                include the `y_col` column with the class/es of each image.
                Values in column can be string/list/tuple if a single class
                or list/tuple if multiple classes.
            - if `class_mode` is `"binary"` or `"sparse"` it must include
                the given `y_col` column with class values as strings.
            - if `class_mode` is `"raw"` or `"multi_output"` it should contain
                the columns specified in `y_col`.
            - if `class_mode` is `"input"` or `None` no extra column is needed.
        directory: string, path to the directory to read images from. If `None`,
            data in `x_col` column should be absolute paths.
        image_data_generator: Instance of `ImageDataGenerator` to use for
            random transformations and normalization. If None, no transformations
            and normalizations are made.
        x_col: string, column in `dataframe` that contains the filenames (or
            absolute paths if `directory` is `None`).
        y_col: string or list, column/s in `dataframe` that has the target data.
        weight_col: string, column in `dataframe` that contains the sample
            weights. Default: `None`.
        target_size: tuple of integers, dimensions to resize input images to.
        color_mode: One of `"rgb"`, `"rgba"`, `"grayscale"`.
            Color mode to read images.
        classes: Optional list of strings, classes to use (e.g. `["dogs", "cats"]`).
            If None, all classes in `y_col` will be used.
        class_mode: one of "binary", "categorical", "input", "multi_output",
            "raw", "sparse" or None. Default: "categorical".
            Mode for yielding the targets:
            - `"binary"`: 1D numpy array of binary labels,
            - `"categorical"`: 2D numpy array of one-hot encoded labels.
                Supports multi-label output.
            - `"input"`: images identical to input images (mainly used to
                work with autoencoders),
            - `"multi_output"`: list with the values of the different columns,
            - `"raw"`: numpy array of values in `y_col` column(s),
            - `"sparse"`: 1D numpy array of integer labels,
            - `None`, no targets are returned (the generator will only yield
                batches of image data, which is useful to use in
                `model.predict_generator()`).
        batch_size: Integer, size of a batch.
        shuffle: Boolean, whether to shuffle the data between epochs.
        seed: Random seed for data shuffling.
        data_format: String, one of `channels_first`, `channels_last`.
        save_to_dir: Optional directory where to save the pictures
            being yielded, in a viewable format. This is useful
            for visualizing the random transformations being
            applied, for debugging purposes.
        save_prefix: String prefix to use for saving sample
            images (if `save_to_dir` is set).
        save_format: Format to use for saving sample images
            (if `save_to_dir` is set).
        subset: Subset of data (`"training"` or `"validation"`) if
            validation_split is set in ImageDataGenerator.
        interpolation: Interpolation method used to resample the image if the
            target size is different from that of the loaded image.
            Supported methods are "nearest", "bilinear", and "bicubic".
            If PIL version 1.1.3 or newer is installed, "lanczos" is also
            supported. If PIL version 3.4.0 or newer is installed, "box" and
            "hamming" are also supported. By default, "nearest" is used.
        dtype: Dtype to use for the generated arrays.
        validate_filenames: Boolean, whether to validate image filenames in
        `x_col`. If `True`, invalid images will be ignored. Disabling this option
        can lead to speed-up in the instantiation of this class. Default: `True`.
    """
    allowed_class_modes = {
        'binary', 'categorical', 'input', 'multi_output', 'raw', 'sparse', None
    }

    def __init__(self,
                 dataframe,
                 input_columns,
                 output_columns=None,
                 weight_column=None,
                 output_modes=None,
                 image_size=(255, 255),
                 color_mode='rgb',
                 image_data_generator=None,
                 batch_size=32,
                 shuffle=True,
                 seed=None,
                 subset=None,
                 interpolation='nearest',
                 dtype='float32'):

        super(DataFrameIterator, self).set_processing_attrs(image_data_generator,
                                                            image_size,
                                                            color_mode,
                                                            subset,
                                                            interpolation)

        if isinstance(input_columns, str):
            input_columns = [input_columns]

        if isinstance(output_columns, str):
            output_columns = [output_columns]
        elif output_columns is None:
            output_columns = []

        dataframe = self._filter_valid_filepaths(dataframe, input_columns)
        if weight_column:
            self.weights = dataframe[weight_column].values.astype(dtype)
        else:
            self.weights = None
        self.dtype = dtype

        output_modes = output_modes or {}

        self.inputs = []
        for col in input_columns:
            self.inputs.append({
                'column': col,
                'values': dataframe[col].tolist()
            })

        self.outputs = []
        for col in output_columns:
            mode = output_modes.get(col)
            output_dict = {'column': col, 'mode': mode}
            output = transform_output(mode, dataframe[col])
            if mode in (None, 'image'):
                output_dict['values'] = output
            elif mode in ('sparse', 'categorical'):
                output_dict['values'] = output[0]
                output_dict['index_to_class'] = output[1]
            self.outputs.append(output_dict)

        super(DataFrameIterator, self).__init__(len(dataframe),
                                                batch_size,
                                                shuffle,
                                                seed)

    def _filter_valid_filepaths(self, df, columns):
        """Keep only dataframe rows with valid filenames

        # Arguments
            df: Pandas dataframe containing filenames in a column
            x_col: string, column in `df` that contains the filenames or filepaths

        # Returns
            absolute paths to image files
        """
        masks = np.full((len(df), len(columns)), True)
        for i, col in enumerate(columns):
            masks[:, i] = df[col].apply(
                validate_filename, args=(self.white_list_formats,)
            )
        mask = np.all(masks, axis=1)
        n_invalid = (~mask).sum()
        if n_invalid:
            warnings.warn(
                'Found {} rows with invalid image filenames. '
                'These rows will be ignored.'
                .format(n_invalid)
            )
        return df[mask]
