import numpy as np

from .utils import img_to_array, load_img


def transform_output(mode, values):
    transformations = {
        None: _raw,
        'sparse': _sparse,
        'categorical': _categorical,
        'image': _image
    }
    return transformations[mode](values)


def _raw(values):
    return values.tolist()


def _sparse(labels):
    classes = __get_classes(labels)
    sorter = np.argsort(classes)
    values = sorter[np.searchsorted(classes, labels, sorter=sorter)]
    return values.tolist(), dict(zip(range(len(classes)), classes))


def _categorical(labels):
    classes = __get_classes(labels)
    sorter = np.argsort(classes)

    values = [sorter[np.searchsorted(classes, label, sorter=sorter)]
              for label in labels]
    values = [list(v) if not isinstance(v, np.integer)else v for v in values]
    return values, dict(zip(range(len(classes)), classes))


def _image(filepaths_series):
    return filepaths_series.tolist()


def __get_classes(labels):
    classes = set()
    for label in labels:
        if isinstance(label, str):
            classes.add(label)
        elif isinstance(label, (list, tuple)):
            classes.update(label)
    return sorted(classes)


def transform_batch(mode, values, index_array, dtype, **kwargs):
    batch_transformations = {
        None: _slice,
        'sparse': _slice,
        'categorical': _binarize,
        'image': _load_images
    }
    return batch_transformations[mode](values, index_array, dtype, **kwargs)


def _slice(values, index_array, dtype, **kwargs):
    return np.array(values, dtype=dtype)[index_array]


def _binarize(values, index_array, dtype, index_to_class, **kwargs):
    batch_y = np.zeros((len(index_array), len(index_to_class)), dtype=dtype)
    for i, n_observation in enumerate(index_array):
        batch_y[i, values[n_observation]] = 1
    return batch_y


def _load_images(filepaths,
                 index_array,
                 dtype,
                 image_shape,
                 color_mode,
                 target_size,
                 interpolation,
                 image_data_generator=None,
                 state=None,
                 inputs=None,
                 inputs_batch=None,
                 output_column=None,
                 **kwargs):
    if output_column:
        index = [i for i, _input in enumerate(inputs)
                 if _input['column'] == output_column]
        if index:
            batch = inputs_batch[index[0]].copy()
            return batch

    batch = np.zeros((len(index_array),) + image_shape, dtype=dtype)
    for i, j in enumerate(index_array):
        img = load_img(
            filepaths[j],
            color_mode=color_mode,
            target_size=target_size,
            interpolation=interpolation
        )
        x = img_to_array(img)
        # Pillow images should be closed after `load_img`,
        # but not PIL images.
        if hasattr(img, 'close'):
            img.close()
        if image_data_generator:
            x = __augment_image(x, image_data_generator, state)
        batch[i] = x
    return batch


def __augment_image(x, image_data_generator, state):
    params = image_data_generator.get_random_transform(x.shape, state=state)
    x = image_data_generator.apply_transform(x, params)
    x = image_data_generator.standardize(x)
    return x
