import numpy as np


def transform_output(mode, values, dtype):
    transformations = {
        None: _raw,
        'sparse': _sparse,
        'categorical': _categorical
    }
    return transformations[mode](values, dtype)


def _raw(values, dtype):
    return values.astype(dtype)


def _sparse(labels, dtype):
    classes = __get_classes(labels)
    sorter = np.argsort(classes)
    values = sorter[np.searchsorted(classes, labels, sorter=sorter)]
    return values.astype(dtype), dict(zip(range(len(classes)), classes))


def _categorical(labels, dtype):
    classes = __get_classes(labels)
    sorter = np.argsort(classes)

    values = [sorter[np.searchsorted(classes, label, sorter=sorter)]
              for label in labels]
    return values, dict(zip(range(len(classes)), classes))


def __get_classes(labels):
    classes = set()
    for label in labels:
        if isinstance(label, str):
            classes.add(label)
        elif isinstance(label, (list, tuple)):
            classes.update(label)
    return sorted(classes)
