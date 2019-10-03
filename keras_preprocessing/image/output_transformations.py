import numpy as np


def transform_output(mode, values, dtype):
    transformations = {
        None: _raw,
        'sparse': _sparse
    }
    return transformations[mode](values, dtype)


def _raw(values, dtype):
    return values.astype(dtype)


def _sparse(labels, dtype):
    print(labels)
    classes = labels.unique()
    classes.sort()
    sorter = np.argsort(classes)
    values = sorter[np.searchsorted(classes, labels, sorter=sorter)]
    return values.astype(dtype), dict(zip(range(len(classes)), classes))
