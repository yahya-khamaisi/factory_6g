"""Load weights trained with the TensorFlow stack into PyTorch modules (Phase 4).

Keras 3 archives (``.keras``) are zip files holding ``model.weights.h5``; layer
weights live under ``layers/<name>/vars/<i>``. Conv2D kernels are stored as
``[kh, kw, in, out]`` and Dense kernels as ``[in, out]``; PyTorch wants
``[out, in, kh, kw]`` and ``[out, in]``.
"""

from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path

import numpy as np


def _natural_key(name: str):
    m = re.match(r"(.*?)(?:_(\d+))?$", name)
    return (m.group(1), int(m.group(2) or 0))


def keras_layer_weights(path: str | Path, weights_name: str = "model.weights.h5") -> list[tuple[str, list[np.ndarray]]]:
    """Ordered (layer name, [vars]) for every layer with weights in a Keras 3 archive or bare .weights.h5 file."""
    import h5py

    path = Path(path)
    if path.suffix == ".keras":
        with zipfile.ZipFile(path) as z:
            handle = h5py.File(io.BytesIO(z.read(weights_name)), "r")
    else:
        handle = h5py.File(path, "r")
    with handle as f:
        root = f["layers"] if "layers" in f else f
        out = []
        for name in sorted(root.keys(), key=_natural_key):
            group = root[name]
            if "vars" not in group or not len(group["vars"]):
                continue
            vars_ = [np.asarray(group["vars"][k]) for k in sorted(group["vars"].keys(), key=int)]
            out.append((name, vars_))
        return out


def conv2d_from_keras(kernel: np.ndarray, bias: np.ndarray):
    import torch

    return torch.as_tensor(np.transpose(kernel, (3, 2, 0, 1)).copy()), torch.as_tensor(bias.copy())


def dense_from_keras(kernel: np.ndarray, bias: np.ndarray):
    import torch

    return torch.as_tensor(kernel.T.copy()), torch.as_tensor(bias.copy())
