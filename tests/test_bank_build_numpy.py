from __future__ import annotations

import numpy as np

from factory6g.components.raytracing import subcarrier_frequencies


def test_numpy_subcarrier_frequencies_match_sionna():
    from sionna.phy.channel import subcarrier_frequencies as sionna_freqs

    for n in (128, 127, 32):
        assert np.allclose(subcarrier_frequencies(n, 30e3), sionna_freqs(n, 30e3).numpy())


def test_components_package_imports_raytracing_without_tensorflow_on_import():
    import importlib
    import sys

    for mod in [m for m in sys.modules if m.startswith("factory6g.components")]:
        del sys.modules[mod]
    components = importlib.import_module("factory6g.components")
    assert "factory6g.components.channel" not in sys.modules
    importlib.import_module("factory6g.components.raytracing")
    assert "factory6g.components.channel" not in sys.modules
    assert components.ChannelModel.__name__ == "ChannelModel"  # still available on first use
